"""Wait for this run, export portable weights, verify serving parity and build reports.

No publication credentials are used; upload remains a separate operation.
"""
import os, sys, pathlib, json, time, shutil, hashlib, copy, gzip, statistics, traceback
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.path.insert(0,str(pathlib.Path(os.environ.get('SECJEV_WORK', './secjev-work')).resolve()/'source/kev'))
WORK = pathlib.Path(os.environ.get('SECJEV_WORK', './secjev-work')).resolve()
R = WORK/'continuation'
RUN = R/'runs/secjev-0.8b-3epochs'
DEST = R/'release/SecJev-0.8B'
BASE = str(WORK/'base')
PROJECT = pathlib.Path(__file__).resolve().parent.parent


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    tmp.replace(path)


def state(stage, **kwargs):
    write(R/'results/release-status.json', {'stage': stage, 'updated_unix': time.time(), 'uploaded': False, **kwargs})


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(x) for x in value]
    if isinstance(value, str) and pathlib.Path(value).is_absolute():
        return 'local-training-path'
    return value


def main():
    state('waiting_for_evaluation')
    while True:
        p = R/'results/pipeline-status.json'
        status = json.loads(p.read_text()) if p.exists() else {}
        if status.get('stage') == 'failed':
            raise RuntimeError('Training/evaluation pipeline failed; see pipeline-status.json')
        if status.get('stage') == 'evaluation_complete':
            break
        time.sleep(60)
    import torch
    from kev.checkpoint import Checkpoint, LoadOptions, write_meta
    from kev.api import SystemOneRequest, to_record
    from kev.data import api_request
    from kev.model import rows_of
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    frozen = json.loads((R/'results/frozen-final-candidate.json').read_text())
    src = RUN/'calibrated'
    assert all(hashlib.sha256((src/p).read_bytes()).hexdigest() == h for p, h in frozen['files'].items())
    assert not DEST.exists(), 'Do not overwrite an existing release candidate'
    shutil.copytree(src, DEST)
    state('export_and_serving_validation')
    checkpoint = Checkpoint(DEST)
    original = copy.deepcopy(checkpoint.meta)
    checkpoint.meta.base = 'Qwen/Qwen3.5-0.8B-Base'
    checkpoint.meta.base_revision = 'dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68'
    checkpoint.meta.holdout = []
    checkpoint.meta.extra = {
        'args': {'lora_targets': original.extra['args']['lora_targets']},
        'name': 'SecJev-0.8B', 'version': '1.0.0',
        'checkpoint_selection': original.extra.get('checkpoint_selection', {}),
        'security_training': clean(original.extra['security_training']) | {'initial_checkpoint': 'jaredpalmer/kev-0.8b@54f4f8777356cd5bbbb6c6919c657f26e6f2f6d8'},
        'temperature_fit': original.extra['temperature_fit'],
        'corpus': {'name': 'SecJev-Corpus', 'version': '1.0.0', 'train_questions': 118818},
    }
    write_meta(DEST, checkpoint.meta)
    exported = Checkpoint(DEST).meta
    assert original.temperature == exported.temperature
    assert all(torch.equal(original.head[k], exported.head[k]) for k in original.head)
    assert hashlib.sha256((DEST/'adapter_model.safetensors').read_bytes()).digest() == hashlib.sha256((src/'adapter_model.safetensors').read_bytes()).digest()
    for p in DEST.glob('*.json'):
        data = clean(json.loads(p.read_text()))
        if p.name == 'adapter_config.json':
            data['base_model_name_or_path'] = checkpoint.meta.base
            data['revision'] = checkpoint.meta.base_revision
        write(p, data)
    # Reference logits were evaluated as isolated question rows. Serving handles full typed requests.
    reference = {}
    with (R/'results/trained-calibration-logits.jsonl').open() as f:
        for line in f:
            row = json.loads(line)
            reference[row['case'], row['qid']] = row['logits']
    requests = []
    with gzip.open(R/'data/calibration.jsonl.gz', 'rt') as f:
        for case, line in enumerate(f):
            rec = json.loads(line)
            requests.append((case, rec))
    selected = {}
    for kind in ['noul', 'choice', 'score']:
        candidates = [(i, rec) for i, rec in requests if any(q['type'] == kind for q in rec['questions'].values())]
        candidates.sort(key=lambda pair: len(json.dumps(pair[1]['state'])))
        for index in [0, len(candidates)//2, len(candidates)-1]:
            i, rec = candidates[index]
            selected[i] = rec
    checkpoint = Checkpoint(DEST)
    checkpoint.meta.base, checkpoint.meta.base_revision = BASE, None
    tokenizer, model = checkpoint.load('cuda:0', LoadOptions(dtype=torch.float32, merge=False, attn='sdpa'))
    max_error = 0.; runs = []
    with torch.inference_mode():
        for case, request in sorted(selected.items()):
            rec, metadata = to_record(SystemOneRequest.model_validate(api_request(request)))
            enc = model.encode(tokenizer, rec, max_state=8192, max_branch=8192, strict=True)
            logits = model.forward(enc)
            for m, z in zip(metadata, logits):
                observed = torch.softmax(z.double().cpu(), -1)
                expected = torch.softmax(torch.tensor(reference[case, m['id']], dtype=torch.float64)/original.temperature, -1)
                max_error = max(max_error, float((observed-expected).abs().max()))
            timings = []; torch.cuda.reset_peak_memory_stats()
            for _ in range(5):
                torch.cuda.synchronize(); start = time.perf_counter()
                probs = [z.softmax(-1).cpu().tolist() for z in model.forward(enc)]
                torch.cuda.synchronize(); timings.append((time.perf_counter()-start)*1000)
            prefix, _, branches = rows_of(enc)
            runs.append({'calibration_scene': case, 'questions': len(metadata),
                'max_question_row_tokens': max(len(prefix)+len(b['ids']) for b in branches),
                'request_packed_tokens': len(enc['ids']), 'forward_ms': timings,
                'median_ms': statistics.median(timings), 'max_ms': max(timings),
                'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20})
    assert max_error < 1e-3, ('Serving parity failed', max_error)
    parity = {'pass': True, 'max_probability_difference': max_error, 'tolerance': 1e-3,
              'temperature': original.temperature, 'adapter_bytes_unchanged': True, 'head_tensors_unchanged': True,
              'comparison': 'Public export full requests versus frozen evaluation isolated-question FP32 logits',
              'gpu': torch.cuda.get_device_name(0), 'dtype': 'FP32', 'adapter': 'unmerged',
              'latency_scope': 'Five warm forward passes per selected calibration scene; excludes load and tokenization; no other SecJev inference on the same GPU',
              'scenes': runs}
    write(R/'results/export-validation.json', parity)
    del model; torch.cuda.empty_cache()
    reportdir = R/'release/evaluation'; reportdir.mkdir()
    reports = {}
    for name in ['parent-validation', 'trained-validation', 'parent-test', 'trained-test', 'export-validation']:
        report = clean(json.loads((R/'results'/f'{name}.json').read_text()))
        reports[name] = report
        write(reportdir/f'{name}.json', report)
    selection = clean(json.loads((R/'results/checkpoint-selection.json').read_text()))
    write(DEST/'checkpoint-selection.json',selection)
    write(reportdir/'checkpoint-selection.json',selection)
    config = clean(json.loads((RUN/'training_config.json').read_text()))
    write(DEST/'training_config.json', config)
    runtime = json.loads((R/'results/runtime.json').read_text())
    write(DEST/'provenance.json', {'base_model': checkpoint.meta.base if not checkpoint.meta.base.startswith('/') else 'Qwen/Qwen3.5-0.8B-Base',
        'base_revision': 'dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68',
        'init_model': 'jaredpalmer/kev-0.8b', 'init_revision': '54f4f8777356cd5bbbb6c6919c657f26e6f2f6d8',
        'kev_source_revision': '90990a5fac2995b9faa3190f7d437e84f2067768',
        'trainable_parameters': runtime['trainable_parameters'], 'frozen_parameters': runtime['frozen_parameters'],
        'run': 'secjev-0.8b-3epochs', 'corpus': 'SecJev-Corpus v1.0.0', 'temperature': original.temperature})
    parent = reports['parent-test']['test']['recalibrated']; trained = reports['trained-test']['test']['recalibrated']
    lines = ['# SecJev-0.8B evaluation', '', 'Checkpoint selected among the three epoch-end candidates using full-development macro-task balanced accuracy. Both models are independently calibrated on the calibration split; test data does not choose the checkpoint or temperature.', '',
             '| Metric | Kev-0.8B parent | SecJev-0.8B |', '|---|---:|---:|']
    for key in ['accuracy', 'macro_task_accuracy', 'macro_task_balanced_accuracy', 'nll', 'brier', 'ece15']:
        lines.append(f'| {key} | {parent[key]:.6f} | {trained[key]:.6f} |')
    lines += ['', '| Task | Questions | Parent accuracy | SecJev accuracy | Parent balanced accuracy | SecJev balanced accuracy |', '|---|---:|---:|---:|---:|---:|']
    for key, t in trained['tasks'].items():
        p = parent['tasks'][key]
        lines.append(f"| {key} | {t['questions']} | {p['accuracy']:.4%} | {t['accuracy']:.4%} | {p['mean_class_recall']:.4%} | {t['mean_class_recall']:.4%} |")
    lines += ['', 'Accuracy counts individual questions; macro-task metrics give each task equal weight. Balanced accuracy averages recalls of semantic labels, not shuffled option positions. Scores are distributions over source-defined ordinal labels, not universal security risk scores.', '',
              'The research split was inspected during dataset construction and is not a blind external benchmark. Controlled Byzantine/Sybil experiments support claims within those experiments. General-domain retention and deployment robustness were not evaluated. No confidence intervals are claimed.', '',
              'See export-validation.json for warm forward-pass measurements and validation of the public artifact. The actual context range and question counts accompany every timed scene.']
    (reportdir/'README.md').write_text('\n'.join(lines)+'\n')
    shutil.copy2(PROJECT/'infer.py', DEST/'infer.py')
    for name in ['LICENSE', 'LICENSE.zh-CN.md', 'THIRD_PARTY_NOTICES.md', 'SOURCES.md', 'UPSTREAM.md']:
        shutil.copy2(PROJECT/name, DEST/name)
    shutil.copytree(PROJECT/'licenses', DEST/'licenses')
    (DEST/'README.md').write_text('''---
language:
- en
license: other
license_name: secjev-use-and-distribution-agreement-v1.0
license_link: LICENSE
base_model: Qwen/Qwen3.5-0.8B-Base
tags:
- security
- decision-model
- lora
---

# SecJev-0.8B

**System One for Security.** A Kev-based decision model adapted on SecJev-Corpus v1.0.0. It maps security evidence and explicit criteria into judgments, choices and ordinal assessments.

This release contains a LoRA adapter, pointer decision head and calibration metadata. The pinned Qwen3.5-0.8B-Base backbone is loaded separately. It scores alternatives without autoregressive answer generation.

Code and usage: https://github.com/UESTC1010/SecJev

The adapter and head originate from jaredpalmer/kev-0.8b, with security adaptation selected on the full development split after three training epochs. See checkpoint-selection.json for the selected epoch and all candidate scores. See provenance.json and training_config.json for exact revisions and parameters. No general-domain replay is used.

## Evaluation

'''+ '\n'.join(lines[2:]) + '''

## Data and terms

The training corpus includes CICIoT2023, ToN-IoT, Twins, VeReMi, ByzFL-based experiments, LANL, AgentDojo and InjecAgent. See SecJev-Corpus/SOURCES.md for attribution and source-specific terms. ToN-IoT's source data is available for academic research; its authors require permission for commercial use. The SecJev Use and Distribution Agreement covers original SecJev contributions and preserves upstream rights and conditions. See LICENSE, LICENSE.zh-CN.md, THIRD_PARTY_NOTICES.md and licenses/. It does not establish clearance for every commercial use.
''')
    for name in ['parent-test.json', 'trained-test.json', 'export-validation.json']:
        shutil.copy2(reportdir/name, DEST/name)
    hashes = {str(p.relative_to(DEST)): hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.rglob('*') if p.is_file()}
    write(DEST/'files.json', hashes)
    assert all(hashlib.sha256((src/p).read_bytes()).hexdigest() == h for p, h in frozen['files'].items())
    state('ready_for_publication', model='release/SecJev-0.8B', evaluation='release/evaluation',
          test_accuracy=trained['accuracy'], parent_test_accuracy=parent['accuracy'],
          serving_parity_max_error=max_error)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        state('failed', error=str(error), traceback=traceback.format_exc())
        raise
