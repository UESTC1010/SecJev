from paths import WORK,SCRIPTS,SOURCE,BASE
"""Frozen original Kev release tests; no checkpoint updates or temperature fitting."""
import os, sys, json, time, pathlib, subprocess, traceback, collections, shutil
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
SRC = pathlib.Path(str(SOURCE))
sys.path.insert(0, str(SRC))
import numpy as np
from kev.suite import load_split, read_manifest, digest, write_json
from kev.benchmark import evaluate_records
from kev.metrics import metrics, grouped_metrics, paired_bootstrap
ROOT = pathlib.Path(str(WORK / 'retention'))
SUITES = {'decision-v7': SRC / 'evals/v7/decision-v7', 'transfer-v4': SRC / 'evals/v4/transfer-v4'}
MODELS = {}

def hashes(path):
    return {p.name: digest(p) for p in pathlib.Path(path).iterdir() if p.is_file()}

def worker(name):
    import torch
    from kev.predictors import LocalPredictor
    from kev.checkpoint import Checkpoint, LoadOptions
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    ck = Checkpoint(MODELS[name])
    ck.meta.base = str(BASE)
    ck.meta.base_revision = None
    predictor = LocalPredictor.__new__(LocalPredictor)
    predictor.run = ck.path
    predictor.device = 'cuda'
    predictor.tok, predictor.model = ck.load('cuda', LoadOptions(dtype=torch.float32, merge=False, attn='sdpa'))
    predictor.temperature = predictor.model.head.temperature
    for suite, path in SUITES.items():
        records = load_split(path, 'test', allow_test=True)
        out = ROOT / name / suite
        if (out / 'report.json').exists() and (out / 'extended.json').exists():
            continue
        if (out / 'report.json').exists() and (out / 'rows.json').exists():
            report = json.loads((out / 'report.json').read_text())
            rows = json.loads((out / 'rows.json').read_text())
            assert report['coverage']['evaluated_questions'] == sum((len(r['questions']) for r in records)) == len(rows)
        else:
            if out.exists():
                out.rename(out.with_name(out.name + f'.interrupted-{time.time_ns()}'))
            report, rows = evaluate_records(records, predictor, out, heldout_sources=read_manifest(path)['holdout_sources'])
        report.update(suite_sha256=digest(path / 'manifest.json'), test_sha256=digest(path / 'test.jsonl'), split='test', model=name, date_facts=False)
        write_json(out / 'report.json', report)
        raw = []
        for r in rows:
            z = np.array(r['logits']) * predictor.temperature
            p = np.exp(z - z.max())
            p /= p.sum()
            raw.append({**r, 'p': p.tolist(), 'logits': z.tolist(), 'inference_temperature': 1.0})
        clean = [r for r in rows if r['variant'] == 'clean']
        raw_clean = [r for r in raw if r['variant'] == 'clean']
        write_json(out / 'extended.json', {'raw_clean': metrics(raw_clean), 'as_shipped_clean': metrics(clean), 'sources': grouped_metrics(clean, 'source'), 'types': grouped_metrics(clean, 'type')})
        print('COMPLETE', name, suite, 'accuracy', report['clean']['acc'], flush=True)
