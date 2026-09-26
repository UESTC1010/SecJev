from paths import WORK,SCRIPTS,SOURCE,BASE
"""Nine independent GPU workers; original three-shard security batching retained."""
import sys, os, pathlib, json, shutil, time, traceback, subprocess, hashlib, pickle, math, fcntl
sys.path.insert(0, str(SCRIPTS))
from eval_support import validate_logits, load_rows, metrics, fit, write, file_hashes, call
from kev.checkpoint import Checkpoint, write_meta
from kev.suite import load_split
import torch
R = pathlib.Path(str(WORK))
RUN = R / 'runs/secjev-2b-3epochs'
CACHE = {}
LOCK = None

def stage(name, **kw):
    write(R / 'results/pipeline-status.json', {'stage': name, 'updated_unix': time.time(), 'gpu_workers': 9, **kw})

def report(rows, t):
    return {'raw': metrics(rows, 1.0), 'recalibrated': metrics(rows, t)}

def model_files(path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in pathlib.Path(path).iterdir() if p.is_file() and p.name != 'optimizer.pt'}

def expected(split):
    if split not in CACHE:
        CACHE[split] = sorted(pickle.load((R / 'data' / f'{split}.pkl').open('rb')), key=lambda r: r['length'])
    return CACHE[split]

def valid_part(tag, split, shard):
    rows = load_rows(R / 'results' / f'{tag}-{split}-shard-{shard}.jsonl')
    wanted = expected(split)[shard::3]
    assert len(rows) == len(wanted)
    for x, y in zip(rows, wanted):
        assert all((x[k] == y[k] for k in ['case', 'qid', 'source', 'task', 'qtype', 'label', 'semantic_label', 'option_keys']))
        assert len(x['logits']) == y['k'] and all((math.isfinite(v) for v in x['logits']))
    return rows

def security_jobs(checkpoint, tag, split):
    fingerprint = model_files(checkpoint)
    meta = R / 'results' / f'{tag}-{split}-inference-source.json'
    if meta.exists():
        assert json.load(open(meta))['model_files'] == fingerprint, 'Different source for cached logits'
    else:
        write(meta, {'model_files': fingerprint, 'split': split, 'shards': 3, 'precision': 'FP32 unmerged; no TF32/fused SDPA'})
    jobs = []
    for shard in range(3):
        if (R / 'results' / f'{tag}-{split}-shard-{shard}.jsonl').exists():
            valid_part(tag, split, shard)
            continue
        jobs.append({'name': f'{tag}-{split}-{shard}', 'kind': 'security', 'tag': tag, 'split': split, 'shard': shard, 'args': [sys.executable, str(SCRIPTS / 'infer_shard.py'), '--checkpoint', str(checkpoint), '--tag', tag, '--split', split, '--shard', str(shard), '--shards', '3']})
    return jobs

def security_rows(tag, split):
    rows = []
    for shard in range(3):
        rows.extend(valid_part(tag, split, shard))
    rows = validate_logits(rows, split)
    p = R / 'results' / f'{tag}-{split}-logits.jsonl'
    tmp = p.with_suffix('.tmp')
    with tmp.open('w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    tmp.replace(p)
    return rows

def valid_general(tag, suite):
    p = R / 'general' / tag / suite
    rows = json.load(open(p / 'rows.json'))
    ext = json.load(open(p / 'extended.json'))
    rep = json.load(open(p / 'report.json'))
    src = pathlib.Path(str(SOURCE / 'evals')) / ('v7/decision-v7' if suite == 'decision-v7' else 'v4/transfer-v4')
    reqs = load_split(src, 'test', allow_test=True)
    assert len(rows) == sum((len(r['questions']) for r in reqs))
    assert len({(r['id'], r['question']) for r in rows}) == len(rows)
    assert rep['split'] == 'test' and rep['model'] == tag and math.isfinite(ext['raw_clean']['acc'])
    assert all((all((math.isfinite(v) for v in r['p'])) for r in rows))

def general_jobs(tag, checkpoint):
    jobs = []
    for suite in ['decision-v7', 'transfer-v4']:
        p = R / 'general' / tag / suite
        if (p / 'extended.json').exists():
            valid_general(tag, suite)
            continue
        jobs.append({'name': f'general-{tag}-{suite}', 'kind': 'general', 'tag': tag, 'suite': suite, 'args': [sys.executable, str(SCRIPTS / 'general_eval.py'), tag, suite]})
    return jobs

def run_jobs(jobs, phase):
    pending = list(jobs)
    active = {}
    done = []
    errors = []
    started = time.time()

    def progress():
        write(R / 'results/evaluation-progress.json', {'phase': phase, 'total_jobs': len(jobs), 'completed_jobs': len(done), 'pending_jobs': len(pending), 'active': {str(g): v[2]['name'] for g, v in active.items()}, 'errors': errors, 'elapsed_s': time.time() - started, 'updated_unix': time.time()})
    while pending or active:
        if (R / 'PAUSE_EVALUATION').exists():
            raise RuntimeError('Pause marker appeared; controller must be stopped with workers')
        for gpu in range(9):
            if gpu in active or not pending or errors:
                continue
            job = pending.pop(0)
            log = R / 'logs' / f"{job['name']}.log"
            if log.exists():
                log.rename(log.with_name(log.name + f'.previous-{time.time_ns()}'))
            f = log.open('w')
            p = subprocess.Popen(job['args'], stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7,8').split(',')[gpu]))
            active[gpu] = (p, f, job)
            print('START', gpu, job['name'], flush=True)
        for gpu, (p, f, job) in list(active.items()):
            rc = p.poll()
            if rc is None:
                continue
            f.close()
            (R / 'logs' / f"{job['name']}.exit").write_text(str(rc) + '\n')
            del active[gpu]
            if rc:
                errors.append({'job': job['name'], 'exit_code': rc})
            else:
                try:
                    if job['kind'] == 'security':
                        valid_part(job['tag'], job['split'], job['shard'])
                    else:
                        valid_general(job['tag'], job['suite'])
                    done.append(job['name'])
                    print('DONE', gpu, job['name'], flush=True)
                except Exception as e:
                    errors.append({'job': job['name'], 'validation_error': str(e)})
        progress()
        if errors and (not active):
            break
        if active:
            time.sleep(2)
    assert not errors, errors

def main():
    global LOCK
    LOCK = (R / 'evaluation.lock').open('a')
    fcntl.flock(LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (R / 'PAUSE_EVALUATION').exists()
    training = json.load(open(R / 'results/train-status.json'))
    assert training['stage'] == 'complete' and training['epochs'] == 3 and (training['steps'] == 2167)
    paths = {f'epoch-{e}': RUN / f'epoch-{e}' for e in [1, 2, 3]}
    paths['kev-stage1'] = R / 'runs/kev-2b-delta/epoch-1'
    for p in paths.values():
        for n in ['adapter_model.safetensors', 'head.pt']:
            assert (p / n).stat().st_size > 0
    initial = {name: model_files(p) for name, p in paths.items()}
    prov = R / 'results/evaluation-resume-source.json'
    if prov.exists():
        assert json.load(open(prov))['models'] == initial
    else:
        write(prov, {'models': initial, 'user_authorized_resume': '2026-09-26; nine GPUs', 'training_repeated': False, 'time': time.time()})
    snap = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits'], text=True)
    assert len(snap.strip().splitlines()) == 9 and all((int(l.split(',')[-1]) < 1024 for l in snap.strip().splitlines())), 'One of nine GPUs no longer idle'
    write(R / 'results/gpu-before-evaluation.json', {'snapshot': snap, 'time': time.time()})
    stage('development_and_calibration_all_epochs')
    jobs = []
    for split in ['development', 'calibration']:
        for e in [1, 2, 3]:
            jobs += security_jobs(paths[f'epoch-{e}'], f'epoch-{e}', split)
    run_jobs(jobs, 'development_and_calibration')
    stage('checkpoint_selection_and_calibration')
    dev = {}
    candidates = []
    for e in [1, 2, 3]:
        dev[e] = security_rows(f'epoch-{e}', 'development')
        m = metrics(dev[e], 1.0)
        m['macro_task_nll'] = sum((t['nll'] for t in m['tasks'].values())) / len(m['tasks'])
        candidates.append({'epoch': e, 'metrics': m, 'checkpoint_files': model_files(paths[f'epoch-{e}'])})
    best = min(candidates, key=lambda c: (-c['metrics']['macro_task_balanced_accuracy'], c['metrics']['macro_task_nll'], c['epoch']))
    write(R / 'results/checkpoint-selection.json', {'selected_epoch': best['epoch'], 'candidates': candidates, 'rule': 'maximum development macro-task semantic-class balanced accuracy; tie macro-task raw NLL, then earlier epoch', 'test_used_for_selection': False})
    temperatures = {}
    frozen = {}
    for e in [1, 2, 3]:
        rows = security_rows(f'epoch-{e}', 'calibration')
        with torch.enable_grad():
            t = fit(rows)
        temperatures[e] = t
        dest = RUN / f'epoch-{e}-calibrated'
        if not dest.exists():
            shutil.copytree(paths[f'epoch-{e}'], dest, ignore=shutil.ignore_patterns('optimizer.pt'))
            ck = Checkpoint(dest)
            ck.meta.temperature = t
            ck.meta.extra['temperature_fit'] = {'split': 'calibration', 'questions': len(rows)}
            write_meta(dest, ck.meta)
        else:
            assert abs(Checkpoint(dest).meta.temperature - t) < 1e-08
        frozen[str(e)] = model_files(dest)
        write(R / 'results' / f'epoch-{e}-validation.json', {'epoch': e, 'temperature': t, 'development': report(dev[e], t), 'calibration': report(rows, t)})
    write(R / 'results/frozen-final-candidate.json', {'selected_epoch': best['epoch'], 'all3predeclared_for_test': True, 'files_by_epoch': frozen, 'test_used_for_selection': False})
    if not (R / 'data/test.jsonl.gz').exists():
        (R / 'data/test.jsonl.gz').symlink_to(str(WORK / 'data' / 'test.jsonl.gz'))
    if not (R / 'data/test.pkl').exists():
        call('prepare-test', [sys.executable, str(SCRIPTS / 'prepare.py'), '--allow-test'])
    stage('security_and_general_tests')
    jobs = []
    for e in [1, 2, 3]:
        jobs += security_jobs(RUN / f'epoch-{e}-calibrated', f'epoch-{e}', 'test')
    jobs += security_jobs(paths['kev-stage1'], 'kev-stage1', 'test')
    for name in ['kev-stage1', 'epoch-1', 'epoch-2', 'epoch-3']:
        jobs += general_jobs(name, paths[name])
    run_jobs(jobs, 'security_and_general_tests')
    rows = security_rows('kev-stage1', 'test')
    write(R / 'results/kev-stage1-test.json', {'test': report(rows, 1.0), 'test_used_for_selection': False})
    for e in [1, 2, 3]:
        rows = security_rows(f'epoch-{e}', 'test')
        write(R / 'results' / f'epoch-{e}-test.json', {'epoch': e, 'temperature': temperatures[e], 'test': report(rows, temperatures[e]), 'test_used_for_selection': False})
        assert model_files(RUN / f'epoch-{e}-calibrated') == frozen[str(e)]
    for name, fp in initial.items():
        assert model_files(paths[name]) == fp, 'Original checkpoint changed'
    stage('reporting')
    call('report', [sys.executable, str(SCRIPTS / 'report.py')])
    snapshot = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader'], text=True)
    write(R / 'results/gpu-after-evaluation.json', {'snapshot': snapshot, 'time': time.time(), 'all_evaluation_workers_exited': True})
    stage('complete', selected_epoch=best['epoch'], all3security_and_general_tests_complete=True, original_checkpoints_unchanged=True)
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        stage('failed', error=str(e), traceback=traceback.format_exc())
        raise
