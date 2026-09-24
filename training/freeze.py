"""Freeze the calibrated final checkpoint before test evaluation, or verify it afterward."""
import argparse, hashlib, json, os, pathlib, time
parser = argparse.ArgumentParser()
parser.add_argument('--verify', action='store_true')
args = parser.parse_args()
root = pathlib.Path(os.environ.get('SECJEV_WORK', './secjev-work')).resolve()
checkpoint = root/'runs/secjev-0.8b-v1/calibrated'
manifest = root/'results/frozen-final-candidate.json'
files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint.iterdir() if p.is_file()}
assert 'head.pt' in files and 'adapter_model.safetensors' in files
if args.verify:
    assert files == json.loads(manifest.read_text())['files']
    print('Frozen checkpoint unchanged')
else:
    assert not manifest.exists(), 'Already frozen; use --verify'
    manifest.write_text(json.dumps({'checkpoint': 'runs/secjev-0.8b-v1/calibrated', 'files': files,
        'selection_rule': 'predeclared one-epoch final checkpoint', 'temperature_source': 'calibration only',
        'frozen_unix': time.time()}, indent=2)+'\n')
    print('Final candidate frozen; test evaluation can now proceed')
