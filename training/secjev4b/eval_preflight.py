from paths import WORK,SCRIPTS,SOURCE,BASE
"""Post-training FP32 inference check; development inputs only."""
import os, sys, pathlib, pickle, json, time
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, str(SOURCE))
import torch
from kev.checkpoint import Checkpoint, LoadOptions
from eval_loading import load_serial
torch.set_num_threads(4)
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
R = pathlib.Path(str(WORK))
ck = Checkpoint(R / 'runs/secjev-4b-3epochs/epoch-1')
ck.meta.base = str(BASE)
ck.meta.base_revision = None
tok, model = load_serial(ck, 'cuda:0', LoadOptions(dtype=torch.float32, merge=False, attn='sdpa', temperature=1.0))
model.eval()
rows = sorted(pickle.load((R / 'data/development.pkl').open('rb')), key=lambda r: r['length'])
results = []
with torch.inference_mode():
    for batch in [rows[:16], rows[-1:]]:
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        logits = model.forward_batch([r['enc'] for r in batch])
        torch.cuda.synchronize()
        assert len(logits) == len(batch)
        for r, z in zip(batch, logits):
            assert len(z[0]) == r['k'] and torch.isfinite(z[0]).all()
        results.append({'questions': len(batch), 'max_tokens': max((r['length'] for r in batch)), 'elapsed_s': time.time() - started, 'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2 ** 20})
(R / 'results/evaluation-preflight.json').write_text(json.dumps({'pass': True, 'precision': 'FP32 unmerged', 'split': 'development', 'cases': results}, indent=2) + '\n')
print('PASS', results, flush=True)
