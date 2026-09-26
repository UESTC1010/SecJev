from paths import WORK,SCRIPTS,SOURCE,BASE
import os, sys, time, json, pickle, datetime, math, contextlib
from common import *
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from kev.data import materialize
from kev.suite import load_split
from kev.train import encode_batch, batch_loss
from types import SimpleNamespace
rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=30))
torch.manual_seed(2)
tok = load_tokenizer(BASE)
m = BucketDecisionModel(BASE, tok, 'cuda:' + str(rank), lora=16, head_dim=256, lora_targets='all', option_isolation=False, special_embeddings=False, dtype=torch.float32, attn='sdpa')
m.lm.config.use_cache = False
m.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
net = DDP(LossModel(m), device_ids=[rank], broadcast_buffers=False, find_unused_parameters=False)
params = [p for p in m.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=5e-05)
a = SimpleNamespace(seed=2, p_none=0.1, p_none_distract=0.12, p_distract=0.15, p_none_pair=0.25, perm_kl=0.0, ord_w=0.0, label_smoothing=0.0, brier_w=0.0, focal_gamma=0.0)
reqs = load_split(R / 'evals/v7/decision-v7', 'train')
from kev.data import load_records
delta = load_records(str(WORK / 'upstream/evals/night2/dates_unknowable.jsonl'))
soft = next((r for r in delta if any((q.get('target') is not None for q in materialize(r)['questions']))))
multi = next((r for r in reqs if len(materialize(r)['questions']) > 1))
vs = encode_batch(m, tok, a, [reqs[0], multi, soft], 0)
rr = [r for v in vs for r in split_variant(v)]
m.eval()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
with torch.no_grad():
    expected, _ = batch_loss(m, a, vs, m.device, {}, None, contextlib.nullcontext())
    actual = net.module(rr)
    err = float(abs(expected - actual))
    assert err < 0.001, err
    assert m.padding_bucket == 1
if rank == 0:
    write(R / 'results/loss-parity.json', {'max_absolute_difference': err, 'tolerance': 0.001, 'includes_multi_question_and_soft_targets': True, 'padding': 'upstream max-row length; extra bucket padding disabled', 'earlier_128_bucket_drift': 0.0019369125366210938})
    print('PARITY PASS', err, 'native upstream padding', flush=True)
m.train()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
rows = sorted(pickle.load((R / 'data/train.pkl').open('rb')), key=lambda r: r['length'])
results = []
for q in [1.0, 0.5, 0.9]:
    end = min(len(rows), max(192, int(len(rows) * q)))
    b = microbatch(rows[end - 1]['length'])
    batch = rows[end - (rank + 1) * b:end - rank * b]
    for rep in range(2):
        opt.zero_grad(set_to_none=True)
        dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = net(batch) / b
        assert torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        opt.step()
        torch.cuda.synchronize()
        stats = torch.tensor([time.time() - t, torch.cuda.max_memory_allocated() / 2 ** 20], device=m.device)
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
        rec = {'quantile': q, 'rep': rep, 'seconds': float(stats[0]), 'peak_mib': float(stats[1]), 'per_gpu_batch': b, 'global_rows': 3 * b, 'length': batch[-1]['length'], 'loss': float(loss), 'grad_norm': float(norm)}
        results.append(rec)
        if rank == 0:
            print('PROBE', json.dumps(rec), flush=True)
if rank == 0:
    write(R / 'results/probe.json', {'pass': True, 'loss_parity_max_absolute': err, 'padding': 'upstream max-row length', 'token_budget_per_gpu': 32768, 'trainable_parameters': sum((p.numel() for p in params)), 'weights_discarded': True, 'results': results})
dist.destroy_process_group()
