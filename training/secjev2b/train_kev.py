from paths import WORK,SCRIPTS,SOURCE,BASE
"""Three-GPU execution of Kev's source-record CE and augmentation recipe.
Effective global batch stays 8 source requests; question rows do not attend to one another.
"""
import sys, os, pathlib, json, random, math, time, datetime, contextlib, argparse, copy
from common import *
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from kev.checkpoint import Meta, write_meta, Checkpoint
from kev.train import training_requests, encode_batch
from kev.suite import read_manifest, digest
from types import SimpleNamespace
SRC = pathlib.Path(str(SOURCE))
SUITE = R / 'evals/v7/decision-v7'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--delta', action='store_true')
    a0 = ap.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=30))
    world = dist.get_world_size()
    assert world == 3
    seed = 1 if a0.delta else 2
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    prov = json.load(open(pathlib.Path(BASE) / 'download_provenance.json'))
    tok = load_tokenizer(BASE)
    meta = Meta(base='Qwen/Qwen3.5-2B-Base', base_revision=prov['revision'], lora=16, head_dim=256, option_isolation=False, special_embeddings=False, temperature=1.0, extra={'args': {'lora_targets': 'all'}})
    m = BucketDecisionModel(BASE, tok, 'cuda:' + str(rank), lora=16, head_dim=256, lora_targets='all', option_isolation=False, special_embeddings=False, dtype=torch.float32, attn='sdpa')
    init = Checkpoint(str(WORK / 'runs/kev-2b-main/epoch-2')).warm_start(m, meta) if a0.delta else None
    m.lm.config.use_cache = False
    m.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    m.train()
    net = DDP(LossModel(m), device_ids=[rank], broadcast_buffers=False, find_unused_parameters=False)
    params = [p for p in m.parameters() if p.requires_grad]
    a = SimpleNamespace(data=str(WORK / 'upstream/evals/night2/dates_unknowable.jsonl') if a0.delta else '', suite=str(SUITE), replay=2000 if a0.delta else 0, seed=seed, train_sources='', public_frac=1.0, synthetic_repeat=1, p_none=0.1, p_none_distract=0.12, p_distract=0.15, p_none_pair=0.25, perm_kl=0.0)
    manifest = read_manifest(SUITE)
    reqs = training_requests(a, tok, manifest, manifest['holdout_sources'])
    epochs = 1 if a0.delta else 2
    assert len(reqs) == (3425 if a0.delta else 12576)
    out = R / 'runs' / ('kev-2b-delta' if a0.delta else 'kev-2b-main')
    tag = 'kev-delta' if a0.delta else 'kev-main'
    lr = 2e-05 if a0.delta else 5e-05
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(reqs) / 8)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    if rank == 0:
        out.mkdir(exist_ok=False)
        write(out / 'training_config.json', {'stage': tag, 'epochs': epochs, 'source_records': len(reqs), 'effective_global_source_batch': 8, 'lr': lr, 'schedule': 'OneCycleLR pct_start=.1 default division factors', 'precision': 'FP32 masters BF16 autocast', 'lora_rank': 16, 'lora_alpha': 32, 'lora_targets': 'all', 'head_dim': 256, 'augmentation': vars(a), 'suite_sha256': digest(SUITE / 'manifest.json'), 'initialization': init, 'base_revision': prov['revision'], 'source_commit': SRC.name, 'recipe_adaptation': '2B has no official recipe; uses 4B peak learning rates, seed2 base / seed1 delta; global batch8, source/variant mean CE retained; 3-GPU question-row execution', 'trainable_parameters': sum((p.numel() for p in params))})
    dist.barrier()
    rng = random.Random(seed)
    step = 0
    start = time.time()
    losses = []
    seen = 0

    def save(name, epoch):
        if rank == 0:
            dest = out / name
            dest.mkdir(exist_ok=False)
            m.lm.peft_config['default'].base_model_name_or_path = meta.base
            m.lm.save_pretrained(dest)
            meta.head = {k: v.detach().cpu() for k, v in m.head.state_dict().items()}
            meta.extra.update(stage=tag, epoch=epoch, step=step, init_source=init)
            write_meta(dest, meta)
            tok.save_pretrained(dest)
            torch.save({'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(), 'epoch': epoch, 'step': step, 'torch_rng': torch.get_rng_state(), 'random_rng': rng.getstate()}, dest / 'optimizer.pt')
        dist.barrier()
    for ep in range(epochs):
        rng.shuffle(reqs)
        used = []
        for offset in range(0, len(reqs), 8):
            chunk = reqs[offset:offset + 8]
            variants = encode_batch(m, tok, a, chunk, ep)
            rows = [r for v in variants for r in split_variant(v)]
            assert abs(sum((r['weight'] for r in rows)) - len(variants)) < 1e-06
            rows.sort(key=lambda r: r['length'])
            b = min(microbatch(max((r['length'] for r in rows))), math.ceil(len(rows) / world))
            micros = [rows[j:j + world * b] for j in range(0, len(rows), world * b)]
            opt.zero_grad(set_to_none=True)
            value = 0.0
            for j, rr in enumerate(micros):
                n = len(rr)
                padded = rr + [{**rr[-1], 'weight': 0.0}] * (world * b - n)
                batch = padded[rank * b:(rank + 1) * b]
                with contextlib.nullcontext() if j == len(micros) - 1 else net.no_sync():
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        loss = net(batch) * world / len(variants)
                    assert torch.isfinite(loss)
                    loss.backward()
                value += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            opt.step()
            sched.step()
            step += 1
            seen += len(chunk)
            used.extend((r['_meta']['id'] for r in chunk))
            mean = torch.tensor(value, device=m.device)
            dist.all_reduce(mean)
            losses.append(float(mean) / world)
            if rank == 0:
                st = {'stage': tag, 'epoch': ep + 1, 'step': step, 'steps': steps, 'source_records_seen': seen, 'loss_step': losses[-1], 'loss_recent_50': sum(losses[-50:]) / len(losses[-50:]), 'elapsed_s': time.time() - start, 'learning_rate': opt.param_groups[0]['lr'], 'grad_norm': float(norm), 'per_gpu_question_batch': b, 'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2 ** 20}
                write(R / 'results' / f'{tag}-status.json', st)
                if step % 10 == 0:
                    print('TRAIN', json.dumps(st), flush=True)
            if step % 500 == 0 and offset + 8 < len(reqs):
                save(f'recovery-step-{step}', ep + 1)
        assert len(used) == len(set(used)) == len(reqs)
        save(f'epoch-{ep + 1}', ep + 1)
    if rank == 0:
        write(R / 'results' / f'{tag}-status.json', {'stage': 'complete', 'epochs': epochs, 'steps': step, 'records_seen': seen, 'elapsed_s': time.time() - start})
    dist.destroy_process_group()
if __name__ == '__main__':
    main()
