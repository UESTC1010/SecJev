from paths import WORK,SCRIPTS,SOURCE,BASE
"""SecJev 2B stage2: warm-start own Kev-style decision checkpoint, three epochs."""
import os, sys, pathlib, json, time, random, math, pickle, contextlib, hashlib, datetime, argparse
sys.path.insert(0, str(SCRIPTS))
from common import BucketDecisionModel, LossModel, microbatch, write, BASE
import torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from kev.model import load_tokenizer
from kev.checkpoint import Meta, write_meta, Checkpoint
R = pathlib.Path(str(WORK))
OUT = R / 'runs/secjev-2b-3epochs'

def batches_for(rows, epoch):
    size, poolsize = (128, 2048) if epoch == 1 else (192, 2304)
    rng = random.Random(20260923 + epoch - 1)
    ids = list(range(len(rows)))
    rng.shuffle(ids)
    batches = []
    for start in range(0, len(ids), poolsize):
        pool = sorted(ids[start:start + poolsize], key=lambda i: rows[i]['length'])
        batches.extend((pool[j:j + size] for j in range(0, len(pool), size)))
    rng.shuffle(batches)
    return batches

def padded_shards(indices, per_rank, world):
    assert 0 < len(indices) <= per_rank * world
    padded = indices + [indices[-1]] * (per_rank * world - len(indices))
    return [[(padded[pos], pos < len(indices)) for pos in range(rank * per_rank, (rank + 1) * per_rank)] for rank in range(world)]

def setup():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=30))
    torch.manual_seed(20260923)
    random.seed(20260923)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    prov = json.loads(pathlib.Path(BASE, 'download_provenance.json').read_text())
    assert prov['revision'] == 'b1485b2fa6dfa1287294f269f5fb618e03d52d7c'
    tok = load_tokenizer(BASE)
    meta = Meta(base='Qwen/Qwen3.5-2B-Base', base_revision=prov['revision'], lora=16, head_dim=256, option_isolation=False, special_embeddings=False, temperature=1.0, extra={'args': {'lora_targets': 'all'}, 'init_from': 'Own Kev-style 2B stage1 checkpoint, then SecJev-Corpus'})
    model = BucketDecisionModel(BASE, tok, 'cuda:' + str(rank), lora=16, head_dim=256, lora_targets='all', option_isolation=False, special_embeddings=False, dtype=torch.float32, attn='sdpa')
    bs = [p for n, p in model.named_parameters() if 'lora_B' in n]
    assert bs and all((torch.count_nonzero(p).item() == 0 for p in bs))
    source = Checkpoint(str(WORK / 'runs/kev-2b-delta/epoch-1')).warm_start(model, meta)
    meta.extra['warm_start'] = source
    assert model.head.temperature == 1.0
    model.lm.config.use_cache = False
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    net = DDP(LossModel(model), device_ids=[rank], broadcast_buffers=False, find_unused_parameters=False)
    params = [p for p in model.parameters() if p.requires_grad]
    assert all(('lora_' in n or n.startswith('head.') for n, p in model.named_parameters() if p.requires_grad))
    if rank == 0:
        write(R / 'results/runtime.json', {'initialization': meta.extra['init_from'], 'base_revision': prov['revision'], 'lora_rank': 16, 'lora_alpha': 32, 'lora_targets': 'all', 'head_dim': 256, 'seed': 20260923, 'trainable_parameters': sum((p.numel() for p in params)), 'world_size': dist.get_world_size(), 'fresh_lora_B_zero_before_warm_start': True, 'kev_checkpoint_loaded': True, 'warm_start': source, 'precision': 'FP32 master/BF16 autocast; training TF32', 'script_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()})
    return (rank, meta, tok, model, net, params)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', action='store_true')
    a = ap.parse_args()
    rank, meta, tok, model, net, params = setup()
    world = dist.get_world_size()
    assert world == 3
    rows = pickle.load((R / 'data/train.pkl').open('rb'))
    profile = json.loads((R / 'results/data-profile.json').read_text())
    scale = len(rows) / profile['train']['records']
    optimizer = torch.optim.AdamW(params, lr=2e-05, weight_decay=0.01)
    if a.probe:
        results = []
        ordered = sorted(rows, key=lambda r: r['length'])
        for q in [1.0, 0.5]:
            end = min(len(ordered), max(192, int(len(ordered) * q)))
            b = microbatch(ordered[end - 1]['length'])
            batch = ordered[end - (rank + 1) * b:end - rank * b]
            optimizer.zero_grad(set_to_none=True)
            dist.barrier()
            torch.cuda.synchronize()
            t = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = net(batch) * scale / b
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            results.append({'quantile': q, 'seconds': time.perf_counter() - t, 'per_gpu_batch': b, 'loss': float(loss), 'grad_norm': float(norm), 'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2 ** 20})
        if rank == 0:
            write(R / 'results/probe.json', {'pass': True, 'probe_weights_discarded': True, 'results': results})
        dist.destroy_process_group()
        return
    schedules = {e: batches_for(rows, e) for e in [1, 2, 3]}
    assert [len(v) for v in schedules.values()] == [929, 619, 619]
    if rank == 0:
        OUT.mkdir(parents=True, exist_ok=False)
        write(OUT / 'training_config.json', {'epochs': 3, 'initialization': meta.extra['init_from'], 'optimizer': 'fresh AdamW, no loaded optimizer moments; retained continuously through all epochs', 'global_question_batches': {'1': 128, '2': 192, '3': 192}, 'steps_per_epoch': [929, 619, 619], 'lr': 'epoch1 peak2e-5 with5%warmup/cosine to2e-6; epochs2+3 peak1e-5 with5%warmup from2e-6/cosine to1e-6', 'objective': 'per-scene mean cross entropy', 'weight_decay': 0.01, 'grad_clip': 1.0, 'world_size': 3, 'comparison': 'same security recipe as 0.8B, adapted to own 2B Kev-style initialization; microbatches reduced for memory', 'train_questions_per_epoch': len(rows), 'replay': 0, 'augmentation': False, 'no_truncation': True, 'context_limit': 8192, 'test_use': 'all3finalepochs predeclared for audit; selection uses full development only'})
    dist.barrier()
    monitor_rows = pickle.load((R / 'data/development.pkl').open('rb'))
    groups = {}
    for row in monitor_rows:
        groups.setdefault((row['source'], row['task'], row['qtype']), []).append(row)
    monitor = []
    for key, rr in sorted(groups.items()):
        rr.sort(key=lambda x: hashlib.sha256(f"monitor:{x['case']}:{x['qid']}".encode()).hexdigest())
        monitor.extend(rr[:4])
    del monitor_rows
    step = 0
    seen_total = 0
    losses = []
    started = time.perf_counter()
    scheduler = None

    def check(epoch):
        dist.barrier()
        model.eval()
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        ce = correct = count = 0
        with torch.no_grad():
            for row in monitor[rank::world]:
                z = model.forward(row['enc'])[0].float()
                ce += float(torch.nn.functional.cross_entropy(z[None], torch.tensor([row['label']], device=model.device)))
                correct += int(int(z.argmax()) == row['label'])
                count += 1
        sums = torch.tensor([ce, correct, count], device=model.device, dtype=torch.float64)
        dist.all_reduce(sums)
        if rank == 0:
            rec = {'epoch': epoch, 'step': step, 'fixed_questions': int(sums[2]), 'loss': float(sums[0] / sums[2]), 'accuracy': float(sums[1] / sums[2]), 'used_for_selection': False}
            with (R / 'results/monitor-history.jsonl').open('a') as f:
                f.write(json.dumps(rec) + '\n')
            print('MONITOR', json.dumps(rec), flush=True)
        model.train()
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.cuda.empty_cache()
        dist.barrier()

    def save(name, epoch, seen_epoch):
        if rank == 0:
            dest = OUT / name
            dest.mkdir(exist_ok=False)
            model.lm.peft_config['default'].base_model_name_or_path = meta.base
            model.lm.save_pretrained(dest)
            meta.head = {k: v.detach().cpu() for k, v in model.head.state_dict().items()}
            meta.temperature = 1.0
            meta.extra.update(security_training={'epoch': epoch, 'step': step, 'seen_total': seen_total, 'seen_epoch': seen_epoch}, temperature_fit='not calibrated')
            write_meta(dest, meta)
            tok.save_pretrained(dest)
            torch.save({'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'global_step': step, 'epoch': epoch, 'torch_rng': torch.get_rng_state()}, dest / 'optimizer.pt')
        dist.barrier()
    check(0)
    for epoch, batches in schedules.items():
        if epoch in [1, 2]:
            peak, total = (2e-05, 929) if epoch == 1 else (1e-05, 1238)
            warmup = round(0.05 * total)
            for g in optimizer.param_groups:
                g['lr'] = g['initial_lr'] = peak

            def factor(s):
                if s < warmup:
                    return (s + 1) / warmup if epoch == 1 else 0.2 + 0.8 * s / warmup
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup)))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
        used = []
        for epoch_step, whole in enumerate(batches, 1):
            per_rank = min(microbatch(max((rows[i]['length'] for i in whole))), math.ceil(len(whole) / world))
            micros = [whole[j:j + world * per_rank] for j in range(0, len(whole), world * per_rank)]
            optimizer.zero_grad(set_to_none=True)
            value = 0.0
            for j, ids in enumerate(micros):
                batch = [rows[i] if real else {**rows[i], 'weight': 0.0} for i, real in padded_shards(ids, per_rank, world)[rank]]
                with contextlib.nullcontext() if j == len(micros) - 1 else net.no_sync():
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        loss = net(batch) * scale / (len(whole) / world)
                    assert torch.isfinite(loss), 'Nonfinite loss'
                    loss.backward()
                value += float(loss.detach())
                used.extend(ids)
                seen_total += len(ids)
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            mean = torch.tensor(value, device=model.device)
            dist.all_reduce(mean)
            losses.append(float(mean) / world)
            if rank == 0:
                elapsed = time.perf_counter() - started
                st = {'stage': 'training', 'epoch': epoch, 'epoch_step': epoch_step, 'epoch_steps': len(batches), 'step': step, 'steps': 2167, 'seen_rows': seen_total, 'loss_step': losses[-1], 'loss_recent_50': sum(losses[-50:]) / len(losses[-50:]), 'grad_norm': float(norm), 'elapsed_s': elapsed, 'estimated_remaining_s': elapsed / step * (2167 - step), 'global_batch': 128 if epoch == 1 else 192, 'per_gpu_batch': per_rank, 'learning_rate': optimizer.param_groups[0]['lr'], 'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2 ** 20}
                write(R / 'results/train-status.json', st)
                print('TRAIN', json.dumps(st), flush=True)
            if step == 10 or step % 100 == 0:
                check(epoch)
            if step % 200 == 0 and epoch_step != len(batches):
                save(f'recovery-step-{step}', epoch, len(used))
        assert len(used) == len(set(used)) == len(rows)
        save(f'epoch-{epoch}', epoch, len(used))
        check(epoch)
    assert seen_total == 3 * len(rows)
    if rank == 0:
        write(R / 'results/train-status.json', {'stage': 'complete', 'epochs': 3, 'steps': step, 'seen_rows': seen_total, 'unique_rows_per_epoch': len(rows), 'elapsed_s': time.perf_counter() - started})
    dist.destroy_process_group()
if __name__ == '__main__':
    main()
