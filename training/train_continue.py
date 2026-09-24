"""Two additional epochs from the immutable epoch-1 checkpoint; three-rank DDP."""
import os, sys, pathlib, json, time, random, math, pickle, contextlib, hashlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import train as upstream_run
import torch
import torch.distributed as dist
from kev.checkpoint import write_meta

WORK = pathlib.Path(os.environ.get('SECJEV_WORK', './secjev-work')).resolve()
R = WORK/'continuation'
INIT = str(WORK/'runs/secjev-0.8b-v1/checkpoint')
OUT = R/'runs/secjev-0.8b-3epochs'
GLOBAL = 192


def batches_for(rows, epoch):
    rng = random.Random(20260923 + epoch - 1)
    ids = list(range(len(rows))); rng.shuffle(ids)
    batches = []
    for start in range(0, len(ids), 2304):
        pool = sorted(ids[start:start+2304], key=lambda i: rows[i]['length'])
        batches.extend(pool[j:j+GLOBAL] for j in range(0, len(pool), GLOBAL))
    rng.shuffle(batches)
    return batches


def padded_shards(indices, per_rank, world):
    assert 0 < len(indices) <= per_rank * world
    padded = indices + [indices[-1]] * (per_rank * world-len(indices))
    return [[(padded[pos], pos < len(indices)) for pos in range(rank*per_rank, (rank+1)*per_rank)] for rank in range(world)]


def main():
    upstream_run.R = R; upstream_run.INIT = INIT
    rank, ck, meta, tok, model, net, params = upstream_run.setup(True)
    world = dist.get_world_size(); assert world == 3
    rows = pickle.load((R/'data/train.pkl').open('rb'))
    profile = json.loads((R/'results/data-profile.json').read_text())
    scale = len(rows)/profile['train']['records']
    optimizer = torch.optim.AdamW(params, lr=1e-5, weight_decay=.01)
    saved = torch.load(pathlib.Path(INIT)/'optimizer.pt', map_location='cpu', weights_only=False)
    assert saved['global_step'] == 929
    optimizer.load_state_dict(saved['optimizer'])
    assert len(optimizer.state) == len(params)
    old_steps = [float(s['step']) for s in optimizer.state.values()]
    assert min(old_steps) == max(old_steps) == 929
    for group in optimizer.param_groups:
        group['lr'] = group['initial_lr'] = 1e-5
    del saved
    schedules = {epoch: batches_for(rows, epoch) for epoch in [2, 3]}
    total_steps = sum(map(len, schedules.values())); warmup = round(.05*total_steps)
    def factor(step):
        if step < warmup:
            return .2 + .8*step/warmup
        return .1 + .9*.5*(1+math.cos(math.pi*(step-warmup)/max(1,total_steps-warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    if rank == 0:
        OUT.mkdir(parents=True, exist_ok=False)
        upstream_run.write(OUT/'training_config.json', {
            'total_epochs': 3, 'additional_epochs': 2, 'starting_checkpoint': INIT,
            'optimizer_moments_restored': True, 'optimizer_initial_step': 929,
            'world_size': world, 'effective_global_question_batch': GLOBAL,
            'per_gpu_batch_max': 64, 'padded_token_budget_per_gpu': 40960,
            'steps_per_epoch': {str(k):len(v) for k,v in schedules.items()},
            'additional_optimizer_steps': total_steps, 'train_questions_per_epoch': len(rows),
            'train_scenes': profile['train']['records'], 'peak_learning_rate': 1e-5,
            'schedule': 'new continuation schedule: 5% warmup from 2e-6 to 1e-5, cosine to 1e-6',
            'optimizer': 'AdamW', 'weight_decay': .01, 'gradient_clip_norm': 1.,
            'precision': 'FP32 master/BF16 autocast, training TF32 allowed',
            'objective': 'mean scene cross entropy, question weight=1/questions_per_scene',
            'checkpoint_candidates': ['epoch-1 final', 'epoch-2 final', 'epoch-3 final'],
            'selection_split': 'development',
            'selection_metric': 'maximum macro-task semantic-class balanced accuracy; then minimum macro-task raw NLL; then earlier epoch',
            'test_used_for_selection': False, 'replay': 0, 'augmentations': False,
            'no_truncation': True, 'context_limit': 8192,
            'script_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        })
    dist.barrier()
    monitor_rows = pickle.load((R/'data/development.pkl').open('rb')); groups = {}
    for row in monitor_rows:
        groups.setdefault((row['source'],row['task'],row['qtype']), []).append(row)
    monitor = []
    for key, rr in sorted(groups.items()):
        rr.sort(key=lambda x:hashlib.sha256(f"monitor:{x['case']}:{x['qid']}".encode()).hexdigest())
        monitor.extend(rr[:4])
    del monitor_rows
    step = 0; seen_total = 0; losses = []; started = time.perf_counter()
    def monitor_check(epoch):
        dist.barrier(); model.eval(); tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        ce = 0.; correct = 0; count = 0
        with torch.no_grad():
            for row in monitor[rank::world]:
                z = model.forward(row['enc'])[0].float()
                ce += float(torch.nn.functional.cross_entropy(z[None], torch.tensor([row['label']],device=model.device)))
                correct += int(int(z.argmax()) == row['label']); count += 1
        sums = torch.tensor([ce,correct,count],dtype=torch.float64,device=model.device)
        dist.all_reduce(sums)
        if rank == 0:
            record = {'epoch':epoch,'continuation_step':step,'fixed_questions':int(sums[2]),'loss':float(sums[0]/sums[2]),'accuracy':float(sums[1]/sums[2]),'used_for_selection':False}
            with (R/'results/monitor-history.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
            print('MONITOR',json.dumps(record),flush=True)
        model.train(); torch.backends.cuda.matmul.allow_tf32 = tf32; torch.cuda.empty_cache(); dist.barrier()
    def save(name, epoch, seen_epoch):
        if rank == 0:
            dest = OUT/name; dest.mkdir(exist_ok=False)
            model.lm.peft_config['default'].base_model_name_or_path = meta.base
            model.lm.save_pretrained(dest)
            meta.head = {k:v.detach().cpu() for k,v in model.head.state_dict().items()}; meta.temperature = 1.
            meta.extra = {**meta.extra, 'security_training': {'epoch':epoch,'continuation_step':step,'optimizer_step':929+step,'seen_additional_rows':seen_total,'seen_rows_current_epoch':seen_epoch,'initial_checkpoint':INIT}, 'temperature_fit':'not calibrated after continuation'}
            write_meta(dest,meta); tok.save_pretrained(dest)
            torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'global_step':929+step,'continuation_step':step,'epoch':epoch,'torch_rng':torch.get_rng_state()},dest/'optimizer.pt')
        dist.barrier()
    monitor_check(1)
    for epoch, batches in schedules.items():
        used = []
        for epoch_step, whole in enumerate(batches,1):
            per_rank = upstream_run.microbatch(max(rows[i]['length'] for i in whole))
            micro = [whole[j:j+world*per_rank] for j in range(0,len(whole),world*per_rank)]
            optimizer.zero_grad(set_to_none=True); total = 0.
            for j, ids in enumerate(micro):
                batch = [rows[i] if real else {**rows[i], 'weight':0.} for i,real in padded_shards(ids,per_rank,world)[rank]]
                with (contextlib.nullcontext() if j == len(micro)-1 else net.no_sync()):
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        loss = net(batch)*scale/(len(whole)/world)
                    if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                    loss.backward()
                total += float(loss.detach()); used.extend(ids); seen_total += len(ids)
            norm = torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); step += 1
            value = torch.tensor(total,device=model.device); dist.all_reduce(value); losses.append(float(value)/world)
            if rank == 0:
                elapsed = time.perf_counter()-started
                status = {'stage':'training','epoch':epoch,'epoch_step':epoch_step,'epoch_steps':len(batches),'step':step,'steps':total_steps,'seen_additional_rows':seen_total,'loss_step':losses[-1],'loss_recent_50':sum(losses[-50:])/len(losses[-50:]),'grad_norm':float(norm),'elapsed_s':elapsed,'estimated_remaining_s':elapsed/step*(total_steps-step),'per_gpu_batch':per_rank,'world_size':world,'global_batch':GLOBAL,'learning_rate':optimizer.param_groups[0]['lr'],'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20}
                upstream_run.write(R/'results/train-status.json',status); print('TRAIN',json.dumps(status),flush=True)
            if step == 10 or step%100 == 0: monitor_check(epoch)
            if step%200 == 0 and epoch_step != len(batches): save(f'recovery-step-{step}',epoch,len(used))
        assert len(used) == len(set(used)) == len(rows)
        save(f'epoch-{epoch}',epoch,len(used)); monitor_check(epoch)
    assert seen_total == 2*len(rows)
    if rank == 0:
        upstream_run.write(R/'results/train-status.json',{'stage':'complete','total_epochs':3,'additional_epochs':2,'steps':step,'seen_additional_rows':seen_total,'each_epoch_unique_rows':len(rows),'elapsed_s':time.perf_counter()-started})
    dist.destroy_process_group()


if __name__ == '__main__': main()
