"""Security delta: upstream Kev numerical model/loss, full independent question rows, DDP."""
import os,sys,pathlib,json,pickle,time,random,math,hashlib,contextlib,copy,datetime
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True';os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
sys.path.insert(0,str(pathlib.Path(os.environ.get('SECJEV_WORK', './secjev-work')).resolve()/'source/kev'))
import torch,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from kev.model import DecisionModel,load_tokenizer
from kev.checkpoint import Checkpoint,Meta,write_meta
from kev.train import question_loss
R=pathlib.Path(os.environ.get('SECJEV_WORK', './secjev-work')).resolve();BASE=str(R/'base');INIT=str(R/'init')
def write(p,data):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(p)
class BucketDecisionModel(DecisionModel):
 padding_bucket=128
 def _pad_rows(self,rows):
  ids,pos,att=super()._pad_rows(rows);extra=(-ids.shape[1])%self.padding_bucket
  if extra:
   ids=torch.nn.functional.pad(ids,(0,extra),value=self.pad_id);pos=torch.nn.functional.pad(pos,(0,extra));att=torch.nn.functional.pad(att,(0,extra))
  return ids,pos,att

def microbatch(length):
 padded=math.ceil(length/128)*128
 return max(b for b in [1,2,4,8,16,32,64] if b*padded<=40960)

class LossModel(torch.nn.Module):
 def __init__(self,m):super().__init__();self.model=m
 def forward(self,batch):
  zs=self.model.forward_batch([r['enc'] for r in batch]);loss=sum(question_loss(z[0].float(),{'label':r['label'],'qtype':r['qtype']},self.model.device,0.)*r['weight'] for z,r in zip(zs,batch))
  return loss

def setup(checkpointing=True):
 rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(4);torch.set_num_interop_threads(1);dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=30));torch.manual_seed(20260923);random.seed(20260923)
 torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 ck=Checkpoint(INIT);meta=copy.deepcopy(ck.meta);meta.head=None;meta.temperature=1.0;prov=json.load(open(BASE+'/download_provenance.json'));assert meta.base_revision==prov['revision'];tok=load_tokenizer(BASE)
 m=BucketDecisionModel(BASE,tok,'cuda:'+str(rank),lora=meta.lora,head_dim=meta.head_dim,lora_targets=meta.extra['args']['lora_targets'],option_isolation=meta.option_isolation,special_embeddings=meta.special_embeddings,dtype=torch.float32,attn='sdpa');source=ck.warm_start(m,meta);m.head.temperature=1.;m.lm.config.use_cache=False
 if checkpointing:m.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 m.train();wrapped=DDP(LossModel(m),device_ids=[rank],broadcast_buffers=False,find_unused_parameters=False)
 params=[p for p in m.parameters() if p.requires_grad];assert all('lora_' in n or n.startswith('head.') for n,p in m.named_parameters() if p.requires_grad)
 if rank==0:write(R/'results/runtime.json',{'init_source':source,'original_meta':{k:v for k,v in ck.meta.to_dict().items() if k!='head'},'trainable_parameters':sum(p.numel() for p in params),'frozen_parameters':sum(p.numel() for p in m.parameters() if not p.requires_grad),'precision':'FP32 master weights, BF16 autocast, TF32 permitted','world_size':dist.get_world_size(),'checkpointing':checkpointing,'gpus':[str(torch.cuda.get_device_properties(i)) for i in range(torch.cuda.device_count())],'evaluator_source_commit':'90990a5fac2995b9faa3190f7d437e84f2067768','script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()})
 return rank,ck,meta,tok,m,wrapped,params

def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['probe','train'],required=True);ap.add_argument('--batch',type=int,default=16);ap.add_argument('--checkpointing',type=int,default=1);a=ap.parse_args();rank,ck,meta,tok,m,net,params=setup(bool(a.checkpointing));rows=pickle.load((R/'data/train.pkl').open('rb'));profile=json.load((R/'results/data-profile.json').open());scale=len(rows)/profile['train']['records'];opt=torch.optim.AdamW(params,lr=2e-5,weight_decay=.01);N=len(rows);ordered=sorted(range(N),key=lambda i:rows[i]['length'])
 def step(batch,denom,sync=True):
  with (contextlib.nullcontext() if sync else net.no_sync()):
   with torch.autocast('cuda',dtype=torch.bfloat16):loss=net(batch)*scale/denom
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
   loss.backward()
  return float(loss.detach())
 if a.mode=='probe':
  results=[]
  # Verify added right padding preserves upstream FP32 logits before using the batching adapter.
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;m.eval();pair=[rows[ordered[N//2]]['enc'],rows[ordered[-1]]['enc']]
  with torch.no_grad():
   m.padding_bucket=1;za=m.forward_batch(pair);m.padding_bucket=128;zb=m.forward_batch(pair)
   err=max(float((a0[0]-b0[0]).abs().max()) for a0,b0 in zip(za,zb));assert err<1e-3,err
  m.train();torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
  if rank==0:write(R/'results/padding-parity.json',{'max_fp32_logit_difference':err,'tolerance':1e-3,'strictly_same_tokens_positions_and_options':True})
  # Timings cover all length deciles and the maximum-length case, using actual training inputs.
  for quantile in [1.0,.05,.15,.25,.35,.45,.55,.65,.75,.85,.95]:
   midpoint=min(N-1,int(N*quantile));a.batch=microbatch(rows[ordered[min(N-1,midpoint+128)]]['length']);start=max(0,min(N-a.batch*2,midpoint-a.batch));ids=ordered[start+rank*a.batch:start+(rank+1)*a.batch];batch=[rows[i] for i in ids]
   for rep in range(3):
    opt.zero_grad(set_to_none=True);dist.barrier();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter();loss=step(batch,a.batch);norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();sec=time.perf_counter()-t;stats=torch.tensor([sec,torch.cuda.max_memory_allocated()/2**20,torch.cuda.max_memory_reserved()/2**20],device='cuda');dist.all_reduce(stats,op=dist.ReduceOp.MAX)
    if rep>0:results.append({'quantile':quantile,'rep':rep,'seconds':stats[0].item(),'peak_allocated_mib':stats[1].item(),'peak_reserved_mib':stats[2].item(),'max_length':max(r['length'] for r in batch),'loss':loss,'grad_norm':float(norm),'global_rows':2*a.batch})
   if rank==0:print('PROBE',a.batch,quantile,json.dumps(results[-1]),flush=True)
  if rank==0:
   write(R/'results'/'probe-adaptive.json',{'batch_policy':'adaptive 40960 padded tokens / GPU; max64; bucket128','world_size':2,'checkpointing':a.checkpointing,'results':results,'no_checkpoint_from_probe':True})
  dist.destroy_process_group();return
 # One epoch, all rows exactly once; no oversampling, augmentation or test inputs. Large effective batch = 128 question rows.
 seed=20260923;rng=random.Random(seed);ids=list(range(N));rng.shuffle(ids)
 # Sort within shuffled 2048-row pools, then shuffle batch order: close lengths with no global length curriculum.
 batches=[]
 for start in range(0,N,2048):
  pool=sorted(ids[start:start+2048],key=lambda i:rows[i]['length'])
  batches += [pool[j:j+128] for j in range(0,len(pool),128)]
 rng.shuffle(batches);updates=len(batches);warmup=max(1,round(.05*updates))
 def factor(s):return (s+1)/warmup if s<warmup else .1+.9*.5*(1+math.cos(math.pi*(s-warmup)/max(1,updates-warmup)))
 scheduler=torch.optim.lr_scheduler.LambdaLR(opt,factor);out=R/'runs/secjev-0.8b-v1';assert not out.exists();dist.barrier()
 if rank==0:
  out.mkdir(parents=True);write(out/'training_config.json',{'epoch':1,'learning_rate':2e-5,'batch_per_gpu_question_rows':'adaptive 1..64, 40960 padded tokens, bucket128','world_size':2,'accumulation':'dynamic to 128 global question rows','effective_global_question_batch':128,'objective':'per-scene mean question cross entropy; per-question weight=1/question_count; normalized to mean training scene weight','context_limit':8192,'records':profile['train']['records'],'questions':N,'no_truncation':True,'replay':0,'augmentations':False,'shuffle_seed':seed,'length_bucketing_pool':2048,'schedule':'5% linear warmup + cosine decay to 10% lr','updates':updates,'precision':'FP32 master/BF16 autocast','cuda_allocator':'expandable_segments:True','run_purpose':'initial SecJev-0.8B release on SecJev-Corpus v1.0.0 (internal v1.7)' ,'gradient_checkpointing':bool(a.checkpointing),'original_checkpoint_revision':json.load(open(INIT+'/inference-provenance.json'))})
 dist.barrier();t0=time.perf_counter();seen=0;global_step=0;losses=[];used=[];opt.zero_grad(set_to_none=True)
 def save(name,include_optimizer=False):
  if rank==0:
   dest=out/name;dest.mkdir(exist_ok=False);m.lm.peft_config['default'].base_model_name_or_path=meta.base;m.lm.save_pretrained(dest);meta.head={k:v.detach().cpu() for k,v in m.head.state_dict().items()};meta.temperature=1.;meta.extra={**meta.extra,'security_training':{'step':global_step,'seen_rows':seen,'initial_checkpoint':INIT},'temperature_fit':'not yet calibrated'};write_meta(dest,meta);tok.save_pretrained(dest)
   if include_optimizer:torch.save({'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'global_step':global_step,'torch_rng':torch.get_rng_state()},dest/'optimizer.pt')
 # A fixed, small development panel checks loss on the same examples, never used for updates or model selection.
 devrows=pickle.load((R/'data/development.pkl').open('rb'));monitor_groups={}
 for row in devrows:monitor_groups.setdefault((row['source'],row['task'],row['qtype']),[]).append(row)
 monitor=[]
 for key,items in sorted(monitor_groups.items()):
  items.sort(key=lambda r:hashlib.sha256(f"monitor:{r['case']}:{r['qid']}".encode()).hexdigest());monitor.extend(items[:4])
 del devrows
 def check_monitor():
  dist.barrier();m.eval();ce=0.;correct=0;n=0
  tf32=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
  with torch.no_grad():
   for row in monitor[rank::2]:
    z=m.forward(row['enc'])[0].float();ce+=float(torch.nn.functional.cross_entropy(z[None],torch.tensor([row['label']],device=m.device)));correct+=int(int(z.argmax())==row['label']);n+=1
  sums=torch.tensor([ce,correct,n],device=m.device,dtype=torch.float64);dist.all_reduce(sums)
  if rank==0:
   record={'step':global_step,'split':'development','fixed_questions':int(sums[2]),'loss':float(sums[0]/sums[2]),'accuracy':float(sums[1]/sums[2]),'used_for_training_or_selection':False}
   with (R/'results/monitor-history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
   print('MONITOR',json.dumps(record),flush=True)
  m.train();torch.backends.cuda.matmul.allow_tf32=tf32;torch.cuda.empty_cache();dist.barrier()
 check_monitor()
 for whole in batches:
  if rank==0:print('UPDATE_START',global_step+1,'max_length',max(rows[i]['length'] for i in whole),flush=True)
  a.batch=microbatch(max(rows[i]['length'] for i in whole));group=[whole[j:j+2*a.batch] for j in range(0,len(whole),2*a.batch)];true_count=len(whole);total=0.
  for j,b in enumerate(group):
   # Final uneven microbatch is padded with zero-weight copies, so DDP shapes/steps match without duplicated supervision.
   real=len(b);padded=b+[b[-1]]*(2*a.batch-real);local=[]
   for pos in range(rank*a.batch,(rank+1)*a.batch):
    row=rows[padded[pos]]
    if pos>=real:row={**row,'weight':0.}
    local.append(row)
   total+=step(local,true_count/2,sync=j==len(group)-1);used.extend(b);seen+=real
  norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step();scheduler.step();opt.zero_grad(set_to_none=True);global_step+=1
  st=torch.tensor([total],device='cuda');dist.all_reduce(st);losses.append(float(st.item()/2))
  if rank==0 and (True):
   status={'stage':'training','run':'secjev-0.8b-v1','step':global_step,'steps':updates,'seen_rows':seen,'total_rows':N,'loss_step':losses[-1],'loss_recent':sum(losses[-10:])/len(losses[-10:]),'loss_recent_50':sum(losses[-50:])/len(losses[-50:]),'grad_norm':float(norm),'elapsed_s':time.perf_counter()-t0,'estimated_remaining_s':(time.perf_counter()-t0)/global_step*(updates-global_step),'batch_per_gpu':a.batch,'effective_global_batch':128,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'learning_rate':opt.param_groups[0]['lr']}
   write(R/'results/train-status.json',status);print('TRAIN',json.dumps(status),flush=True)
  if global_step==10 or global_step%50==0:check_monitor()
  if global_step%50==0:save(f'step-{global_step}',include_optimizer=True);dist.barrier()
 assert len(used)==N and len(set(used))==N and seen==N;save('checkpoint',include_optimizer=True);dist.barrier()
 if rank==0:write(R/'results/train-status.json',{'stage':'complete','steps':global_step,'seen_rows':seen,'unique_rows':len(set(used)),'elapsed_s':time.perf_counter()-t0,'checkpoint':str(out/'checkpoint'),'calibration':'pending'})
 dist.destroy_process_group()
if __name__=='__main__':main()
