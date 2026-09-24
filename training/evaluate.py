"""Frozen-checkpoint FP32 evaluation. Calibration never uses development or test labels."""
import os,sys,pathlib,json,pickle,math,copy,shutil,collections,time,argparse
os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
sys.path.insert(0,str(pathlib.Path(os.environ.get('SECJEV_WORK', './secjev-work')).resolve()/'source/kev'))
import torch
from kev.checkpoint import Checkpoint,LoadOptions,write_meta
R=pathlib.Path(os.environ.get('SECJEV_WORK', './secjev-work')).resolve();BASE=str(R/'base');RUN=R/'runs/secjev-0.8b-v1'
def write(p,x):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def summary(rows,temp):
 c=collections.Counter();bylabel=collections.defaultdict(lambda:[0,0]);ece=[[0.,0,0] for _ in range(15)];mae=0.;ns=0;accepted=0;wrongaccepted=0
 for r in rows:
  p=torch.softmax(torch.tensor(r['logits'],dtype=torch.float64)/temp,0);pred=int(p.argmax());ok=pred==r['label'];confidence=float(p.max());c['n']+=1;c['correct']+=ok;c['nll']-=math.log(max(float(p[r['label']]),1e-30));c['brier']+=float((p*p).sum()-2*p[r['label']]+1);bylabel[(r['task'],str(r['semantic_label']))][0]+=ok;bylabel[(r['task'],str(r['semantic_label']))][1]+=1
  b=ece[min(14,int(confidence*15))];b[0]+=confidence;b[1]+=ok;b[2]+=1
  if confidence>=.9:accepted+=1;wrongaccepted+=not ok
  if r['qtype']=='score':mae+=abs(sum(i*float(v) for i,v in enumerate(p))-r['label']);ns+=1
 return {'questions':c['n'],'accuracy':c['correct']/c['n'],'mean_class_recall':sum(v[0]/v[1] for v in bylabel.values())/len(bylabel),'nll':c['nll']/c['n'],'brier':c['brier']/c['n'],'ece15':sum(abs(b[0]-b[1]) for b in ece)/c['n'],'temperature':temp,'coverage_at_confidence_0_9':accepted/c['n'],'error_rate_at_confidence_0_9':wrongaccepted/accepted if accepted else None,'ordinal_expected_level_mae':mae/ns if ns else None}
def metrics(rows,temp):
 out=summary(rows,temp);groups=collections.defaultdict(list)
 for r in rows:groups[(r['source'],r['task'])].append(r)
 out['tasks']={s+':'+t:summary(rr,temp) for (s,t),rr in sorted(groups.items())};out['macro_task_accuracy']=sum(v['accuracy'] for v in out['tasks'].values())/len(out['tasks']);out['macro_task_balanced_accuracy']=sum(v['mean_class_recall'] for v in out['tasks'].values())/len(out['tasks']);return out

def fit(rows):
 K=max(len(r['logits']) for r in rows);z=torch.full((len(rows),K),-1e4,dtype=torch.float64);y=torch.tensor([r['label'] for r in rows])
 for i,r in enumerate(rows):z[i,:len(r['logits'])]=torch.tensor(r['logits'],dtype=torch.float64)
 lt=torch.tensor(0.,dtype=torch.float64,requires_grad=True);opt=torch.optim.LBFGS([lt],max_iter=60,line_search_fn='strong_wolfe')
 def closure():
  opt.zero_grad();loss=torch.nn.functional.cross_entropy(z/lt.clamp(-4,4).exp(),y);loss.backward();return loss
 opt.step(closure);t=float(lt.detach().clamp(-4,4).exp());assert math.isfinite(t);return t

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['parent','trained'],required=True);ap.add_argument('--allow-test',action='store_true');a=ap.parse_args()
 if a.allow_test:assert (R/'results/frozen-final-candidate.json').exists()
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
 path=str(R/'init') if a.model=='parent' else str(RUN/'checkpoint');ck=Checkpoint(path);original=copy.deepcopy(ck.meta);ck.meta.base=BASE;ck.meta.base_revision=None;tok,m=ck.load('cuda:0',LoadOptions(dtype=torch.float32,merge=False,attn='sdpa',temperature=1.));m.eval();results={};started=time.time()
 for split in (['test'] if a.allow_test else ['calibration','development']):
  rows=pickle.load((R/'data'/f'{split}.pkl').open('rb'));ordered=sorted(rows,key=lambda r:r['length']);out=[];cursor=0
  with torch.inference_mode():
   while cursor<len(ordered):
    n=min(16,len(ordered)-cursor)
    while n>1 and (n*ordered[cursor+n-1]['length']>8192 or n*ordered[cursor+n-1]['length']**2>8000000):n-=1
    batch=ordered[cursor:cursor+n];zs=m.forward_batch([r['enc'] for r in batch]);torch.cuda.synchronize()
    for r,z in zip(batch,zs):
     v=z[0].float().cpu().tolist();assert all(math.isfinite(x) for x in v);out.append({k:r[k] for k in ['case','qid','source','task','qtype','label','semantic_label','option_keys']}|{'logits':v})
    cursor+=n
    if cursor%500<n:print(a.model,split,cursor,len(rows),flush=True)
  assert len(out)==len(rows) and len({(r['case'],r['qid']) for r in out})==len(rows);out.sort(key=lambda r:(r['case'],r['qid']));results[split]=out
  with (R/'results'/f'{a.model}-{split}-logits.jsonl').open('w') as f:
   for r in out:f.write(json.dumps(r)+'\n')
 del m;torch.cuda.empty_cache()
 if a.allow_test:temp=json.load((R/'results'/f'{a.model}-validation.json').open())['temperature']
 else:
  with torch.enable_grad():temp=fit(results['calibration'])
 report={'model':a.model,'temperature_fitted_only_on':'calibration','temperature':temp,'elapsed_s':time.time()-started,'inference':'FP32 unmerged, raw logits, TF32/flash SDPA off','test_used_for_training_or_selection':False}
 for split,rr in results.items():report[split]={'raw':metrics(rr,1.),'recalibrated':metrics(rr,temp)}
 if a.model=='trained' and not a.allow_test:
  dest=RUN/'calibrated';shutil.copytree(path,dest,ignore=shutil.ignore_patterns('optimizer.pt'));original.temperature=temp;original.extra['temperature_fit']={'split':'calibration','questions':len(results['calibration']),'data_sha256':json.load((R/'data/manifest.json').open())['calibration']['sha256']};write_meta(dest,original);report['checkpoint']=str(dest)
 write(R/'results'/f"{a.model}-{'test' if a.allow_test else 'validation'}.json",report);print('COMPLETE',a.model,list(results),'temperature',temp,flush=True)
if __name__=='__main__':main()
