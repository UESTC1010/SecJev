"""FP32 inference-only sharding. Selection and temperature fitting run separately."""
import os, sys, pathlib, argparse, pickle, json, math, time
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TOKENIZERS_PARALLELISM']='false'
sys.path.insert(0,str(pathlib.Path(os.environ.get('SECJEV_WORK', './secjev-work')).resolve()/'source/kev'))
import torch
from kev.checkpoint import Checkpoint, LoadOptions
WORK=pathlib.Path(os.environ.get('SECJEV_WORK', './secjev-work')).resolve()
R=WORK/'continuation'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--split',choices=['development','calibration','test'],required=True);ap.add_argument('--tag',required=True);ap.add_argument('--shard',type=int,required=True);ap.add_argument('--shards',type=int,default=3);a=ap.parse_args()
    if a.split=='test': assert (R/'results/frozen-final-candidate.json').exists()
    torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    ck=Checkpoint(a.checkpoint);ck.meta.base=str(WORK/'base');ck.meta.base_revision=None
    tok,model=ck.load('cuda:0',LoadOptions(dtype=torch.float32,merge=False,attn='sdpa',temperature=1.))
    ordered=sorted(pickle.load((R/'data'/f'{a.split}.pkl').open('rb')),key=lambda r:r['length'])[a.shard::a.shards]
    out=[];cursor=0;started=time.time()
    with torch.inference_mode():
        while cursor<len(ordered):
            n=min(16,len(ordered)-cursor)
            while n>1 and (n*ordered[cursor+n-1]['length']>8192 or n*ordered[cursor+n-1]['length']**2>8000000):n-=1
            batch=ordered[cursor:cursor+n];logits=model.forward_batch([r['enc'] for r in batch]);torch.cuda.synchronize()
            for row,z in zip(batch,logits):
                values=z[0].float().cpu().tolist();assert all(math.isfinite(x) for x in values)
                out.append({k:row[k] for k in ['case','qid','source','task','qtype','label','semantic_label','option_keys']}|{'logits':values})
            cursor+=n
            if cursor%500<n:print(a.tag,a.split,'shard',a.shard,cursor,len(ordered),flush=True)
    path=R/'results'/f'{a.tag}-{a.split}-shard-{a.shard}.jsonl';tmp=path.with_suffix('.tmp')
    with tmp.open('w') as f:
        for row in out:f.write(json.dumps(row)+'\n')
    tmp.replace(path)
    print('COMPLETE',len(out),'seconds',time.time()-started,flush=True)


if __name__=='__main__':main()
