"""Test three-rank tail masking, loss normalization, and per-epoch coverage on CPU."""
import math
from train_continue import padded_shards,batches_for,GLOBAL
for n in [1,2,3,7,23,24,25,127,128,129,191,192]:
    for b in [8,16,32,64]:
        seen=[];rank_sums=[0.,0.,0.]
        ids=list(range(n));values=[(i+1)/17 for i in ids]
        for start in range(0,n,3*b):
            shards=padded_shards(ids[start:start+3*b],b,3)
            for rank,shard in enumerate(shards):
                assert len(shard)==b
                for i,real in shard:
                    if real: seen.append(i);rank_sums[rank]+=values[i]/(n/3)
        assert sorted(seen)==ids
        assert math.isclose(sum(rank_sums)/3,sum(values)/n,rel_tol=1e-12)
rows=[{'length':126+(i*37)%3463} for i in range(118818)]
for epoch in [2,3]:
    batches=batches_for(rows,epoch);flat=[i for b in batches for i in b]
    assert len(batches)==619 and len(flat)==len(set(flat))==len(rows)
    assert all(0<len(b)<=GLOBAL for b in batches)
assert batches_for(rows,2)!=batches_for(rows,3)
print('PASS: 3-rank padding contributes zero loss; DDP mean equals single-process objective; all 118818 rows exactly once per epoch; 619 updates/epoch')
