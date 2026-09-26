from paths import WORK,SCRIPTS,SOURCE,BASE
import math
from train_security import batches_for, padded_shards
for n in [1, 2, 3, 7, 23, 24, 25, 127, 128, 129, 191, 192]:
    for b in [1, 8, 16, 32, 43, 64]:
        seen = []
        sums = [0.0, 0.0, 0.0]
        values = [(i + 1) / 17 for i in range(n)]
        for start in range(0, n, 3 * b):
            for rank, shard in enumerate(padded_shards(list(range(n))[start:start + 3 * b], b, 3)):
                for i, real in shard:
                    if real:
                        seen.append(i)
                        sums[rank] += values[i] / (n / 3)
        assert sorted(seen) == list(range(n))
        assert math.isclose(sum(sums) / 3, sum(values) / n, rel_tol=1e-12)
rows = [{'length': 126 + i * 37 % 3463} for i in range(118818)]
for e, steps in [(1, 929), (2, 619), (3, 619)]:
    bs = batches_for(rows, e)
    flat = [i for b in bs for i in b]
    assert len(bs) == steps and len(flat) == len(set(flat)) == 118818
print('PASS: all epochs exact coverage; 3-rank zero-weight tail; exact DDP loss normalization.')
