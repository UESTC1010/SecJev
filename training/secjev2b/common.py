from paths import WORK,SCRIPTS,SOURCE,BASE
import sys, os, pathlib, math
sys.path.insert(0, str(SCRIPTS))
from train_support import BucketDecisionModel, write
BucketDecisionModel.padding_bucket = 1
from kev.model import load_tokenizer, rows_of
from kev.train import question_loss
import torch
R = pathlib.Path(str(WORK))
BASE = str(BASE)

class LossModel(torch.nn.Module):

    def __init__(self, m):
        super().__init__()
        self.model = m

    def forward(self, batch):
        zs = self.model.forward_batch([r['enc'] for r in batch])
        return sum((question_loss(z[0].float(), r, self.model.device, 0.0) * r['weight'] for z, r in zip(zs, batch)))

def microbatch(length):
    budget = int(os.environ.get('SECJEV_TOKEN_BUDGET', '32768'))
    padded = math.ceil(length / 128) * 128
    return max(1, max([b for b in [1, 2, 4, 8, 16, 32, 64] if b * padded <= budget], default=1))

def split_variant(v):
    S, Sp, branches = rows_of(v.enc)
    out = []
    for q, b in zip(v.rec['questions'], branches):
        en = {'ids': S + b['ids'], 'pos': Sp + b['pos'], 'seg': [0] * len(S) + [1] * len(b['ids']), 'decide_idx': [len(S) + b['decide']], 'opt_idx': [[len(S) + x for x in b['opts']]], 'option_isolation': False}
        out.append({'enc': en, 'length': len(en['ids']), 'label': q['label'], 'qtype': q['qtype'], 'target': q.get('target'), 'weight': 1 / len(branches)})
    return out
