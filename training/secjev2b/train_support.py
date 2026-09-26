from paths import *
import json,torch
from kev.model import DecisionModel
def write(p, data):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)

class BucketDecisionModel(DecisionModel):
    padding_bucket = 128

    def _pad_rows(self, rows):
        ids, pos, att = super()._pad_rows(rows)
        extra = -ids.shape[1] % self.padding_bucket
        if extra:
            ids = torch.nn.functional.pad(ids, (0, extra), value=self.pad_id)
            pos = torch.nn.functional.pad(pos, (0, extra))
            att = torch.nn.functional.pad(att, (0, extra))
        return (ids, pos, att)
