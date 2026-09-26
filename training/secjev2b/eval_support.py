from paths import *
from metrics import metrics,fit,write
import pathlib,json,pickle,hashlib,subprocess
R=WORK
def call(name, args, env=None):
    with (R / 'logs' / f'{name}.log').open('w') as f:
        rc = subprocess.call(args, stdout=f, stderr=subprocess.STDOUT, env=env)
    (R / 'logs' / f'{name}.exit').write_text(str(rc) + '\n')
    assert rc == 0, (name, rc)

def load_rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f]

def validate_logits(rows, split):
    expected = pickle.load((R / 'data' / f'{split}.pkl').open('rb'))
    lookup = {(x['case'], x['qid']): x for x in expected}
    assert len(rows) == len(lookup) == len({(x['case'], x['qid']) for x in rows})
    for x in rows:
        row = lookup[x['case'], x['qid']]
        assert all((x[k] == row[k] for k in ['source', 'task', 'qtype', 'label', 'semantic_label', 'option_keys']))
        assert len(x['logits']) == row['k']
    return sorted(rows, key=lambda x: (x['case'], x['qid']))

def file_hashes(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}
