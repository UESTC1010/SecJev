from paths import WORK,SCRIPTS,SOURCE,BASE
import os, sys, pathlib, gzip, json, pickle, hashlib, collections, time
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.path.insert(0, str(SOURCE))
from kev.model import encode, rows_of, load_tokenizer
from kev.data import materialize
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--allow-test', action='store_true')
args = ap.parse_args()
R = pathlib.Path(str(WORK))
tok = load_tokenizer(str(BASE))
manifest = json.load((R / 'data/manifest.json').open())
report = {}
for split, info in manifest.items():
    if (split == 'test') != args.allow_test:
        continue
    p = R / 'data' / f'{split}.jsonl.gz'
    assert hashlib.sha256(p.read_bytes()).hexdigest() == info['sha256']
    rows = []
    scenes = 0
    for i, l in enumerate(gzip.open(p, 'rt')):
        req = json.loads(l)
        assert req['_meta']['split_subset'] == split
        assert split != 'test' or args.allow_test
        rec = materialize(req)
        e = encode(tok, rec, max_state=8192, max_branch=8192, strict=True)
        assert not e['state_truncated']
        S, Sp, br = rows_of(e)
        n = len(br)
        for qid, (q, b) in enumerate(zip(rec['questions'], br)):
            en = {'ids': S + b['ids'], 'pos': Sp + b['pos'], 'seg': [0] * len(S) + [1] * len(b['ids']), 'decide_idx': [len(S) + b['decide']], 'opt_idx': [[len(S) + x for x in b['opts']]], 'option_isolation': False}
            assert rows_of(en) == (S, Sp, [b])
            rawq = req['questions'][q['qid']]
            keys = list(rawq['criteria']) if rawq['type'] == 'choice' else [False, True] if rawq['type'] == 'noul' else list(range(len(rawq['criteria'])))
            assert keys[q['label']] == rawq['label']
            rows.append({'semantic_label': rawq['label'], 'option_keys': keys, 'enc': en, 'label': q['label'], 'qtype': q['qtype'], 'weight': 1 / n, 'case': i, 'qid': q['qid'], 'source': req['_meta']['source'], 'task': q['src'], 'k': len(q['options']), 'length': len(en['ids'])})
        scenes += 1
    assert scenes == info['records'] and len(rows) == info['questions']
    assert abs(sum((x['weight'] for x in rows)) - scenes) < 1e-06
    with (R / 'data' / f'{split}.pkl').open('wb') as f:
        pickle.dump(rows, f, protocol=5)
    lengths = sorted((x['length'] for x in rows))
    report[split] = {'records': scenes, 'questions': len(rows), 'tokens': sum(lengths), 'max_length': max(lengths), 'quantiles': {str(q): lengths[min(len(lengths) - 1, int(q * len(lengths)))] for q in [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]}, 'no_truncation': True, 'pkl_sha256': hashlib.sha256((R / 'data' / f'{split}.pkl').read_bytes()).hexdigest()}
    print(split, json.dumps(report[split]), flush=True)
(R / 'results' / ('test-data-profile.json' if args.allow_test else 'data-profile.json')).write_text(json.dumps(report, indent=2) + '\n')
