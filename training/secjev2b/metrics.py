import collections,math,json,torch
def write(p, x):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)

def summary(rows, temp):
    c = collections.Counter()
    bylabel = collections.defaultdict(lambda: [0, 0])
    ece = [[0.0, 0, 0] for _ in range(15)]
    mae = 0.0
    ns = 0
    accepted = 0
    wrongaccepted = 0
    for r in rows:
        p = torch.softmax(torch.tensor(r['logits'], dtype=torch.float64) / temp, 0)
        pred = int(p.argmax())
        ok = pred == r['label']
        confidence = float(p.max())
        c['n'] += 1
        c['correct'] += ok
        c['nll'] -= math.log(max(float(p[r['label']]), 1e-30))
        c['brier'] += float((p * p).sum() - 2 * p[r['label']] + 1)
        bylabel[r['task'], str(r['semantic_label'])][0] += ok
        bylabel[r['task'], str(r['semantic_label'])][1] += 1
        b = ece[min(14, int(confidence * 15))]
        b[0] += confidence
        b[1] += ok
        b[2] += 1
        if confidence >= 0.9:
            accepted += 1
            wrongaccepted += not ok
        if r['qtype'] == 'score':
            mae += abs(sum((i * float(v) for i, v in enumerate(p))) - r['label'])
            ns += 1
    return {'questions': c['n'], 'accuracy': c['correct'] / c['n'], 'mean_class_recall': sum((v[0] / v[1] for v in bylabel.values())) / len(bylabel), 'nll': c['nll'] / c['n'], 'brier': c['brier'] / c['n'], 'ece15': sum((abs(b[0] - b[1]) for b in ece)) / c['n'], 'temperature': temp, 'coverage_at_confidence_0_9': accepted / c['n'], 'error_rate_at_confidence_0_9': wrongaccepted / accepted if accepted else None, 'ordinal_expected_level_mae': mae / ns if ns else None}

def metrics(rows, temp):
    out = summary(rows, temp)
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r['source'], r['task']].append(r)
    out['tasks'] = {s + ':' + t: summary(rr, temp) for (s, t), rr in sorted(groups.items())}
    out['macro_task_accuracy'] = sum((v['accuracy'] for v in out['tasks'].values())) / len(out['tasks'])
    out['macro_task_balanced_accuracy'] = sum((v['mean_class_recall'] for v in out['tasks'].values())) / len(out['tasks'])
    return out

def fit(rows):
    K = max((len(r['logits']) for r in rows))
    z = torch.full((len(rows), K), -10000.0, dtype=torch.float64)
    y = torch.tensor([r['label'] for r in rows])
    for i, r in enumerate(rows):
        z[i, :len(r['logits'])] = torch.tensor(r['logits'], dtype=torch.float64)
    lt = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([lt], max_iter=60, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(z / lt.clamp(-4, 4).exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    t = float(lt.detach().clamp(-4, 4).exp())
    assert math.isfinite(t)
    return t
