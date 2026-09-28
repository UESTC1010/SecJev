from paths import WORK,SCRIPTS,SOURCE,BASE
import pathlib, json, sys, shutil
sys.path.insert(0, str(SCRIPTS))
from eval_helpers import write
R = pathlib.Path(str(WORK))
O = R / 'report'
O.mkdir(exist_ok=True)
sel = json.load(open(R / 'results/checkpoint-selection.json'))
rows = []
selected = R / f"runs/secjev-9b-3epochs/epoch-{sel['selected_epoch']}-calibrated"
dest = R / 'runs/secjev-9b-3epochs/checkpoint'
if not dest.exists():
    shutil.copytree(selected, dest)
assert (dest / 'head.pt').read_bytes() == (selected / 'head.pt').read_bytes()
assert (dest / 'adapter_model.safetensors').read_bytes() == (selected / 'adapter_model.safetensors').read_bytes()
for i in [1, 2, 3]:
    v = json.load(open(R / f'results/epoch-{i}-validation.json'))
    t = json.load(open(R / f'results/epoch-{i}-test.json'))
    g = {s: json.load(open(R / f'general/epoch-{i}/{s}/extended.json')) for s in ['decision-v7', 'transfer-v4']}
    rows.append({'epoch': i, 'development': v['development'], 'test': t['test'], 'general': g, 'temperature': v['temperature']})
parent = json.load(open(R / 'results/kev-9b-test.json'))
gparent = {s: json.load(open(R / f'general/kev-9b/{s}/extended.json')) for s in ['decision-v7', 'transfer-v4']}
write(O / 'results.json', {'selected_epoch': sel['selected_epoch'], 'epochs': rows, 'kev_9b_security': parent, 'kev_9b_general': gparent, 'base_revision': '68c46c4b3498877f3ef123c856ecfde50c39f404', 'test_used_for_selection': False})
lines = ['# SecJev-9B 训练与评测结果', '', '官方 Kev-9B → SecJev-Corpus 三轮。训练使用全部 9 张 GPU。使用 LoRA r16 和决策头，BF16 冻结骨干、FP32 LoRA/决策头/优化器，BF16 混合精度训练；评测使用 BF16 骨干、FP32 LoRA 和决策头。', '', f"开发验证集选择第 {sel['selected_epoch']} 轮；三个 checkpoint 均按预定方案测试，测试集不用于选择。", '', '| 模型 | 总准确率 | 任务宏平均准确率 | 任务宏平均平衡准确率 |', '|---|---:|---:|---:|']
for name, m in [('官方 Kev-9B', parent['test']['raw'])] + [(f"SecJev epoch {r['epoch']}", r['test']['raw']) for r in rows]:
    lines.append(f"| {name} | {m['accuracy']:.2%} | {m['macro_task_accuracy']:.2%} | {m['macro_task_balanced_accuracy']:.2%} |")
lines += ['', '| 安全任务 | 题数 | 官方 Kev-9B | Epoch1 | Epoch2 | Epoch3 |', '|---|---:|---:|---:|---:|---:|']
for task, m in parent['test']['raw']['tasks'].items():
    lines.append(f"| {task} | {m['questions']} | {m['accuracy']:.2%} | " + ' | '.join((f"{r['test']['raw']['tasks'][task]['accuracy']:.2%}" for r in rows)) + ' |')
lines += ['', '| 模型 | decision-v7 | transfer-v4 |', '|---|---:|---:|']
for name, g in [('官方 Kev-9B', gparent)] + [(f"SecJev epoch {r['epoch']}", r['general']) for r in rows]:
    lines.append(f"| {name} | {g['decision-v7']['raw_clean']['acc']:.2%} | {g['transfer-v4']['raw_clean']['acc']:.2%} |")
lines += ['', '准确率均为 argmax 判定。各安全 checkpoint 的概率温度只用校准集拟合；详见 results.json 的 NLL、Brier、ECE 等指标。通用测试没有日期事实补充。', '', '从官方 jaredpalmer/kev-9b 的 LoRA 和决策头继续训练，底座为 Qwen3.5-9B-Base。安全训练沿用现有0.8B/2B配方，所有问题完整编码。使用既有研究测试集，不能称为新的盲测。']
(O / '报告.md').write_text('\n'.join(lines) + '\n')
