# SecJev-4B 训练与评测结果

官方 Kev-4B → SecJev-Corpus 三轮。训练使用全部 9 张 GPU。使用 LoRA r16 和决策头，FP32 参数、BF16 训练；评测为 FP32。

开发验证集 14,912 题选择第 3 轮，校准集 14,821 题只用于温度拟合。安全测试集为完整的 21,634 题、14 项任务；三个 checkpoint 均按预定方案测试，测试集不用于选择。

总准确率按题目加权；任务宏平均对 14 项任务等权；任务宏平均平衡准确率先在每项任务内对语义类别召回率等权，再对任务等权。

| 模型 | 总准确率 | 任务宏平均准确率 | 任务宏平均平衡准确率 |
|---|---:|---:|---:|
| 官方 Kev-4B | 50.49% | 67.88% | 67.16% |
| SecJev epoch 1 | 85.88% | 93.38% | 90.74% |
| SecJev epoch 2 | 87.12% | 94.15% | 91.69% |
| SecJev epoch 3 | 88.01% | 94.51% | 92.42% |

| 安全任务 | 题数 | 官方 Kev-4B | Epoch1 | Epoch2 | Epoch3 |
|---|---:|---:|---:|---:|---:|
| agentdojo_tool_outputs:agentdojo_boundary_action | 426 | 54.23% | 99.53% | 99.53% | 99.53% |
| agentdojo_tool_outputs:agentdojo_embedded_injection | 426 | 73.00% | 98.36% | 98.36% | 98.36% |
| byzfl_mnist:fl_attack_generated_update | 800 | 79.88% | 93.88% | 95.25% | 95.88% |
| ciciot2023:ciciot_attack | 6000 | 26.28% | 90.85% | 89.95% | 91.72% |
| injecagent:injecagent_requested_operation_v13 | 124 | 62.90% | 100.00% | 100.00% | 100.00% |
| lanl_auth_policy:lanl_known_logon_action_v16 | 540 | 79.81% | 100.00% | 100.00% | 100.00% |
| lanl_auth_policy:lanl_observable_policy | 540 | 99.44% | 100.00% | 100.00% | 100.00% |
| ton_iot_network:ton_attack | 3500 | 50.66% | 98.83% | 98.94% | 98.80% |
| twins_streamlet:twins_evidence_level | 659 | 81.18% | 99.09% | 99.85% | 99.54% |
| twins_streamlet:twins_policy_action | 659 | 71.78% | 99.39% | 99.70% | 99.85% |
| twins_streamlet:twins_quorum | 659 | 78.15% | 99.70% | 99.85% | 100.00% |
| twins_streamlet:twins_visible_conflict | 659 | 97.12% | 99.70% | 99.70% | 99.85% |
| veremi_nextgen:veremi_message_manipulation | 3321 | 47.76% | 65.46% | 68.38% | 70.34% |
| veremi_nextgen:veremi_supported_attack_type_v16 | 3321 | 48.09% | 62.48% | 68.59% | 69.23% |

通用评测完整运行 decision-v7 的 1,440 题和 transfer-v4 的 764 题（含扰动变体）。下表沿用原先口径，只汇总 clean 子集，分别为 1,200 题和 656 题。

| 模型 | decision-v7 | transfer-v4 |
|---|---:|---:|
| 官方 Kev-4B | 87.08% | 83.69% |
| SecJev epoch 1 | 86.67% | 83.84% |
| SecJev epoch 2 | 86.50% | 83.69% |
| SecJev epoch 3 | 86.58% | 83.84% |

准确率均为 argmax 判定。各安全 checkpoint 的概率温度只用校准集拟合；详见 results.json 的 NLL、Brier、ECE 等指标。通用测试没有日期事实补充。

从官方 jaredpalmer/kev-4b 的 LoRA 和决策头继续训练，底座为 Qwen3.5-4B-Base。安全训练沿用现有0.8B/2B配方，所有问题完整编码。使用既有研究测试集，不能称为新的盲测。
