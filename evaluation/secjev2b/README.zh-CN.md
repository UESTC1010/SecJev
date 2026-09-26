# SecJev-2B 两阶段训练结果

Qwen3.5-2B-Base → Kev 主训练及补充训练 → SecJev-Corpus 三轮。使用 LoRA r16 和决策头，FP32 参数、BF16 训练；评测为 FP32。

开发验证集选择第 3 轮；三个 checkpoint 均按预定方案测试，测试集不用于选择。

| 模型 | 总准确率 | 任务宏平均准确率 | 任务宏平均平衡准确率 |
|---|---:|---:|---:|
| Kev 阶段 | 40.44% | 52.15% | 51.34% |
| SecJev epoch 1 | 84.76% | 91.63% | 88.15% |
| SecJev epoch 2 | 84.91% | 92.12% | 88.59% |
| SecJev epoch 3 | 86.48% | 92.98% | 90.31% |

| 安全任务 | 题数 | Kev 阶段 | Epoch1 | Epoch2 | Epoch3 |
|---|---:|---:|---:|---:|---:|
| agentdojo_tool_outputs:agentdojo_boundary_action | 426 | 40.85% | 91.31% | 92.02% | 92.96% |
| agentdojo_tool_outputs:agentdojo_embedded_injection | 426 | 29.81% | 90.85% | 92.02% | 92.72% |
| byzfl_mnist:fl_attack_generated_update | 800 | 80.00% | 95.75% | 97.62% | 97.50% |
| ciciot2023:ciciot_attack | 6000 | 25.00% | 90.82% | 89.45% | 92.13% |
| injecagent:injecagent_requested_operation_v13 | 124 | 51.61% | 97.58% | 98.39% | 98.39% |
| lanl_auth_policy:lanl_known_logon_action_v16 | 540 | 72.78% | 98.70% | 98.52% | 99.81% |
| lanl_auth_policy:lanl_observable_policy | 540 | 99.44% | 100.00% | 100.00% | 100.00% |
| ton_iot_network:ton_attack | 3500 | 25.00% | 97.86% | 97.83% | 98.97% |
| twins_streamlet:twins_evidence_level | 659 | 40.06% | 99.09% | 99.39% | 99.85% |
| twins_streamlet:twins_policy_action | 659 | 39.45% | 98.94% | 99.24% | 100.00% |
| twins_streamlet:twins_quorum | 659 | 64.95% | 98.94% | 98.79% | 99.85% |
| twins_streamlet:twins_visible_conflict | 659 | 70.11% | 98.94% | 99.85% | 99.70% |
| veremi_nextgen:veremi_message_manipulation | 3321 | 48.12% | 63.90% | 63.78% | 66.21% |
| veremi_nextgen:veremi_supported_attack_type_v16 | 3321 | 42.97% | 60.10% | 62.75% | 63.69% |

| 模型 | decision-v7 | transfer-v4 |
|---|---:|---:|
| Kev 阶段 | 84.00% | 75.91% |
| SecJev epoch 1 | 84.50% | 75.91% |
| SecJev epoch 2 | 84.67% | 76.22% |
| SecJev epoch 3 | 84.50% | 75.91% |

准确率均为 argmax 判定。各安全 checkpoint 的概率温度只用校准集拟合；详见 results.json 的 NLL、Brier、ECE 等指标。通用测试没有日期事实补充。

Kev 没有官方 2B 配方：本次沿用其数据、决策头、LoRA、增广、CE、2轮主训练和1轮补充，学习率采用4B的5e-5/2e-5，固定一个种子，没有进行多种子择优。安全训练沿用现有0.8B配方，所有问题完整编码。使用既有研究测试集，不能称为新的盲测。
