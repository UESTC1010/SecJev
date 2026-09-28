# SecJev-9B results

Epoch 3 was selected on the 14,912-question development split. Temperatures were fitted on a separate 14,821-question calibration split before testing. All three checkpoints were predeclared for testing.

Security test: **21,634 questions across 14 tasks**.

| Model | Micro accuracy | Macro-task accuracy | Macro-task balanced accuracy |
|---|---:|---:|---:|
| Original Kev-9B | 50.87% | 69.82% | 70.36% |
| Epoch 1 | 86.18% | 93.49% | 90.68% |
| Epoch 2 | 87.18% | 94.08% | 91.27% |
| Epoch 3 | 88.11% | 94.51% | 92.19% |

| Task | Questions | Original Kev-9B | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 3 balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| agentdojo_tool_outputs:agentdojo_boundary_action | 426 | 61.97% | 98.12% | 98.83% | 98.83% | 97.40% |
| agentdojo_tool_outputs:agentdojo_embedded_injection | 426 | 64.55% | 98.12% | 98.83% | 99.06% | 98.66% |
| byzfl_mnist:fl_attack_generated_update | 800 | 80.25% | 93.75% | 93.88% | 94.88% | 90.23% |
| ciciot2023:ciciot_attack | 6000 | 37.93% | 91.03% | 90.27% | 92.10% | 88.64% |
| injecagent:injecagent_requested_operation_v13 | 124 | 57.26% | 100.00% | 100.00% | 100.00% | 100.00% |
| lanl_auth_policy:lanl_known_logon_action_v16 | 540 | 96.11% | 100.00% | 99.63% | 100.00% | 100.00% |
| lanl_auth_policy:lanl_observable_policy | 540 | 99.81% | 100.00% | 100.00% | 100.00% | 100.00% |
| ton_iot_network:ton_attack | 3500 | 24.83% | 98.69% | 98.89% | 98.51% | 97.18% |
| twins_streamlet:twins_evidence_level | 659 | 81.94% | 100.00% | 99.85% | 100.00% | 100.00% |
| twins_streamlet:twins_policy_action | 659 | 88.16% | 100.00% | 100.00% | 100.00% | 100.00% |
| twins_streamlet:twins_quorum | 659 | 91.81% | 99.54% | 99.85% | 99.85% | 99.79% |
| twins_streamlet:twins_visible_conflict | 659 | 96.97% | 100.00% | 100.00% | 100.00% | 100.00% |
| veremi_nextgen:veremi_message_manipulation | 3321 | 47.76% | 67.63% | 69.26% | 71.39% | 72.24% |
| veremi_nextgen:veremi_supported_attack_type_v16 | 3321 | 48.15% | 62.00% | 67.90% | 68.53% | 46.53% |
| **Macro-average** | — | **69.82%** | **93.49%** | **94.08%** | **94.51%** | **92.19%** |

Macro metrics give tasks equal weight. Balanced accuracy averages semantic-class recall within each task, then across tasks. Ordered ratings use argmax accuracy. VeReMi remains the weakest area: 71.39% for message-manipulation labels and 68.53% for attack-type classification.

| General suite (clean subset) | Original Kev-9B | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---:|---:|---:|---:|
| decision-v7 (1,200 questions) | 87.50% | 87.58% | 87.42% | 87.33% |
| transfer-v4 (656 questions) | 85.06% | 84.60% | 84.60% | 84.76% |

Full suites, including controls and variants, contain 1,440 and 764 questions. No requests were rejected or truncated. Native benchmark IDs and counting are preserved, including identical requests in some permutation/control pairs; audit.json records these pairs.

All models use unmerged BF16 backbones and original FP32 LoRA/head tensors. These scores come from previously inspected research splits; source-label and explicit-policy tasks do not measure live attack detection. See results.json for probability metrics and all checkpoints.

## Development selection

| Epoch | Micro accuracy | Macro-task accuracy | Macro-task balanced accuracy |
|---|---:|---:|---:|
| 1 | 89.55% | 92.83% | 90.14% |
| 2 | 90.60% | 93.86% | 91.54% |
| 3 | 90.98% | 94.02% | 92.22% |
