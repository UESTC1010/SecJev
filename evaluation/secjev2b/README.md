# SecJev-2B results

The released checkpoint is epoch 3, selected on all 14,912 development questions by macro-task semantic-class balanced accuracy. Temperature is fitted only on the 14,821 calibration questions. All three epoch-end checkpoints were evaluated as specified before testing.

The security test uses all 21,634 questions in SecJev-Corpus v1.0.0 (14 tasks). The Kev-style stage is our own 2B decision model before security training; it is not an official Kev-2B release.

| Model | Micro accuracy | Macro-task accuracy | Macro-task balanced accuracy |
|---|---:|---:|---:|
| Kev-style stage | 40.44% | 52.15% | 51.34% |
| Epoch 1 | 84.76% | 91.63% | 88.15% |
| Epoch 2 | 84.91% | 92.12% | 88.59% |
| Epoch 3 | 86.48% | 92.98% | 90.31% |

| Task | Test questions | Kev-style stage | SecJev-2B | SecJev-2B balanced accuracy |
|---|---:|---:|---:|---:|
| agentdojo_tool_outputs:agentdojo_boundary_action | 426 | 40.85% | 92.96% | 89.55% |
| agentdojo_tool_outputs:agentdojo_embedded_injection | 426 | 29.81% | 92.72% | 89.02% |
| byzfl_mnist:fl_attack_generated_update | 800 | 80.00% | 97.50% | 96.09% |
| ciciot2023:ciciot_attack | 6,000 | 25.00% | 92.13% | 88.80% |
| injecagent:injecagent_requested_operation_v13 | 124 | 51.61% | 98.39% | 98.15% |
| lanl_auth_policy:lanl_known_logon_action_v16 | 540 | 72.78% | 99.81% | 99.72% |
| lanl_auth_policy:lanl_observable_policy | 540 | 99.44% | 100.00% | 100.00% |
| ton_iot_network:ton_attack | 3,500 | 25.00% | 98.97% | 98.17% |
| twins_streamlet:twins_evidence_level | 659 | 40.06% | 99.85% | 99.83% |
| twins_streamlet:twins_policy_action | 659 | 39.45% | 100.00% | 100.00% |
| twins_streamlet:twins_quorum | 659 | 64.95% | 99.85% | 99.79% |
| twins_streamlet:twins_visible_conflict | 659 | 70.11% | 99.70% | 99.50% |
| veremi_nextgen:veremi_message_manipulation | 3,321 | 48.12% | 66.21% | 66.67% |
| veremi_nextgen:veremi_supported_attack_type_v16 | 3,321 | 42.97% | 63.69% | 39.10% |
| **Macro-average · 14 tasks** | — | **52.15%** | **92.98%** | **90.31%** |

Macro averages give each task equal weight. Balanced accuracy averages recall over semantic labels rather than shuffled answer positions. Score questions use the argmax ordinal label.

| General suite (clean questions) | Kev-style stage | SecJev-2B |
|---|---:|---:|
| decision-v7 | 84.00% | 84.50% |
| transfer-v4 | 75.91% | 75.91% |

Security calibration after temperature fitting: NLL 0.384723, Brier 0.193513, ECE (15 bins) 3.28%. Temperature: 1.2923921937.

All inference uses FP32 unmerged weights; TF32 and fused SDPA are disabled. General-suite percentages use the clean subsets (1,200 decision-v7 and 656 transfer-v4 questions); the full suites including robustness variants were also evaluated. No date-fact augmentation was enabled. See [results.json](results.json) for raw and calibrated probabilities metrics.

These are previously inspected research splits, not a new blind benchmark. Vehicle-message tasks remain the weakest: 66.21% message-manipulation accuracy and 63.69% ten-way accuracy. Reported scores predict source labels or follow input policies; they are not live deployment detection rates.
