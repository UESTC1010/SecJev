# SecJev-4B results

Epoch 3 was selected on all 14,912 development questions. Calibration uses a separate 14,821-question split. Test scores below cover all 21,634 security questions and 14 tasks. All three checkpoints were specified for testing before test access.

| Model | Micro accuracy | Macro-task accuracy | Macro-task balanced accuracy |
|---|---:|---:|---:|
| Original Kev-4B | 50.49% | 67.88% | 67.16% |
| Epoch 1 | 85.88% | 93.38% | 90.74% |
| Epoch 2 | 87.12% | 94.15% | 91.69% |
| Epoch 3 | 88.01% | 94.51% | 92.42% |

| Task | Questions | Original Kev-4B | SecJev-4B | SecJev balanced accuracy |
|---|---:|---:|---:|---:|
| agentdojo_tool_outputs:agentdojo_boundary_action | 426 | 54.23% | 99.53% | 98.96% |
| agentdojo_tool_outputs:agentdojo_embedded_injection | 426 | 73.00% | 98.36% | 96.35% |
| byzfl_mnist:fl_attack_generated_update | 800 | 79.88% | 95.88% | 92.97% |
| ciciot2023:ciciot_attack | 6000 | 26.28% | 91.72% | 87.41% |
| injecagent:injecagent_requested_operation_v13 | 124 | 62.90% | 100.00% | 100.00% |
| lanl_auth_policy:lanl_known_logon_action_v16 | 540 | 79.81% | 100.00% | 100.00% |
| lanl_auth_policy:lanl_observable_policy | 540 | 99.44% | 100.00% | 100.00% |
| ton_iot_network:ton_attack | 3500 | 50.66% | 98.80% | 98.21% |
| twins_streamlet:twins_evidence_level | 659 | 81.18% | 99.54% | 99.50% |
| twins_streamlet:twins_policy_action | 659 | 71.78% | 99.85% | 99.83% |
| twins_streamlet:twins_quorum | 659 | 78.15% | 100.00% | 100.00% |
| twins_streamlet:twins_visible_conflict | 659 | 97.12% | 99.85% | 99.89% |
| veremi_nextgen:veremi_message_manipulation | 3321 | 47.76% | 70.34% | 71.07% |
| veremi_nextgen:veremi_supported_attack_type_v16 | 3321 | 48.09% | 69.23% | 49.71% |
| **Macro-average** | — | **67.88%** | **94.51%** | **92.42%** |

Macro metrics weight tasks equally. Balanced accuracy averages recall over semantic classes within each task. Ordinal questions use argmax accuracy. Vehicle-message manipulation (70.34%) and attack-type classification (69.23%) remain the weakest tasks.

| General suite (clean subset) | Original Kev-4B | SecJev-4B |
|---|---:|---:|
| decision-v7 (1,200) | 87.08% | 86.58% |
| transfer-v4 (656) | 83.69% | 83.84% |

Full general suites, including variants, contain 1,440 and 764 questions. Inference is FP32 unmerged, TF32/fused SDPA off. All original multiquestion semantics are retained. These are previously inspected research splits; source-label and policy scores are not live attack-detection rates. See results.json for calibration metrics and every checkpoint.
