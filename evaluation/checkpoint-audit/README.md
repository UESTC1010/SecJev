# Checkpoint selection audit

Retrospective test comparison requested after the epoch-3 model had already been selected and tested. All three checkpoint files are unchanged. The original release selection remains epoch 3; these new test results do not silently change it.

| Epoch | Development accuracy | Development macro-task balanced accuracy | Test accuracy | Test macro-task balanced accuracy | Raw test NLL | Calibrated test NLL | Calibrated ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 88.5998% | 87.2496% | 82.5183% | 84.8766% | 0.479466 | 0.463283 | 2.9950% |
| 2 | 89.4246% | 88.5079% | 83.5953% | 86.4010% | 0.458113 | 0.431387 | 3.3178% |
| 3 | 90.3098% | 90.3572% | 85.0328% | 87.9975% | 0.434838 | 0.412389 | 3.3372% |

Development primary ranking: [3, 2, 1]. Test primary ranking: [3, 2, 1].
The test comparison supports selecting epoch 3 on this metric.

| Task | Test questions | Epoch 1 accuracy | Epoch 2 accuracy | Epoch 3 accuracy | Epoch 1 balanced accuracy | Epoch 2 balanced accuracy | Epoch 3 balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| agentdojo_tool_outputs:agentdojo_boundary_action | 426 | 80.2817% | 81.6901% | 80.2817% | 73.9773% | 76.3636% | 78.0398% |
| agentdojo_tool_outputs:agentdojo_embedded_injection | 426 | 80.9859% | 81.2207% | 79.8122% | 79.6023% | 79.7538% | 79.5833% |
| byzfl_mnist:fl_attack_generated_update | 800 | 85.3750% | 90.7500% | 92.6250% | 78.6719% | 90.2344% | 91.8750% |
| ciciot2023:ciciot_attack | 6000 | 90.1833% | 89.4500% | 91.8000% | 83.3222% | 80.8556% | 88.0667% |
| injecagent:injecagent_requested_operation_v13 | 124 | 99.1935% | 100.0000% | 100.0000% | 99.0741% | 100.0000% | 100.0000% |
| lanl_auth_policy:lanl_known_logon_action_v16 | 540 | 96.8519% | 98.3333% | 98.8889% | 96.0894% | 97.9035% | 98.5837% |
| lanl_auth_policy:lanl_observable_policy | 540 | 100.0000% | 99.0741% | 99.8148% | 100.0000% | 99.3873% | 99.8775% |
| ton_iot_network:ton_attack | 3500 | 97.4857% | 98.3143% | 98.0286% | 96.7619% | 97.1619% | 97.0095% |
| twins_streamlet:twins_evidence_level | 659 | 98.1791% | 98.1791% | 98.7860% | 97.9924% | 98.0024% | 98.6625% |
| twins_streamlet:twins_policy_action | 659 | 97.5721% | 97.8756% | 98.4825% | 97.3224% | 97.6624% | 98.3274% |
| twins_streamlet:twins_quorum | 659 | 99.3930% | 99.3930% | 99.6965% | 99.4334% | 99.1453% | 99.7647% |
| twins_streamlet:twins_visible_conflict | 659 | 98.4825% | 98.9378% | 99.2413% | 97.7726% | 98.5263% | 98.7437% |
| veremi_nextgen:veremi_message_manipulation | 3321 | 58.6269% | 60.6745% | 63.1135% | 58.8792% | 60.0293% | 63.5495% |
| veremi_nextgen:veremi_supported_attack_type_v16 | 3321 | 58.1752% | 61.8187% | 64.1373% | 29.3733% | 34.5882% | 39.8818% |

Each epoch has its own temperature fitted only on calibration. Positive temperature does not change argmax accuracy. Development and test use the same labels, per-question scoring, and macro-task weighting. All test comparisons pair exactly the same scene/question IDs. See comparison.json for corrected-versus-regressed question counts.

Descriptive comparison on these fixed splits. Related questions/captures are not assumed independent; no significance or unseen-distribution guarantee claimed.
