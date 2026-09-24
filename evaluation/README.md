# SecJev-0.8B evaluation

Checkpoint selected among the three epoch-end candidates using full-development macro-task balanced accuracy. Both models are independently calibrated on the calibration split; test data does not choose the checkpoint or temperature.

| Metric | Kev-0.8B parent | SecJev-0.8B |
|---|---:|---:|
| accuracy | 0.405334 | 0.850328 |
| macro_task_accuracy | 0.509352 | 0.903363 |
| macro_task_balanced_accuracy | 0.518562 | 0.879975 |
| nll | 0.971910 | 0.412389 |
| brier | 0.577171 | 0.212686 |
| ece15 | 0.170698 | 0.033372 |

| Task | Questions | Parent accuracy | SecJev accuracy | Parent balanced accuracy | SecJev balanced accuracy |
|---|---:|---:|---:|---:|---:|
| agentdojo_tool_outputs:agentdojo_boundary_action | 426 | 51.8779% | 80.2817% | 59.3371% | 78.0398% |
| agentdojo_tool_outputs:agentdojo_embedded_injection | 426 | 32.8638% | 79.8122% | 56.2973% | 79.5833% |
| byzfl_mnist:fl_attack_generated_update | 800 | 79.6250% | 92.6250% | 49.7656% | 91.8750% |
| ciciot2023:ciciot_attack | 6000 | 25.0000% | 91.8000% | 50.0000% | 88.0667% |
| injecagent:injecagent_requested_operation_v13 | 124 | 44.3548% | 100.0000% | 50.7143% | 100.0000% |
| lanl_auth_policy:lanl_known_logon_action_v16 | 540 | 72.4074% | 98.8889% | 66.1017% | 98.5837% |
| lanl_auth_policy:lanl_observable_policy | 540 | 80.7407% | 99.8148% | 87.2549% | 99.8775% |
| ton_iot_network:ton_attack | 3500 | 25.0000% | 98.0286% | 50.0000% | 97.0095% |
| twins_streamlet:twins_evidence_level | 659 | 41.1229% | 98.7860% | 38.0365% | 98.6625% |
| twins_streamlet:twins_policy_action | 659 | 47.6480% | 98.4825% | 47.3485% | 98.3274% |
| twins_streamlet:twins_quorum | 659 | 53.4143% | 99.6965% | 46.6933% | 99.7647% |
| twins_streamlet:twins_visible_conflict | 659 | 64.0364% | 99.2413% | 64.5450% | 98.7437% |
| veremi_nextgen:veremi_message_manipulation | 3321 | 47.7567% | 63.1135% | 50.0000% | 63.5495% |
| veremi_nextgen:veremi_supported_attack_type_v16 | 3321 | 47.2448% | 64.1373% | 9.8928% | 39.8818% |

Accuracy counts individual questions; macro-task metrics give each task equal weight. Balanced accuracy averages recalls of semantic labels, not shuffled option positions. Scores are distributions over source-defined ordinal labels, not universal security risk scores.

The research split was inspected during dataset construction and is not a blind external benchmark. Controlled Byzantine/Sybil experiments support claims within those experiments. General-domain retention is reported in the [separate retention evaluation](retention/README.md). Deployment robustness was not evaluated. No confidence intervals are claimed for this security-test table.

See export-validation.json for warm forward-pass measurements and validation of the public artifact. The actual context range and question counts accompany every timed scene.

The homepage presents the same 14 tasks with readable labels. [task-results.json](task-results.json) maps those labels to the exact source/task IDs and stores the unrounded metrics. Macro-average accuracy is the unweighted mean of 14 task accuracies; macro-average balanced accuracy is the unweighted mean of the 14 per-task semantic-class recall averages. Values are averaged before rounding.
