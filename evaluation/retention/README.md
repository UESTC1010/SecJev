# Original Kev capability retention

The frozen SecJev-0.8B epoch-3 candidate was compared with its original Kev-0.8B parent on the official release test suites, using the pinned official predictor and scoring code. All 2,204 questions were evaluated with no rejection or truncation. Headline accuracy uses 1,856 clean questions; the remaining 348 robustness variants are reported separately.

| Official test suite | Clean questions | Kev-0.8B | SecJev-0.8B | Accuracy change | Original correct answers retained |
|---|---:|---:|---:|---:|---:|
| decision-v7 | 1,200 | 83.50% | 83.08% | −0.42 pp | 97.80% |
| transfer-v4 | 656 | 68.45% | 66.01% | −2.44 pp | 91.54% |

On decision-v7, 22 formerly correct answers were lost and 17 formerly wrong answers became correct. On transfer-v4, 38 were lost and 22 gained. Source-stratified, record-clustered paired bootstrap 95% intervals for the micro accuracy changes were [−1.50, +0.58] and [−4.73, −0.30] percentage points respectively (2,000 draws, seed 42). The task-macro intervals are separately stored in comparison.json and should not be confused with micro accuracy.

Most tested answer accuracy was retained, with a measurable transfer loss. Confidence reliability degraded more: using each checkpoint's stored temperature, transfer ECE increased from 2.56% to 21.10%, and high-confidence errors (p ≥ 0.9 and wrong, divided by all questions) from 1.07% to 13.57%. No temperatures were fitted on these tests. Raw T=1 transfer NLL also worsened from 0.8682 to 1.3573, so the difference is not solely caused by stored temperature values. These ECE measurements use the official 10 bins, unlike the security evaluation's 15 bins.

FP32, unmerged LoRA, SDPA, TF32 and fused SDPA disabled, strict original context limits, no date preprocessing. Predictions preserve full requests and the official independent-question semantics. Model files were unchanged after evaluation. The original parent reproduced the published rounded 68.4% transfer score; its 83.50% decision score differs by approximately one question from the model card's 83.4%. The exact cause of that small discrepancy was not isolated. All retention comparisons use the actual same-environment parent run.

The current security-test gain (40.53% → 85.03%) therefore comes with some general-task regression and substantial confidence miscalibration outside security. This evaluation did not modify training or checkpoint selection. Results characterize these suites, not all general capability or deployment conditions.

See [the detailed report](README.zh-CN.md), [comparison.json](comparison.json), and [plan.json](plan.json) for per-source, per-type, per-task results and provenance. Official source: [Kev-0.8B model card](https://github.com/jaredpalmer/kev/blob/90990a5fac2995b9faa3190f7d437e84f2067768/docs/model-cards/kev-0.8b.md).
