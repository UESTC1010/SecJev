# Using SecJev-0.8B

The checkpoint contains a trained LoRA adapter, pointer decision head, calibration temperature and tokenizer files. The pinned **Qwen3.5-0.8B-Base** backbone is loaded separately. This is an adapter release, not a standalone copy of all backbone weights.

Use the environment and pinned upstream source prepared in [TRAINING.md](TRAINING.md). Download the released checkpoint to a local directory, then run:

```bash
python infer.py --model /path/to/SecJev-0.8B --request examples/certificate.json --device cuda
```

For an already downloaded backbone, add `--base /path/to/Qwen3.5-0.8B-Base`. CPU execution is also selectable with `--device cpu`; GPU evaluation results do not imply measured CPU latency.

Input follows Kev's System One request shape: a shared `state` and a mapping of typed `questions`. The example has an explicit consensus policy and asks which certificate action follows it. The policy-derived expected answer is `reject`; the script prints the model's actual answer.

Output includes:

- `answers`: Kev-compatible typed judgments, choices and scores.
- `probabilities`: unrounded probabilities for every candidate; use these for calibration analysis.
- `temperature`: the stored calibration temperature used for inference.
- `forward_ms`: elapsed forward-pass time for this request, excluding model loading and tokenization.

`noul` returns the probability of true; `choice` returns the maximum-probability option; `score` returns the probability-weighted category index. The model scores candidates in a forward pass. It does not generate an answer token by token.

The default reference path is FP32 with the adapter unmerged. Every state-plus-question row must fit within 8,192 tokens; oversized inputs raise an error. The corpus's observed maximum is approximately 3,588 tokens, so the configured limit is not evidence of tested quality throughout an 8K context.

Compatibility refers to request/response structure and Kev conventions, not identical Jev internals, confidence formulas or predictions. Probability calibration is evaluated on this corpus; deployment on a new distribution needs its own evaluation.
