# Using SecJev-0.8B

The checkpoint contains a trained LoRA adapter, pointer decision head, calibration temperature and tokenizer files. The pinned **Qwen3.5-0.8B-Base** backbone is loaded separately. This is an adapter release, not a standalone copy of all backbone weights.

Install the dependencies, prepare the pinned backbone and Kev source, and unpack the model:

```bash
python -m pip install -r requirements.txt
export SECJEV_WORK="$PWD/secjev-work"
python training/bootstrap.py --inference-only
curl -L --fail -o SecJev-0.8B-v1.0.0.tar.gz https://github.com/UESTC1010/SecJev/releases/download/v1.0.0/SecJev-0.8B-v1.0.0.tar.gz
tar -xzf SecJev-0.8B-v1.0.0.tar.gz
python infer.py --model ./SecJev-0.8B --base "$SECJEV_WORK/base" --request examples/certificate.json --device cuda
```

The inference setup downloads the pinned Qwen backbone and Kev source; it does not download the training corpus or original Kev adapter. GPU use requires a compatible CUDA PyTorch installation. The release archive contains the adapter and head, so its size is smaller than the full 0.8B backbone.

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

## SecJev-2B

The 2B release uses the same request format and `infer.py`. Download the model package and use its included bootstrap to obtain the exact backbone and Kev code:

```bash
curl -L --fail -o SecJev-2B-v1.1.0.tar.gz https://github.com/UESTC1010/SecJev/releases/download/v1.1.0/SecJev-2B-v1.1.0.tar.gz
tar -xzf SecJev-2B-v1.1.0.tar.gz
export SECJEV_WORK="$PWD/secjev-2b-work"
python SecJev-2B/bootstrap.py --inference-only
python SecJev-2B/infer.py --model ./SecJev-2B --base "$SECJEV_WORK/base" --request SecJev-2B/examples/certificate.json --device cuda
```

Install `requirements.txt` first. Use `--device cpu` for CPU inference. The archive contains actual trained LoRA weights and a decision head, not the multi-gigabyte Qwen backbone. Bootstrap downloads `Qwen/Qwen3.5-2B-Base` at revision `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`. The released temperature (1.2923921937) is applied automatically. No training dataset is required for inference.

For the hand-written certificate example, the released 2B checkpoint returned `accept` in the CPU smoke test, although the stated policy requires `reject`. The package includes the actual `example-output.json`. This example demonstrates the request format; use the per-task evaluation to assess model performance.
