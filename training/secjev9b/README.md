# SecJev-9B training

Start from official Kev-9B on Qwen3.5-9B-Base and train three epochs on SecJev-Corpus v1.0.0. Exact upstream revisions are pinned in `bootstrap.py`.

```bash
python -m pip install -r requirements.txt
export SECJEV_WORK="$PWD/secjev-9b-work"
python training/secjev9b/bootstrap.py --corpus /path/to/SecJev-Corpus
python training/secjev9b/prepare.py
python training/secjev9b/check_batching.py
export SECJEV_TOKEN_BUDGET=2048
# Nine idle GPUs and a fresh work directory.
torchrun --standalone --nproc_per_node=9 training/secjev9b/train_security.py --probe
torchrun --standalone --nproc_per_node=9 training/secjev9b/train_security.py
python training/secjev9b/evaluate_nine.py
```

Measured on nine RTX 3090 24GB cards. Frozen BF16 backbone; FP32 LoRA, decision head and AdamW states; BF16 autocast. LoRA r16/alpha32 covers attention, MLP and DeltaNet, with a 256-dimensional decision head. Gradient checkpointing and clipping 1.0. Global batches 128/192/192 are accumulated from length-dependent microbatches. Each epoch covers all 118,818 questions once: 929/619/619 updates. Epoch 1 peaks at LR 2e-5; epochs 2–3 share a cosine schedule peaking at 1e-5. Seed 20260923.

The 56-question panel is for monitoring only. Full development (14,912 questions) selects by macro-task semantic-class balanced accuracy, then macro-task raw NLL and earlier epoch. Temperatures are fitted on calibration only (14,821 questions) and frozen before the 21,634-question safety test. All three checkpoints are evaluated under this predeclared protocol, alongside official Kev-9B and the decision-v7/transfer-v4 suites.

Evaluation uses an unmerged BF16 backbone and the original FP32 LoRA/head. `eval_loading.py` restores the original FP32 adapter tensors after the upstream loader's dtype conversion. TF32 and fused SDPA are off, with native padding and no input truncation. Reference PyTorch causal convolution runs on CUDA.

Epoch 3 was selected. Training took 14h14m and evaluation 2h28m. Runtime: Python 3.11, PyTorch 2.6.0+cu126, transformers 5.17.0, PEFT 0.21.0, FLA 0.4.2, Triton 3.2.0. Scripts preserve the measured implementation with portable paths; training is not repeated during release packaging.
