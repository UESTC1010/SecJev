# SecJev-4B training

Warm-start the official Kev-4B adapter and decision head on Qwen3.5-4B-Base, then train three epochs on SecJev-Corpus v1.0.0. The backbone and adapter revisions are pinned in `bootstrap.py`.

```bash
python -m pip install -r requirements.txt
export SECJEV_WORK="$PWD/secjev-4b-work"
python training/secjev4b/bootstrap.py --corpus /path/to/SecJev-Corpus
python training/secjev4b/prepare.py
python training/secjev4b/check_batching.py
export SECJEV_TOKEN_BUDGET=4096
# Use nine idle GPUs; a fresh work directory is required.
torchrun --standalone --nproc_per_node=9 training/secjev4b/train_security.py --probe
torchrun --standalone --nproc_per_node=9 training/secjev4b/train_security.py
python training/secjev4b/evaluate_nine.py
```

Measured on nine RTX 3090 24GB cards. LoRA r16/alpha32 covers attention, MLP and DeltaNet; decision head dimension 256. FP32 parameters, BF16 autocast, gradient checkpointing, fresh AdamW and gradient clipping 1.0. Global batches are 128/192/192, with gradient accumulation over length-dependent microbatches. Each epoch sees all 118,818 questions once: 929/619/619 optimizer steps. Epoch 1 peaks at LR 2e-5; epochs 2–3 share a cosine schedule peaking at 1e-5. The seed is 20260923.

The 56-question development panel only monitors training. Full development (14,912 questions) selects among all three checkpoints by macro-task semantic-class balanced accuracy, then raw NLL and earlier epoch. Temperatures use calibration only (14,821 questions). Candidates are frozen before opening the 21,634-question test. Original Kev-4B is the security and general-capability baseline. Evaluation uses FP32 unmerged weights, no TF32 or fused SDPA. All original question rows and native padding are retained.

The released checkpoint is epoch 3. Training took 9h42m and full evaluation 3h29m. Runtime: Python 3.11, PyTorch 2.6.0+cu126, transformers 5.17.0, PEFT 0.21.0, FLA 0.4.2, Triton 3.2.0; reference PyTorch causal convolution. These scripts are the measured implementation with local paths made portable. Packaging checks verify imports and nine-rank batching; the complete training was not repeated for release.
