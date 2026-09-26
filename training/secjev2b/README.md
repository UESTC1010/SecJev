# SecJev-2B: training and evaluation

The release starts from pinned Qwen3.5-2B-Base, learns the Kev decision task, then trains on SecJev-Corpus for three epochs. It does not start from an official Kev-2B checkpoint: there was none in the family used for this run.

Install the repository `requirements.txt` in a CUDA environment. The measured stack used Python 3.11, PyTorch 2.6.0+cu126, transformers 5.17.0, PEFT 0.21.0, FLA 0.4.2 and Triton 3.2.0. The optional causal-conv1d extension was absent; the run used the PyTorch reference convolution.

```bash
export SECJEV_WORK="$PWD/secjev-2b-work"
python training/secjev2b/bootstrap.py --corpus /path/to/SecJev-Corpus
python training/secjev2b/prepare.py
python training/secjev2b/check_batching.py
export SECJEV_TOKEN_BUDGET=32768
# Select three idle GPUs for the measured training recipe.
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 training/secjev2b/probe.py
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 training/secjev2b/train_kev.py
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 training/secjev2b/train_kev.py --delta
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 training/secjev2b/train_security.py
# After training, use nine idle GPUs for validation, calibration and test.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python training/secjev2b/evaluate_nine.py
```

Use a fresh work directory. Training uses the train split; the fixed small development panel only monitors loss. Full development selects the checkpoint by macro-task balanced accuracy, with raw NLL and earlier epoch as tie-breakers. Each checkpoint is calibrated separately, then frozen before test access. The chosen calibrated checkpoint is under `$SECJEV_WORK/runs/secjev-2b-3epochs/checkpoint`.

The Kev stage uses 12,576 training requests for two epochs (effective batch 8, peak LR 5e-5), followed by 1,425 dates/unknowable requests plus 2,000 replay requests for one epoch (LR 2e-5). It preserves Kev's augmentation and source/variant/question mean loss. Security training uses 118,818 questions per epoch, global batches 128/192/192, 929/619/619 updates, peak LR 2e-5 in epoch 1 and 1e-5 across epochs 2–3. LoRA r16/alpha32 covers attention, MLP and DeltaNet; head dimension256; base/master parameters are FP32, with BF16 autocast and gradient checkpointing. Native padding and independent causal question rows are used for the hybrid backbone.

These are the measured implementations with filesystem paths made portable. The packaged code was import/syntax checked and its batching invariants checked; the complete training was not rerun merely to package the release. [Evaluation](../../evaluation/secjev2b/README.md) includes all three checkpoints. Existing 0.8B code and results remain separate.
