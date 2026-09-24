# Reproducing SecJev-0.8B

The release starts from the published Kev-0.8B adapter and decision head on **SecJev-Corpus v1.0.0**. The initial one-epoch phase below is followed by **two additional epochs**, with checkpoint selection on the complete development split. An earlier SecJev experiment is not used for initialization.

## Environment and inputs

The measured run uses Linux, Python 3.11, PyTorch 2.6.0 with CUDA 12.6, Transformers 5.17.0, PEFT 0.21.0, Triton 3.2.0 and FLA 0.4.2. `causal_conv1d` is absent in this run; the reference convolution implementation is used. Do not compare these timings to an optimized build without remeasuring.

Create a dedicated environment, install `requirements.txt`, and unpack the corpus archive. The bootstrap script downloads the exact upstream source and weight revisions listed in [UPSTREAM.md](UPSTREAM.md). No existing security checkpoint is required.

```bash
export SECJEV_WORK="$PWD/secjev-work"
python -m pip install -r requirements.txt
python training/bootstrap.py --corpus /path/to/SecJev-Corpus
python training/prepare.py
python training/check_metrics.py
```

Preparation verifies public corpus content hashes and encodes training, calibration and development separately. It rejects oversized inputs rather than truncating them. Each question is an independent causal row, preserving its original state, criteria order and tokens. Labels and provenance are excluded from model inputs.

## Initial one-epoch recipe

| Setting | Value |
|---|---|
| Supervision | 118,818 questions in 81,410 scenes |
| Epochs / optimizer updates | 1 / 929 |
| Trainable components | Existing LoRA adapter and pointer decision head |
| Effective batch | 128 question rows across two GPUs |
| Per-GPU microbatch | Adaptive powers of two, maximum 64, 40,960 padded-token budget |
| Optimizer | AdamW; learning rate 2e-5; weight decay 0.01 |
| Schedule | 5% warmup; cosine decay to 10% of peak learning rate |
| Precision | FP32 master parameters; BF16 autocast; TF32 enabled during training |
| Loss | Mean cross-entropy per scene; question weight = 1 / questions in the scene |
| Padding / grouping | 128-token buckets; length sorting within shuffled 2,048-row pools |
| Seed | 20260923 |
| Gradient checkpointing / clipping | Enabled / norm 1.0 |
| Replay / augmentation | None / none |

Choose two idle GPUs. The measured setup reserved about 14 GB per GPU in the initial probe; actual requirements vary with the software and GPU.

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 training/train.py --mode probe --batch 64 --checkpointing 1
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 training/train.py --mode train --batch 64 --checkpointing 1
```

Probe updates are discarded. Formal training reloads the pinned original checkpoint. The run directory must not already exist. Every supervised row is used exactly once; padding copies in the final uneven batch have zero loss weight.

## Continuation and checkpoint selection

The release procedure continues from the first epoch's uncalibrated checkpoint and saved AdamW moments for two more epochs. The first epoch remains a candidate; its weights are retained unchanged.

| Continuation setting | Value |
|---|---|
| GPUs | 3 × RTX 3090 |
| Effective global batch | 192 question rows |
| Per-GPU microbatch | Adaptive, maximum 64; 40,960 padded-token budget |
| Additional updates | 619 per epoch, 1,238 total |
| Optimizer | Restore first-epoch AdamW moments at step 929 |
| Learning rate | New 5% warmup from 2e-6 to 1e-5, then cosine decay to 1e-6 |
| Supervision | Same corpus and per-scene weighting; each row exactly once per epoch |
| Candidate checkpoints | Final checkpoints of epochs 1, 2 and 3 |
| Selection data | All 14,912 development questions |
| Primary selection metric | Highest macro-task semantic-class balanced accuracy |
| Tiebreaks | Lowest macro-task raw NLL, then earlier epoch |

The small 56-question monitoring panel is only a training diagnostic. It does not select the checkpoint. Selection metrics use raw logits; scalar temperature is fitted on calibration data only after the checkpoint is selected. The first epoch's already-computed calibration/development logits can be reused when appropriate.

After selection and calibration, the candidate files are frozen. Only then are the 21,634 test questions evaluated against both the original Kev-0.8B baseline and the selected model. Test scores do not choose a checkpoint. Three-way inference sharding distributes independent question rows; candidate probabilities and labels are merged by original scene/question identifiers.

The complete continuation, selection, calibration, test and export workflow is included in `training/pipeline_continue.py`. After the initial one-epoch training command finishes, run:

```bash
# Evaluate the initial checkpoint and baseline on calibration/development only.
CUDA_VISIBLE_DEVICES=0 python training/evaluate.py --model parent
CUDA_VISIBLE_DEVICES=0 python training/evaluate.py --model trained
# Continue for epochs 2 and 3; select, calibrate, freeze, test and export.
CUDA_VISIBLE_DEVICES=0,1,2 python training/pipeline_continue.py
```

Choose three idle GPUs and replace the device IDs as needed. The controller honors the order in `CUDA_VISIBLE_DEVICES`. Use a fresh `SECJEV_WORK` directory for each complete run. Do not run the legacy one-epoch `freeze.py` or prepare the test split before this controller: it freezes the selected three-epoch candidate itself.

Epochs 2 and 3 are saved under `$SECJEV_WORK/continuation/runs/secjev-0.8b-3epochs/`. The portable selected checkpoint is exported to `$SECJEV_WORK/continuation/release/SecJev-0.8B/`; reports and logs are under `continuation/results/` and `continuation/logs/`.

`train_continue.py` contains the measured two-epoch continuation, `infer_shard.py` performs FP32 evaluation, and `finish_release.py` exports weights and verifies serving probabilities. The released scripts replace the original container mount paths with `SECJEV_WORK`; the training objective, batches, optimizer schedule and selection rule are unchanged. The original training and export were run on GPUs; the path-portable packaging was syntax-checked and its batch coverage and loss normalization checked on CPU, without rerunning all three epochs.

Evaluation uses FP32, unmerged adapters, TF32 disabled and calibrated probabilities. It reports micro accuracy, per-task accuracy, semantic-class balanced accuracy, NLL, Brier score, ECE, high-confidence coverage/error, and ordinal expected-level error. Shuffled option positions are not treated as semantic classes.

The corpus splits were inspected during construction, so this is a held-out research evaluation rather than a blind external benchmark. General-domain retention is not measured by this recipe.
