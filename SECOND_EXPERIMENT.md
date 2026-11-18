# Second experiment — GradCache-32 training integration

## Scope
This branch starts from `first_experiment` (GradCache gradient verification passed).
**Only** the training path for an effective batch of 32 is changed. CLIP pretraining,
pooling identity initialization, logit scale, contrastive loss, and data augmentation
are retained. XBM, head-first LP-FT, and SigLIP are **not** part of this pilot.

For X-Pool, video representations depend on each text, so GradCache must compute
*one* 32×32 text-conditioned pooling / similarity matrix and contrastive loss.
Accumulating independent small-chunk losses is **not equivalent**.

## Code
- `--batch_size=32 --grad_cache_chunk_size=2` encodes/backprops two
  examples per chunk while retaining all 31 in-batch negatives per query.
- `--grad_cache_chunk_size=0` is the unchanged original path.
- Tokenized text and the effective-batch video tensor stay on CPU; individual
  chunks move to CUDA. Gradients are clipped and optimizer/scheduler advance
  **once** per effective batch.
- `--eval_batch_size` defaults to 4 under GradCache; otherwise matches
  `--batch_size`. Validation itself is regular (no GradCache) forward.
- `--grad_cache_check_recompute` checks that feature recomputation is
  deterministic, with overhead; leave it off for timing experiments.
- Checkpoints save `grad_cache_chunk_size` and `eval_batch_size` and enforce
  exact settings on resume. Never resume a batch-4 checkpoint as batch 32.

## Smoke (10 optimizer steps)

Windows with `X_Pool_Experiment` and `xpool` side by side:

```bat
run_second_experiment.bat
```

Uses RTX 3060 Ti environment, MSR-VTT 9K, CLIP ViT-B/32, FP32, 12 frames,
batch 32 / chunk 2, 2 loader workers, seed 24, no periodic retrieval eval.
Writes `outputs/gradcache32_smoke_<timestamp>/training_profile.json`,
`training_steps.csv` and `checkpoint-last.pth`; console log goes under
`outputs/gradcache32_smoke_<timestamp>_console.log`. Does **not** run 5 epochs.
GPU smoke execution is **not performed** by this source-code commit.

Manual equivalent from repository root (adjust the videos path):

```powershell
..\xpool\.venv\Scripts\python.exe train.py --exp_name=gradcache32_smoke_manual --videos_dir=..\xpool\data\MSRVTT\vids --dataset_name=MSRVTT --msrvtt_train_file=9k --huggingface --batch_size=32 --grad_cache_chunk_size=2 --eval_batch_size=4 --num_workers=2 --noclip_lr=3e-5 --transformer_dropout=0.3 --no_tensorboard --evals_per_epoch=0 --max_train_steps=10 --profile_training --grad_cache_check_recompute --log_step=1
```

Run CPU tests with the existing environment:

```powershell
..\xpool\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Pilot after smoke passes (not yet executed)

Use a *new* experiment name. Turn recompute checking off while measuring speed.

```powershell
..\xpool\.venv\Scripts\python.exe train.py --exp_name=gradcache32_pilot300 --videos_dir=..\xpool\data\MSRVTT\vids --dataset_name=MSRVTT --msrvtt_train_file=9k --huggingface --batch_size=32 --grad_cache_chunk_size=2 --eval_batch_size=4 --num_workers=2 --noclip_lr=3e-5 --transformer_dropout=0.3 --no_tensorboard --evals_per_epoch=0 --max_train_steps=300 --profile_training --log_step=25
```

This is **9,600 caption/video-pair exposures** (32×300). A batch-4 control
requires 2,400 updates for the same 9,600 exposures, with the original
`--grad_cache_chunk_size=0`; record both separately. Even with matched
pair exposures, optimizer-update counts and negative sets are different,
so do not treat this as a controlled *gradient* comparison.
Keep CLIP initialization, head initialization, loss, data split, seed,
augmentations, LR schedule and evaluation protocol otherwise fixed.
Raw training loss is not directly comparable across batch sizes.

## Gate before any long run
1. Check `status=finished`, `stop_reason=step_limit`,
   `completed_steps=10`, effective `batch_size=32`,
   `grad_cache_chunk_size=2`, finite loss.
2. From `training_steps.csv`, check step durations and
   `peak_reserved_mib`. PyTorch memory excludes desktop VRAM; inspect
   `nvidia-smi` separately. Require comfortable GPU memory headroom.
3. If OOM, try `--grad_cache_chunk_size=1`. The 32×32 pooling still consumes
   memory and may set a lower bound.
4. Verify resume on identical flags/seed, and use distinct output directories.
   Do not run two sessions into the same experiment directory.
5. Compare wall-time/sample and evaluate R@1/R@5/R@10 on a fixed held-out
   split if proceeding. Do not tune against the official 1K-A test set.

## Background
[Gao et al. (2021), Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup](https://arxiv.org/abs/2101.06983).
[First experiment results](results/first_experiment_20261115/README.md):
batch-4 GradCache 2+2 gradient error 2.4e-6–5.6e-6 after disabling TF32.
**No measured batch-32 VRAM, throughput or retrieval gain yet.**
