"""Start or continue an MSR-VTT training run with optional session limits."""

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import re

from config.base_config import HF_CLIP_MODEL
from measure_training import ROOT, run_training, training_command
from modules.training_state import training_directory_lock


LEGACY_EXPERIMENT = 'msrvtt9k_split_training'
LEGACY_CHECKPOINT = ROOT / 'outputs/msrvtt9k_train_vram_20261107_054714/checkpoint-epoch2.pth'


def default_asset(relative_path):
    local = ROOT / relative_path
    return local if local.is_dir() else ROOT.parent / 'xpool' / relative_path


def select_checkpoint(output, requested, *, legacy_checkpoint=None):
    latest = output / 'checkpoint-last.pth'
    if requested is not None:
        checkpoint = requested.expanduser().resolve()
        if not checkpoint.is_file():
            raise ValueError(f'Checkpoint does not exist: {checkpoint}')
        if latest.is_file() and checkpoint != latest.resolve():
            raise ValueError('Use a new experiment name to start from a different checkpoint')
        return checkpoint
    if latest.is_file():
        return latest
    if legacy_checkpoint is not None and legacy_checkpoint.is_file():
        return legacy_checkpoint
    if output.exists() and any(path.name != '.session.lock' for path in output.iterdir()):
        raise ValueError(f'Existing run has no checkpoint-last.pth: {output}. '
                         'Use a new --exp-name or specify a checkpoint with --resume.')
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hours', type=float, required=True,
                        help='Wall-time budget for this session (0 disables the time limit)')
    parser.add_argument('--exp-name', default=LEGACY_EXPERIMENT)
    parser.add_argument('--resume', type=Path, help='Explicit checkpoint for a new output directory')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--loss', choices=('clip', 'supcon'), default='clip')
    chunk_modes = parser.add_mutually_exclusive_group()
    chunk_modes.add_argument('--grad-cache-chunk-size', type=int, default=0)
    chunk_modes.add_argument('--accumulation-chunk-size', type=int, default=0)
    parser.add_argument('--check-recompute', action='store_true', help='Check GradCache feature replay during smoke tests')
    parser.add_argument('--videos-dir', type=Path, default=default_asset('data/MSRVTT/vids'))
    local_clip = default_asset(HF_CLIP_MODEL)
    parser.add_argument('--clip-model-path', type=Path, default=local_clip if local_clip.is_dir() else None)
    parser.add_argument('--seed', type=int, default=24)
    parser.add_argument('--log-step', type=int, default=100)
    parser.add_argument('--max-steps', type=int, default=0, help='Extra per-session step cap for smoke and pilot runs')
    parser.add_argument('--dry-run', action='store_true', help='Print the resolved command without training')
    args = parser.parse_args(argv)
    seconds = args.hours * 3600
    if not math.isfinite(seconds) or seconds < 0:
        parser.error('--hours must be finite and nonnegative')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.exp_name):
        parser.error('--exp-name must be a single directory name')
    if args.batch_size <= 0 or not all(
        0 <= size <= args.batch_size for size in (args.grad_cache_chunk_size, args.accumulation_chunk_size)
    ):
        parser.error('Batch size must be positive and chunk size must be between 0 and batch size')
    if args.check_recompute and not args.grad_cache_chunk_size:
        parser.error('--check-recompute requires --grad-cache-chunk-size')
    if args.max_steps < 0 or args.log_step <= 0 or args.seed < 0:
        parser.error('--max-steps and --seed must be nonnegative, and --log-step must be positive')
    videos_dir = args.videos_dir.expanduser().resolve()
    if not videos_dir.is_dir():
        parser.error(f'MSR-VTT videos directory not found: {videos_dir}')
    clip_path = None
    if args.clip_model_path is not None:
        clip_path = args.clip_model_path.expanduser().resolve()
        for filename in ('config.json', 'pytorch_model.bin', 'vocab.json', 'merges.txt'):
            if not (clip_path / filename).is_file():
                parser.error(f'CLIP asset not found: {clip_path / filename}. Set --clip-model-path.')

    output = ROOT / 'outputs' / args.exp_name
    legacy_checkpoint = None
    if (args.exp_name == LEGACY_EXPERIMENT and args.batch_size == 4 and args.seed == 24
            and args.loss == 'clip' and not (args.grad_cache_chunk_size or args.accumulation_chunk_size)):
        legacy_checkpoint = LEGACY_CHECKPOINT
    options = dict(
        videos_dir=videos_dir, clip_model_path=clip_path,
        grad_cache_chunk_size=args.grad_cache_chunk_size,
        accumulation_chunk_size=args.accumulation_chunk_size,
        check_recompute=args.check_recompute, seed=args.seed, log_step=args.log_step, loss=args.loss,
    )

    def session():
        try:
            checkpoint = select_checkpoint(output, args.resume, legacy_checkpoint=legacy_checkpoint)
        except ValueError as error:
            parser.error(str(error))
        command_args = (args.exp_name, args.batch_size, seconds, args.max_steps, checkpoint)
        command = training_command(*command_args, **options)
        print(json.dumps({
            'hours': args.hours, 'resume': str(checkpoint) if checkpoint else None,
            'output': str(output), 'command': command,
            'total_epochs': 5, 'batch_size': args.batch_size, 'loss': args.loss,
        }, indent=2), flush=True)
        if args.dry_run:
            return 0
        result = run_training(*command_args, **options)
        print(json.dumps(result, indent=2), flush=True)
        return result['exit_code']

    if args.dry_run:
        return session()
    if os.name == 'nt':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        output.mkdir(parents=True, exist_ok=True)
        with training_directory_lock(str(output), '.session.lock'):
            return session()
    finally:
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    raise SystemExit(main())
