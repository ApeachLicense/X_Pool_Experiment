"""Continue the saved MSR-VTT run for a per-session number of hours."""

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import re
import sys

from measure_training import ROOT, run_training, training_command
from modules.training_state import training_directory_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hours', type=float, required=True, help='Wall-time budget for this session')
    parser.add_argument('--exp-name', default='msrvtt9k_split_training')
    parser.add_argument('--resume', type=Path, help='Override automatic selection of the latest checkpoint')
    parser.add_argument('--dry-run', action='store_true', help='Print the selected checkpoint and command')
    args = parser.parse_args()
    if not math.isfinite(args.hours) or args.hours <= 0:
        parser.error('--hours must be finite and positive')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.exp_name):
        parser.error('--exp-name must be a single directory name')
    latest = ROOT / 'outputs' / args.exp_name / 'checkpoint-last.pth'
    legacy = ROOT / 'outputs/msrvtt9k_train_vram_20261107_054714/checkpoint-epoch2.pth'
    checkpoint = args.resume
    if checkpoint is None:
        checkpoint = latest if latest.is_file() else legacy
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f'Checkpoint does not exist: {checkpoint}')
    seconds = args.hours * 3600
    command = training_command(args.exp_name, 4, seconds, 0, checkpoint)
    print(json.dumps({
        'hours': args.hours, 'resume': str(checkpoint),
        'output': str(ROOT / 'outputs' / args.exp_name), 'command': command,
        'total_epochs': 5, 'batch_size': 4,
    }, indent=2), flush=True)
    if args.dry_run:
        return 0
    if os.name == 'nt':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        output = ROOT / 'outputs' / args.exp_name
        output.mkdir(parents=True, exist_ok=True)
        with training_directory_lock(str(output), '.session.lock'):
            result = run_training(args.exp_name, 4, seconds, 0, checkpoint)
        print(json.dumps(result, indent=2), flush=True)
        return result['exit_code']
    finally:
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    sys.exit(main())
