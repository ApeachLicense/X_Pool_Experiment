"""Compare matched-pair-exposure batch-4, GradCache-32 and accumulation profiles.

Only speed/memory/throughput are compared here. R@1 needs held-out evaluation,
and raw training losses use different negative sets and are not comparable.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def read_run(directory, warmup_steps):
    directory = Path(directory)
    profile_file = directory / 'training_profile.json'
    profile = json.loads(profile_file.read_text(encoding='utf-8'))
    # Resumed sessions write their CSV under sessions/<id>, even though the
    # latest profile stays at the run root.
    step_file = directory / 'training_steps.csv'
    if profile.get('steps_csv'):
        recorded = Path(profile['steps_csv'])
        if recorded.parent.parent.name == 'sessions':
            step_file = directory / 'sessions' / recorded.parent.name / recorded.name
    with step_file.open(encoding='utf-8', newline='') as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f'No optimizer steps were recorded in {step_file}')
    if profile.get('completed_steps') != len(rows):
        raise ValueError(
            f'Profile/CSV completed-step mismatch in {directory}: '
            f'{profile.get("completed_steps")} != {len(rows)}'
        )
    if len(rows) <= warmup_steps:
        raise ValueError(f'Need more than {warmup_steps} steps in {directory}')
    useful = rows[warmup_steps:]
    durations = [float(row['step_seconds']) for row in useful]
    if not all(math.isfinite(x) and x > 0 for x in durations):
        raise ValueError(f'Invalid step times in {step_file}')
    samples = sum(int(row['batch_size']) for row in useful)
    total_seconds = sum(durations)
    monitor_file = directory / 'monitor_report.json'
    monitor = json.loads(monitor_file.read_text(encoding='utf-8')) if monitor_file.is_file() else {}
    # An earlier telemetry run may remain alongside a direct train.py invocation.
    monitored = monitor.get('training_profile', {})
    has_current_telemetry = monitored.get('started_at_utc') == profile.get('started_at_utc')
    return {
        'directory': str(directory),
        'status': profile.get('status'),
        'stop_reason': profile.get('stop_reason'),
        'batch_size': int(profile['batch_size']),
        'grad_cache_chunk_size': int(profile.get('grad_cache_chunk_size', 0)),
        'accumulation_chunk_size': int(profile.get('accumulation_chunk_size', 0)),
        'total_optimizer_steps': int(profile['completed_steps']),
        'total_seen_pairs': int(profile['processed_pairs']),
        'measured_steps': len(useful),
        'warmup_steps_excluded': warmup_steps,
        'mean_step_seconds': statistics.mean(durations),
        'median_step_seconds': statistics.median(durations),
        'measured_pairs_per_second': samples / total_seconds,
        'max_pytorch_peak_reserved_mib': max(float(row['peak_reserved_mib']) for row in rows),
        'max_pytorch_peak_allocated_mib': max(float(row['peak_allocated_mib']) for row in rows),
        'gpu_peak_used_mib': monitor.get('gpu_peak_used_mib') if has_current_telemetry and monitor.get('gpu_samples') else None,
        'video_id_steps': profile.get('video_id_steps'),
        'duplicate_video_batches': profile.get('duplicate_video_batches'),
        'duplicate_negative_batches': profile.get('duplicate_negative_batches'),
        'profile_path': str(profile_file),
    }


def compare_runs(baseline, gradcache, warmup_steps=10, accumulation=None):
    base = read_run(baseline, warmup_steps)
    cache = read_run(gradcache, warmup_steps)
    warnings = []
    if base['batch_size'] != 4 or base['grad_cache_chunk_size'] or base['accumulation_chunk_size']:
        warnings.append('Expected a standard batch-4 baseline')
    if cache['batch_size'] != 32 or not cache['grad_cache_chunk_size'] or cache['accumulation_chunk_size']:
        warnings.append('Expected effective batch 32 with GradCache enabled')
    runs = {'baseline': base, 'gradcache': cache}
    if accumulation is not None:
        control = read_run(accumulation, warmup_steps)
        runs['accumulation'] = control
        if control['batch_size'] != 32 or control['accumulation_chunk_size'] != 4 or control['grad_cache_chunk_size']:
            warnings.append('Expected effective batch 32 with independent chunk-4 losses')
    matched = len({run['total_seen_pairs'] for run in runs.values()}) == 1
    if not matched:
        warnings.append('Pair exposures differ; compare matching exposure budgets')
    if any(run['status'] != 'finished' for run in runs.values()):
        warnings.append('At least one run did not finish successfully')
    return {
        **runs,
        'total_pair_exposures_matched': matched,
        'measured_throughput_ratio_gradcache_over_baseline':
            cache['measured_pairs_per_second'] / base['measured_pairs_per_second'],
        'warnings': warnings,
        'interpretation': (
            'Operational pilot: throughput, memory and run stability. The exclusion '
            'is for timing startup only. Training loss and retrieval quality require separate comparisons.'
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--gradcache', type=Path, required=True)
    parser.add_argument('--accumulation', type=Path)
    parser.add_argument('--warmup-steps', type=int, default=10)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.warmup_steps < 0:
        parser.error('--warmup-steps must be >= 0')
    result = compare_runs(args.baseline, args.gradcache, args.warmup_steps, args.accumulation)
    output = json.dumps(result, indent=2, ensure_ascii=False)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
