"""Summarize saved query rankings and plot checkpoint/failure diagnostics.

This script does not load model weights or run model inference. Plotting requires
matplotlib and numpy; the evaluation environment can remain unchanged.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import textwrap


LABELS = ['official', 'epoch1', 'legacy6h', 'epoch2', 'epoch3', 'epoch4', 'epoch5']
BUCKETS = [('1', 1, 1), ('2-5', 2, 5), ('6-10', 6, 10), ('11-50', 11, 50), ('51-100', 51, 100), ('101-1000', 101, 1000)]


def load_rows(path):
    with path.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ['query_index', 'rank', 'gt_score_ties', 'gt_attention_argmax', 'top1_attention_argmax']:
            row[key] = int(row[key])
        for key in ['gt_score', 'top1_score', 'gt_minus_top1', 'gt_attention_entropy', 'top1_attention_entropy', 'gt_attention_peak', 'top1_attention_peak']:
            row[key] = float(row[key])
        for key in ['gt_attention', 'top1_attention', 'top10_video_ids', 'top10_scores']:
            row[key] = json.loads(row[key])
    return rows


def mean(values):
    return sum(values) / len(values) if values else None


def summarize(root):
    evaluations = {label: json.loads((root / label / 'evaluation.json').read_text(encoding='utf-8')) for label in LABELS}
    rows = {label: load_rows(root / label / 'queries.csv') for label in LABELS}
    ids = [r['video_id'] for r in rows['official']]
    assert len(ids) == len(set(ids)) == 1000
    assert all([r['video_id'] for r in rows[label]] == ids for label in LABELS)
    for label in LABELS:
        for key, threshold in [('R1', 1), ('R5', 5), ('R10', 10)]:
            assert abs(evaluations[label]['metrics'][key] - sum(r['rank'] <= threshold for r in rows[label]) / 10) < 1e-9
    best = max(LABELS[1:], key=lambda label: (evaluations[label]['metrics']['R1'], -evaluations[label]['checkpoint_metadata']['global_step']))
    comparisons = {}
    for label in LABELS[1:]:
        a, b = rows['official'], rows[label]
        groups = {
            'both_hit': [i for i in range(1000) if a[i]['rank'] == b[i]['rank'] == 1],
            'official_only_hit': [i for i in range(1000) if a[i]['rank'] == 1 and b[i]['rank'] > 1],
            'trained_only_hit': [i for i in range(1000) if a[i]['rank'] > 1 and b[i]['rank'] == 1],
            'both_miss': [i for i in range(1000) if a[i]['rank'] > 1 and b[i]['rank'] > 1],
        }
        assert sum(len(v) for v in groups.values()) == 1000
        attention = {}
        for name, indices in groups.items():
            attention[name] = {
                model: {
                    'gt_normalized_entropy_mean': mean([rows[model][i]['gt_attention_entropy'] for i in indices]),
                    'gt_peak_weight_mean': mean([rows[model][i]['gt_attention_peak'] for i in indices]),
                } for model in ['official', label]
            }
        comparisons[label] = {
            'r1_groups': {k: len(v) for k, v in groups.items()},
            'official_only_hit_trained_ranks': {name: sum(lo <= b[i]['rank'] <= hi for i in groups['official_only_hit']) for name, lo, hi in BUCKETS},
            'rank_improved': sum(b[i]['rank'] < a[i]['rank'] for i in range(1000)),
            'rank_equal': sum(b[i]['rank'] == a[i]['rank'] for i in range(1000)),
            'rank_worsened': sum(b[i]['rank'] > a[i]['rank'] for i in range(1000)),
            'mean_rank_difference_trained_minus_official': mean([b[i]['rank'] - a[i]['rank'] for i in range(1000)]),
            'attention_by_r1_group': attention,
        }
    buckets = {label: {name: sum(lo <= r['rank'] <= hi for r in rows[label]) for name, lo, hi in BUCKETS} for label in LABELS}
    with (root / 'metrics_summary.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['label', 'epoch', 'global_step', 'R1', 'R5', 'R10', 'MedR', 'MeanR', 'evaluation_seconds'])
        writer.writeheader()
        for label in LABELS:
            report = evaluations[label]
            writer.writerow(dict(label=label, epoch=report['checkpoint_metadata'].get('epoch'), global_step=report['checkpoint_metadata'].get('global_step'),
                                 **{k: report['metrics'][k] for k in ['R1', 'R5', 'R10', 'MedR', 'MeanR']}, evaluation_seconds=report['wall_time_seconds']))
    with (root / 'rank_comparison.csv').open('w', encoding='utf-8', newline='') as handle:
        names = ['query_index', 'video_id', 'caption'] + [f'{label}_{suffix}' for label in LABELS for suffix in ['rank', 'top1_video_id']]
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for i in range(1000):
            item = {k: rows['official'][i][k] for k in ['query_index', 'video_id', 'caption']}
            for label in LABELS:
                for suffix in ['rank', 'top1_video_id']:
                    item[label + '_' + suffix] = rows[label][i][suffix]
            writer.writerow(item)

    a, b = rows['official'], rows[best]
    regressions = sorted([i for i in range(1000) if a[i]['rank'] == 1 and b[i]['rank'] > 1], key=lambda i: (-b[i]['rank'], i))
    recoveries = sorted([i for i in range(1000) if a[i]['rank'] > 1 and b[i]['rank'] == 1], key=lambda i: (-a[i]['rank'], i))
    shared = sorted([i for i in range(1000) if a[i]['rank'] > 10 and b[i]['rank'] > 10], key=lambda i: (-min(a[i]['rank'], b[i]['rank']), i))
    controls = [i for i in range(1000) if a[i]['rank'] == b[i]['rank'] == 1]
    selection = [
        ('largest_regression', regressions[0], 'Official R@1 hit with the largest trained rank; ties use query order.'),
        ('median_regression', regressions[len(regressions)//2], 'Middle item after sorting official-only hits by trained rank.'),
        ('near_miss_regression', regressions[-1], 'Smallest trained rank among official-only hits; deterministic query-order tie break.'),
        ('largest_recovery', recoveries[0], 'Trained R@1 hit with the largest official rank.'),
        ('median_recovery', recoveries[len(recoveries)//2], 'Middle item after sorting trained-only hits by official rank.'),
        ('shared_hard_failure', shared[0], 'Largest minimum rank among queries outside both top-10 lists.'),
        ('shared_hit_control', controls[0], 'First query that is a top-1 hit for both models.'),
    ]
    cases = []
    for number, (category, idx, rule) in enumerate(selection, 1):
        cases.append({'case_id': number, 'selection': category, 'selection_rule': rule, 'query_index': idx,
                      'video_id': ids[idx], 'caption': a[idx]['caption'], 'models': {label: rows[label][idx] for label in dict.fromkeys(['official', best, 'epoch5'])}})
    result = {
        'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'queries': 1000,
        'highest_test_r1_trained_checkpoint': best,
        'selection_warning': 'Highest R@1 is a descriptive comparison on the same 1K-A test set, not a validation-selected model or a new independent generalization result.',
        'rank_buckets': buckets, 'comparisons_to_official': comparisons,
        'metrics': {label: evaluations[label]['metrics'] for label in LABELS},
        'overall_gt_attention': {label: {'normalized_entropy_mean': mean([r['gt_attention_entropy'] for r in rows[label]]),
                                         'peak_weight_mean': mean([r['gt_attention_peak'] for r in rows[label]])} for label in LABELS},
        'epoch4_to_epoch5_r1': {
            'both_hit': sum(rows['epoch4'][i]['rank'] == rows['epoch5'][i]['rank'] == 1 for i in range(1000)),
            'epoch4_only_hit': sum(rows['epoch4'][i]['rank'] == 1 and rows['epoch5'][i]['rank'] > 1 for i in range(1000)),
            'epoch5_only_hit': sum(rows['epoch4'][i]['rank'] > 1 and rows['epoch5'][i]['rank'] == 1 for i in range(1000)),
            'both_miss': sum(rows['epoch4'][i]['rank'] > 1 and rows['epoch5'][i]['rank'] > 1 for i in range(1000)),
        },
        'attention_definition': 'Normalized Shannon entropy across 12 frames: 0 concentrated, 1 uniform; raw learned attention is not a causal explanation.',
        'case_selection': 'Seven deterministic rank-based illustrative cases; not a representative random sample or an exhaustive failure taxonomy.',
    }
    (root / 'analysis_summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    (root / 'casebook.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return result, evaluations, rows, cases


def plot(root, summary, evaluations, rows, cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    directory = root / 'figures'
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    stages = LABELS[1:]
    positions = [evaluations[label]['checkpoint_metadata']['global_step']/45000 for label in stages]
    for metric in ['R1', 'R5', 'R10']:
        ax.plot(positions, [evaluations[label]['metrics'][metric] for label in stages], marker='o', label=metric)
    ax.axhline(46.9, color='#444444', linestyle='--', label='Official checkpoint R@1: 46.9')
    ax.axvline(62216/45000, color='#999999', linestyle=':', linewidth=.8)
    ax.annotate('6h: before resume', xy=(62216/45000, evaluations['legacy6h']['metrics']['R1']), xytext=(1.55,42.5),
                textcoords='data', fontsize=8, arrowprops={'arrowstyle': '->', 'color': '#666666', 'lw': .8})
    ax.set(xlabel='Recorded training progress: global step / 45,000 (batch size 4)', ylabel='Recall (%)', xticks=[1, 2, 3, 4, 5], title='MSR-VTT 1K-A: existing checkpoint evaluation')
    ax.legend(loc='best');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(directory / 'checkpoint_recall.png', dpi=180);plt.close(fig)

    best = summary['highest_test_r1_trained_checkpoint']
    labels = list(dict.fromkeys(['official', best, 'epoch5']))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(BUCKETS));width = .8 / len(labels)
    for j, label in enumerate(labels):
        values = [summary['rank_buckets'][label][name] for name, _, _ in BUCKETS]
        ax.bar(x + (j-(len(labels)-1)/2)*width, values, width=width, label=label)
    ax.set(xticks=x, xticklabels=[b[0] for b in BUCKETS], xlabel='Ground-truth rank (1-based)', ylabel='Queries (out of 1000)', title='Rank distribution: top-1 accuracy and retrieval tail')
    ax.legend();fig.tight_layout();fig.savefig(directory / 'rank_buckets.png', dpi=180);plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    a = np.array([r['rank'] for r in rows['official']]);b = np.array([r['rank'] for r in rows[best]])
    groups = [('both hit', (a==1)&(b==1), '#2ca02c'), ('official only hit', (a==1)&(b>1), '#d62728'),
              ('trained only hit', (a>1)&(b==1), '#1f77b4'), ('both miss', (a>1)&(b>1), '#888888')]
    for name, mask, color in groups:
        ax.scatter(a[mask], b[mask], alpha=.45, s=18, color=color, label=f'{name} (n={mask.sum()})')
    ax.plot([1,1000],[1,1000], color='black', linewidth=.7)
    ax.set(xscale='log', yscale='log', xlabel='Official checkpoint rank', ylabel=f'{best} rank', title='Same 1000 queries; above diagonal = worse trained rank')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(directory / 'paired_ranks.png', dpi=180);plt.close(fig)

    for case in cases:
        matrix = [];names = []
        for label, row in case['models'].items():
            for kind in ['gt', 'top1']:
                matrix.append(row[kind + '_attention'])
                video = row['video_id'] if kind == 'gt' else row['top1_video_id']
                names.append(f'{label}: {kind} {video}\nGT rank {row["rank"]}')
        fig, ax = plt.subplots(figsize=(11, 3.6 if len(names)==4 else 4.5))
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap='viridis', aspect='auto')
        ax.set(yticks=np.arange(len(names)), yticklabels=names, xticks=np.arange(12), xticklabels=np.arange(1,13), xlabel='Sampled frame position (uniform 12-frame sampling)',
               title=f'Case {case["case_id"]}: {case["selection"]}\n' + textwrap.fill(case['caption'], 95))
        fig.colorbar(im, ax=ax, label='Attention probability', fraction=.025)
        fig.tight_layout();fig.savefig(directory / f'attention_case_{case["case_id"]:02d}.png', dpi=160, bbox_inches='tight');plt.close(fig)
    return {'python': sys.version, 'matplotlib': matplotlib.__version__, 'numpy': np.__version__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=Path(__file__).resolve().parent / 'results/checkpoint_analysis_20261109')
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    result, evaluations, rows, cases = summarize(args.results_dir)
    if args.plots:
        versions = plot(args.results_dir, result, evaluations, rows, cases)
        (args.results_dir / 'plotting_environment.json').write_text(json.dumps(versions, indent=2)+'\n', encoding='utf-8')
    best = result['highest_test_r1_trained_checkpoint']
    print(json.dumps({'highest_test_r1_trained_checkpoint': best,
                      'metrics': result['metrics'],
                      'paired_R1_counts': result['comparisons_to_official'][best]['r1_groups'],
                      'cases': [{'case_id': c['case_id'], 'video_id': c['video_id'], 'caption': c['caption']} for c in cases]}, indent=2))


if __name__ == '__main__':
    main()
