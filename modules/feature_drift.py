"""How far do stored CLIP features drift from the current model's features?

A cross-batch memory bank (XBM, Wang et al., CVPR 2020; ContAccum, Kim et al.,
NeurIPS 2024) reuses features computed some steps ago as extra negatives. For
X-Pool the bank holds text features and per-frame features; the text-conditioned
pooling is always recomputed with the current pooling module. The bank is only
valid while the stored features stay close to what the current encoder would give.

The functions here compare two feature snapshots of the same probe samples and
express the difference as an error in the training logits.
"""

from collections import defaultdict
from contextlib import contextmanager
import math

import torch
import torch.nn.functional as F

from modules.metrics import sim_matrix_training


LOGIT_SCALE_MAX = math.log(100)

PAIR_METRICS = (
    'text_cosine_distance', 'frame_cosine_distance',
    'text_relative_l2', 'frame_relative_l2',
    'stale_video_logit_error_mean', 'stale_video_logit_error_p95', 'stale_video_logit_error_max',
    'stale_text_logit_error_mean', 'stale_text_logit_error_p95', 'stale_text_logit_error_max',
    'negative_logit_std', 'positive_margin_mean',
    'hardest_negative_agreement_stale_video',
)


@contextmanager
def evaluation_mode(model):
    """Temporarily switch to eval (no pooling dropout) and restore the previous mode."""
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def feature_change(old, new):
    """Mean cosine distance and relative L2 change between two feature tensors."""
    cosine_distance = 1 - F.cosine_similarity(old, new, dim=-1)
    relative_l2 = (new - old).norm(dim=-1) / new.norm(dim=-1).clamp_min(1e-12)
    return float(cosine_distance.mean()), float(relative_l2.mean())


@torch.no_grad()
def probe_logits(model, text, frames, pooling_type, device):
    """Training logits (text x video) of probe features with the current pooling module."""
    text = text.to(device)
    frames = frames.to(device)
    with evaluation_mode(model):
        pooled = model.pool(text, frames)
    sims = sim_matrix_training(text, pooled, pooling_type)
    scale = model.clip.logit_scale.detach().clamp(max=LOGIT_SCALE_MAX).exp()
    return (sims * scale).float().cpu()


def _off_diagonal(matrix):
    size = matrix.shape[0]
    mask = ~torch.eye(size, dtype=torch.bool)
    return matrix[mask]


def _error_summary(error):
    values = _off_diagonal(error)
    ordered = torch.sort(values).values
    p95 = ordered[min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered))) - 1)]
    return float(values.mean()), float(p95), float(values.max())


def compare_snapshots(model, old, new, pooling_type, device):
    """Compare an older snapshot with the newest one, using the current pooling module.

    old, new: dicts with CPU tensors ``text`` (P x D) and ``frames`` (P x F x D) of
    the same P probe pairs; row i of text matches row i of frames.

    stale_video: current texts against stored (old) frame features - what the
        text-to-video loss sees when the video bank is stale.
    stale_text: stored (old) texts against current frame features - what the
        video-to-text loss sees when the text bank is stale.
    Errors are absolute logit differences on the negatives (off-diagonal entries).
    """
    fresh = probe_logits(model, new['text'], new['frames'], pooling_type, device)
    stale_video = probe_logits(model, new['text'], old['frames'], pooling_type, device)
    stale_text = probe_logits(model, old['text'], new['frames'], pooling_type, device)

    text_cos, text_l2 = feature_change(old['text'], new['text'])
    frame_cos, frame_l2 = feature_change(old['frames'], new['frames'])
    video_mean, video_p95, video_max = _error_summary((stale_video - fresh).abs())
    text_mean, text_p95, text_max = _error_summary((stale_text - fresh).abs())

    size = fresh.shape[0]
    diagonal = torch.eye(size, dtype=torch.bool)
    fresh_negatives = fresh.masked_fill(diagonal, float('-inf'))
    stale_negatives = stale_video.masked_fill(diagonal, float('-inf'))
    margin = torch.diag(fresh) - fresh_negatives.max(dim=1).values
    agreement = (fresh_negatives.argmax(dim=1) == stale_negatives.argmax(dim=1)).float().mean()

    return {
        'text_cosine_distance': text_cos, 'frame_cosine_distance': frame_cos,
        'text_relative_l2': text_l2, 'frame_relative_l2': frame_l2,
        'stale_video_logit_error_mean': video_mean, 'stale_video_logit_error_p95': video_p95,
        'stale_video_logit_error_max': video_max,
        'stale_text_logit_error_mean': text_mean, 'stale_text_logit_error_p95': text_p95,
        'stale_text_logit_error_max': text_max,
        'negative_logit_std': float(_off_diagonal(fresh).std()),
        'positive_margin_mean': float(margin.mean()),
        'hardest_negative_agreement_stale_video': float(agreement),
    }


def summarize_by_lag(records, batch_size):
    """Average pair records that share the same step distance.

    ``bank_entries`` is how many samples are enqueued in ``lag_steps`` steps, i.e. the
    bank size whose oldest entry has that age.
    """
    groups = defaultdict(list)
    for record in records:
        groups[record['lag_steps']].append(record)
    rows = []
    for lag in sorted(groups):
        items = groups[lag]
        row = {'lag_steps': lag, 'bank_entries': lag * batch_size, 'pairs': len(items)}
        for name in PAIR_METRICS:
            row[name] = sum(item[name] for item in items) / len(items)
        rows.append(row)
    return rows


def suggest_bank_size(lag_rows, tolerance_ratio):
    """Largest measured bank size whose mean logit error stays within the tolerance.

    The rule is a heuristic, not a result from the literature: the mean stale-logit
    error (worse of the two directions) must not exceed ``tolerance_ratio`` times the
    spread (std) of the negative logits, for this lag and every shorter one.
    Returns (bank_entries, reason).
    """
    if not lag_rows:
        return 0, 'no measurements'
    accepted = None
    for row in lag_rows:
        error = max(row['stale_video_logit_error_mean'], row['stale_text_logit_error_mean'])
        if error <= tolerance_ratio * row['negative_logit_std']:
            accepted = row
        else:
            break
    if accepted is None:
        return 0, 'the shortest measured lag already exceeds the tolerance'
    if accepted is lag_rows[-1]:
        return accepted['bank_entries'], 'every measured lag is within the tolerance; a longer run could allow more'
    return accepted['bank_entries'], 'the next longer lag exceeds the tolerance'
