"""Large-batch contrastive gradients from small chunks (gradient caching) for X-Pool.

Method: Gao et al., "Scaling Deep Contrastive Learning Batch Size under Memory
Limited Setup" (RepL4NLP 2021), https://arxiv.org/abs/2101.06983

X-Pool differs from a plain dual encoder: the video embedding depends on the text,
so the similarity matrix needs a pooling over every text-video combination. The
split used here is

    encode  (CLIP text / image towers)   per sample, heavy  -> run chunk by chunk
    pool + similarity + loss             across samples, light -> run once on all

Requirements for the result to equal the full-batch gradient:
  * ``model.encode`` gives the same output for a sample whatever chunk it is in and
    however often it is called (no dropout, no batch statistics). The Hugging Face
    CLIP used here has dropout 0 and only LayerNorm.
  * The pooling dropout is drawn once, in the single full pooling call.
  * One loss over the whole similarity matrix, not a sum of per-chunk losses.
  * The caller zeroes gradients before, and clips/steps the optimizer once after.
"""

from collections.abc import Mapping

import torch

from modules.metrics import sim_matrix_training


def batch_size_of(data):
    return data['video'].shape[0]


def slice_batch(data, start, stop):
    """Rows [start, stop) of a collated batch. Tensors, lists and nested dicts are sliced."""
    chunk = {}
    for key, value in data.items():
        if isinstance(value, Mapping):
            chunk[key] = {name: item[start:stop] for name, item in value.items()}
        else:
            chunk[key] = value[start:stop]
    return chunk


def split_batch(data, chunk_size):
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    total = batch_size_of(data)
    return [slice_batch(data, start, min(start + chunk_size, total))
            for start in range(0, total, chunk_size)]


def move_batch(data, device):
    """Move the tensors of a batch to ``device``; strings and other values are kept."""
    moved = {}
    for key, value in data.items():
        if isinstance(value, Mapping):
            moved[key] = {name: item.to(device) for name, item in value.items()}
        elif isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def grad_cache_backward(model, loss_fn, data, chunk_size, pooling_type,
                        device=None, before_pool=None, check_recompute=True):
    """Accumulate into ``.grad`` the gradient of one loss over the whole batch.

    model: provides ``encode(data)``, ``pool(text, frames)`` and ``clip.logit_scale``.
    data: collated batch with a tokenized ``text`` (dict of tensors) and ``video``.
    chunk_size: samples encoded with a graph at one time (what fits in memory).
    device: if given, each chunk is moved there just before it is encoded, so the
        whole batch can stay on the CPU.
    before_pool: optional callable run right before the full pooling call, e.g. to
        seed the pooling dropout in an equivalence test.

    Returns a dict with the detached loss and similarity matrix. The caller owns
    ``zero_grad`` (before) and gradient clipping + ``optimizer.step`` (after).
    """
    chunks = split_batch(data, chunk_size)

    def load(chunk):
        return chunk if device is None else move_batch(chunk, device)

    # 1) Features of every sample without a graph.
    with torch.no_grad():
        cached = [model.encode(load(chunk)) for chunk in chunks]
    text_features = torch.cat([text for text, _ in cached]).requires_grad_()
    video_features = torch.cat([frames for _, frames in cached]).requires_grad_()

    # 2) One pooling + loss over all text-video combinations. The pooling module
    #    and logit_scale get their gradients here; the feature gradients are cached.
    if before_pool is not None:
        before_pool()
    pooled = model.pool(text_features, video_features)
    sims = sim_matrix_training(text_features, pooled, pooling_type)
    loss = loss_fn(sims, model.clip.logit_scale)
    loss.backward()
    text_grad, video_grad = text_features.grad, video_features.grad

    # 3) Encode each chunk again with a graph and push the cached gradients through.
    offset = 0
    recompute_diff = 0.0
    for chunk, (cached_text, cached_frames) in zip(chunks, cached):
        text, frames = model.encode(load(chunk))
        count = text.shape[0]
        surrogate = (text * text_grad[offset:offset + count]).sum() \
            + (frames * video_grad[offset:offset + count]).sum()
        surrogate.backward()
        if check_recompute:
            recompute_diff = max(recompute_diff,
                                 float((text.detach() - cached_text).abs().max()),
                                 float((frames.detach() - cached_frames).abs().max()))
        offset += count

    return {
        'loss': loss.detach(),
        'sims': sims.detach(),
        'num_chunks': len(chunks),
        # 0 (or float noise) when encode is deterministic; large when it is not.
        'feature_recompute_max_abs_diff': recompute_diff if check_recompute else None,
    }
