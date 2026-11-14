"""Three short checks to run before any long small-batch experiment.

    gradcheck  Does a chunked GradCache step reproduce the full-batch gradient?
    steptime   How much of a training step is video decoding vs GPU work?
    drift      How fast do stored CLIP features go stale (memory-bank size)?

Nothing here writes a training checkpoint or touches an existing experiment
folder. Results go to outputs/first_experiment/<check>_<UTC time>/.
See FIRST_EXPERIMENT.md for commands and how to read the numbers.
"""

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from datasets.data_factory import DataFactory
from datasets.model_transforms import init_transform_dict
from datasets.video_capture import VideoCapture
from model.clip_transformer import CLIPTransformer
from modules.feature_drift import compare_snapshots, evaluation_mode, suggest_bank_size, summarize_by_lag
from modules.grad_cache import batch_size_of, grad_cache_backward, move_batch, split_batch
from modules.loss import CLIPLoss
from modules.metrics import sim_matrix_training
from modules.optimization import AdamW, get_cosine_schedule_with_warmup
from modules.training_state import capture_random_state, restore_random_state


ROOT = Path(__file__).resolve().parent
TRAINING_EPOCHS = 5
WARMUP_PROPORTION = 0.1


# ----------------------------------------------------------------------------
# Shared setup
# ----------------------------------------------------------------------------

def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def seed_everything(seed):
    """Same seeding and determinism flags as train.py."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def reseed(seed, device):
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def build_config(args):
    """The settings of the recorded batch-4 run, without AllConfig's directory side effects."""
    return SimpleNamespace(
        dataset_name='MSRVTT', msrvtt_train_file='9k', videos_dir=args.videos_dir,
        num_frames=args.num_frames, video_sample_type='uniform', input_res=args.input_res,
        huggingface=True, arch='clip_transformer', clip_arch='ViT-B/32', embed_dim=args.embed_dim,
        loss='clip', clip_lr=args.clip_lr, noclip_lr=args.noclip_lr, weight_decay=0.2,
        batch_size=args.batch_size, num_epochs=TRAINING_EPOCHS, warmup_proportion=WARMUP_PROPORTION,
        pooling_type='transformer', k=-1, attention_temperature=0.01, num_mha_heads=1,
        transformer_dropout=0.3, num_workers=args.num_workers, seed=args.seed,
    )


def build_tokenizer():
    from transformers import CLIPTokenizer
    return CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32', TOKENIZERS_PARALLELISM=False)


def tokenize(data, tokenizer):
    """Copy of a collated batch whose captions are replaced by token tensors (as in Trainer)."""
    tokens = tokenizer(list(data['text']), return_tensors='pt', padding=True, truncation=True)
    result = dict(data)
    result['text'] = {name: value for name, value in tokens.items()}
    return result


def build_model(config, checkpoint_path=None):
    model = CLIPTransformer(config)
    checkpoint = None
    if checkpoint_path is not None:
        checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
        state = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(state)
    return model, checkpoint


def build_optimizer(model, config):
    """Same parameter groups as train.py."""
    named = list(model.named_parameters())
    groups = [
        {'params': [p for n, p in named if 'clip.' in n], 'lr': config.clip_lr},
        {'params': [p for n, p in named if 'clip.' not in n], 'lr': config.noclip_lr},
    ]
    return AdamW(groups, weight_decay=config.weight_decay)


def train_step(model, loss_fn, optimizer, scheduler, data, pooling_type):
    """One optimizer update, identical to the body of Trainer._train_epoch."""
    text_embeds, video_embeds_pooled = model(data)
    sims = sim_matrix_training(text_embeds, video_embeds_pooled, pooling_type)
    loss = loss_fn(sims, model.clip.logit_scale)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    optimizer.zero_grad()
    torch.clamp_(model.clip.logit_scale.data, max=np.log(100))
    return loss.detach()


def environment_info(device):
    info = {
        'python': sys.version.split()[0], 'torch': torch.__version__, 'device': str(device),
        'utc_time': datetime.now(timezone.utc).isoformat(),
    }
    if device.type == 'cuda':
        info['gpu'] = torch.cuda.get_device_name(device)
        info['gpu_total_mib'] = round(torch.cuda.get_device_properties(device).total_memory / 2**20, 1)
    return info


def measured(device, function):
    """Run ``function`` and return (result, seconds, peak allocated MiB or None)."""
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    start = time.perf_counter()
    result = function()
    sync(device)
    seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None
    return result, seconds, peak


# ----------------------------------------------------------------------------
# 1. gradcheck
# ----------------------------------------------------------------------------

def gradient_group(name):
    if name.startswith('clip.text_model') or name.startswith('clip.text_projection'):
        return 'clip_text'
    if name.startswith('clip.vision_model') or name.startswith('clip.visual_projection'):
        return 'clip_vision'
    if name == 'clip.logit_scale':
        return 'logit_scale'
    if name.startswith('pool_frames'):
        return 'pooling'
    return 'other'


def collect_gradients(model):
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            gradients[name] = torch.zeros(parameter.shape, dtype=parameter.dtype)
        else:
            gradients[name] = parameter.grad.detach().cpu().clone()
    return gradients


def compare_gradients(reference, candidate):
    """Relative L2 error, cosine and largest absolute difference per parameter group."""
    sums = defaultdict(lambda: defaultdict(float))
    for name, expected in reference.items():
        expected = expected.double()
        actual = candidate[name].double()
        difference = actual - expected
        for key in (gradient_group(name), 'all'):
            entry = sums[key]
            entry['difference'] += float((difference ** 2).sum())
            entry['reference'] += float((expected ** 2).sum())
            entry['candidate'] += float((actual ** 2).sum())
            entry['dot'] += float((expected * actual).sum())
            entry['max_abs_diff'] = max(entry['max_abs_diff'], float(difference.abs().max()))
    result = {}
    for key, entry in sums.items():
        reference_norm = math.sqrt(entry['reference'])
        candidate_norm = math.sqrt(entry['candidate'])
        if reference_norm > 0:
            relative = math.sqrt(entry['difference']) / reference_norm
        else:
            relative = 0.0 if entry['difference'] == 0 else None
        cosine = entry['dot'] / (reference_norm * candidate_norm) if reference_norm > 0 and candidate_norm > 0 else None
        result[key] = {
            'relative_l2_error': relative, 'cosine': cosine,
            'max_abs_diff': entry['max_abs_diff'], 'reference_norm': reference_norm,
        }
    return result


def reference_backward(model, loss_fn, data, pooling_type, before_pool):
    """Ordinary full-batch forward/backward: the gradient GradCache must reproduce."""
    text_features, video_features = model.encode(data)
    before_pool()
    pooled = model.pool(text_features, video_features)
    sims = sim_matrix_training(text_features, pooled, pooling_type)
    loss = loss_fn(sims, model.clip.logit_scale)
    loss.backward()
    return loss.detach()


def accumulation_backward(model, loss_fn, data, chunk_size, pooling_type, before_pool):
    """Plain gradient accumulation: a separate small contrastive loss per chunk.

    Reported only as a contrast. Each chunk sees only its own negatives, so this
    is NOT expected to match the full-batch gradient.
    """
    total = batch_size_of(data)
    before_pool()
    loss_sum = 0.0
    for chunk in split_batch(data, chunk_size):
        text_features, pooled = model(chunk)
        sims = sim_matrix_training(text_features, pooled, pooling_type)
        loss = loss_fn(sims, model.clip.logit_scale) * (batch_size_of(chunk) / total)
        loss.backward()
        loss_sum += float(loss.detach())
    return loss_sum


@torch.no_grad()
def forward_split_check(model, data):
    """forward() must give exactly what pool(encode()) gives (no dropout in eval mode)."""
    with evaluation_mode(model):
        text_a, frames_a, pooled_a = model(data, return_all_frames=True)
        text_b, frames_b = model.encode(data)
        pooled_b = model.pool(text_b, frames_b)
    return {
        'bitwise_equal': bool(torch.equal(text_a, text_b) and torch.equal(frames_a, frames_b)
                              and torch.equal(pooled_a, pooled_b)),
        'max_abs_diff': float((pooled_a - pooled_b).abs().max()),
    }


def run_gradcheck(model, loss_fn, batches, chunk_size, pooling_type, device,
                  seed=24, tolerance=1e-2, min_loss=1e-4, num_informative=3, max_batches=None):
    """Compare full-batch, GradCache and plain-accumulation gradients on real batches.

    A batch whose loss is below ``min_loss`` has an almost zero gradient, so a
    relative error on it is mostly float noise. Such batches are reported but do not
    count toward the verdict; batches are drawn until ``num_informative`` remain.
    """
    model.train()
    results = []
    forward_check = None
    informative = 0
    for index, batch in enumerate(batches):
        if informative >= num_informative or (max_batches is not None and index >= max_batches):
            break
        data = move_batch(batch, device)
        if forward_check is None:
            forward_check = forward_split_check(model, data)
        batch_seed = seed + index

        def seed_pool():
            reseed(batch_seed, device)

        # Each path starts from a different random stream, so an encoder that draws
        # random numbers cannot match by coincidence. Only the pooling dropout is
        # aligned, through seed_pool.
        model.zero_grad()
        reseed(batch_seed + 1000003, device)
        reference_loss, reference_seconds, reference_peak = measured(
            device, lambda: reference_backward(model, loss_fn, data, pooling_type, seed_pool))
        reference = collect_gradients(model)

        model.zero_grad()
        reseed(batch_seed + 2000003, device)
        cache_output, cache_seconds, cache_peak = measured(
            device, lambda: grad_cache_backward(model, loss_fn, data, chunk_size, pooling_type,
                                                before_pool=seed_pool))
        cache = compare_gradients(reference, collect_gradients(model))

        model.zero_grad()
        accumulation_loss, accumulation_seconds, accumulation_peak = measured(
            device, lambda: accumulation_backward(model, loss_fn, data, chunk_size, pooling_type, seed_pool))
        accumulation = compare_gradients(reference, collect_gradients(model))
        model.zero_grad()

        loss_value = float(reference_loss)
        is_informative = loss_value >= min_loss and cache['all']['reference_norm'] > 0
        error = cache['all']['relative_l2_error']
        passed = error is not None and error <= tolerance
        informative += int(is_informative)
        result = {
            'batch_index': index, 'batch_size': batch_size_of(data), 'chunk_size': chunk_size,
            'video_ids': [str(v) for v in batch.get('video_id', [])],
            'reference_loss': loss_value, 'grad_cache_loss': float(cache_output['loss']),
            'accumulation_loss': accumulation_loss,
            'informative': is_informative, 'grad_cache_passed': passed,
            'feature_recompute_max_abs_diff': cache_output['feature_recompute_max_abs_diff'],
            'grad_cache_vs_full': cache, 'accumulation_vs_full': accumulation,
            'seconds': {'full': reference_seconds, 'grad_cache': cache_seconds, 'accumulation': accumulation_seconds},
            'peak_allocated_mib': {'full': reference_peak, 'grad_cache': cache_peak, 'accumulation': accumulation_peak},
        }
        results.append(result)
        print('batch {}: loss {:.6f}{} | grad_cache rel_err {:.3e} cos {} | accumulation rel_err {:.3e} cos {}'.format(
            index, loss_value, '' if is_informative else ' (too small, not counted)',
            error if error is not None else float('nan'), _format(cache['all']['cosine']),
            accumulation['all']['relative_l2_error'] if accumulation['all']['relative_l2_error'] is not None else float('nan'),
            _format(accumulation['all']['cosine'])), flush=True)

    counted = [r for r in results if r['informative']]
    passed = bool(counted) and all(r['grad_cache_passed'] for r in counted) \
        and forward_check is not None and forward_check['bitwise_equal']
    return {
        'passed': passed, 'tolerance_relative_l2': tolerance, 'min_loss': min_loss,
        'informative_batches': len(counted), 'requested_informative_batches': num_informative,
        'forward_equals_encode_then_pool': forward_check, 'batches': results,
    }


def _format(value):
    return 'n/a' if value is None else '{:.6f}'.format(value)


def gradcheck(args, config, device, output):
    seed_everything(config.seed)
    tokenizer = build_tokenizer()
    loader = DataFactory.get_data_loader(config, split_type='train')
    model, _ = build_model(config, args.checkpoint)
    model = model.to(device)
    loss_fn = CLIPLoss().to(device)
    batches = (tokenize(batch, tokenizer) for batch in loader)
    report = run_gradcheck(model, loss_fn, batches, args.chunk_size, config.pooling_type, device,
                           seed=config.seed, tolerance=args.tolerance, min_loss=args.min_loss,
                           num_informative=args.num_batches, max_batches=args.num_batches * 5)
    del batches  # closes the loader workers
    report.update(environment=environment_info(device), checkpoint=str(args.checkpoint) if args.checkpoint else None)
    write_json(output / 'gradcheck_report.json', report)
    print('forward == pool(encode): {}'.format(report['forward_equals_encode_then_pool']))
    print('GRADCHECK {} ({} informative batches, tolerance {:g})'.format(
        'PASSED' if report['passed'] else 'FAILED', report['informative_batches'], args.tolerance))
    return 0 if report['passed'] else 1


# ----------------------------------------------------------------------------
# 2. steptime
# ----------------------------------------------------------------------------

def run_steptime(model, loss_fn, optimizer, loader, tokenizer, pooling_type, device,
                 steps, warmup, keep=8):
    """Time the phases of real training steps and of a no-grad encode.

    data_wait is the time the main process is blocked waiting for the loader. With
    worker processes it is close to zero whenever the workers keep up with the GPU.
    """
    model.train()
    rows = []
    kept = []
    iterator = iter(loader)
    for step in range(steps + warmup):
        start = time.perf_counter()
        batch = next(iterator)
        loaded = time.perf_counter()
        batch = tokenize(batch, tokenizer)
        tokenized = time.perf_counter()
        data = move_batch(batch, device)
        sync(device)
        moved = time.perf_counter()
        loss = train_step(model, loss_fn, optimizer, None, data, pooling_type)
        sync(device)
        finished = time.perf_counter()
        rows.append({
            'step': step + 1, 'warmup': step < warmup, 'loss': float(loss),
            'data_wait': loaded - start, 'tokenize': tokenized - loaded,
            'to_device': moved - tokenized, 'gpu_step': finished - moved, 'total': finished - start,
        })
        kept = (kept + [data])[-keep:]
    del iterator

    # No-grad encode of one chunk: the extra pass GradCache adds per chunk.
    encode_seconds = []
    with torch.no_grad():
        for repeat in range(2):
            for data in kept:
                sync(device)
                start = time.perf_counter()
                model.encode(data)
                sync(device)
                if repeat == 1:
                    encode_seconds.append(time.perf_counter() - start)
    return rows, encode_seconds


def measure_decode(dataset, count, seed):
    """Single-process cost of producing one training sample, split into decode and transform."""
    order = random.Random(seed).sample(range(len(dataset)), min(count, len(dataset)))
    rows = []
    for index in order:
        video_path, _, video_id = dataset._get_vidpath_and_caption_by_index(index)
        start = time.perf_counter()
        frames, _ = VideoCapture.load_frames_from_video(video_path, dataset.config.num_frames,
                                                        dataset.config.video_sample_type)
        decoded = time.perf_counter()
        if dataset.img_transforms is not None:
            dataset.img_transforms(frames)
        transformed = time.perf_counter()
        rows.append({'video_id': video_id, 'height': int(frames.shape[2]), 'width': int(frames.shape[3]),
                     'decode': decoded - start, 'transform': transformed - decoded})
    return rows


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def summarize_steptime(rows, encode_seconds, decode_rows, batch_size, num_workers,
                       pairs_per_epoch, epochs=TRAINING_EPOCHS, grad_cache_batch=32):
    """Turn raw timings into shares, a bottleneck verdict and a GradCache time estimate."""
    measured_rows = [row for row in rows if not row['warmup']]
    phases = {name: _mean(row[name] for row in measured_rows)
              for name in ('data_wait', 'tokenize', 'to_device', 'gpu_step', 'total')}
    total = phases['total']
    shares = {name: (phases[name] / total if total > 0 else 0.0)
              for name in ('data_wait', 'tokenize', 'to_device', 'gpu_step')}

    decode = _mean(row['decode'] for row in decode_rows)
    transform = _mean(row['transform'] for row in decode_rows)
    sample_cpu = decode + transform
    workers = max(1, num_workers)
    supply = workers / sample_cpu if sample_cpu > 0 else None        # samples/s the loader can produce
    busy = phases['tokenize'] + phases['to_device'] + phases['gpu_step']
    demand = batch_size / busy if busy > 0 else None                 # samples/s the GPU could consume

    if supply is not None and demand is not None and supply < demand:
        verdict = 'data-bound: the loader produces fewer samples per second than the GPU could consume'
    elif shares['data_wait'] >= 0.2:
        verdict = 'data-bound: at least 20% of each step is spent waiting for the loader'
    else:
        verdict = 'gpu-bound: the loader keeps up; faster decoding would change little'

    encode = _mean(encode_seconds)
    updates_per_epoch = pairs_per_epoch / batch_size
    chunks = grad_cache_batch / batch_size
    return {
        'measured_steps': len(measured_rows), 'batch_size': batch_size, 'num_workers': num_workers,
        'mean_seconds': phases, 'share_of_step': shares,
        'decode_seconds_per_video': decode, 'transform_seconds_per_video': transform,
        'decoded_videos': len(decode_rows),
        'loader_supply_samples_per_second': supply, 'gpu_demand_samples_per_second': demand,
        'verdict': verdict,
        'speedup_if_data_wait_removed': total / (total - phases['data_wait']) if total > phases['data_wait'] else None,
        'projected_hours_vanilla': updates_per_epoch * epochs * total / 3600,
        'no_grad_encode_seconds_per_chunk': encode,
        # Estimate only: GradCache adds one no-grad encode per chunk to the same work.
        'grad_cache_estimate': {
            'effective_batch': grad_cache_batch, 'chunks_per_update': chunks,
            'seconds_per_update': chunks * (total + encode),
            'time_ratio_to_vanilla': (total + encode) / total if total > 0 else None,
            'projected_hours': updates_per_epoch * epochs * (total + encode) / 3600,
        },
    }


def steptime(args, config, device, output):
    seed_everything(config.seed)
    tokenizer = build_tokenizer()
    loader = DataFactory.get_data_loader(config, split_type='train')
    model, _ = build_model(config)
    model = model.to(device)
    loss_fn = CLIPLoss().to(device)
    optimizer = build_optimizer(model, config)
    rows, encode_seconds = run_steptime(model, loss_fn, optimizer, loader, tokenizer, config.pooling_type,
                                        device, args.steps, args.warmup)
    dataset = loader.dataset.dataset
    decode_rows = measure_decode(dataset, args.decode_videos, config.seed)
    summary = summarize_steptime(rows, encode_seconds, decode_rows, config.batch_size, config.num_workers,
                                 pairs_per_epoch=len(dataset))
    summary['environment'] = environment_info(device)
    write_csv(output / 'step_times.csv', rows)
    write_csv(output / 'decode_times.csv', decode_rows)
    write_json(output / 'steptime_report.json', summary)

    seconds, shares = summary['mean_seconds'], summary['share_of_step']
    print('mean step {:.3f}s over {} steps'.format(seconds['total'], summary['measured_steps']))
    for name in ('data_wait', 'tokenize', 'to_device', 'gpu_step'):
        print('  {:<10} {:.4f}s  {:5.1f}%'.format(name, seconds[name], 100 * shares[name]))
    print('one sample on one CPU process: decode {:.3f}s + transform {:.3f}s'.format(
        summary['decode_seconds_per_video'], summary['transform_seconds_per_video']))
    print('loader supply {} samples/s vs GPU demand {} samples/s'.format(
        _rate(summary['loader_supply_samples_per_second']), _rate(summary['gpu_demand_samples_per_second'])))
    print('VERDICT ' + summary['verdict'])
    estimate = summary['grad_cache_estimate']
    print('projected 5 epochs: vanilla {:.1f}h, GradCache-{} about {:.1f}h (x{:.2f}, estimate)'.format(
        summary['projected_hours_vanilla'], estimate['effective_batch'], estimate['projected_hours'],
        estimate['time_ratio_to_vanilla']))
    return 0


def _rate(value):
    return 'n/a' if value is None else '{:.1f}'.format(value)


# ----------------------------------------------------------------------------
# 3. drift
# ----------------------------------------------------------------------------

def load_probe(dataset, transform, probe_size):
    """A fixed set of distinct training videos, each with its first caption, center-cropped."""
    videos = list(dataset.train_vids)
    positions = sorted(set(np.linspace(0, len(videos) - 1, min(probe_size, len(videos))).astype(int).tolist()))
    video_ids, captions, clips = [], [], []
    for position in positions:
        video_id = videos[position]
        frames, _ = VideoCapture.load_frames_from_video(
            os.path.join(dataset.videos_dir, video_id + '.mp4'),
            dataset.config.num_frames, dataset.config.video_sample_type)
        video_ids.append(video_id)
        captions.append(dataset.vid2caption[video_id][0])
        clips.append(transform(frames))
    return {'video_id': video_ids, 'text': captions, 'video': torch.stack(clips)}


@torch.no_grad()
def encode_probe(model, probe, chunk_size, device):
    """Current features of the probe set, on the CPU, without disturbing training randomness."""
    state = capture_random_state()
    try:
        with evaluation_mode(model):
            parts = [model.encode(move_batch(chunk, device)) for chunk in split_batch(probe, chunk_size)]
    finally:
        restore_random_state(state)
    return {'text': torch.cat([text for text, _ in parts]).float().cpu(),
            'frames': torch.cat([frames for _, frames in parts]).float().cpu()}


def run_drift(model, loss_fn, optimizer, scheduler, loader, tokenizer, probe, pooling_type, device,
              steps, every, chunk_size):
    """Train normally and, every ``every`` steps, compare the probe features with all older snapshots."""
    model.train()
    snapshots = [(0, encode_probe(model, probe, chunk_size, device))]
    records = []
    losses = []
    iterator = iter(loader)
    for step in range(1, steps + 1):
        data = move_batch(tokenize(next(iterator), tokenizer), device)
        losses.append(float(train_step(model, loss_fn, optimizer, scheduler, data, pooling_type)))
        if step % every == 0:
            current = encode_probe(model, probe, chunk_size, device)
            for old_step, old in snapshots:
                record = {'from_step': old_step, 'to_step': step, 'lag_steps': step - old_step}
                record.update(compare_snapshots(model, old, current, pooling_type, device))
                records.append(record)
            snapshots.append((step, current))
            newest = records[-1]
            print('step {}: lag {} steps -> frame cos-dist {:.2e}, stale logit error video {:.4f} text {:.4f}, loss(mean) {:.4f}'.format(
                step, newest['lag_steps'], newest['frame_cosine_distance'],
                newest['stale_video_logit_error_mean'], newest['stale_text_logit_error_mean'],
                _mean(losses[-every:])), flush=True)
    del iterator
    return records, losses


def drift(args, config, device, output):
    seed_everything(config.seed)
    tokenizer = build_tokenizer()
    loader = DataFactory.get_data_loader(config, split_type='train')
    model, checkpoint = build_model(config, args.checkpoint)
    model = model.to(device)
    loss_fn = CLIPLoss().to(device)
    optimizer = build_optimizer(model, config)

    lr_mode = args.lr_mode
    if lr_mode == 'auto':
        lr_mode = 'checkpoint' if checkpoint is not None and 'optimizer' in checkpoint else 'peak'
    scheduler = None
    if lr_mode == 'checkpoint':
        if checkpoint is None or 'optimizer' not in checkpoint:
            raise SystemExit('--lr-mode checkpoint needs --checkpoint with a saved optimizer state')
        # Restores Adam moments and the learning rates in effect when the checkpoint was saved.
        optimizer.load_state_dict(checkpoint['optimizer'])
    elif lr_mode == 'schedule':
        total = len(loader) * config.num_epochs
        scheduler = get_cosine_schedule_with_warmup(optimizer, int(config.warmup_proportion * total), total)
    del checkpoint
    learning_rates = [group['lr'] for group in optimizer.param_groups]
    if not any(rate > 0 for rate in learning_rates):
        print('WARNING: every learning rate is 0 (end of a cosine schedule?); features will not move. '
              'Use an earlier checkpoint or --lr-mode peak.', flush=True)

    dataset = loader.dataset.dataset
    transform = init_transform_dict(config.input_res)['clip_test']
    probe = load_probe(dataset, transform, args.probe_size)
    probe = tokenize(probe, tokenizer)
    print('probe: {} videos; lr mode {}; clip lr {:g}, pooling lr {:g}'.format(
        len(probe['video_id']), lr_mode, learning_rates[0], learning_rates[1]), flush=True)

    records, losses = run_drift(model, loss_fn, optimizer, scheduler, loader, tokenizer, probe,
                                config.pooling_type, device, args.steps, args.every, config.batch_size)
    lag_rows = summarize_by_lag(records, config.batch_size)
    bank_entries, reason = suggest_bank_size(lag_rows, args.tolerance_ratio)
    report = {
        'steps': args.steps, 'measure_every': args.every, 'probe_videos': probe['video_id'],
        'batch_size': config.batch_size, 'lr_mode': lr_mode, 'learning_rates_at_start': learning_rates,
        'checkpoint': str(args.checkpoint) if args.checkpoint else None,
        'mean_training_loss': _mean(losses),
        'tolerance_ratio': args.tolerance_ratio,
        'suggested_max_bank_entries': bank_entries, 'suggestion_reason': reason,
        'suggestion_rule': 'heuristic: mean stale-logit error <= tolerance_ratio * std of negative logits',
        'by_lag': lag_rows, 'environment': environment_info(device),
    }
    write_csv(output / 'drift_pairs.csv', records)
    write_csv(output / 'drift_by_lag.csv', lag_rows)
    write_json(output / 'drift_report.json', report)

    print('lag_steps bank_entries frame_cos_dist text_cos_dist logit_err_video logit_err_text neg_logit_std hardest_neg_agree')
    for row in lag_rows:
        print('{:9d} {:12d} {:14.3e} {:13.3e} {:15.4f} {:14.4f} {:13.4f} {:17.3f}'.format(
            row['lag_steps'], row['bank_entries'], row['frame_cosine_distance'], row['text_cosine_distance'],
            row['stale_video_logit_error_mean'], row['stale_text_logit_error_mean'],
            row['negative_logit_std'], row['hardest_negative_agreement_stale_video']))
    print('SUGGESTED max bank entries: {} ({}; heuristic, tolerance ratio {:g})'.format(
        bank_entries, reason, args.tolerance_ratio))
    return 0


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------

def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--source-root', type=Path, default=ROOT,
                        help='Folder that holds data/MSRVTT (and a local openai/clip-vit-base-patch32 if used)')
    common.add_argument('--videos-dir', default='data/MSRVTT/vids')
    common.add_argument('--batch-size', type=int, default=4)
    common.add_argument('--num-workers', type=int, default=2)
    common.add_argument('--seed', type=int, default=24)
    common.add_argument('--device', default='cuda')
    common.add_argument('--name', default=None, help='Output folder name; default <check>_<UTC time>')
    common.add_argument('--clip-lr', type=float, default=1e-6)
    common.add_argument('--noclip-lr', type=float, default=3e-5)
    common.add_argument('--num-frames', type=int, default=12)
    common.add_argument('--input-res', type=int, default=224)
    common.add_argument('--embed-dim', type=int, default=512)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)

    check = commands.add_parser('gradcheck', parents=[common], help='GradCache vs full-batch gradient')
    check.add_argument('--chunk-size', type=int, default=2)
    check.add_argument('--num-batches', type=int, default=3, help='Informative batches to compare')
    check.add_argument('--tolerance', type=float, default=1e-2, help='Allowed relative L2 gradient error')
    check.add_argument('--min-loss', type=float, default=1e-4, help='Batches below this loss are not counted')
    check.add_argument('--checkpoint', type=Path, default=None, help='Optional trained checkpoint to test on')

    timing = commands.add_parser('steptime', parents=[common], help='Share of data loading vs GPU work in a step')
    timing.add_argument('--steps', type=int, default=100)
    timing.add_argument('--warmup', type=int, default=10, help='Initial steps left out of the averages')
    timing.add_argument('--decode-videos', type=int, default=40)

    stale = commands.add_parser('drift', parents=[common], help='Feature drift over training steps')
    stale.add_argument('--steps', type=int, default=600)
    stale.add_argument('--every', type=int, default=50)
    stale.add_argument('--probe-size', type=int, default=32)
    stale.add_argument('--checkpoint', type=Path, default=None, help='Start from this training checkpoint')
    stale.add_argument('--lr-mode', choices=['auto', 'peak', 'schedule', 'checkpoint'], default='auto',
                       help='peak: constant --clip-lr/--noclip-lr (worst case). schedule: the real warmup from '
                            'step 0. checkpoint: optimizer state and learning rates stored in --checkpoint. '
                            'auto: checkpoint if one is given, else peak')
    stale.add_argument('--tolerance-ratio', type=float, default=0.1)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error('--batch-size must be positive and --num-workers non-negative')
    if args.command == 'gradcheck' and not 0 < args.chunk_size < args.batch_size:
        parser.error('--chunk-size must be smaller than --batch-size')
    if args.command == 'drift' and (args.every <= 0 or args.steps < args.every):
        parser.error('--steps must be at least --every')
    for name in ('checkpoint',):
        if getattr(args, name, None) is not None:
            setattr(args, name, getattr(args, name).resolve())

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is not available; pass --device cpu only for a functional smoke test')

    name = args.name or '{}_{}'.format(args.command, datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    output = ROOT / 'outputs' / 'first_experiment' / name
    output.mkdir(parents=True, exist_ok=False)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.chdir(args.source_root.resolve())

    config = build_config(args)
    write_json(output / 'arguments.json', {key: str(value) if isinstance(value, Path) else value
                                           for key, value in vars(args).items()})
    code = {'gradcheck': gradcheck, 'steptime': steptime, 'drift': drift}[args.command](args, config, device, output)
    print('results: {}'.format(output))
    return code


if __name__ == '__main__':
    sys.exit(main())
