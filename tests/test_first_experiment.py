import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import csv
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

import first_experiment as fe
from model.clip_transformer import CLIPTransformer
from modules.feature_drift import PAIR_METRICS, compare_snapshots, suggest_bank_size, summarize_by_lag
from modules.grad_cache import grad_cache_backward, move_batch, split_batch
from modules.loss import CLIPLoss


EMBED, FRAMES, RES, VOCAB = 16, 3, 32, 100
POOLING = 'transformer'


def tiny_clip():
    """A randomly initialised CLIP with the real architecture, small enough for the CPU."""
    from transformers import CLIPConfig, CLIPModel
    config = CLIPConfig(
        text_config_dict=dict(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                              num_attention_heads=4, max_position_embeddings=16),
        vision_config_dict=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                num_attention_heads=4, image_size=RES, patch_size=16),
        projection_dim=EMBED)
    clip = CLIPModel(config)
    clip.logit_scale.data.fill_(float(np.log(20)))
    return clip


def make_model(seed=0):
    torch.manual_seed(seed)
    config = SimpleNamespace(huggingface=True, clip_arch='ViT-B/32', embed_dim=EMBED, num_mha_heads=1,
                             transformer_dropout=0.3, input_res=RES, num_frames=FRAMES)
    with patch('transformers.CLIPModel.from_pretrained', return_value=tiny_clip()):
        model = CLIPTransformer(config)
    # The pooling starts as an identity map; perturb it so every parameter matters.
    with torch.no_grad():
        for parameter in model.pool_frames.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))
    return model


def make_batch(size, seed=0, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    length = 6
    input_ids = torch.randint(1, VOCAB - 1, (size, length), generator=generator)
    attention_mask = torch.ones(size, length, dtype=torch.long)
    for row in range(size):
        end = 2 + row % (length - 2)          # captions of different lengths
        input_ids[row, end] = VOCAB - 1       # end-of-text is the largest id, as in CLIP
        input_ids[row, end + 1:] = 0
        attention_mask[row, end + 1:] = 0
    return {
        'video_id': ['video{}'.format(index) for index in range(size)],
        'video': torch.randn(size, FRAMES, 3, RES, RES, generator=generator).to(dtype),
        'text': {'input_ids': input_ids, 'attention_mask': attention_mask},
    }


def seeder(seed):
    return lambda: torch.manual_seed(seed)


def full_gradient(model, data, seed):
    model.zero_grad()
    loss = fe.reference_backward(model, CLIPLoss(), data, POOLING, seeder(seed))
    return loss, fe.collect_gradients(model)


class FakeTokenizer:
    def __call__(self, texts, return_tensors='pt', padding=True, truncation=True):
        rows = [[1 + (sum(map(ord, word)) % (VOCAB - 2)) for word in text.split()][:10] + [VOCAB - 1]
                for text in texts]
        width = max(len(row) for row in rows)
        input_ids = torch.zeros(len(rows), width, dtype=torch.long)
        attention_mask = torch.zeros(len(rows), width, dtype=torch.long)
        for index, row in enumerate(rows):
            input_ids[index, :len(row)] = torch.tensor(row)
            attention_mask[index, :len(row)] = 1
        return {'input_ids': input_ids, 'attention_mask': attention_mask}


class GradCacheTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_forward_is_unchanged_by_the_encode_pool_split(self):
        model = make_model().eval()
        data = make_batch(4)
        with torch.no_grad():
            text, frames, pooled = model(data, return_all_frames=True)
            text_only, pooled_only = model(data)
            split_text, split_frames = model.encode(data)
            split_pooled = model.pool(split_text, split_frames)
        self.assertEqual(tuple(frames.shape), (4, FRAMES, EMBED))
        self.assertEqual(tuple(pooled.shape), (4, 4, EMBED))
        for left, right in [(text, split_text), (frames, split_frames), (pooled, split_pooled),
                            (text, text_only), (pooled, pooled_only)]:
            self.assertTrue(torch.equal(left, right))
        self.assertTrue(fe.forward_split_check(model, data)['bitwise_equal'])

    def test_split_and_move_keep_rows_aligned(self):
        data = make_batch(5)
        chunks = split_batch(data, 2)
        self.assertEqual([chunk['video'].shape[0] for chunk in chunks], [2, 2, 1])
        self.assertEqual(sum((chunk['video_id'] for chunk in chunks), []), data['video_id'])
        self.assertTrue(torch.equal(torch.cat([c['text']['input_ids'] for c in chunks]), data['text']['input_ids']))
        self.assertTrue(torch.equal(torch.cat([c['video'] for c in chunks]), data['video']))
        moved = move_batch(chunks[0], torch.device('cpu'))
        self.assertEqual(moved['video_id'], chunks[0]['video_id'])
        with self.assertRaises(ValueError):
            split_batch(data, 0)

    def test_grad_cache_reproduces_the_full_batch_gradient_in_double_precision(self):
        model = make_model().double().train()
        data = make_batch(4, dtype=torch.float64)
        loss, reference = full_gradient(model, data, seed=7)
        groups = {fe.gradient_group(name) for name in reference}
        self.assertEqual(groups, {'clip_text', 'clip_vision', 'logit_scale', 'pooling'})
        for chunk_size in (1, 2, 3):
            model.zero_grad()
            output = grad_cache_backward(model, CLIPLoss(), data, chunk_size, POOLING, before_pool=seeder(7))
            comparison = fe.compare_gradients(reference, fe.collect_gradients(model))
            self.assertAlmostEqual(float(output['loss']), float(loss), places=12)
            self.assertEqual(output['feature_recompute_max_abs_diff'], 0.0)
            for group, values in comparison.items():
                self.assertGreater(values['reference_norm'], 0, group)
                self.assertLess(values['relative_l2_error'], 1e-9, (chunk_size, group))
        # Keeping the batch on the CPU and moving chunks one at a time gives the same result.
        model.zero_grad()
        grad_cache_backward(model, CLIPLoss(), data, 2, POOLING, device=torch.device('cpu'), before_pool=seeder(7))
        self.assertLess(fe.compare_gradients(reference, fe.collect_gradients(model))['all']['relative_l2_error'], 1e-9)

    def test_the_comparison_can_fail(self):
        model = make_model().double().train()
        data = make_batch(4, dtype=torch.float64)
        _, reference = full_gradient(model, data, seed=7)

        # A different pooling dropout mask is a different gradient: the seed hook matters.
        _, other_mask = full_gradient(model, data, seed=8)
        self.assertGreater(fe.compare_gradients(reference, other_mask)['all']['relative_l2_error'], 1e-3)

        # Per-chunk losses (plain accumulation) do not see cross-chunk negatives.
        model.zero_grad()
        fe.accumulation_backward(model, CLIPLoss(), data, 2, POOLING, seeder(7))
        accumulation = fe.compare_gradients(reference, fe.collect_gradients(model))['all']
        self.assertGreater(accumulation['relative_l2_error'], 0.05)

    def test_run_gradcheck_passes_on_a_deterministic_encoder_and_fails_on_a_noisy_one(self):
        batches = [make_batch(4, seed=index) for index in range(3)]
        model = make_model().train()
        report = fe.run_gradcheck(model, CLIPLoss(), batches, 2, POOLING, torch.device('cpu'), num_informative=3)
        self.assertTrue(report['passed'])
        self.assertEqual(report['informative_batches'], 3)
        self.assertTrue(report['forward_equals_encode_then_pool']['bitwise_equal'])
        for batch in report['batches']:
            self.assertLess(batch['grad_cache_vs_full']['all']['relative_l2_error'], 1e-3)
            self.assertGreater(batch['accumulation_vs_full']['all']['relative_l2_error'], 0.05)
        self.assertTrue(all(parameter.grad is None or not parameter.grad.any() for parameter in model.parameters()))
        json.dumps(report)

        noisy = make_model().train()
        encode = noisy.encode
        noisy.encode = lambda data: tuple(F.dropout(value, 0.2, training=True) for value in encode(data))
        report = fe.run_gradcheck(noisy, CLIPLoss(), batches, 2, POOLING, torch.device('cpu'), num_informative=3)
        self.assertFalse(report['passed'])
        self.assertGreater(report['batches'][0]['feature_recompute_max_abs_diff'], 0)

    def test_batches_with_a_vanishing_loss_are_not_counted(self):
        model = make_model().train()
        batches = [make_batch(4, seed=index) for index in range(4)]
        report = fe.run_gradcheck(model, CLIPLoss(), batches, 2, POOLING, torch.device('cpu'),
                                  min_loss=1e9, num_informative=2, max_batches=3)
        self.assertFalse(report['passed'])
        self.assertEqual(len(report['batches']), 3)
        self.assertEqual(report['informative_batches'], 0)


class DriftTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = make_model().train()
        self.probe = make_batch(6, seed=3)
        self.device = torch.device('cpu')

    def snapshot(self):
        return fe.encode_probe(self.model, self.probe, 4, self.device)

    def test_identical_snapshots_have_no_error_and_probing_leaves_the_model_untouched(self):
        state = torch.get_rng_state()
        first = self.snapshot()
        second = self.snapshot()
        self.assertTrue(torch.equal(first['text'], second['text']))
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        self.assertTrue(self.model.training)
        result = compare_snapshots(self.model, first, second, POOLING, self.device)
        self.assertEqual(result['stale_video_logit_error_max'], 0.0)
        self.assertEqual(result['stale_text_logit_error_max'], 0.0)
        self.assertLess(result['frame_cosine_distance'], 1e-6)
        self.assertEqual(result['hardest_negative_agreement_stale_video'], 1.0)
        self.assertGreater(result['negative_logit_std'], 0)

    def test_error_grows_with_the_size_of_the_parameter_change(self):
        old = self.snapshot()
        errors = []
        for scale in (1e-3, 1e-1):
            model = make_model().train()
            with torch.no_grad():
                for parameter in model.clip.parameters():
                    parameter.add_(scale * torch.randn_like(parameter))
            new = fe.encode_probe(model, self.probe, 4, self.device)
            result = compare_snapshots(model, old, new, POOLING, self.device)
            self.assertGreater(result['frame_cosine_distance'], 0)
            errors.append((result['stale_video_logit_error_mean'], result['stale_text_logit_error_mean']))
        self.assertGreater(errors[1][0], errors[0][0])
        self.assertGreater(errors[1][1], errors[0][1])

    def test_only_the_stale_side_contributes_to_each_direction(self):
        old = self.snapshot()
        new = {'text': old['text'].clone(), 'frames': old['frames'] + 0.05 * torch.randn_like(old['frames'])}
        result = compare_snapshots(self.model, old, new, POOLING, self.device)
        self.assertGreater(result['stale_video_logit_error_mean'], 0)
        self.assertEqual(result['stale_text_logit_error_max'], 0.0)

    def test_lag_summary_and_bank_size_rule(self):
        def record(start, stop, error):
            item = {name: 0.0 for name in PAIR_METRICS}
            item.update(from_step=start, to_step=stop, lag_steps=stop - start, negative_logit_std=2.0,
                        stale_video_logit_error_mean=error, stale_text_logit_error_mean=error / 2)
            return item
        records = [record(0, 50, 0.05), record(50, 100, 0.07), record(0, 100, 0.15), record(0, 150, 0.9)]
        rows = summarize_by_lag(records, batch_size=4)
        self.assertEqual([row['lag_steps'] for row in rows], [50, 100, 150])
        self.assertEqual([row['bank_entries'] for row in rows], [200, 400, 600])
        self.assertEqual(rows[0]['pairs'], 2)
        self.assertAlmostEqual(rows[0]['stale_video_logit_error_mean'], 0.06)
        self.assertEqual(suggest_bank_size(rows, 0.1)[0], 400)      # 0.15 <= 0.2 < 0.9
        self.assertEqual(suggest_bank_size(rows, 0.01)[0], 0)
        self.assertEqual(suggest_bank_size(rows, 1.0)[0], 600)
        self.assertEqual(suggest_bank_size([], 0.1)[0], 0)


class StepTimeTests(unittest.TestCase):
    @staticmethod
    def rows(data_wait, gpu_step):
        rows = []
        for step in range(6):
            warmup = step < 2
            factor = 10 if warmup else 1    # slow warmup steps must not affect the averages
            rows.append({'step': step + 1, 'warmup': warmup, 'loss': 0.1, 'data_wait': data_wait * factor,
                         'tokenize': 0.01, 'to_device': 0.01, 'gpu_step': gpu_step,
                         'total': data_wait * factor + 0.02 + gpu_step})
        return rows

    def test_gpu_bound_summary(self):
        decode = [{'decode': 0.03, 'transform': 0.01}] * 5
        summary = fe.summarize_steptime(self.rows(0.0, 0.30), [0.08, 0.12], decode, batch_size=4, num_workers=2,
                                        pairs_per_epoch=180000)
        self.assertEqual(summary['measured_steps'], 4)
        self.assertAlmostEqual(summary['mean_seconds']['total'], 0.32)
        self.assertAlmostEqual(sum(summary['share_of_step'].values()), 1.0)
        self.assertTrue(summary['verdict'].startswith('gpu-bound'))
        self.assertAlmostEqual(summary['loader_supply_samples_per_second'], 50.0)
        self.assertAlmostEqual(summary['gpu_demand_samples_per_second'], 12.5)
        self.assertAlmostEqual(summary['projected_hours_vanilla'], 45000 * 5 * 0.32 / 3600)
        estimate = summary['grad_cache_estimate']
        self.assertEqual(estimate['chunks_per_update'], 8)
        self.assertAlmostEqual(estimate['time_ratio_to_vanilla'], (0.32 + 0.10) / 0.32)
        json.dumps(summary)

    def test_data_bound_summary(self):
        decode = [{'decode': 0.5, 'transform': 0.1}] * 3
        summary = fe.summarize_steptime(self.rows(0.9, 0.30), [0.1], decode, batch_size=4, num_workers=2,
                                        pairs_per_epoch=180000)
        self.assertTrue(summary['verdict'].startswith('data-bound'))
        self.assertGreater(summary['speedup_if_data_wait_removed'], 3)


class CommandLineTests(unittest.TestCase):
    """The three commands end to end on the CPU: real loader and video decoding, tiny random CLIP."""

    def setUp(self):
        torch.set_num_threads(1)
        self.cwd = os.getcwd()
        self.rng = (torch.get_rng_state(), np.random.get_state())
        self.outputs = Path(__file__).resolve().parents[1] / 'outputs'
        self.outputs.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix='first_experiment_test_', dir=self.outputs)
        self.source = Path(self.directory.name).resolve()
        self.names = []
        try:
            self.make_dataset()
        except Exception:
            self.tearDown()
            raise

    def tearDown(self):
        os.chdir(self.cwd)
        torch.use_deterministic_algorithms(False)
        torch.set_rng_state(self.rng[0])
        np.random.set_state(self.rng[1])
        for name in self.names:
            shutil.rmtree(self.outputs / 'first_experiment' / name, ignore_errors=True)
        self.directory.cleanup()

    def make_dataset(self):
        import cv2
        folder = self.source / 'data' / 'MSRVTT'
        videos = folder / 'vids'
        videos.mkdir(parents=True)
        ids = ['video{}'.format(index) for index in range(10)]
        rng = np.random.RandomState(0)
        for video_id in ids:
            writer = cv2.VideoWriter(str(videos / (video_id + '.mp4')), cv2.VideoWriter_fourcc(*'mp4v'), 10, (64, 48))
            if not writer.isOpened():
                self.skipTest('This OpenCV build cannot write mp4 test videos')
            for _ in range(15):
                writer.write(rng.randint(0, 255, (48, 64, 3), dtype=np.uint8))
            writer.release()
        if not cv2.VideoCapture(str(videos / 'video0.mp4')).isOpened():
            self.skipTest('This OpenCV build cannot read the generated test videos')
        with (folder / 'MSRVTT_train.9k.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['video_id'])
            writer.writerows([[video_id] for video_id in ids])
        with (folder / 'MSRVTT_JSFUSION_test.csv').open('w', encoding='utf-8', newline='') as handle:
            csv.writer(handle).writerows([['key', 'vid_key', 'video_id', 'sentence'],
                                          ['ret0', 'msr0', 'video0', 'a test caption']])
        words = ['red', 'blue', 'dog', 'car', 'runs', 'sings', 'cooks', 'game', 'person', 'talks']
        sentences = [{'caption': 'a {} {} {}'.format(words[i], words[(i + j + 1) % 10], words[(3 * i + j) % 10]),
                      'video_id': video_id, 'sen_id': 2 * i + j}
                     for i, video_id in enumerate(ids) for j in range(2)]
        (folder / 'MSRVTT_data.json').write_text(json.dumps({'sentences': sentences}), encoding='utf-8')

    def run_command(self, command, *options):
        name = 'unittest_{}_{}'.format(command, os.getpid())
        self.names.append(name)
        arguments = [command, '--source-root', str(self.source), '--device', 'cpu', '--num-workers', '0',
                     '--input-res', str(RES), '--embed-dim', str(EMBED), '--num-frames', str(FRAMES),
                     '--name', name] + list(options)
        with patch('transformers.CLIPModel.from_pretrained', return_value=tiny_clip()), \
                patch.object(fe, 'build_tokenizer', return_value=FakeTokenizer()):
            torch.manual_seed(0)
            code = fe.main(arguments)
        return code, self.outputs / 'first_experiment' / name

    def test_gradcheck_command(self):
        code, output = self.run_command('gradcheck', '--chunk-size', '2', '--num-batches', '2')
        report = json.loads((output / 'gradcheck_report.json').read_text(encoding='utf-8'))
        self.assertEqual(code, 0)
        self.assertTrue(report['passed'])
        self.assertEqual(report['informative_batches'], 2)
        self.assertEqual(len(report['batches'][0]['video_ids']), 4)
        self.assertTrue((output / 'arguments.json').exists())

    def test_steptime_command(self):
        code, output = self.run_command('steptime', '--steps', '3', '--warmup', '1', '--decode-videos', '4')
        report = json.loads((output / 'steptime_report.json').read_text(encoding='utf-8'))
        self.assertEqual(code, 0)
        self.assertEqual(report['measured_steps'], 3)
        self.assertGreater(report['mean_seconds']['gpu_step'], 0)
        self.assertGreater(report['decode_seconds_per_video'], 0)
        self.assertGreater(report['no_grad_encode_seconds_per_chunk'], 0)
        self.assertEqual(report['decoded_videos'], 4)
        with (output / 'step_times.csv').open(encoding='utf-8', newline='') as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 4)

    def test_drift_command(self):
        code, output = self.run_command('drift', '--steps', '4', '--every', '2', '--probe-size', '6',
                                        '--clip-lr', '1e-3', '--noclip-lr', '1e-3')
        report = json.loads((output / 'drift_report.json').read_text(encoding='utf-8'))
        self.assertEqual(code, 0)
        self.assertEqual(report['lr_mode'], 'peak')
        self.assertEqual(len(report['probe_videos']), 6)
        self.assertEqual([row['lag_steps'] for row in report['by_lag']], [2, 4])
        self.assertEqual([row['bank_entries'] for row in report['by_lag']], [8, 16])
        self.assertEqual([row['pairs'] for row in report['by_lag']], [2, 1])
        self.assertGreater(report['by_lag'][0]['frame_cosine_distance'], 0)
        self.assertGreater(report['by_lag'][1]['stale_video_logit_error_mean'], 0)
        with (output / 'drift_pairs.csv').open(encoding='utf-8', newline='') as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 3)

    def test_drift_from_a_checkpoint_uses_its_optimizer_state_and_learning_rates(self):
        model = make_model(seed=5).train()
        optimizer = fe.build_optimizer(model, SimpleNamespace(clip_lr=5e-4, noclip_lr=7e-4, weight_decay=0.2))
        fe.train_step(model, CLIPLoss(), optimizer, None, make_batch(4), POOLING)
        checkpoint = self.source / 'checkpoint-epoch1.pth'
        torch.save({'epoch': 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint)
        code, output = self.run_command('drift', '--steps', '2', '--every', '1', '--probe-size', '4',
                                        '--checkpoint', str(checkpoint))
        report = json.loads((output / 'drift_report.json').read_text(encoding='utf-8'))
        self.assertEqual(code, 0)
        self.assertEqual(report['lr_mode'], 'checkpoint')
        self.assertEqual(report['learning_rates_at_start'], [5e-4, 7e-4])
        self.assertGreater(report['by_lag'][0]['frame_cosine_distance'], 0)


if __name__ == '__main__':
    unittest.main()
