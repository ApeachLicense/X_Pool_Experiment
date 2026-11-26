"""CPU checks for chunked training and checkpoint selection."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.utils.data import Dataset

from datasets.data_factory import DataFactory
from modules.grad_cache import accumulation_backward, grad_cache_backward, split_batch
from modules.loss import CLIPLoss
from modules.metrics import sim_matrix_training
from modules.optimization import AdamW
from trainer.trainer import Trainer


class TinyModel(nn.Module):
    """Deterministic encoder, stochastic text-conditioned pooling."""
    def __init__(self):
        super().__init__()
        self.clip = nn.Module()
        self.clip.text_proj = nn.Linear(8, 8)
        self.clip.frame_proj = nn.Linear(8, 8)
        self.clip.logit_scale = nn.Parameter(torch.tensor(1.5))
        self.pool_frames = nn.Sequential(nn.Linear(8, 8), nn.Tanh(), nn.Dropout(0.2))

    def encode(self, batch):
        text = batch['text']
        if isinstance(text, dict):
            text = text['input_ids']
        return self.clip.text_proj(text.float()), self.clip.frame_proj(batch['video'].float())

    def pool(self, text, frames):
        # (videos, texts, channels): all-pair conditioned pooling
        return self.pool_frames(frames.mean(dim=1)[:, None, :] + 0.1 * text[None, :, :])

    def forward(self, data):
        text, frames = self.encode(data)
        return text, self.pool(text, frames)


class TinyDataset(Dataset):
    def __len__(self):
        return 13

    def __getitem__(self, index):
        x = torch.sin(torch.arange(8).float() * 0.7 + index)
        frames = torch.stack([x + 0.1 * i for i in range(3)])
        return {'text': x, 'video': frames, 'video_id': index}


class CPUTrainer(Trainer):
    def _prepare_device(self):
        return torch.device('cpu')


def make_batch(count=6):
    rows = [TinyDataset()[i] for i in range(count)]
    return {
        'text': {'input_ids': torch.stack([row['text'] for row in rows])},
        'video': torch.stack([row['video'] for row in rows]),
        'video_id': [row['video_id'] for row in rows],
    }


def make_trainer(directory, chunk_size, accumulation_chunk_size=0):
    config = SimpleNamespace(
        model_path=str(directory), dataset_name='TOY', msrvtt_train_file='9k',
        num_frames=3, video_sample_type='uniform', input_res=224,
        arch='clip_transformer', huggingface=True, clip_arch='ViT-B/32',
        embed_dim=8, loss='clip', batch_size=6, grad_cache_chunk_size=chunk_size,
        accumulation_chunk_size=accumulation_chunk_size,
        grad_cache_check_recompute=True, eval_batch_size=2, num_epochs=2,
        clip_lr=0.0005, noclip_lr=0.0005, weight_decay=0.2,
        warmup_proportion=0.1, seed=24, num_mha_heads=1,
        transformer_dropout=0.2, num_workers=0, log_step=100,
        evals_per_epoch=0, eval_window_size=5, save_every=1,
        max_train_steps=1, max_train_seconds=0,
        profile_training=True, pooling_type='transformer', resume=None,
    )
    loader = DataFactory.training_loader(TinyDataset(), config)
    model = TinyModel()
    optimizer = AdamW([
        {'params': list(model.clip.parameters()), 'lr': config.clip_lr},
        {'params': list(model.pool_frames.parameters()), 'lr': config.noclip_lr},
    ], weight_decay=config.weight_decay)
    return CPUTrainer(model, CLIPLoss(), None, optimizer, config,
                      loader, None, None, None, None)


def load_checkpoint(path):
    # PyTorch >=2.6 changed torch.load's default weights_only value.
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


class ChunkedTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(100)

    def test_full_batch_and_gradcache_match_with_pooling_dropout(self):
        full = TinyModel().train()
        cached = copy.deepcopy(full).train()
        batch = make_batch(6)
        loss_fn = CLIPLoss()
        torch.manual_seed(2026)
        text, pooled = full(batch)
        expected_loss = loss_fn(sim_matrix_training(text, pooled, 'transformer'),
                                full.clip.logit_scale)
        expected_loss.backward()
        expected = {name: parameter.grad.detach().clone()
                    for name, parameter in full.named_parameters()}

        torch.manual_seed(2026)
        report = grad_cache_backward(cached, loss_fn, batch, 2, 'transformer',
                                     device=torch.device('cpu'), check_recompute=True)
        self.assertEqual(report['num_chunks'], 3)
        self.assertEqual(report['feature_recompute_max_abs_diff'], 0.0)
        self.assertAlmostEqual(float(report['loss']), float(expected_loss), places=6)
        for name, parameter in cached.named_parameters():
            rel = (parameter.grad - expected[name]).norm() / expected[name].norm().clamp_min(1e-8)
            self.assertLess(float(rel), 1e-5, name)

    def test_trainer_updates_once_and_saves_effective_batch_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = make_trainer(Path(folder), chunk_size=2)
            before = copy.deepcopy(trainer.model.state_dict())
            trainer.train()
            self.assertEqual(trainer.global_step, 1)
            self.assertTrue(any(not torch.equal(before[key], value)
                                for key, value in trainer.model.state_dict().items()))
            checkpoint = load_checkpoint(Path(folder) / 'checkpoint-last.pth')
            self.assertEqual(checkpoint['global_step'], 1)
            self.assertEqual(checkpoint['data_state']['next_batch'], 1)
            self.assertEqual(checkpoint['training_settings']['batch_size'], 6)
            self.assertEqual(checkpoint['training_settings']['grad_cache_chunk_size'], 2)
            self.assertEqual(checkpoint['training_settings']['eval_batch_size'], 2)

            changed = make_trainer(Path(folder), chunk_size=0)
            with self.assertRaisesRegex(ValueError, 'grad_cache_chunk_size'):
                changed.load_checkpoint(str(Path(folder) / 'checkpoint-last.pth'),
                                        resume_training=True)

    def test_short_final_chunk_is_supported(self):
        model = TinyModel().train()
        torch.manual_seed(3)
        report = grad_cache_backward(model, CLIPLoss(), make_batch(5), 2,
                                     'transformer', device=torch.device('cpu'))
        self.assertEqual(report['num_chunks'], 3)
        self.assertTrue(torch.isfinite(report['loss']))

    def test_accumulation_matches_weighted_independent_losses_with_short_tail(self):
        model = TinyModel().train()
        reference = copy.deepcopy(model)
        data = make_batch(5)
        loss_fn = CLIPLoss()
        torch.manual_seed(42)
        losses = []
        for chunk in split_batch(data, 3):
            text, pooled = reference(chunk)
            scores = sim_matrix_training(text, pooled, 'transformer')
            losses.append(loss_fn(scores, reference.clip.logit_scale))
        expected = (3 * losses[0] + 2 * losses[1]) / 5
        expected.backward()

        torch.manual_seed(42)
        result = accumulation_backward(model, loss_fn, data, 3, 'transformer', device='cpu')
        torch.testing.assert_allclose(result, expected.detach())
        for actual, desired in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_allclose(actual.grad, desired.grad, atol=1e-6, rtol=1e-5)

    def test_accumulation_updates_once_and_rejects_incompatible_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            trainer = make_trainer(directory, chunk_size=0, accumulation_chunk_size=2)
            trainer.lr_scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=1)
            before_step = trainer.lr_scheduler.last_epoch
            trainer.train()
            self.assertEqual(trainer.lr_scheduler.last_epoch, before_step + 1)
            path = directory / 'checkpoint-last.pth'
            checkpoint = load_checkpoint(path)
            self.assertEqual(checkpoint['global_step'], 1)
            self.assertEqual(checkpoint['data_state']['next_batch'], 1)
            self.assertEqual({int(value['step']) for value in checkpoint['optimizer']['state'].values()}, {1})
            self.assertEqual(checkpoint['training_settings']['accumulation_chunk_size'], 2)
            with self.assertRaisesRegex(ValueError, 'accumulation_chunk_size'):
                make_trainer(directory, chunk_size=0).load_checkpoint(str(path), resume_training=True)
            del checkpoint['training_settings']['accumulation_chunk_size']
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, 'accumulation_chunk_size'):
                trainer.load_checkpoint(str(path), resume_training=True)

    def test_session_starts_fresh_then_selects_its_own_checkpoint(self):
        from run_training_session import select_checkpoint

        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / '.session.lock').touch()
            self.assertIsNone(select_checkpoint(directory, None))
            (directory / 'training.log').touch()
            with self.assertRaisesRegex(ValueError, 'no checkpoint-last'):
                select_checkpoint(directory, None)
            latest = directory / 'checkpoint-last.pth'
            latest.touch()
            self.assertEqual(select_checkpoint(directory, None), latest)

    def test_legacy_checkpoint_fallback_prefers_latest(self):
        from run_training_session import select_checkpoint

        with tempfile.TemporaryDirectory() as folder:
            legacy = Path(folder) / 'checkpoint-epoch2.pth'
            legacy.touch()
            output = Path(folder) / 'run'
            output.mkdir()
            self.assertEqual(select_checkpoint(output, None, legacy_checkpoint=legacy), legacy)
            self.assertIsNone(select_checkpoint(output, None))
            latest = output / 'checkpoint-last.pth'
            latest.touch()
            self.assertEqual(select_checkpoint(output, None, legacy_checkpoint=legacy), latest)


if __name__ == '__main__':
    unittest.main()
