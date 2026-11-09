import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import json
from itertools import count
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from datasets.data_factory import DataFactory
from modules.loss import CLIPLoss
from modules.optimization import AdamW, get_cosine_schedule_with_warmup
from modules.training_state import training_directory_lock
from trainer.trainer import Trainer


class ToyDataset(Dataset):
    def __len__(self):
        return 19

    def __getitem__(self, index):
        noise = torch.rand(8) + torch.tensor(np.random.random(8), dtype=torch.float32)
        noise += random.random()
        features = torch.arange(8).float() / 8 + index / 19
        return {'video_id': index, 'video': features + noise / 10, 'text': features - noise / 10}


class ToyClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = ToyClip()
        self.head = nn.Linear(8, 8)
        self.dropout = nn.Dropout(0.3)
        self.history = []

    def forward(self, data):
        self.history.append((data['video_id'].tolist(), data['video'].clone(), data['text'].clone()))
        text = self.clip.proj(data['text'])
        video = self.head(self.clip.proj(data['video']))
        return text + self.dropout(text) / 4, video + self.dropout(video) / 4


class CPUTrainer(Trainer):
    def _prepare_device(self):
        return torch.device('cpu')


def make_trainer(output, max_steps, resume=None, workers=2, seconds=0):
    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)
    output.mkdir(parents=True, exist_ok=True)
    config = SimpleNamespace(
        model_path=str(output), dataset_name='TOY', msrvtt_train_file='9k',
        num_frames=12, video_sample_type='uniform', input_res=224,
        arch='toy', huggingface=True, clip_arch='ViT-B/32', embed_dim=8, loss='clip',
        batch_size=4, num_epochs=3, clip_lr=0.005, noclip_lr=0.004,
        weight_decay=0.2, warmup_proportion=0.1, seed=24, num_mha_heads=1,
        transformer_dropout=0.3, num_workers=workers, log_step=100,
        evals_per_epoch=0, eval_window_size=5, save_every=1,
        max_train_steps=max_steps, max_train_seconds=seconds,
        profile_training=True, pooling_type='avg', resume=str(resume) if resume else None,
    )
    loader = DataFactory.training_loader(ToyDataset(), config)
    model = ToyModel()
    optimizer = AdamW([
        {'params': list(model.clip.parameters()), 'lr': config.clip_lr},
        {'params': list(model.head.parameters()), 'lr': config.noclip_lr},
    ], weight_decay=config.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 1, 15)
    trainer = CPUTrainer(model, CLIPLoss(), None, optimizer, config, loader, None, None, scheduler)
    if resume:
        trainer.load_checkpoint(str(resume), resume_training=True)
    return trainer


class ResumeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        root = Path(__file__).resolve().parents[1] / 'outputs'
        root.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix='resume_test_', dir=root)
        self.root = Path(self.directory.name).resolve()
        self.assertTrue(self.root.is_relative_to(root.resolve()))

    def tearDown(self):
        self.directory.cleanup()

    def assert_state_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_prefetch_and_multiple_resumes_match_continuous_training(self):
        full = make_trainer(self.root / 'full', 12)
        full.train()
        first = make_trainer(self.root / 'first', 3)
        first.train()
        path = self.root / 'first/checkpoint-last.pth'
        saved = torch.load(path, map_location='cpu')
        self.assertEqual(saved['data_state']['next_batch'], 3)
        second = make_trainer(self.root / 'second', 4, path, workers=0)
        second.train()
        third = make_trainer(self.root / 'third', 5, self.root / 'second/checkpoint-last.pth')
        third.train()
        self.assertEqual(third.global_step, 12)
        self.assert_state_equal(full.model.state_dict(), third.model.state_dict())
        self.assert_state_equal(full.optimizer.state_dict(), third.optimizer.state_dict())
        self.assert_state_equal(full.lr_scheduler.state_dict(), third.lr_scheduler.state_dict())
        history = first.model.history + second.model.history + third.model.history
        self.assert_state_equal(full.model.history, history)

    def test_epoch_boundary_and_final_short_batch(self):
        full = make_trainer(self.root / 'full', 0, workers=0)
        full.train()
        first = make_trainer(self.root / 'first', 5, workers=0)
        first.train()
        path = self.root / 'first/checkpoint-last.pth'
        self.assertFalse(torch.load(path)['partial_epoch'])
        resumed = make_trainer(self.root / 'second', 0, path, workers=0)
        self.assertEqual(resumed.start_epoch, 2)
        resumed.train()
        self.assertEqual(resumed.global_step, 15)
        self.assert_state_equal(full.model.state_dict(), resumed.model.state_dict())
        self.assert_state_equal(full.optimizer.state_dict(), resumed.optimizer.state_dict())
        finished = make_trainer(self.root / 'finished', 0, self.root / 'second/checkpoint-last.pth', workers=0)
        finished.train()
        self.assertEqual(finished.global_step, 15)
        self.assertEqual(finished.model.history, [])

    def test_time_budget_restarts_and_previous_csv_is_preserved(self):
        output = self.root / 'same'
        first = make_trainer(output, 0, workers=0, seconds=0.000001)
        with patch('modules.training_profile.time.monotonic', side_effect=count(0, 0.05).__next__):
            first.train()
        original_csv = (output / 'training_steps.csv').read_bytes()
        resumed = make_trainer(output, 0, output / 'checkpoint-last.pth', workers=0, seconds=0.000001)
        with patch('modules.training_profile.time.monotonic', side_effect=count(0, 0.05).__next__):
            resumed.train()
        self.assertEqual(resumed.global_step, 2)
        self.assertEqual((output / 'training_steps.csv').read_bytes(), original_csv)
        profile = json.loads((output / 'training_profile.json').read_text())
        self.assertEqual(profile['completed_steps'], 1)
        self.assertEqual(profile['initial_global_step'], 1)
        self.assertEqual(profile['stop_reason'], 'time_limit')
        self.assertNotEqual(Path(profile['steps_csv']), output / 'training_steps.csv')

    def test_legacy_migration_preserves_position_and_scheduler(self):
        first = make_trainer(self.root / 'first', 7, workers=0)
        first.train()
        state = torch.load(self.root / 'first/checkpoint-last.pth')
        state.pop('data_state')
        state.pop('random_state')
        state.pop('training_settings')
        path = self.root / 'legacy.pth'
        torch.save(state, path)
        resumed = make_trainer(self.root / 'resumed', 1, path, workers=0)
        self.assertEqual(resumed.start_epoch, 2)
        self.assertEqual(resumed.train_data_loader.batch_sampler.next_batch, 2)
        self.assertEqual(resumed.lr_scheduler.last_epoch, 7)
        resumed.train()
        self.assertEqual(resumed.global_step, 8)
        self.assertEqual(resumed.legacy_transition_step, 7)

    def test_changed_training_settings_and_concurrent_writer_are_rejected(self):
        first = make_trainer(self.root / 'first', 2, workers=0)
        first.train()
        resumed = make_trainer(self.root / 'second', 1, workers=0)
        resumed.config.batch_size = 2
        with self.assertRaisesRegex(ValueError, 'batch_size'):
            resumed.load_checkpoint(str(self.root / 'first/checkpoint-last.pth'), resume_training=True)
        with training_directory_lock(str(self.root)):
            with self.assertRaises(RuntimeError):
                with training_directory_lock(str(self.root)):
                    self.fail('Concurrent writer acquired the same lock')
        with training_directory_lock(str(self.root)):
            pass


if __name__ == '__main__':
    unittest.main()
