"""Reproducible shuffle and augmentation independent of worker prefetch."""

import math
import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class SeededDataset(Dataset):
    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.seed = seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        epoch, index = key
        seed = (self.seed + (epoch - 1) * len(self) + index) % (2**32)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        try:
            random.seed(seed)
            np.random.seed(seed)
            # Seed only the CPU generator. Dataset access must not reset CUDA dropout.
            torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
            return self.dataset[index]
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)


class ResumableBatchSampler(Sampler):
    def __init__(self, dataset_size, batch_size, seed):
        if dataset_size <= 0 or batch_size <= 0:
            raise ValueError('Dataset size and batch size must be positive')
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed if seed >= 0 else int(torch.empty((), dtype=torch.int64).random_().item())
        self.epoch = None
        self.next_batch = 0
        self.order = None
        self.set_epoch(1)

    @property
    def total_batches(self):
        return math.ceil(self.dataset_size / self.batch_size)

    def set_epoch(self, epoch):
        if self.epoch != epoch:
            generator = torch.Generator().manual_seed(self.seed + epoch - 1)
            self.order = torch.randperm(self.dataset_size, generator=generator)
            self.epoch = epoch
            self.next_batch = 0

    def __iter__(self):
        # Prefetch can exhaust this iterator before the optimizer finishes a batch.
        # Only mark_completed() advances the checkpoint cursor.
        start = self.next_batch * self.batch_size
        for offset in range(start, self.dataset_size, self.batch_size):
            indices = self.order[offset:offset + self.batch_size].tolist()
            yield [(self.epoch, index) for index in indices]

    def __len__(self):
        return self.total_batches - self.next_batch

    def mark_completed(self, batch_index):
        if batch_index != self.next_batch:
            raise RuntimeError('Training batches must be committed in order')
        self.next_batch += 1

    def state_dict(self):
        return {
            'dataset_size': self.dataset_size, 'batch_size': self.batch_size,
            'seed': self.seed, 'epoch': self.epoch, 'next_batch': self.next_batch,
            'order': self.order.clone(),
        }

    def load_state_dict(self, state):
        for name in ('dataset_size', 'batch_size'):
            if state[name] != getattr(self, name):
                raise ValueError(f'Cannot resume with a different {name}')
        if not 0 <= state['next_batch'] <= self.total_batches:
            raise ValueError('Checkpoint batch position is outside the epoch')
        order = state['order'].cpu()
        if order.dtype != torch.int64 or order.shape != (self.dataset_size,):
            raise ValueError('Checkpoint shuffle has an invalid shape or dtype')
        if not torch.equal(torch.sort(order).values, torch.arange(self.dataset_size)):
            raise ValueError('Checkpoint shuffle is not a permutation of the dataset')
        self.seed = state['seed']
        self.epoch = state['epoch']
        self.next_batch = state['next_batch']
        self.order = order.clone()
