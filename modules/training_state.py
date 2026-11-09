"""Checkpoint random state and protect a training directory from concurrent writers."""

from contextlib import contextmanager
import os
import random

import numpy as np
import torch


RESUME_SETTINGS = (
    'dataset_name', 'msrvtt_train_file', 'num_frames', 'video_sample_type', 'input_res',
    'arch', 'huggingface', 'clip_arch', 'embed_dim', 'loss', 'batch_size', 'num_epochs',
    'clip_lr', 'noclip_lr', 'weight_decay', 'warmup_proportion', 'seed',
    'num_mha_heads', 'transformer_dropout',
)


def capture_random_state():
    return {
        'python': random.getstate(), 'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_random_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA devices must match the checkpoint to restore dropout randomness')
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


@contextmanager
def training_directory_lock(directory, filename='.training.lock'):
    path = os.path.join(directory, filename)
    with open(path, 'a+b') as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f'Another training session is using {directory}') from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
