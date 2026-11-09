import torch
import os
import json
import shutil
from abc import abstractmethod
from config.base_config import Config
from modules.training_profile import TrainingProfile
from modules.training_state import RESUME_SETTINGS, capture_random_state, restore_random_state


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, loss, metrics, optimizer, config: Config, writer=None):
        self.config = config
        # setup GPU device if available, move model into configured device
        self.device = self._prepare_device()
        self.model = model.to(self.device)

        self.loss = loss.to(self.device)
        self.metrics = metrics
        self.optimizer = optimizer
        self.start_epoch = 1
        self.global_step = 0

        self.num_epochs = config.num_epochs
        self.writer = writer
        self.checkpoint_dir = config.model_path

        self.log_step = config.log_step
        self.evals_per_epoch = config.evals_per_epoch
        self.training_profile = None
        self.legacy_transition_step = None

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def _valid_epoch_step(self, epoch, step, num_steps):
        """
        Training logic for a step in an epoch
        :param epoch: Current epoch number
               step: Current step in epoch
               num_steps: Number of steps in epoch
        """
        raise NotImplementedError


    def train(self):
        if self.config.profile_training or self.config.max_train_seconds or self.config.max_train_steps:
            self.training_profile = TrainingProfile(
                self.config, self.device, len(self.train_data_loader.dataset), self.global_step
            )
        try:
            for epoch in range(self.start_epoch, self.num_epochs + 1):
                result = self._train_epoch(epoch)
                stopped = self.training_profile is not None and self.training_profile.stop_reason is not None
                if epoch % self.config.save_every == 0 or stopped:
                    self._save_checkpoint(epoch, save_best=False)
                if stopped:
                    print(f"Training stopped at step {self.global_step}: {self.training_profile.stop_reason}")
                    break
        except BaseException as error:
            if self.training_profile is not None:
                self.training_profile.close('failed', repr(error))
            raise
        else:
            if self.training_profile is not None:
                self.training_profile.close('finished')

    def validate(self):
        self._valid_epoch_step(0,0,0)

    def _prepare_device(self):
        """
        setup GPU device if available, move model into configured device
        """
        use_gpu = torch.cuda.is_available()
        device = torch.device('cuda:0' if use_gpu else 'cpu')
        return device

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param save_best: if True, save checkpoint to 'model_best.pth'
        """

        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'random_state': capture_random_state(),
            'training_settings': {
                name: getattr(self.config, name) for name in RESUME_SETTINGS
                if hasattr(self.config, name)
            },
            'legacy_transition_step': self.legacy_transition_step,
        }
        sampler = self.train_data_loader.batch_sampler
        state['data_state'] = sampler.state_dict()
        state['partial_epoch'] = sampler.next_batch < sampler.total_batches
        scheduler = getattr(self, 'lr_scheduler', None)
        state['lr_scheduler'] = scheduler.state_dict() if scheduler is not None else None
        if hasattr(self, 'window_metric'):
            state['metrics'] = {
                'best': self.best, 'best_window': self.best_window,
                'window': {name: list(values) for name, values in self.window_metric.items()},
            }

        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path + '.tmp')
            os.replace(best_path + '.tmp', best_path)
            print("Saving current best: model_best.pth ...")
            filename = best_path
        else:
            filename = os.path.join(self.checkpoint_dir, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename + '.tmp')
            os.replace(filename + '.tmp', filename)
            print("Saving checkpoint: {} ...".format(filename))
        latest = os.path.join(self.checkpoint_dir, 'checkpoint-last.pth')
        shutil.copyfile(filename, latest + '.tmp')
        os.replace(latest + '.tmp', latest)


    def load_checkpoint(self, model_name, resume_training=False):
        """
        Load from saved checkpoints
        :param model_name: Model name experiment to be loaded
        """
        checkpoint_path = model_name if os.path.isfile(model_name) else os.path.join(self.checkpoint_dir, model_name)
        print("Loading checkpoint: {} ...".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if resume_training:
            self._validate_resume_settings(checkpoint, checkpoint_path)
        self.start_epoch = checkpoint['epoch'] + 1 if 'epoch' in checkpoint else 1
        state_dict = checkpoint['state_dict']
        
        self.model.load_state_dict(state_dict)

        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        if resume_training:
            self._restore_training_progress(checkpoint)

        print("Checkpoint loaded")

    def _validate_resume_settings(self, checkpoint, checkpoint_path):
        saved = checkpoint.get('training_settings')
        if saved is None:
            profile_path = os.path.join(os.path.dirname(checkpoint_path), 'training_profile.json')
            if os.path.isfile(profile_path):
                with open(profile_path, encoding='utf-8') as handle:
                    saved = json.load(handle).get('settings', {})
        for name in RESUME_SETTINGS:
            if saved and name in saved and getattr(self.config, name) != saved[name]:
                raise ValueError(f'Resume requires {name}={saved[name]!r}, got {getattr(self.config, name)!r}')

    def _restore_training_progress(self, checkpoint):
        sampler = self.train_data_loader.batch_sampler
        steps_per_epoch = sampler.total_batches
        optimizer_steps = {int(value['step']) for value in checkpoint['optimizer']['state'].values()}
        if 'global_step' in checkpoint:
            self.global_step = checkpoint['global_step']
        elif len(optimizer_steps) == 1:
            self.global_step = optimizer_steps.pop()
        else:
            raise ValueError('The checkpoint has no unambiguous completed optimizer step count')
        if not 0 <= self.global_step <= steps_per_epoch * self.num_epochs:
            raise ValueError('Checkpoint progress exceeds the configured training schedule')
        self.start_epoch = self.global_step // steps_per_epoch + 1
        batch_offset = self.global_step % steps_per_epoch

        scheduler = getattr(self, 'lr_scheduler', None)
        if scheduler is not None:
            if checkpoint.get('lr_scheduler') is not None:
                scheduler.load_state_dict(checkpoint['lr_scheduler'])
            else:
                # Older official checkpoints predate scheduler snapshots.
                scheduler.last_epoch = self.global_step
                scheduler._step_count = self.global_step + 1
                scheduler._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        elif checkpoint.get('lr_scheduler') is not None:
            raise ValueError('This checkpoint requires a learning-rate scheduler')

        if 'data_state' in checkpoint and 'random_state' in checkpoint:
            sampler.load_state_dict(checkpoint['data_state'])
            completed = (sampler.epoch - 1) * steps_per_epoch + sampler.next_batch
            if completed != self.global_step:
                raise ValueError('Checkpoint shuffle cursor and optimizer progress disagree')
            self.train_data_loader.dataset.seed = sampler.seed
            self.legacy_transition_step = checkpoint.get('legacy_transition_step')
            restore_random_state(checkpoint['random_state'])
        else:
            sampler.set_epoch(self.start_epoch)
            sampler.next_batch = batch_offset
            self.legacy_transition_step = self.global_step
            print('Legacy checkpoint has no shuffle/RNG snapshot. '
                  'Seeded shuffle and augmentation start from this continuation.')

        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            self.best = metrics['best']
            self.best_window = metrics['best_window']
            for name, values in metrics['window'].items():
                self.window_metric[name].extend(values)
        print(f'Resuming at global step {self.global_step}, epoch {self.start_epoch}, batch {batch_offset}')

