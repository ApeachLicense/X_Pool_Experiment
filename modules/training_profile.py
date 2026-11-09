"""Durable step measurements and an optional wall-time limit for training."""

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import torch


class TrainingProfile:
    def __init__(self, config, device, dataset_size, initial_global_step=0):
        self.config = config
        self.device = device
        self.dataset_size = dataset_size
        self.started = time.monotonic()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.previous_step = self.started
        self.last_report = self.started
        self.steps = 0
        self.initial_global_step = initial_global_step
        self.examples = 0
        self.loss_sum = 0.0
        self.last_loss = None
        self.stop_reason = None
        self.output_dir = Path(config.model_path)
        session_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        self.session_dir = self.output_dir / 'sessions' / session_id
        self.session_dir.mkdir(parents=True)
        self.csv_path = self.output_dir / 'training_steps.csv'
        if self.csv_path.exists():
            self.csv_path = self.session_dir / 'training_steps.csv'
        self.file = self.csv_path.open("x", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            "elapsed_seconds", "step", "epoch", "batch_index", "batch_size", "loss",
            "step_seconds", "allocated_mib", "reserved_mib", "peak_allocated_mib",
            "peak_reserved_mib", "clip_lr", "noclip_lr",
        ])
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.write_report("running")

    def record_step(self, epoch, batch_index, batch_size, loss, optimizer):
        now = time.monotonic()
        elapsed = now - self.started
        if not math.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at training step {self.steps + 1}")
        self.steps += 1
        self.examples += batch_size
        self.loss_sum += loss
        self.last_loss = loss
        memory = self.memory()
        self.writer.writerow([
            round(elapsed, 6), self.initial_global_step + self.steps, epoch, batch_index, batch_size, loss,
            round(now - self.previous_step, 6),
            memory["allocated_mib"], memory["reserved_mib"],
            memory["peak_allocated_mib"], memory["peak_reserved_mib"],
            optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"],
        ])
        self.file.flush()
        self.previous_step = now
        if self.config.max_train_seconds and elapsed >= self.config.max_train_seconds:
            self.stop_reason = "time_limit"
        elif self.config.max_train_steps and self.steps >= self.config.max_train_steps:
            self.stop_reason = "step_limit"
        if self.stop_reason or now - self.last_report >= 30:
            self.write_report("running")
            self.last_report = now
        return self.stop_reason is not None

    def memory(self):
        if self.device.type != "cuda":
            return dict.fromkeys([
                "allocated_mib", "reserved_mib", "peak_allocated_mib", "peak_reserved_mib"
            ], 0.0)
        return {
            "allocated_mib": round(torch.cuda.memory_allocated(self.device) / 2**20, 3),
            "reserved_mib": round(torch.cuda.memory_reserved(self.device) / 2**20, 3),
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated(self.device) / 2**20, 3),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved(self.device) / 2**20, 3),
        }

    def write_report(self, status, error=None):
        elapsed = time.monotonic() - self.started
        report = {
            "status": status,
            "error": error,
            "started_at_utc": self.started_at,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 3),
            "requested_seconds": self.config.max_train_seconds,
            "requested_steps": self.config.max_train_steps,
            "stop_reason": self.stop_reason,
            "completed_steps": self.steps,
            "initial_global_step": self.initial_global_step,
            "global_step": self.initial_global_step + self.steps,
            "steps_csv": str(self.csv_path),
            "processed_pairs": self.examples,
            "training_pairs_per_epoch": self.dataset_size,
            "batch_size": self.config.batch_size,
            "precision": "float32",
            "device": str(self.device),
            "num_frames": self.config.num_frames,
            "input_resolution": self.config.input_res,
            "last_loss": self.last_loss,
            "mean_loss": self.loss_sum / self.steps if self.steps else None,
            "mean_step_seconds": elapsed / self.steps if self.steps else None,
            "pairs_per_second": self.examples / elapsed if elapsed else None,
            "pytorch_memory": self.memory(),
            "settings": vars(self.config),
        }
        temporary = self.output_dir / "training_profile.json.tmp"
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(self.output_dir / "training_profile.json")
        session_report = self.session_dir / 'training_profile.json.tmp'
        session_report.write_text(json.dumps(report, indent=2), encoding='utf-8')
        session_report.replace(self.session_dir / 'training_profile.json')

    def close(self, status, error=None):
        try:
            self.write_report(status, error)
        finally:
            self.file.close()
