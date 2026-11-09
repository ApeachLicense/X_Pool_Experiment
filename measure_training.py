"""Probe small batches, then monitor a bounded run through the official train.py."""

import argparse
import csv
import ctypes
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def training_command(name, batch_size, seconds, steps, resume=None):
    command = [
        sys.executable, "-u", "train.py", f"--exp_name={name}",
        "--videos_dir=data/MSRVTT/vids", "--dataset_name=MSRVTT", "--msrvtt_train_file=9k",
        f"--batch_size={batch_size}", "--num_workers=2", "--huggingface", "--no_tensorboard",
        "--noclip_lr=3e-5", "--transformer_dropout=0.3", "--evals_per_epoch=0",
        "--log_step=100", "--profile_training",
        f"--max_train_seconds={seconds}", f"--max_train_steps={steps}",
    ]
    if resume is not None:
        command.append(f'--resume={resume}')
    return command


def run_training(name, batch_size, seconds, steps, resume=None):
    output = ROOT / "outputs" / name
    output.mkdir(parents=True, exist_ok=True)
    session_id = datetime.now(timezone.utc).strftime('monitor_%Y%m%d_%H%M%S_%f')
    session_output = output / 'sessions' / session_id
    session_output.mkdir(parents=True)
    log_path = output / 'training.log'
    gpu_path = output / 'gpu_samples.csv'
    if log_path.exists():
        log_path = session_output / 'training.log'
    if gpu_path.exists():
        gpu_path = session_output / 'gpu_samples.csv'
    command = training_command(name, batch_size, seconds, steps, resume)
    environment = dict(os.environ, PYTHONUTF8="1", TOKENIZERS_PARALLELISM="false")
    environment["HF_HOME"] = str(ROOT / ".cache" / "huggingface")
    environment["TRANSFORMERS_CACHE"] = str(ROOT / ".cache" / "transformers")
    start = time.monotonic()
    summary = {
        "status": "starting", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command, "gpu_sample_interval_seconds": 5,
        "gpu_memory_note": "Device memory includes the desktop and other processes",
        "precision": "float32", "batch_size": batch_size,
        "gpu_peak_used_mib": 0.0, "gpu_max_temperature_c": 0.0,
        "gpu_samples": 0, "gpu_memory_before_mib": None,
        "telemetry_errors": [],
        "training_log": str(log_path), "gpu_samples_csv": str(gpu_path),
        "requested_seconds": seconds, "resume_checkpoint": str(resume) if resume else None,
    }
    with log_path.open("x", encoding="utf-8") as log:
        with gpu_path.open("x", newline="", encoding="utf-8") as gpu_file:
            writer = csv.writer(gpu_file)
            writer.writerow([
                "utc_time", "elapsed_seconds", "device_used_mib", "device_total_mib",
                "gpu_utilization_percent", "temperature_c", "power_w",
            ])
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            summary.update(status="running", training_pid=process.pid)
            write_json(output / "monitor_report.json", summary)
            last_progress = start
            while process.poll() is None:
                elapsed = time.monotonic() - start
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--id=0", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, check=True, timeout=15,
                    )
                    values = [float(value.strip()) for value in next(csv.reader(result.stdout.splitlines()))]
                    writer.writerow([datetime.now(timezone.utc).isoformat(), round(elapsed, 3), *values])
                    gpu_file.flush()
                    if summary["gpu_samples"] == 0:
                        summary["gpu_memory_before_mib"] = values[0]
                    summary["gpu_samples"] += 1
                    summary["gpu_peak_used_mib"] = max(summary["gpu_peak_used_mib"], values[0])
                    summary["gpu_max_temperature_c"] = max(summary["gpu_max_temperature_c"], values[3])
                except (subprocess.SubprocessError, ValueError) as error:
                    summary["telemetry_errors"].append(str(error))
                summary["wall_time_seconds"] = round(elapsed, 3)
                summary["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                write_json(output / "monitor_report.json", summary)
                write_json(session_output / "monitor_report.json", summary)
                if time.monotonic() - last_progress >= 300:
                    print(f"{name} running {elapsed/3600:.2f} h, peak GPU {summary['gpu_peak_used_mib']:.0f} MiB", flush=True)
                    last_progress = time.monotonic()
                if elapsed > seconds + 600:
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=True)
                    summary["supervisor_timeout"] = True
                    break
                time.sleep(5)
            return_code = process.wait()
    summary.update(
        status="finished" if return_code == 0 else "failed", exit_code=return_code,
        ended_at_utc=datetime.now(timezone.utc).isoformat(),
        wall_time_seconds=round(time.monotonic() - start, 3),
    )
    profile = output / "training_profile.json"
    if profile.is_file():
        saved_profile = json.loads(profile.read_text(encoding="utf-8"))
        if saved_profile['updated_at_utc'] >= summary['started_at_utc']:
            summary["training_profile"] = saved_profile
    write_json(output / "monitor_report.json", summary)
    write_json(session_output / "monitor_report.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=6)
    args = parser.parse_args()
    if args.hours <= 0 or not math.isfinite(args.hours):
        parser.error("--hours must be finite and positive")
    with (ROOT / "data/MSRVTT/MSRVTT_train.9k.csv").open(encoding="utf-8") as handle:
        train_ids = {row["video_id"] for row in csv.DictReader(handle)}
    with (ROOT / "data/MSRVTT/MSRVTT_JSFUSION_test.csv").open(encoding="utf-8") as handle:
        test_ids = {row["video_id"] for row in csv.DictReader(handle)}
    if len(train_ids) != 9000 or train_ids & test_ids:
        raise RuntimeError("The official training/evaluation split verification failed")
    missing = [name for name in train_ids if not (ROOT / "data/MSRVTT/vids" / f"{name}.mp4").is_file()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} training videos. Run prepare_msrvtt_train.py first.")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    workflow_path = ROOT / "outputs" / f"training_measurement_{timestamp}.json"
    workflow = {
        "status": "probing", "requested_hours": args.hours, "probes": [],
        "dataset": "MSR-VTT 9K training split", "evaluation_during_training": False,
        "gpu": "NVIDIA GeForce RTX 3060 Ti 8GB",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    # Hold a system power request only while this process is alive.
    sleep_request = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    workflow["windows_system_sleep_request_active"] = bool(sleep_request)
    write_json(workflow_path, workflow)
    try:
        chosen_batch = None
        for batch_size in (2, 4):
            name = f"vram_probe_b{batch_size}_{timestamp}"
            print(f"Probing batch {batch_size} for up to 20 optimizer updates", flush=True)
            probe = run_training(name, batch_size, 120, 20)
            workflow["probes"].append(probe)
            write_json(workflow_path, workflow)
            if probe["exit_code"] == 0 and probe["training_profile"]["completed_steps"] >= 2:
                reserved = probe["training_profile"]["pytorch_memory"]["peak_reserved_mib"]
                conservative_peak = reserved + (probe["gpu_memory_before_mib"] or 0)
                if probe["gpu_samples"] and max(probe["gpu_peak_used_mib"], conservative_peak) <= 7168:
                    chosen_batch = batch_size
            elif batch_size == 2:
                raise RuntimeError(f"Batch 2 probe failed. Inspect outputs/{name}/training.log")
        if chosen_batch is None:
            raise RuntimeError("No tested batch left at least 1 GiB of device memory headroom")
        name = f"msrvtt9k_train_vram_{timestamp}"
        workflow.update(status="running", selected_batch_size=chosen_batch, training_output=str(ROOT / "outputs" / name))
        write_json(workflow_path, workflow)
        print(f"Starting {args.hours:g}-hour training measurement with batch {chosen_batch}", flush=True)
        result = run_training(name, chosen_batch, args.hours * 3600, 0)
        workflow["result"] = result
        workflow["status"] = "finished" if result["exit_code"] == 0 else "failed"
        write_json(workflow_path, workflow)
        print(json.dumps(workflow, indent=2), flush=True)
        return result["exit_code"]
    except BaseException as error:
        workflow.update(status="failed", error=repr(error))
        write_json(workflow_path, workflow)
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    sys.exit(main())
