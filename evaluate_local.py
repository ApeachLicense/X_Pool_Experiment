"""Run the official MSR-VTT evaluation and record wall time and GPU memory."""

import json
from importlib.metadata import version
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time


def main():
    root = Path(__file__).resolve().parent
    output_dir = root / "outputs" / "msrvtt9k"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-u", "test.py",
        "--exp_name=msrvtt9k",
        "--videos_dir=data/MSRVTT/vids",
        "--batch_size=4",
        "--num_workers=2",
        "--huggingface",
        "--load_epoch=-1",
        "--dataset_name=MSRVTT",
        "--msrvtt_train_file=9k",
        "--no_tensorboard",
    ]
    samples = []
    finished = threading.Event()

    def monitor_gpu():
        while not finished.is_set():
            result = subprocess.run(
                ["nvidia-smi", "--id=0", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True,
            )
            samples.append(int(result.stdout.strip()))
            finished.wait(1)

    monitor = threading.Thread(target=monitor_gpu, daemon=True)
    monitor.start()
    start = time.perf_counter()
    environment = dict(os.environ, PYTHONUTF8="1")
    environment["HF_HOME"] = str(root / ".cache" / "huggingface")
    environment["TRANSFORMERS_CACHE"] = str(root / ".cache" / "transformers")
    lines = []
    try:
        with (output_dir / "evaluation.log").open("w", encoding="utf-8") as log:
            with subprocess.Popen(
                command, cwd=root, env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            ) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                    lines.append(line)
                return_code = process.wait()
    finally:
        elapsed = time.perf_counter() - start
        finished.set()
        monitor.join(timeout=5)

    text = "".join(lines)
    metrics = {
        name: float(value)
        for name, value in re.findall(
            r"(?:^|\n)\s*(R@1|R@5|R@10|MedR|MeanR):\s*([0-9.]+)", text
        )
    }
    report = {
        "command": command,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "python_version": sys.version,
        "packages": {
            name: version(name)
            for name in ("torch", "torchvision", "transformers", "opencv-python", "numpy")
        },
        "exit_code": return_code,
        "wall_time_seconds": round(elapsed, 3),
        "gpu_memory_sample_interval_seconds": 1,
        "gpu_device_memory_before_mib": samples[0] if samples else None,
        "gpu_device_memory_peak_mib": max(samples) if samples else None,
        "gpu_memory_note": "nvidia-smi device total includes desktop and other processes",
        "metrics": metrics,
    }
    (output_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return return_code


if __name__ == "__main__":
    sys.exit(main())
