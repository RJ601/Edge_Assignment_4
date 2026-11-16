"""
benchmark.py
============
Automated benchmark of all 6 student models (3 baseline + 3 distilled).

For every model it records:
  * parameter count (millions)
  * PyTorch checkpoint size and ONNX file size (MB)
  * test accuracy measured with ONNX Runtime on the CPU
  * single-image (batch = 1) CPU latency in PyTorch and in ONNX Runtime (ms, median)

Output:
    output/benchmark.csv

Usage:
    python benchmark.py
"""

import csv
import statistics
import time

import numpy as np
import torch

import config
from data import get_test_loader
from models import count_parameters, load_model
from onnx_inference import OnnxPredictor


def median_latency_ms(run_once) -> float:
    """Time a zero-argument function: warm-up runs, then the median of timed runs (ms)."""
    for _ in range(config.LATENCY_WARMUP_RUNS):
        run_once()
    times = []
    for _ in range(config.LATENCY_TIMED_RUNS):
        start = time.perf_counter()
        run_once()
        times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times)


def benchmark_model(model_name: str, variant: str, test_loader):
    """Return one CSV row (dict) for a model, or None if its files are missing."""
    run_name = f"{variant}_{model_name}"
    ckpt, onnx_file = config.checkpoint_path(run_name), config.onnx_path(run_name)
    if not ckpt.exists() or not onnx_file.exists():
        print(f"[skip] {run_name}: checkpoint or ONNX file missing")
        return None

    # --- PyTorch CPU latency (batch of 1) ---
    model = load_model(model_name, ckpt, torch.device("cpu"))
    sample = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)

    def torch_once():
        with torch.inference_mode():
            model(sample)

    torch_latency = median_latency_ms(torch_once)

    # --- ONNX Runtime CPU latency (batch of 1) and accuracy ---
    predictor = OnnxPredictor(run_name, num_threads=1)   # 1 thread, like PyTorch above
    sample_np = sample.numpy()
    onnx_latency = median_latency_ms(lambda: predictor.run(sample_np))
    accuracy = predictor.evaluate(test_loader)

    row = {
        "model": model_name,
        "variant": variant,
        "params_million": round(count_parameters(model) / 1e6, 3),
        "checkpoint_size_mb": round(ckpt.stat().st_size / 1e6, 2),
        "onnx_size_mb": round(onnx_file.stat().st_size / 1e6, 2),
        "onnx_test_accuracy": round(accuracy, 2),
        "pytorch_cpu_latency_ms": round(torch_latency, 2),
        "onnx_cpu_latency_ms": round(onnx_latency, 2),
    }
    print(row)
    return row


def main() -> None:
    config.ensure_output_dirs()
    torch.set_num_threads(1)   # single-thread timing mimics a small edge CPU
    test_loader = get_test_loader(num_workers=0)

    rows = []
    for name in config.STUDENT_NAMES:
        for variant in ("baseline", "kd"):
            row = benchmark_model(name, variant, test_loader)
            if row:
                rows.append(row)

    if rows:
        with open(config.BENCHMARK_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nBenchmark saved -> {config.BENCHMARK_CSV}")


if __name__ == "__main__":
    main()
