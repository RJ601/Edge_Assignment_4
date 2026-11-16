"""
benchmark.py
============
Automated benchmark of the final NAS model: PyTorch vs ONNX Runtime on the CPU.

Records parameter count, file sizes, test accuracy and single-image CPU
latency (median over repeated runs, 1 thread) for both runtimes, and whether
the model satisfies the deployment constraints from config.py.

Output: output/benchmark.csv

Usage:
    python benchmark.py
"""

import csv
import statistics
import time

import numpy as np
import torch

import config
from data import CifarData
from export_onnx import load_trained_model
from hardware import check_constraints, measure_latency_ms
from model_builder import count_parameters
from onnx_inference import OnnxPredictor
from trainer import evaluate_accuracy


def onnx_latency_ms(predictor: OnnxPredictor) -> float:
    """Median ONNX Runtime latency (ms) for one 32x32 image, same protocol as hardware.py."""
    sample = np.random.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).astype(np.float32)
    for _ in range(config.LATENCY_WARMUP_RUNS):
        predictor.run(sample)
    times = []
    for _ in range(config.LATENCY_TIMED_RUNS):
        start = time.perf_counter()
        predictor.run(sample)
        times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times)


def main() -> None:
    config.ensure_output_dirs()
    model = load_trained_model()
    data = CifarData()                     # CPU tensors; the test split is used for accuracy
    predictor = OnnxPredictor(num_threads=config.LATENCY_NUM_THREADS)

    pytorch_latency = measure_latency_ms(model)
    checkpoint_mb = config.BEST_MODEL_PTH.stat().st_size / 1e6
    meets, _ = check_constraints(pytorch_latency, checkpoint_mb)

    row = {
        "model": "nas_best",
        "params_million": round(count_parameters(model) / 1e6, 4),
        "checkpoint_size_mb": round(checkpoint_mb, 3),
        "onnx_size_mb": round(config.BEST_MODEL_ONNX.stat().st_size / 1e6, 3),
        "pytorch_test_accuracy": round(evaluate_accuracy(model, data.test_x, data.test_y), 2),
        "onnx_test_accuracy": round(predictor.evaluate(data.test_x, data.test_y), 2),
        "pytorch_cpu_latency_ms": round(pytorch_latency, 3),
        "onnx_cpu_latency_ms": round(onnx_latency_ms(predictor), 3),
        "meets_constraints": meets,
    }
    with open(config.BENCHMARK_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(row)
    print(f"Benchmark saved -> {config.BENCHMARK_CSV}")


if __name__ == "__main__":
    main()
