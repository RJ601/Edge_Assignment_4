"""
hardware.py
===========
Hardware-aware measurements and deployment constraints.

For a model on the CPU this file measures
  * latency          : median time of one single-image forward pass (ms)
  * model size       : size of the saved weights file (MB)
  * peak memory      : largest increase of the process memory (RSS) while running
                       inference (MB). Needs the `psutil` package; returns NaN if
                       it is not installed ("where supported").
and decides whether the model satisfies the deployment constraints
(config.MAX_LATENCY_MS and config.MAX_MODEL_SIZE_MB).

Latency and size are measured on real runs / real files - not estimated from
FLOPs or parameter counts.
"""

import math
import os
import statistics
import tempfile
import threading
import time
from typing import Tuple

import torch
import torch.nn as nn

import config
from model_builder import count_parameters


def _example_input() -> torch.Tensor:
    """One CIFAR-sized image (batch of 1)."""
    return torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)


def measure_model_size_mb(model: nn.Module) -> float:
    """Save the weights to a temporary file and return its size in MB."""
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "model.pth")
        torch.save(model.state_dict(), path)
        return os.path.getsize(path) / 1e6


def measure_latency_ms(model: nn.Module) -> float:
    """
    Median CPU latency of a batch-of-1 forward pass in milliseconds.

    Runs with config.LATENCY_NUM_THREADS threads (restored afterwards), after
    config.LATENCY_WARMUP_RUNS untimed warm-up runs.
    """
    model.eval()
    x = _example_input()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.LATENCY_NUM_THREADS)
    try:
        with torch.inference_mode():
            for _ in range(config.LATENCY_WARMUP_RUNS):
                model(x)
            times = []
            for _ in range(config.LATENCY_TIMED_RUNS):
                start = time.perf_counter()
                model(x)
                times.append((time.perf_counter() - start) * 1000.0)
    finally:
        torch.set_num_threads(previous_threads)
    return statistics.median(times)


def measure_peak_memory_mb(model: nn.Module) -> float:
    """
    Peak extra memory (MB) used by the process while running inference.

    A background thread samples the process resident memory every millisecond;
    the result is (highest sample - memory before inference).
    Returns NaN when psutil is unavailable.
    """
    try:
        import psutil
    except ImportError:
        return math.nan

    model.eval()
    x = _example_input()
    process = psutil.Process(os.getpid())
    baseline = process.memory_info().rss
    peak = [baseline]
    stop = threading.Event()

    def sampler() -> None:
        while not stop.is_set():
            peak[0] = max(peak[0], process.memory_info().rss)
            time.sleep(0.001)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    try:
        with torch.inference_mode():
            for _ in range(config.MEMORY_RUNS):
                model(x)
    finally:
        stop.set()
        thread.join()
    peak[0] = max(peak[0], process.memory_info().rss)
    return (peak[0] - baseline) / 1e6


def profile_model(model: nn.Module) -> dict:
    """Measure everything for a CPU model; returns a dict of the measurements."""
    model = model.cpu().eval()
    return {
        "params_million": count_parameters(model) / 1e6,
        "peak_memory_mb": measure_peak_memory_mb(model),   # first: before caches warm up
        "latency_ms": measure_latency_ms(model),
        "model_size_mb": measure_model_size_mb(model),
    }


def check_constraints(latency_ms: float, model_size_mb: float) -> Tuple[bool, str]:
    """
    Check the deployment constraints.

    Returns (is_feasible, reason). `reason` is '' for feasible models,
    otherwise a text that explains which limit was violated.
    """
    reasons = []
    if latency_ms > config.MAX_LATENCY_MS:
        reasons.append(f"latency {latency_ms:.1f} ms > {config.MAX_LATENCY_MS} ms")
    if model_size_mb > config.MAX_MODEL_SIZE_MB:
        reasons.append(f"size {model_size_mb:.2f} MB > {config.MAX_MODEL_SIZE_MB} MB")
    return (len(reasons) == 0), "; ".join(reasons)
