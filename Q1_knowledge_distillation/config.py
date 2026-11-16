"""
config.py
=========
Central configuration for Question 1 (Knowledge Distillation on CIFAR-100).

Every path and every hyperparameter used anywhere in this folder is defined
here, so that all scripts stay consistent and a value only needs to be
changed in ONE place.

Nothing in this file runs any code besides creating path objects, so it is safe
to import from every other script.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Folder that contains this file (works no matter where the scripts are run from).
ROOT_DIR = Path(__file__).resolve().parent

# Where the CIFAR-100 dataset lives. torchvision expects the extracted folder
# "cifar-100-python" to be INSIDE this directory.
DATA_DIR = ROOT_DIR / "data"

# Every file produced by a run is written below this single folder.
OUTPUT_DIR = ROOT_DIR / "output"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"   # trained model weights (.pth)
LOG_DIR = OUTPUT_DIR / "logs"                 # per-epoch metric CSV files
ONNX_DIR = OUTPUT_DIR / "onnx"                # exported ONNX models
PLOT_DIR = OUTPUT_DIR / "plots"               # PNG figures
BENCHMARK_CSV = OUTPUT_DIR / "benchmark.csv"  # final benchmarking table

# ---------------------------------------------------------------------------
# Dataset / input pipeline
# ---------------------------------------------------------------------------
NUM_CLASSES = 100        # CIFAR-100 has 100 classes
IMAGE_SIZE = 64          # 32x32 CIFAR images are up-sampled to 64x64 before the network
BATCH_SIZE = 128
NUM_WORKERS = 2          # data-loading worker processes
SEED = 42                # random seed for reproducibility

# Normalisation statistics. Pretrained ImageNet weights expect ImageNet statistics.
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
USE_PRETRAINED = True    # start every network from ImageNet weights (much faster convergence)
STUDENT_NAMES = ["mobilenetv2", "squeezenet", "efficientnet_lite0"]
TEACHER_NAME = "resnet50"

# ---------------------------------------------------------------------------
# Optimisation (identical recipe for teacher, baselines and distilled students)
# ---------------------------------------------------------------------------
EPOCHS_TEACHER = 15
EPOCHS_STUDENT = 15
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# SGD learning rate per architecture. SqueezeNet has no batch-norm, so it
# gets a smaller learning rate to avoid divergence.
LEARNING_RATES = {
    "resnet50": 0.02,
    "mobilenetv2": 0.02,
    "squeezenet": 0.01,
    "efficientnet_lite0": 0.02,
}

# ---------------------------------------------------------------------------
# Knowledge-distillation hyperparameters
# ---------------------------------------------------------------------------
TEMPERATURE = 4.0        # softmax temperature T used for the soft targets
ALPHA = 0.7              # weight of the soft (teacher) loss; (1 - ALPHA) weights the hard-label loss
FEATURE_WEIGHT = 1.0     # weight of the feature-distillation (hint) loss

# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------
LATENCY_WARMUP_RUNS = 20   # untimed runs to warm up caches
LATENCY_TIMED_RUNS = 100   # timed runs; the median is reported
ONNX_OPSET = 17            # ONNX operator-set version used for export


def ensure_output_dirs() -> None:
    """Create every output sub-folder if it does not exist yet."""
    for folder in (CHECKPOINT_DIR, LOG_DIR, ONNX_DIR, PLOT_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def checkpoint_path(run_name: str) -> Path:
    """Return the checkpoint file path for a run, e.g. 'kd_mobilenetv2'."""
    return CHECKPOINT_DIR / f"{run_name}.pth"


def log_path(run_name: str) -> Path:
    """Return the metric-log CSV path for a run, e.g. 'baseline_squeezenet'."""
    return LOG_DIR / f"{run_name}.csv"


def onnx_path(run_name: str) -> Path:
    """Return the ONNX file path for a run, e.g. 'kd_efficientnet_lite0'."""
    return ONNX_DIR / f"{run_name}.onnx"
