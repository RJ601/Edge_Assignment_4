"""
config.py
=========
Central configuration for Question 2 (hardware-aware evolutionary Neural
Architecture Search on CIFAR-10).

Every path, search-space option, deployment constraint and hyperparameter is
defined here so that a value only needs to be changed in ONE place.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"                  # must contain the folder 'cifar-10-batches-py'
OUTPUT_DIR = ROOT_DIR / "output"              # everything a run produces goes here
PLOT_DIR = OUTPUT_DIR / "plots"

SEARCH_HISTORY_CSV = OUTPUT_DIR / "search_history.csv"      # one row per candidate
SEARCH_HISTORY_JSON = OUTPUT_DIR / "search_history.json"    # same data, JSON
BEST_ARCH_JSON = OUTPUT_DIR / "best_architecture.json"      # winner of the search
FINAL_LOG_CSV = OUTPUT_DIR / "final_training_log.csv"       # per-epoch log of the final training
BEST_MODEL_PTH = OUTPUT_DIR / "best_model.pth"              # trained weights of the winner
BEST_MODEL_ONNX = OUTPUT_DIR / "best_model.onnx"            # ONNX export of the winner
BENCHMARK_CSV = OUTPUT_DIR / "benchmark.csv"                # PyTorch vs ONNX Runtime comparison

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
NUM_CLASSES = 10
IMAGE_SIZE = 32            # native CIFAR-10 size; networks resize internally to their own resolution
VAL_SIZE = 5000            # images held out from the 50k training set to score candidates
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
SEED = 42

# ---------------------------------------------------------------------------
# Search space (what the evolutionary search can choose from)
# ---------------------------------------------------------------------------
RESOLUTIONS = [16, 24, 32]                          # variable input resolution
STEM_CHANNELS = [8, 16, 24, 32]                     # filters of the first conv layer
MIN_BLOCKS, MAX_BLOCKS = 3, 8                       # variable convolutional depth
BLOCK_CHANNELS = [16, 24, 32, 48, 64, 96, 128]      # variable number of filters per block
KERNEL_SIZES = [3, 5, 7]                            # different kernel sizes
BLOCK_TYPES = ["standard", "depthwise_separable"]   # normal conv vs depthwise-separable conv
ACTIVATIONS = ["relu", "relu6", "hardswish", "silu"]  # different activation functions
STRIDES = [1, 2]                                    # stride 2 halves the feature-map size
MIN_FEATURE_SIZE = 2                                # feature maps are never down-sampled below this

# ---------------------------------------------------------------------------
# Deployment constraints (candidates violating them are REJECTED, never trained)
# ---------------------------------------------------------------------------
MAX_LATENCY_MS = 10.0       # max CPU inference latency for ONE image (single thread)
MAX_MODEL_SIZE_MB = 2.0     # max model storage size (float32 weights saved to disk)

# ---------------------------------------------------------------------------
# Hardware measurement settings
# ---------------------------------------------------------------------------
LATENCY_WARMUP_RUNS = 10    # untimed runs before measuring
LATENCY_TIMED_RUNS = 50     # timed runs; the median is the latency
LATENCY_NUM_THREADS = 1     # 1 thread mimics a small edge CPU and gives stable numbers
MEMORY_RUNS = 20            # forward passes while peak memory is sampled

# ---------------------------------------------------------------------------
# Evolutionary search
# ---------------------------------------------------------------------------
POPULATION_SIZE = 10        # architectures kept alive
GENERATIONS = 6             # number of evolution rounds after the initial population
TOURNAMENT_SIZE = 3         # parent = best of this many random individuals
MAX_ATTEMPTS_FACTOR = 20    # give up on a generation after this x POPULATION_SIZE tries

# Fitness = validation accuracy - penalties (all values in 0..1 / fractions of the limits).
LATENCY_PENALTY = 0.05      # subtracted: 0.05 * latency / MAX_LATENCY_MS
SIZE_PENALTY = 0.05         # subtracted: 0.05 * size / MAX_MODEL_SIZE_MB

# ---------------------------------------------------------------------------
# Training of candidates (short) and of the final architecture (long)
# ---------------------------------------------------------------------------
BATCH_SIZE = 128
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-4
SEARCH_EPOCHS = 3           # epochs per candidate during the search
FINAL_EPOCHS = 30           # epochs when the best architecture is trained from scratch


def ensure_output_dirs() -> None:
    """Create the output folders if they do not exist yet."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
