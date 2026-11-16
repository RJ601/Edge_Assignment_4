"""
plot.py
=======
Creates one plot per metric recorded by the pipeline (all saved in output/plots/).

From the architecture search (output/search_history.csv), against the generation number:
    val_accuracy.png     validation accuracy of every trained candidate (+ best per generation)
    latency_ms.png       measured CPU latency of every candidate (+ the limit)
    model_size_mb.png    measured model size of every candidate (+ the limit)
    peak_memory_mb.png   measured peak inference memory of every candidate

From the final training (output/final_training_log.csv), against the epoch:
    train_loss.png       training loss
    test_accuracy.png    test accuracy

In the search plots green dots are feasible candidates and red crosses are
candidates that were rejected for violating a constraint.

Usage:
    python plot.py
"""

import matplotlib
matplotlib.use("Agg")   # file output only; works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


def plot_by_generation(history: pd.DataFrame, column: str, ylabel: str, title: str,
                       file_name: str, limit: float = None, best_line: bool = False) -> None:
    """
    Scatter one recorded metric of every candidate against its generation.

    limit     : draws a horizontal dashed line (a deployment constraint)
    best_line : also draws the best (max) value per generation
    """
    if history[column].isna().all():
        print(f"[skip] {column}: no values recorded")
        return

    # Small deterministic horizontal jitter so dots of one generation do not overlap.
    jitter = np.random.default_rng(0).uniform(-0.2, 0.2, len(history))
    x = history["generation"] + jitter
    feasible = history["feasible"].astype(bool)

    plt.figure(figsize=(8, 5))
    plt.scatter(x[feasible], history.loc[feasible, column], color="tab:green",
                alpha=0.8, label="feasible")
    if (~feasible).any() and column != "val_accuracy":   # rejected ones have no accuracy
        plt.scatter(x[~feasible], history.loc[~feasible, column], color="tab:red",
                    marker="x", label="rejected")
    if best_line:
        best = history[feasible].groupby("generation")[column].max()
        plt.plot(best.index, best.values, color="black", linewidth=2, label="best per generation")
    if limit is not None:
        plt.axhline(limit, color="tab:red", linestyle="--", label=f"limit ({limit})")

    plt.xlabel("Generation (0 = random initial population)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(sorted(history["generation"].unique()))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_file = config.PLOT_DIR / file_name
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"saved {out_file}")


def plot_by_epoch(log: pd.DataFrame, column: str, ylabel: str, title: str, file_name: str) -> None:
    """Line plot of one metric of the final training against the epoch."""
    plt.figure(figsize=(8, 5))
    plt.plot(log["epoch"], log[column], linewidth=2, color="tab:blue")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_file = config.PLOT_DIR / file_name
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"saved {out_file}")


def main() -> None:
    config.ensure_output_dirs()

    if config.SEARCH_HISTORY_CSV.exists():
        history = pd.read_csv(config.SEARCH_HISTORY_CSV)
        plot_by_generation(history, "val_accuracy", "Validation accuracy (%)",
                           "Search: validation accuracy of candidates", "val_accuracy.png",
                           best_line=True)
        plot_by_generation(history, "latency_ms", "CPU latency (ms)",
                           "Search: measured CPU latency of candidates", "latency_ms.png",
                           limit=config.MAX_LATENCY_MS)
        plot_by_generation(history, "model_size_mb", "Model size (MB)",
                           "Search: measured model size of candidates", "model_size_mb.png",
                           limit=config.MAX_MODEL_SIZE_MB)
        plot_by_generation(history, "peak_memory_mb", "Peak inference memory (MB)",
                           "Search: peak inference memory of candidates", "peak_memory_mb.png")
    else:
        print(f"[skip] {config.SEARCH_HISTORY_CSV} not found (run evolution.py)")

    if config.FINAL_LOG_CSV.exists():
        log = pd.read_csv(config.FINAL_LOG_CSV)
        plot_by_epoch(log, "train_loss", "Cross-entropy loss",
                      "Final training: training loss", "train_loss.png")
        plot_by_epoch(log, "test_accuracy", "Test accuracy (%)",
                      "Final training: test accuracy", "test_accuracy.png")
    else:
        print(f"[skip] {config.FINAL_LOG_CSV} not found (run train_best.py)")


if __name__ == "__main__":
    main()
