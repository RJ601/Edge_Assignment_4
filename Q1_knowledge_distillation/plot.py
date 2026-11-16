"""
plot.py
=======
Creates the plots of the metrics recorded during training. ONE plot per metric:

    output/plots/train_loss.png      - hard-label cross-entropy on the training set, per epoch
    output/plots/test_accuracy.png   - test accuracy (%), per epoch

Every run found in output/logs/ (teacher, baselines, distilled students) is drawn
as one line. Colour = architecture, dashed = baseline, solid = distilled.

Usage:
    python plot.py
"""

import matplotlib
matplotlib.use("Agg")   # draw to files only; works without a display (laptop, Colab, server)
import matplotlib.pyplot as plt
import pandas as pd

import config

# One fixed colour per architecture so the comparison is easy to read.
COLORS = {"resnet50": "black", "mobilenetv2": "tab:blue",
          "squeezenet": "tab:orange", "efficientnet_lite0": "tab:green"}
LINESTYLES = {"teacher": "-", "baseline": "--", "kd": "-"}


def load_logs() -> dict:
    """Read every CSV in output/logs/ into {run_name: DataFrame}."""
    logs = {}
    for csv_file in sorted(config.LOG_DIR.glob("*.csv")):
        logs[csv_file.stem] = pd.read_csv(csv_file)
    return logs


def plot_metric(logs: dict, column: str, ylabel: str, title: str, file_name: str) -> None:
    """Draw one metric (a column of every log) against the epoch number."""
    plt.figure(figsize=(8, 5))
    for run_name, frame in logs.items():
        variant, model_name = run_name.split("_", 1)    # e.g. 'kd', 'mobilenetv2'
        plt.plot(frame["epoch"], frame[column], label=run_name,
                 color=COLORS.get(model_name, "gray"),
                 linestyle=LINESTYLES.get(variant, "-"), linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_file = config.PLOT_DIR / file_name
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"saved {out_file}")


def main() -> None:
    config.ensure_output_dirs()
    logs = load_logs()
    if not logs:
        raise SystemExit(f"No log files found in {config.LOG_DIR}. Train the models first.")
    plot_metric(logs, "train_loss", "Cross-entropy (training set)",
                "Training loss per epoch", "train_loss.png")
    plot_metric(logs, "test_accuracy", "Accuracy (%)",
                "Test accuracy per epoch", "test_accuracy.png")


if __name__ == "__main__":
    main()
