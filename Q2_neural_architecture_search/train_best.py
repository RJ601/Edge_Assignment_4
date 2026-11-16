"""
train_best.py
=============
Train the best architecture found by the search FROM SCRATCH (fresh random
weights) on the full 50,000 training images, for FINAL_EPOCHS epochs, and
evaluate it on the 10,000 test images after every epoch.

Needs: output/best_architecture.json (created by evolution.py)
Output:
    output/best_model.pth            trained weights (last epoch)
    output/final_training_log.csv    epoch, train_loss, test_accuracy

Usage:
    python train_best.py [--epochs 30] [--data_dir ./data] [--download]
"""

import argparse
import json
from pathlib import Path

import torch

import config
from data import CifarData
from model_builder import build_model, count_parameters
from search_space import Architecture
from trainer import get_device, train_model


def load_best_architecture() -> Architecture:
    """Read output/best_architecture.json and rebuild the Architecture."""
    if not config.BEST_ARCH_JSON.exists():
        raise FileNotFoundError(f"{config.BEST_ARCH_JSON} not found. Run evolution.py first.")
    with open(config.BEST_ARCH_JSON) as f:
        return Architecture.from_dict(json.load(f)["architecture"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the best NAS architecture from scratch")
    parser.add_argument("--epochs", type=int, default=config.FINAL_EPOCHS)
    parser.add_argument("--data_dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    config.ensure_output_dirs()
    torch.manual_seed(config.SEED)
    device = get_device()
    print(f"Device: {device}")

    arch = load_best_architecture()
    model = build_model(arch)                      # new random weights = "from scratch"
    print(f"Architecture: resolution {arch.resolution}, {len(arch.blocks)} blocks, "
          f"{count_parameters(model) / 1e6:.3f} M parameters")

    data = CifarData(args.data_dir, args.download).to(device)
    full_x, full_y = data.full_train()
    final_acc = train_model(model, full_x, full_y, data.test_x, data.test_y,
                            args.epochs, device, log_path=str(config.FINAL_LOG_CSV))

    torch.save(model.cpu().state_dict(), config.BEST_MODEL_PTH)
    print(f"Final test accuracy: {final_acc:.2f}% | saved -> {config.BEST_MODEL_PTH}")


if __name__ == "__main__":
    main()
