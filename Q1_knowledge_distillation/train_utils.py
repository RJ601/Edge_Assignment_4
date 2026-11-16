"""
train_utils.py
==============
Training helpers shared by train_teacher.py, train_baseline.py and
train_distill.py.

The key idea: all three scripts differ ONLY in how the loss of one batch is
computed. So `fit()` contains the generic training loop and receives a
`compute_loss(images, labels)` function from the calling script.
"""

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn

import config


def set_seed(seed: int = config.SEED) -> None:
    """Seed every random number generator for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Use the GPU if one is available, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the command-line options shared by all training scripts."""
    parser.add_argument("--data_dir", type=Path, default=config.DATA_DIR,
                        help="folder that contains 'cifar-100-python'")
    parser.add_argument("--download", action="store_true",
                        help="download CIFAR-100 into --data_dir if it is missing")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--no_pretrained", action="store_true",
                        help="train from random initialisation instead of ImageNet weights")
    return parser


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> Tuple[float, float]:
    """
    Evaluate a classifier on a DataLoader.

    Returns:
        (average cross-entropy loss, accuracy in percent)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits.float(), labels).item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


def build_optimizer(parameters, model_name: str):
    """SGD with momentum and weight decay; learning rate depends on the architecture."""
    return torch.optim.SGD(parameters, lr=config.LEARNING_RATES[model_name],
                           momentum=config.MOMENTUM, nesterov=True,
                           weight_decay=config.WEIGHT_DECAY)


def fit(run_name: str,
        model: nn.Module,
        train_loader,
        test_loader,
        compute_loss: Callable,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        device: torch.device) -> None:
    """
    Generic training loop.

    Args:
        run_name     : e.g. 'baseline_mobilenetv2'; names the log and checkpoint files.
        model        : the network being trained (only its weights are saved).
        compute_loss : function(images, labels) -> (loss_to_optimise, hard_label_ce_for_logging)
        optimizer    : optimiser over the model (and any helper modules such as the KD regressor).
        epochs       : number of passes over the training set.

    After every epoch the model is evaluated on the test set and one row
    (epoch, train_loss, test_accuracy) is appended to output/logs/<run_name>.csv.
    The weights after the LAST epoch are saved (no checkpoint picking on the
    test set, so the reported accuracy is not optimistically biased).
    """
    config.ensure_output_dirs()
    model.to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = device.type == "cuda"                      # mixed precision only on GPU
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    with open(config.log_path(run_name), "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["epoch", "train_loss", "test_accuracy"])

        for epoch in range(1, epochs + 1):
            model.train()
            start = time.time()
            loss_sum, seen = 0.0, 0

            for images, labels in train_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    loss, hard_loss = compute_loss(images, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                loss_sum += hard_loss.item() * labels.size(0)
                seen += labels.size(0)

            scheduler.step()
            train_loss = loss_sum / seen
            _, test_acc = evaluate(model, test_loader, device)

            writer.writerow([epoch, f"{train_loss:.4f}", f"{test_acc:.2f}"])
            log_file.flush()   # keep the log valid even if the run is interrupted
            print(f"[{run_name}] epoch {epoch:02d}/{epochs} | train CE {train_loss:.4f} "
                  f"| test acc {test_acc:.2f}% | {time.time() - start:.0f}s")

    torch.save(model.state_dict(), config.checkpoint_path(run_name))
    print(f"[{run_name}] saved -> {config.checkpoint_path(run_name)}")
