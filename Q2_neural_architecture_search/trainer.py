"""
trainer.py
==========
Automated training and validation of one network on in-memory CIFAR-10 tensors.

The same function is used for
  * the short training of each candidate during the search (evaluated on the validation split), and
  * the long final training of the best architecture (evaluated on the test set every epoch).
"""

import csv
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from data import iterate_batches


def get_device() -> torch.device:
    """Use the GPU if available, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                      batch_size: int = 1000) -> float:
    """Accuracy of `model` on (x, y) in PERCENT (0-100)."""
    model.eval()
    correct = 0
    for images, labels in iterate_batches(x, y, batch_size):
        correct += (model(images).argmax(dim=1) == labels).sum().item()
    return 100.0 * correct / x.size(0)


def train_model(model: nn.Module,
                train_x: torch.Tensor, train_y: torch.Tensor,
                eval_x: torch.Tensor, eval_y: torch.Tensor,
                epochs: int,
                device: torch.device,
                log_path: Optional[str] = None) -> float:
    """
    Train `model` from its current weights and return the final accuracy (%)
    on (eval_x, eval_y).

    Optimiser: AdamW with a cosine learning-rate schedule; mixed precision on GPU.

    If `log_path` is given, the model is evaluated after EVERY epoch and the
    rows (epoch, train_loss, test_accuracy) are written to that CSV file.
    Otherwise it is evaluated only once, after the last epoch (faster).
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                                  weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log_file = open(log_path, "w", newline="") if log_path else None
    writer = csv.writer(log_file) if log_file else None
    if writer:
        writer.writerow(["epoch", "train_loss", "test_accuracy"])

    accuracy = float("nan")
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            loss_sum, seen = 0.0, 0
            for images, labels in iterate_batches(train_x, train_y, config.BATCH_SIZE,
                                                  shuffle=True, augment=True):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    loss = F.cross_entropy(model(images).float(), labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_sum += loss.item() * labels.size(0)
                seen += labels.size(0)
            scheduler.step()

            if writer or epoch == epochs:
                accuracy = evaluate_accuracy(model, eval_x, eval_y)
            if writer:
                writer.writerow([epoch, f"{loss_sum / seen:.4f}", f"{accuracy:.2f}"])
                log_file.flush()
                print(f"epoch {epoch:02d}/{epochs} | train loss {loss_sum / seen:.4f} "
                      f"| test acc {accuracy:.2f}%")
    finally:
        if log_file:
            log_file.close()
    return accuracy
