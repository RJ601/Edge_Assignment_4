"""
train_baseline.py
=================
Baseline pipeline: train each student with ordinary cross-entropy and NO
knowledge distillation. These are the reference models that show how much
distillation helps.

Output (per student):
    output/checkpoints/baseline_<student>.pth
    output/logs/baseline_<student>.csv

Usage:
    python train_baseline.py [--students mobilenetv2 squeezenet efficientnet_lite0] [--epochs 15]
"""

import argparse

import torch.nn.functional as F

import config
from data import get_dataloaders
from models import build_model, count_parameters
from train_utils import add_common_args, build_optimizer, fit, get_device, set_seed


def train_baseline(name: str, args, device, train_loader, test_loader) -> None:
    """Train one student without distillation."""
    set_seed()   # same seed for each student -> fair comparison with the KD run
    student = build_model(name, pretrained=not args.no_pretrained)
    print(f"\n=== Baseline {name} ({count_parameters(student) / 1e6:.2f} M parameters) ===")

    def compute_loss(images, labels):
        """Hard-label cross-entropy only."""
        loss = F.cross_entropy(student(images).float(), labels)
        return loss, loss.detach()

    optimizer = build_optimizer(student.parameters(), name)
    fit(f"baseline_{name}", student, train_loader, test_loader,
        compute_loss, optimizer, args.epochs, device)


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Train baseline students"))
    parser.add_argument("--epochs", type=int, default=config.EPOCHS_STUDENT)
    parser.add_argument("--students", nargs="+", default=config.STUDENT_NAMES,
                        choices=config.STUDENT_NAMES)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    train_loader, test_loader = get_dataloaders(
        args.data_dir, args.batch_size, args.num_workers, args.download)

    for name in args.students:
        train_baseline(name, args, device, train_loader, test_loader)


if __name__ == "__main__":
    main()
