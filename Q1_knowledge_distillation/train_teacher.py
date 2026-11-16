"""
train_teacher.py
================
Train the ResNet-50 teacher on CIFAR-100 with ordinary cross-entropy.

Output:
    output/checkpoints/teacher_resnet50.pth
    output/logs/teacher_resnet50.csv

Usage:
    python train_teacher.py [--epochs 15] [--data_dir ./data] [--download]
"""

import argparse

import torch.nn.functional as F

import config
from data import get_dataloaders
from models import build_model, count_parameters
from train_utils import add_common_args, build_optimizer, fit, get_device, set_seed


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Train the ResNet-50 teacher"))
    parser.add_argument("--epochs", type=int, default=config.EPOCHS_TEACHER)
    args = parser.parse_args()

    set_seed()
    device = get_device()
    print(f"Device: {device}")

    train_loader, test_loader = get_dataloaders(
        args.data_dir, args.batch_size, args.num_workers, args.download)

    teacher = build_model(config.TEACHER_NAME, pretrained=not args.no_pretrained)
    print(f"Teacher parameters: {count_parameters(teacher) / 1e6:.2f} M")

    def compute_loss(images, labels):
        """Plain supervised loss: the loss to optimise is also the logged loss."""
        loss = F.cross_entropy(teacher(images).float(), labels)
        return loss, loss.detach()

    optimizer = build_optimizer(teacher.parameters(), config.TEACHER_NAME)
    fit(f"teacher_{config.TEACHER_NAME}", teacher, train_loader, test_loader,
        compute_loss, optimizer, args.epochs, device)


if __name__ == "__main__":
    main()
