"""
train_distill.py
================
Train each student with knowledge distillation from the trained ResNet-50 teacher.

Loss = ALPHA * soft_target_loss(T) + (1 - ALPHA) * hard_label_loss + FEATURE_WEIGHT * feature_loss
(see distillation.py). Requires train_teacher.py to have been run first.

Output (per student):
    output/checkpoints/kd_<student>.pth
    output/logs/kd_<student>.csv

Usage:
    python train_distill.py [--students mobilenetv2 ...] [--epochs 15]
"""

import argparse

import torch

import config
from data import get_dataloaders
from distillation import DistillationLoss
from models import build_model, count_parameters, load_model
from train_utils import add_common_args, build_optimizer, fit, get_device, set_seed


def train_distilled(name: str, args, device, teacher, train_loader, test_loader) -> None:
    """Train one student using the frozen teacher as supervisor."""
    set_seed()   # identical seed to the baseline run
    student = build_model(name, pretrained=not args.no_pretrained)
    print(f"\n=== Distilling into {name} ({count_parameters(student) / 1e6:.2f} M parameters) ===")

    # The distillation loss owns the feature "regressor" (trainable, training-only).
    criterion = DistillationLoss(student.feature_channels, teacher.feature_channels).to(device)

    def compute_loss(images, labels):
        """One distillation step: teacher predicts (no gradient), student learns."""
        with torch.no_grad():
            teacher_logits, teacher_feat = teacher(images, return_features=True)
        student_logits, student_feat = student(images, return_features=True)
        return criterion(student_logits, student_feat, teacher_logits, teacher_feat, labels)

    # Optimise the student AND the regressor together.
    parameters = list(student.parameters()) + list(criterion.parameters())
    optimizer = build_optimizer(parameters, name)
    fit(f"kd_{name}", student, train_loader, test_loader,
        compute_loss, optimizer, args.epochs, device)


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Distil the teacher into students"))
    parser.add_argument("--epochs", type=int, default=config.EPOCHS_STUDENT)
    parser.add_argument("--students", nargs="+", default=config.STUDENT_NAMES,
                        choices=config.STUDENT_NAMES)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    train_loader, test_loader = get_dataloaders(
        args.data_dir, args.batch_size, args.num_workers, args.download)

    # Load the trained teacher once and freeze it.
    teacher_file = config.checkpoint_path(f"teacher_{config.TEACHER_NAME}")
    if not teacher_file.exists():
        raise FileNotFoundError(f"{teacher_file} not found. Run train_teacher.py first.")
    teacher = load_model(config.TEACHER_NAME, teacher_file, device)
    for p in teacher.parameters():
        p.requires_grad_(False)

    for name in args.students:
        train_distilled(name, args, device, teacher, train_loader, test_loader)


if __name__ == "__main__":
    main()
