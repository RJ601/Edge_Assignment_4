"""
distillation.py
===============
The knowledge-distillation loss.

Total loss = ALPHA * soft_loss + (1 - ALPHA) * hard_loss + FEATURE_WEIGHT * feature_loss

  hard_loss    : ordinary cross-entropy with the ground-truth labels.
  soft_loss    : KL divergence between the temperature-softened student and
                 teacher class distributions (multiplied by T^2 so its
                 gradient magnitude stays comparable when T changes).
  feature_loss : FitNets-style "hint" loss. The student's last feature map is
                 pooled to the teacher's spatial size, projected to the
                 teacher's channel count by a learnable 1x1 convolution
                 (the "regressor"), and compared with the teacher's feature
                 map using mean-squared error.

The regressor only exists during training; it is not part of the student and
is never saved or exported.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


def soft_target_loss(student_logits: torch.Tensor,
                     teacher_logits: torch.Tensor,
                     temperature: float) -> torch.Tensor:
    """
    Temperature-based soft-target loss.

    softmax(z / T) produces a softer distribution as T grows, exposing the
    teacher's "dark knowledge" (relative similarity between wrong classes).
    """
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    # 'batchmean' = sum over classes, mean over the batch (mathematically correct KL).
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    return kl * (temperature ** 2)


class DistillationLoss(nn.Module):
    """Combined hard-label + soft-label + feature distillation loss."""

    def __init__(self, student_channels: int, teacher_channels: int,
                 temperature: float = config.TEMPERATURE,
                 alpha: float = config.ALPHA,
                 feature_weight: float = config.FEATURE_WEIGHT):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.feature_weight = feature_weight
        # Learnable 1x1 conv that maps student feature channels -> teacher feature channels.
        self.regressor = nn.Conv2d(student_channels, teacher_channels, kernel_size=1)

    def feature_loss(self, student_feat: torch.Tensor,
                     teacher_feat: torch.Tensor) -> torch.Tensor:
        """MSE between the projected student feature map and the teacher feature map."""
        # Match spatial size first (student and teacher maps can differ in H x W).
        student_feat = F.adaptive_avg_pool2d(student_feat, teacher_feat.shape[-2:])
        projected = self.regressor(student_feat)
        return F.mse_loss(projected.float(), teacher_feat.float())

    def forward(self, student_logits, student_feat, teacher_logits, teacher_feat, labels):
        """
        Compute the total loss.

        Returns:
            total_loss : tensor to back-propagate
            hard_loss  : the ground-truth cross-entropy (detached, for logging)
        """
        student_logits = student_logits.float()
        teacher_logits = teacher_logits.float()

        hard = F.cross_entropy(student_logits, labels)
        soft = soft_target_loss(student_logits, teacher_logits, self.temperature)
        feat = self.feature_loss(student_feat, teacher_feat)

        total = self.alpha * soft + (1.0 - self.alpha) * hard + self.feature_weight * feat
        return total, hard.detach()
