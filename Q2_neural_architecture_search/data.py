"""
data.py
=======
CIFAR-10 data handling for the NAS pipeline.

To make the architecture search fast, the WHOLE dataset is held in memory as
normalised float tensors (and moved to the GPU once, when one is available).
Mini-batches are then sliced directly from these tensors and augmented on the
device, which avoids the slow per-image DataLoader overhead.

Splits
------
  train (45,000)  : used to train candidates during the search
  val   ( 5,000)  : used to score candidates (the fitness)
  test  (10,000)  : touched ONLY for the final model's reporting
"""

from pathlib import Path
from typing import Iterator, Tuple

import torch
import torch.nn.functional as F
import torchvision

import config


def _to_normalised_tensor(images_uint8) -> torch.Tensor:
    """(N, 32, 32, 3) uint8 numpy array -> normalised float tensor (N, 3, 32, 32)."""
    x = torch.from_numpy(images_uint8).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(config.CIFAR10_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(config.CIFAR10_STD).view(1, 3, 1, 1)
    return (x - mean) / std


class CifarData:
    """Holds the train / validation / test tensors of CIFAR-10."""

    def __init__(self, data_dir: Path = config.DATA_DIR, download: bool = False):
        train_set = torchvision.datasets.CIFAR10(str(data_dir), train=True, download=download)
        test_set = torchvision.datasets.CIFAR10(str(data_dir), train=False, download=download)
        self.class_names = train_set.classes

        all_x = _to_normalised_tensor(train_set.data)
        all_y = torch.tensor(train_set.targets, dtype=torch.long)

        # Fixed random split so every candidate sees exactly the same train/val images.
        generator = torch.Generator().manual_seed(config.SEED)
        perm = torch.randperm(len(all_x), generator=generator)
        val_idx, train_idx = perm[:config.VAL_SIZE], perm[config.VAL_SIZE:]

        self.train_x, self.train_y = all_x[train_idx], all_y[train_idx]
        self.val_x, self.val_y = all_x[val_idx], all_y[val_idx]
        self.test_x = _to_normalised_tensor(test_set.data)
        self.test_y = torch.tensor(test_set.targets, dtype=torch.long)

    def to(self, device: torch.device) -> "CifarData":
        """Move every tensor to `device` (done once; returns self)."""
        for name in ("train_x", "train_y", "val_x", "val_y", "test_x", "test_y"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def full_train(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train + validation images together (all 50,000), for the final training."""
        return torch.cat([self.train_x, self.val_x]), torch.cat([self.train_y, self.val_y])


def augment_batch(x: torch.Tensor) -> torch.Tensor:
    """
    GPU-friendly augmentation of a batch:
      * random horizontal flip (decided independently per image)
      * random crop after 4-pixel zero padding (one random offset per batch)
    """
    flip = torch.rand(x.size(0), device=x.device) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
    padded = F.pad(x, (4, 4, 4, 4))
    top, left = torch.randint(0, 9, (2,)).tolist()
    return padded[:, :, top:top + config.IMAGE_SIZE, left:left + config.IMAGE_SIZE]


def iterate_batches(x: torch.Tensor, y: torch.Tensor, batch_size: int,
                    shuffle: bool = False, augment: bool = False
                    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Yield (images, labels) mini-batches from in-memory tensors.

    With shuffle=True the order is random and an incomplete last batch is dropped
    (training). With shuffle=False every sample is yielded once (evaluation).
    """
    n = x.size(0)
    order = torch.randperm(n, device=x.device) if shuffle else torch.arange(n, device=x.device)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        if shuffle and idx.numel() < batch_size:
            break
        images = x[idx]
        yield (augment_batch(images) if augment else images), y[idx]
