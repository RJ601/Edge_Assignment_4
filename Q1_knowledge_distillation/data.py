"""
data.py
=======
Data loading for CIFAR-100.

Provides:
  * training / test transforms (augmentation + resize + normalisation),
  * DataLoaders for training and testing,
  * the inference transform used for single images (PyTorch and ONNX Runtime),
  * the list of the 100 class names.

The dataset is expected on disk (download it manually, or pass download=True).
"""

from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

import config


def get_train_transform() -> T.Compose:
    """
    Training transform.

    1. RandomCrop(32, padding=4) and RandomHorizontalFlip are standard CIFAR
       augmentations applied on the original 32x32 image.
    2. Resize up-samples to IMAGE_SIZE x IMAGE_SIZE for the ImageNet-style networks.
    3. ToTensor + Normalize converts to a normalised float tensor.
    """
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.Resize(config.IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(config.NORM_MEAN, config.NORM_STD),
    ])


def get_inference_transform() -> T.Compose:
    """
    Deterministic transform used for the test set and for single-image inference.

    Resize((S, S)) forces an exact square size, so images of any resolution
    can be fed to the models.
    """
    return T.Compose([
        T.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(config.NORM_MEAN, config.NORM_STD),
    ])


def get_test_loader(data_dir: Path = config.DATA_DIR,
                    batch_size: int = config.BATCH_SIZE,
                    num_workers: int = config.NUM_WORKERS,
                    download: bool = False) -> DataLoader:
    """Return a DataLoader over the 10,000 CIFAR-100 test images."""
    test_set = torchvision.datasets.CIFAR100(
        root=str(data_dir), train=False, download=download,
        transform=get_inference_transform())
    return DataLoader(test_set, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())


def get_dataloaders(data_dir: Path = config.DATA_DIR,
                    batch_size: int = config.BATCH_SIZE,
                    num_workers: int = config.NUM_WORKERS,
                    download: bool = False):
    """
    Return (train_loader, test_loader) for CIFAR-100.

    The training loader shuffles and augments; the test loader does neither.
    """
    train_set = torchvision.datasets.CIFAR100(
        root=str(data_dir), train=True, download=download,
        transform=get_train_transform())
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True,
                              pin_memory=torch.cuda.is_available())
    test_loader = get_test_loader(data_dir, batch_size, num_workers, download)
    return train_loader, test_loader


def get_class_names(data_dir: Path = config.DATA_DIR, download: bool = False) -> list:
    """Return the list of 100 CIFAR-100 class names (index -> name)."""
    dataset = torchvision.datasets.CIFAR100(
        root=str(data_dir), train=False, download=download)
    return dataset.classes
