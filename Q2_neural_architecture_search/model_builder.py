"""
model_builder.py
================
The automated architecture generator.

`build_model(arch)` turns an `Architecture` genome (see search_space.py) into a
ready-to-train PyTorch network:

    input (3x32x32) --[resize to arch.resolution]--> stem conv --> N blocks
        --> global average pooling --> linear classifier (10 classes)

Each block is either
  * "standard"            : Conv(k x k) -> BatchNorm -> Activation
  * "depthwise_separable" : DepthwiseConv(k x k) -> BN -> Act -> Pointwise Conv(1x1) -> BN -> Act

The resize to the candidate's resolution is part of the network itself, so
every model accepts a normal 32x32 CIFAR image, also after ONNX export.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from search_space import Architecture, BlockGene

ACTIVATION_CLASSES = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hardswish": nn.Hardswish,
    "silu": nn.SiLU,
}


def make_activation(name: str) -> nn.Module:
    """Create an activation layer from its name."""
    return ACTIVATION_CLASSES[name]()


def conv_bn_act(in_ch: int, out_ch: int, kernel: int, stride: int,
                activation: str, groups: int = 1) -> nn.Sequential:
    """Conv (same-style padding) -> BatchNorm -> Activation."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2,
                  groups=groups, bias=False),
        nn.BatchNorm2d(out_ch),
        make_activation(activation))


def make_block(in_ch: int, gene: BlockGene) -> nn.Module:
    """Create one block from its gene."""
    if gene.block_type == "standard":
        return conv_bn_act(in_ch, gene.out_channels, gene.kernel_size,
                           gene.stride, gene.activation)
    if gene.block_type == "depthwise_separable":
        return nn.Sequential(
            # depthwise: one k x k filter per input channel (groups == channels)
            conv_bn_act(in_ch, in_ch, gene.kernel_size, gene.stride,
                        gene.activation, groups=in_ch),
            # pointwise: 1x1 conv that mixes channels and sets the output width
            conv_bn_act(in_ch, gene.out_channels, 1, 1, gene.activation))
    raise ValueError(f"Unknown block type: {gene.block_type}")


class GeneratedNetwork(nn.Module):
    """A CNN generated from an Architecture genome."""

    def __init__(self, arch: Architecture, num_classes: int = config.NUM_CLASSES):
        super().__init__()
        self.resolution = arch.resolution
        self.needs_resize = arch.resolution != config.IMAGE_SIZE   # static decision

        layers = [conv_bn_act(3, arch.stem_channels, 3, 1, "relu")]   # stem
        in_ch = arch.stem_channels
        for gene in arch.blocks:
            layers.append(make_block(in_ch, gene))
            in_ch = gene.out_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.needs_resize:
            x = F.interpolate(x, size=(self.resolution, self.resolution),
                              mode="bilinear", align_corners=False)
        x = self.features(x)
        return self.classifier(torch.flatten(self.pool(x), 1))


def build_model(arch: Architecture) -> GeneratedNetwork:
    """Public entry point of the architecture generator."""
    return GeneratedNetwork(arch)


def count_parameters(model: nn.Module) -> int:
    """Total number of parameters."""
    return sum(p.numel() for p in model.parameters())
