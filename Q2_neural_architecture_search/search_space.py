"""
search_space.py
===============
The searchable CNN architecture space.

An architecture is described by a small, serialisable "genome":

    Architecture
      resolution     : input resolution the network works at
      stem_channels  : filters of the first conv layer
      blocks         : list of 3..8 BlockGene, each with
                         out_channels, kernel_size, block_type, activation, stride

This file is pure Python (no PyTorch). It contains everything the
evolutionary search needs to manipulate genomes:
    random_architecture()  - sample a random genome
    mutate()               - produce a slightly changed copy
    repair()               - fix genomes that would shrink the feature map too much
"""

import copy
import json
import math
import random
from dataclasses import asdict, dataclass, field
from typing import List

import config


@dataclass
class BlockGene:
    """Description of one convolutional block."""
    out_channels: int
    kernel_size: int
    block_type: str      # "standard" or "depthwise_separable"
    activation: str      # "relu", "relu6", "hardswish" or "silu"
    stride: int          # 1 or 2


@dataclass
class Architecture:
    """Full description of one candidate network."""
    resolution: int
    stem_channels: int
    blocks: List[BlockGene] = field(default_factory=list)

    # ---- serialisation helpers -------------------------------------------------
    def to_dict(self) -> dict:
        """Convert to a plain dict (JSON friendly)."""
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "Architecture":
        """Rebuild an Architecture from the dict produced by to_dict()."""
        return Architecture(
            resolution=data["resolution"],
            stem_channels=data["stem_channels"],
            blocks=[BlockGene(**b) for b in data["blocks"]])

    def key(self) -> str:
        """Unique text id; two identical architectures have the same key (used to skip duplicates)."""
        return json.dumps(self.to_dict(), sort_keys=True)


# Options for every block field, used by random sampling and by mutation.
BLOCK_OPTIONS = {
    "out_channels": config.BLOCK_CHANNELS,
    "kernel_size": config.KERNEL_SIZES,
    "block_type": config.BLOCK_TYPES,
    "activation": config.ACTIVATIONS,
    "stride": config.STRIDES,
}


def random_block(rng: random.Random) -> BlockGene:
    """Sample one block with every field chosen uniformly at random."""
    return BlockGene(**{name: rng.choice(options) for name, options in BLOCK_OPTIONS.items()})


def repair(arch: Architecture) -> Architecture:
    """
    Make the genome valid: a stride-2 block is changed to stride 1 whenever it
    would shrink the feature map below config.MIN_FEATURE_SIZE.
    Modifies and returns the same object.
    """
    size = arch.resolution
    for block in arch.blocks:
        if block.stride == 2:
            if math.ceil(size / 2) < config.MIN_FEATURE_SIZE:
                block.stride = 1
            else:
                size = math.ceil(size / 2)
    return arch


def random_architecture(rng: random.Random) -> Architecture:
    """Sample a random valid architecture."""
    num_blocks = rng.randint(config.MIN_BLOCKS, config.MAX_BLOCKS)
    arch = Architecture(
        resolution=rng.choice(config.RESOLUTIONS),
        stem_channels=rng.choice(config.STEM_CHANNELS),
        blocks=[random_block(rng) for _ in range(num_blocks)])
    return repair(arch)


# ---------------------------------------------------------------------------
# Mutation operators
# ---------------------------------------------------------------------------
def _mutate_resolution(arch: Architecture, rng: random.Random) -> None:
    """Pick a different input resolution."""
    arch.resolution = rng.choice([r for r in config.RESOLUTIONS if r != arch.resolution])


def _mutate_stem(arch: Architecture, rng: random.Random) -> None:
    """Pick a different number of stem filters."""
    arch.stem_channels = rng.choice([c for c in config.STEM_CHANNELS if c != arch.stem_channels])


def _add_block(arch: Architecture, rng: random.Random) -> None:
    """Insert a random block at a random position (deeper network)."""
    arch.blocks.insert(rng.randint(0, len(arch.blocks)), random_block(rng))


def _remove_block(arch: Architecture, rng: random.Random) -> None:
    """Delete a random block (shallower network)."""
    arch.blocks.pop(rng.randrange(len(arch.blocks)))


def _modify_block(arch: Architecture, rng: random.Random) -> None:
    """Change ONE field (channels, kernel, type, activation or stride) of one block."""
    block = rng.choice(arch.blocks)
    field_name = rng.choice(list(BLOCK_OPTIONS))
    current = getattr(block, field_name)
    setattr(block, field_name, rng.choice([o for o in BLOCK_OPTIONS[field_name] if o != current]))


def mutate(arch: Architecture, rng: random.Random) -> Architecture:
    """
    Return a mutated COPY of `arch` (the parent is never modified).

    One or two random mutation operators are applied. Operators that are not
    possible (e.g. removing a block when the network already has the minimum
    depth) are skipped. "modify block" is listed three times so that small
    changes are more likely than big structural ones.
    """
    child = copy.deepcopy(arch)
    for _ in range(rng.choice([1, 2])):
        operators = [_mutate_resolution, _mutate_stem,
                     _modify_block, _modify_block, _modify_block]
        if len(child.blocks) < config.MAX_BLOCKS:
            operators.append(_add_block)
        if len(child.blocks) > config.MIN_BLOCKS:
            operators.append(_remove_block)
        rng.choice(operators)(child, rng)
    return repair(child)
