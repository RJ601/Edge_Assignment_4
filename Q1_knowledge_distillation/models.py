"""
models.py
=========
Teacher and student network definitions.

All networks share one small interface (FeatureNet) so that the rest of the
code (training, distillation, inference, ONNX export) treats them identically:

    extract_features(x) -> last convolutional feature map   (used for feature distillation)
    classify(feature_map) -> class logits
    forward(x)           -> class logits
    forward(x, return_features=True) -> (logits, feature_map)

Networks
--------
  * ResNet50Teacher        - the large teacher (torchvision ResNet-50)
  * MobileNetV2Student     - torchvision MobileNetV2
  * SqueezeNetStudent      - torchvision SqueezeNet 1.1
  * EfficientNetLite0Student - EfficientNet-Lite0 from the `timm` library
                              (torchvision does not ship Lite0)
"""

import torch
import torch.nn as nn
import torchvision.models as tvm

import config


class FeatureNet(nn.Module):
    """
    Base class that splits a classifier into 'feature extractor' and 'head'.

    Subclasses must set `self.feature_channels` (number of channels of the
    feature map returned by extract_features) and implement the two methods.
    """

    feature_channels: int = 0

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the last convolutional feature map, shape (B, C, H, W)."""
        raise NotImplementedError

    def classify(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Turn a feature map into class logits, shape (B, num_classes)."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """Plain forward pass; optionally also returns the feature map."""
        feature_map = self.extract_features(x)
        logits = self.classify(feature_map)
        if return_features:
            return logits, feature_map
        return logits


class ResNet50Teacher(FeatureNet):
    """ResNet-50 teacher. Feature map has 2048 channels."""

    def __init__(self, num_classes: int, pretrained: bool):
        super().__init__()
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base = tvm.resnet50(weights=weights)
        # Everything up to (and including) the last residual stage.
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)      # new 100-class head
        self.feature_channels = 2048

    def extract_features(self, x):
        return self.backbone(x)

    def classify(self, feature_map):
        return self.fc(torch.flatten(self.pool(feature_map), 1))


class MobileNetV2Student(FeatureNet):
    """MobileNetV2 student. Feature map has 1280 channels."""

    def __init__(self, num_classes: int, pretrained: bool):
        super().__init__()
        weights = tvm.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        base = tvm.mobilenet_v2(weights=weights)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, num_classes))
        self.feature_channels = 1280

    def extract_features(self, x):
        return self.features(x)

    def classify(self, feature_map):
        return self.classifier(torch.flatten(self.pool(feature_map), 1))


class SqueezeNetStudent(FeatureNet):
    """
    SqueezeNet 1.1 student. Feature map has 512 channels.

    The original SqueezeNet head is Dropout -> 1x1 Conv -> ReLU -> AvgPool.
    The final ReLU is removed here so that the logits can be negative; this
    lets them match the teacher's logits during distillation.
    """

    def __init__(self, num_classes: int, pretrained: bool):
        super().__init__()
        weights = tvm.SqueezeNet1_1_Weights.IMAGENET1K_V1 if pretrained else None
        base = tvm.squeezenet1_1(weights=weights)
        self.features = base.features
        final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        nn.init.normal_(final_conv.weight, mean=0.0, std=0.01)
        nn.init.zeros_(final_conv.bias)
        self.head = nn.Sequential(nn.Dropout(0.5), final_conv, nn.AdaptiveAvgPool2d(1))
        self.feature_channels = 512

    def extract_features(self, x):
        return self.features(x)

    def classify(self, feature_map):
        return torch.flatten(self.head(feature_map), 1)


class EfficientNetLite0Student(FeatureNet):
    """EfficientNet-Lite0 student (timm implementation). Feature map has 1280 channels."""

    def __init__(self, num_classes: int, pretrained: bool):
        super().__init__()
        import timm  # imported here so the other models work even if timm is missing
        self.model = timm.create_model(
            "efficientnet_lite0", pretrained=pretrained, num_classes=num_classes)
        self.feature_channels = self.model.num_features   # 1280

    def extract_features(self, x):
        return self.model.forward_features(x)

    def classify(self, feature_map):
        return self.model.forward_head(feature_map)


# Registry mapping a model name to its class.
MODEL_REGISTRY = {
    "resnet50": ResNet50Teacher,
    "mobilenetv2": MobileNetV2Student,
    "squeezenet": SqueezeNetStudent,
    "efficientnet_lite0": EfficientNetLite0Student,
}


def build_model(name: str, pretrained: bool = config.USE_PRETRAINED,
                num_classes: int = config.NUM_CLASSES) -> FeatureNet:
    """Create a model by name (see MODEL_REGISTRY)."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](num_classes=num_classes, pretrained=pretrained)


def load_model(name: str, checkpoint_file, device: torch.device) -> FeatureNet:
    """
    Build a model WITHOUT downloading pretrained weights, load a trained
    checkpoint into it, move it to `device` and switch to eval mode.
    """
    model = build_model(name, pretrained=False)
    state_dict = torch.load(checkpoint_file, map_location="cpu")
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def count_parameters(model: nn.Module) -> int:
    """Total number of parameters of a model."""
    return sum(p.numel() for p in model.parameters())
