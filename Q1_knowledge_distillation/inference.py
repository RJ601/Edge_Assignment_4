"""
inference.py
============
Unified PyTorch inference pipeline for ALL trained student models
(baseline and distilled variants share the same code path).

Two modes:
  * with --image : classify one image and print the top-5 predictions of each selected model
  * without      : evaluate each selected model on the CIFAR-100 test set

Usage examples:
    python inference.py --model all --variant both                 # test-set accuracy of all 6 models
    python inference.py --model mobilenetv2 --variant kd --image cat.png
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

import config
from data import get_class_names, get_inference_transform, get_test_loader
from models import load_model
from train_utils import evaluate, get_device


class StudentPredictor:
    """Loads one trained student checkpoint and classifies PIL images."""

    def __init__(self, model_name: str, variant: str, device: torch.device):
        run_name = f"{variant}_{model_name}"
        self.run_name = run_name
        self.device = device
        self.transform = get_inference_transform()
        self.model = load_model(model_name, config.checkpoint_path(run_name), device)

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 5):
        """Return a list of (class_index, probability) for the top_k classes."""
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(tensor), dim=1)[0]
        values, indices = probs.topk(top_k)
        return list(zip(indices.tolist(), values.tolist()))


def select_names(model_arg: str) -> list:
    """Turn '--model all' or a single name into a list of model names."""
    return config.STUDENT_NAMES if model_arg == "all" else [model_arg]


def select_variants(variant_arg: str) -> list:
    """Turn '--variant both' or a single variant into a list."""
    return ["baseline", "kd"] if variant_arg == "both" else [variant_arg]


def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch inference for the student models")
    parser.add_argument("--model", default="all", choices=["all"] + config.STUDENT_NAMES)
    parser.add_argument("--variant", default="both", choices=["both", "baseline", "kd"])
    parser.add_argument("--image", type=Path, default=None, help="path of an image to classify")
    parser.add_argument("--data_dir", type=Path, default=config.DATA_DIR)
    args = parser.parse_args()

    device = get_device()
    class_names = get_class_names(args.data_dir)
    test_loader = None if args.image else get_test_loader(args.data_dir)

    for name in select_names(args.model):
        for variant in select_variants(args.variant):
            predictor = StudentPredictor(name, variant, device)
            if args.image:
                print(f"\n{predictor.run_name}:")
                for index, prob in predictor.predict(Image.open(args.image)):
                    print(f"  {class_names[index]:<15s} {prob * 100:5.1f}%")
            else:
                _, accuracy = evaluate(predictor.model, test_loader, device)
                print(f"{predictor.run_name:<30s} test accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
