"""
onnx_inference.py
=================
CPU inference with ONNX Runtime for the exported student models.

Contains the reusable OnnxPredictor class (also imported by benchmark.py) and
a small command-line interface:
  * with --image : classify one image with each selected model
  * without      : evaluate each selected model on the CIFAR-100 test set

Usage examples:
    python onnx_inference.py --model all --variant both
    python onnx_inference.py --model squeezenet --variant kd --image cat.png
"""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

import config
from data import get_class_names, get_inference_transform, get_test_loader
from inference import select_names, select_variants


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class OnnxPredictor:
    """Runs one exported ONNX model on the CPU."""

    def __init__(self, run_name: str, num_threads: int = 0):
        options = ort.SessionOptions()
        options.intra_op_num_threads = num_threads   # 0 = let ONNX Runtime decide
        self.run_name = run_name
        self.session = ort.InferenceSession(
            str(config.onnx_path(run_name)), sess_options=options,
            providers=["CPUExecutionProvider"])
        self.transform = get_inference_transform()

    def run(self, batch: np.ndarray) -> np.ndarray:
        """Run a float32 batch of shape (B, 3, H, W); returns logits (B, 100)."""
        return self.session.run(None, {"input": batch})[0]

    def predict(self, image: Image.Image, top_k: int = 5):
        """Return a list of (class_index, probability) for one PIL image."""
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).numpy()
        probs = softmax(self.run(tensor))[0]
        top = np.argsort(-probs)[:top_k]
        return [(int(i), float(probs[i])) for i in top]

    def evaluate(self, test_loader) -> float:
        """Test-set accuracy (percent) computed entirely with ONNX Runtime."""
        correct, total = 0, 0
        for images, labels in test_loader:
            logits = self.run(images.numpy())
            correct += int((logits.argmax(axis=1) == labels.numpy()).sum())
            total += labels.size(0)
        return 100.0 * correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX Runtime CPU inference")
    parser.add_argument("--model", default="all", choices=["all"] + config.STUDENT_NAMES)
    parser.add_argument("--variant", default="both", choices=["both", "baseline", "kd"])
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--data_dir", type=Path, default=config.DATA_DIR)
    args = parser.parse_args()

    class_names = get_class_names(args.data_dir)
    test_loader = None if args.image else get_test_loader(args.data_dir, num_workers=0)

    for name in select_names(args.model):
        for variant in select_variants(args.variant):
            predictor = OnnxPredictor(f"{variant}_{name}")
            if args.image:
                print(f"\n{predictor.run_name}:")
                for index, prob in predictor.predict(Image.open(args.image)):
                    print(f"  {class_names[index]:<15s} {prob * 100:5.1f}%")
            else:
                print(f"{predictor.run_name:<30s} ONNX test accuracy: "
                      f"{predictor.evaluate(test_loader):.2f}%")


if __name__ == "__main__":
    main()
