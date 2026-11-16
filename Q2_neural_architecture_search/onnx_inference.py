"""
onnx_inference.py
=================
CPU inference pipeline with ONNX Runtime for the exported best architecture.

Contains the reusable OnnxPredictor class (also used by benchmark.py) and a
command-line interface:
  * with --image : classify one image and print the top-3 predictions
  * without      : evaluate on the CIFAR-10 test set

Usage:
    python onnx_inference.py                     # test-set accuracy
    python onnx_inference.py --image my_image.png
"""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

import config
from data import CifarData, iterate_batches

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    PIL image -> float32 array of shape (1, 3, 32, 32), normalised like the training data.
    Any input size is accepted; it is resized to 32x32 first.
    """
    image = image.convert("RGB").resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0                  # (H, W, 3) in 0..1
    mean = np.array(config.CIFAR10_MEAN, dtype=np.float32)
    std = np.array(config.CIFAR10_STD, dtype=np.float32)
    array = (array - mean) / std
    return array.transpose(2, 0, 1)[None]                                # (1, 3, H, W)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


class OnnxPredictor:
    """Runs the exported ONNX model on the CPU."""

    def __init__(self, onnx_file: Path = config.BEST_MODEL_ONNX, num_threads: int = 0):
        options = ort.SessionOptions()
        options.intra_op_num_threads = num_threads      # 0 = ONNX Runtime default
        self.session = ort.InferenceSession(str(onnx_file), sess_options=options,
                                            providers=["CPUExecutionProvider"])

    def run(self, batch: np.ndarray) -> np.ndarray:
        """Logits (B, 10) for a float32 batch (B, 3, 32, 32)."""
        return self.session.run(None, {"input": batch})[0]

    def predict(self, image: Image.Image, top_k: int = 3):
        """Top-k (class_name, probability) pairs for one PIL image."""
        probs = softmax(self.run(preprocess_image(image)))[0]
        top = np.argsort(-probs)[:top_k]
        return [(CLASS_NAMES[i], float(probs[i])) for i in top]

    def evaluate(self, x, y, batch_size: int = 500) -> float:
        """Accuracy (%) on tensors (x, y) computed entirely with ONNX Runtime."""
        correct = 0
        for images, labels in iterate_batches(x, y, batch_size):
            logits = self.run(images.numpy())
            correct += int((logits.argmax(axis=1) == labels.numpy()).sum())
        return 100.0 * correct / x.size(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX Runtime CPU inference")
    parser.add_argument("--image", type=Path, default=None, help="image file to classify")
    parser.add_argument("--data_dir", type=Path, default=config.DATA_DIR)
    args = parser.parse_args()

    predictor = OnnxPredictor()
    if args.image:
        for name, prob in predictor.predict(Image.open(args.image)):
            print(f"{name:<12s} {prob * 100:5.1f}%")
    else:
        data = CifarData(args.data_dir)
        print(f"ONNX Runtime test accuracy: {predictor.evaluate(data.test_x, data.test_y):.2f}%")


if __name__ == "__main__":
    main()
