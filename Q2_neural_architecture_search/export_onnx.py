"""
export_onnx.py
==============
Export the trained best architecture to ONNX and verify that ONNX Runtime
gives the same outputs as PyTorch.

The exported model takes a normalised 32x32 CIFAR image (batch size can vary).
If the architecture works at another resolution, the resize is already part
of the exported graph.

Needs: output/best_architecture.json and output/best_model.pth
Output: output/best_model.onnx

Usage:
    python export_onnx.py
"""

import numpy as np
import onnxruntime as ort
import torch

import config
from model_builder import build_model
from train_best import load_best_architecture


def load_trained_model() -> torch.nn.Module:
    """Rebuild the best architecture and load its trained weights (CPU, eval mode)."""
    if not config.BEST_MODEL_PTH.exists():
        raise FileNotFoundError(f"{config.BEST_MODEL_PTH} not found. Run train_best.py first.")
    model = build_model(load_best_architecture())
    model.load_state_dict(torch.load(config.BEST_MODEL_PTH, map_location="cpu"))
    return model.eval()


def main() -> None:
    config.ensure_output_dirs()
    model = load_trained_model()
    dummy = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)

    export_args = dict(
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17)
    try:
        # Newer PyTorch: force the stable TorchScript-based exporter.
        torch.onnx.export(model, dummy, str(config.BEST_MODEL_ONNX), dynamo=False, **export_args)
    except TypeError:
        # Older PyTorch has no `dynamo` argument.
        torch.onnx.export(model, dummy, str(config.BEST_MODEL_ONNX), **export_args)

    # Verify: same input through PyTorch and ONNX Runtime must give (almost) the same logits.
    session = ort.InferenceSession(str(config.BEST_MODEL_ONNX),
                                   providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"input": dummy.numpy()})[0]
    with torch.no_grad():
        torch_out = model(dummy).numpy()
    print(f"Exported -> {config.BEST_MODEL_ONNX} "
          f"({config.BEST_MODEL_ONNX.stat().st_size / 1e6:.3f} MB)")
    print(f"Max |ONNX - PyTorch| output difference: {np.abs(onnx_out - torch_out).max():.2e}")


if __name__ == "__main__":
    main()
