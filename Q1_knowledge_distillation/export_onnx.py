"""
export_onnx.py
==============
Export every trained student (3 baselines + 3 distilled) to ONNX and verify
that ONNX Runtime reproduces the PyTorch outputs.

Output:
    output/onnx/<variant>_<student>.onnx

Usage:
    python export_onnx.py
"""

import numpy as np
import onnxruntime as ort
import torch

import config
from models import load_model


def export_one(model_name: str, variant: str) -> None:
    """Export a single checkpoint to ONNX and check numerical agreement."""
    run_name = f"{variant}_{model_name}"
    ckpt = config.checkpoint_path(run_name)
    if not ckpt.exists():
        print(f"[skip] {ckpt} not found")
        return

    model = load_model(model_name, ckpt, torch.device("cpu"))
    dummy = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    out_file = config.onnx_path(run_name)

    export_args = dict(
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},  # variable batch size
        opset_version=config.ONNX_OPSET)
    try:
        # Newer PyTorch versions: force the stable TorchScript-based exporter.
        torch.onnx.export(model, dummy, str(out_file), dynamo=False, **export_args)
    except TypeError:
        # Older PyTorch versions do not have the `dynamo` argument.
        torch.onnx.export(model, dummy, str(out_file), **export_args)

    # Verification: run the same input through PyTorch and ONNX Runtime.
    session = ort.InferenceSession(str(out_file), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"input": dummy.numpy()})[0]
    with torch.no_grad():
        torch_out = model(dummy).numpy()
    max_diff = float(np.abs(onnx_out - torch_out).max())
    size_mb = out_file.stat().st_size / 1e6
    print(f"[ok] {run_name:<30s} {size_mb:6.2f} MB | max |ONNX - PyTorch| = {max_diff:.2e}")


def main() -> None:
    config.ensure_output_dirs()
    for name in config.STUDENT_NAMES:
        for variant in ("baseline", "kd"):
            export_one(name, variant)


if __name__ == "__main__":
    main()
