"""Genuine legacy TorchScript-to-ONNX export with dynamic batch and sequence axes."""

from pathlib import Path

import onnx
import torch

from .model import BertStyleEncoder


def export_encoder(
    model: BertStyleEncoder,
    destination: str | Path,
    *,
    opset: int = 17,
) -> Path:
    """Export an evaluated CPU model and validate the resulting ONNX graph."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    input_ids = torch.ones((2, 12), dtype=torch.long)
    attention_mask = torch.ones((2, 12), dtype=torch.long)
    attention_mask[1, 8:] = 0
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "last_hidden_state": {0: "batch", 1: "sequence"},
    }
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        path,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(path))
    return path
