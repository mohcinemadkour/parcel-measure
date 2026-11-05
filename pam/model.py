"""Model definition (training side). Serving uses the exported ONNX file."""
from pathlib import Path

import torch
import segmentation_models_pytorch as smp

from .config import N_CLASSES


def build_model(encoder: str = "resnet34", pretrained_path: str | None = None) -> torch.nn.Module:
    model = smp.Unet(encoder_name=encoder, encoder_weights=None, classes=N_CLASSES)
    if pretrained_path and Path(pretrained_path).exists():
        sd = torch.load(pretrained_path, map_location="cpu")
        sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
        res = torch.nn.Module.load_state_dict(model.encoder, sd, strict=False)
        missing, unexpected = res.missing_keys, res.unexpected_keys
        print(f"loaded ImageNet weights: {len(sd)} tensors, missing={len(missing)}, unexpected={len(unexpected)}")
    return model
