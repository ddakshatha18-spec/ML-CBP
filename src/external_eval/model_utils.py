"""PyTorch glue: device choice, model construction, checkpoint loading and a single-image Predictor.

Preprocessing is identical to every number reported so far:
RGB -> resize to size x size (bilinear; non-square images are squashed) -> [0, 1] -> ImageNet mean/std
-> argmax over classes. No EXIF rotation is applied to dataset images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

import torch
import segmentation_models_pytorch as smp

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def pick_device(requested: str = "auto") -> torch.device:
    """auto = mps > cuda > cpu. An unavailable explicit request falls back to cpu with a printed note."""
    requested = (requested or "auto").lower()
    has_mps = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    if requested == "auto":
        return torch.device("mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu")
    if requested == "mps" and not has_mps:
        print("note: mps not available, using cpu")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("note: cuda not available, using cpu")
        return torch.device("cpu")
    if requested not in ("cpu", "mps", "cuda"):
        raise ValueError("device must be auto, cpu, mps or cuda")
    return torch.device(requested)


def build_unet(num_classes: int, imagenet: bool = False) -> torch.nn.Module:
    """The final-model architecture: smp.Unet with a ResNet34 encoder."""
    return smp.Unet(encoder_name="resnet34", encoder_weights="imagenet" if imagenet else None,
                    in_channels=3, classes=num_classes)


def read_state_dict(path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Checkpoints are not in git; check the file name in checkpoints/."
        )
    ckpt = torch.load(path, map_location="cpu")
    return ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt


def load_lutseg_model(path, device: torch.device) -> torch.nn.Module:
    """Load a 6-class LUTSeg ResNet34 U-Net checkpoint (same way as the existing evaluators)."""
    model = build_unet(6, imagenet=False)
    model.load_state_dict(read_state_dict(path))
    return model.to(device).eval()


def init_from_lutseg(model: torch.nn.Module, path) -> None:
    """Copy every LUTSeg weight except the final 6-class head into a model with a different class count."""
    state = {k: v for k, v in read_state_dict(path).items() if not k.startswith("segmentation_head")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in missing if not k.startswith("segmentation_head")]
    if bad or unexpected:
        raise RuntimeError(f"Unexpected key mismatch while initialising from {path}: missing={bad}, unexpected={unexpected}")


class Predictor:
    """image (PIL, RGB) -> class-id mask (uint8).

    By default the mask has the image's ORIGINAL size (nearest-neighbour upsample of the ``size`` x ``size``
    prediction). ``out_size=(W, H)`` returns a mask of that size instead; ``out_size=(size, size)`` returns
    the raw model grid with no resampling.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device, size: int = 256):
        self.model = model.eval()
        self.device = device
        self.size = int(size)

    def _tensor(self, image: Image.Image) -> torch.Tensor:
        small = image.convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        arr = (np.asarray(small, dtype=np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def __call__(self, image: Image.Image, out_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        grid = self.model(self._tensor(image)).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        target = out_size if out_size is not None else image.size  # PIL size is (W, H)
        if tuple(target) != (self.size, self.size):
            grid = np.asarray(Image.fromarray(grid).resize(tuple(target), Image.NEAREST))
        return grid
