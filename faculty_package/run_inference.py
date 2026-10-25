#!/usr/bin/env python3
"""Run the wound tissue segmentation model (ResNet34 U-Net) on one image or a folder.

Research prototype only. Not a medical device. Not for treatment decisions.

Examples:
    python run_inference.py --input my_image.jpg --output results_out
    python run_inference.py --input folder_of_images --output results_out --device cpu
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import segmentation_models_pytorch as smp

CLASS_NAMES = ["Background", "Epithelial", "Slough", "Granulation", "Necrotic", "Other"]
# RGB colours used for the overlay (class 0 is left untouched)
COLORS = np.array(
    [[0, 0, 0], [80, 160, 255], [255, 220, 50], [60, 200, 90], [230, 50, 50], [200, 80, 220]],
    dtype=np.uint8,
)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet statistics
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "resnet34_unet_weighted_seed42.pth"


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(weights, device):
    weights = Path(weights)
    if not weights.exists():
        sys.exit(f"ERROR: weights file not found: {weights}")
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=6)
    # weights_only=True: the file must contain tensors only (no arbitrary Python objects)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def predict(model, image, size, device):
    """Return a uint8 mask (height x width of the ORIGINAL image) with class ids 0..5."""
    small = image.resize((size, size), Image.BILINEAR)
    x = (np.asarray(small, dtype=np.float32) / 255.0 - MEAN) / STD
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).argmax(1)[0].cpu().numpy().astype(np.uint8)
    return np.asarray(Image.fromarray(pred).resize(image.size, Image.NEAREST))


def measure(mask):
    """Wound area as a percent of the whole image; tissue shares as a percent of wound pixels.

    No physical scale is available, so these are NOT cm^2 values.
    """
    wound = int((mask > 0).sum())
    out = {"wound_area_pct_of_image": round(100.0 * wound / mask.size, 3)}
    for i in range(1, 6):
        n = int((mask == i).sum())
        out[f"{CLASS_NAMES[i].lower()}_pct_of_wound"] = round(100.0 * n / wound, 3) if wound else None
    return out


def overlay(image, mask):
    arr = np.asarray(image, dtype=np.float32).copy()
    fg = mask > 0
    arr[fg] = 0.5 * arr[fg] + 0.5 * COLORS[mask[fg]]
    return Image.fromarray(arr.astype(np.uint8))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="an image file or a folder of images")
    ap.add_argument("--output", default="inference_output", help="folder for masks, overlays, summary.csv")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="path to the .pth weights file")
    ap.add_argument("--size", type=int, default=256, help="model input size (default 256, as in the reported results)")
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    args = ap.parse_args()

    src = Path(args.input)
    if src.is_dir():
        files = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    elif src.is_file():
        files = [src]
    else:
        sys.exit(f"ERROR: input not found: {src}")
    if not files:
        sys.exit("ERROR: no .jpg/.jpeg/.png images found")

    device = pick_device(args.device)
    print(f"Device: {device} | images: {len(files)} | weights: {args.weights}")
    model = load_model(args.weights, device)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in files:
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:  # unreadable or corrupt image
            print(f"SKIPPED {path}: {type(exc).__name__}")
            continue
        mask = predict(model, image, args.size, device)
        stem = path.stem
        Image.fromarray(mask).save(out_dir / f"{stem}_mask.png")  # pixel values = class ids 0..5
        overlay(image, mask).save(out_dir / f"{stem}_overlay.png")
        row = {"image": str(path), **measure(mask)}
        rows.append(row)
        print(path.name, "|", {k: v for k, v in row.items() if k != "image"})

    if rows:
        with open(out_dir / "summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved masks, overlays and summary.csv to {out_dir}")
    print("Research prototype only. Not for treatment decisions.")


if __name__ == "__main__":
    main()
