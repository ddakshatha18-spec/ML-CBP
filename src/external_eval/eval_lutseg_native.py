"""Re-score LUTSeg checkpoints the way the base paper (TiSage) scores them, next to our usual 256 x 256 way.

    PYTHONPATH=src python -m external_eval.eval_lutseg_native --device cpu \
        --checkpoint checkpoints/best_model_resnet34_weighted.pth checkpoints/best_model_r34w_s1.pth checkpoints/best_model_r34w_s2.pth

--eval-res 256     our earlier protocol (image and mask resized to 256 x 256, nearest for the mask)
--eval-res native  each image at its own resolution (prediction made at 256 x 256 and upsampled with nearest neighbour),
                   like TiSage's method/eval/evaluate_checkpoint.py, which averages per-class IoU / Dice over all 6
                   classes including Background and the never-predicted 'Other' (counted as 0).
Run both and compare: the first should reproduce your reported numbers (use --device cpu for exact reproduction).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import OUR_CLASSES
from .mapping import metrics_from_confusion, native_confusion
from .model_utils import Predictor, load_lutseg_model, pick_device


def read_pairs(root: Path, split_file: str = "val.txt"):
    pairs = []
    for line in (root / split_file).read_text().splitlines():
        line = line.strip()
        if line:
            ip, mp = line.split()
            pairs.append((root / ip, root / mp))
    if not pairs:
        raise ValueError(f"No samples in {root / split_file}")
    return pairs


def score_checkpoint(path, pairs, device, size, eval_res):
    predict = Predictor(load_lutseg_model(path, device), device, size)
    conf = np.zeros((6, 6), dtype=np.int64)
    for ip, mp in pairs:
        img = Image.open(ip).convert("RGB")  # no EXIF rotation: masks are on the raw pixel grid
        gt_img = Image.open(mp).convert("L")
        if eval_res == "256":
            gt = np.array(gt_img.resize((256, 256), Image.NEAREST))
            pred = predict(img, out_size=(256, 256))
        else:
            gt = np.array(gt_img)
            pred = predict(img)
        conf += native_confusion(gt, pred, 6, 6)  # 255 (ignore) is excluded
    return metrics_from_confusion(conf, OUR_CLASSES)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", nargs="+", default=["checkpoints/best_model_resnet34_weighted.pth"])
    ap.add_argument("--lutseg-root", default="data/LUTSeg")
    ap.add_argument("--split-file", default="val.txt")
    ap.add_argument("--eval-res", nargs="+", default=["256", "native"], choices=["256", "native"])
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output", default="outputs/external_eval/lutseg_rescore")
    args = ap.parse_args()

    device = pick_device(args.device)
    pairs = read_pairs(Path(args.lutseg_root), args.split_file)
    print(f"device {device} | {len(pairs)} images from {args.split_file}")
    out = {}
    for res in args.eval_res:
        ious, dices = [], []
        print(f"\n--- eval resolution: {res} ---")
        for ck in args.checkpoint:
            m = score_checkpoint(ck, pairs, device, args.size, res)
            ious.append(m["mean_iou"])
            dices.append(m["mean_dice"])
            print(f"  {Path(ck).name:<40} mIoU {m['mean_iou']:6.2f}  Dice {m['mean_dice']:6.2f}   "
                  + " ".join(f"{n[:4]} {m['classes'][n]['iou']:5.1f}" for n in OUR_CLASSES))
            out.setdefault(res, {})[Path(ck).name] = m
        if len(ious) > 1:
            print(f"  mean over {len(ious)} checkpoints: mIoU {np.mean(ious):.2f} +/- {np.std(ious, ddof=1):.2f}"
                  f" | Dice {np.mean(dices):.2f} +/- {np.std(dices, ddof=1):.2f}   (sample std)")
        out[res]["_summary"] = {"mean_iou": float(np.mean(ious)), "mean_dice": float(np.mean(dices)),
                                "std_iou": float(np.std(ious, ddof=1)) if len(ious) > 1 else None,
                                "std_dice": float(np.std(dices, ddof=1)) if len(dices) > 1 else None,
                                "n_checkpoints": len(ious)}
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "lutseg_rescore.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved {args.output}/lutseg_rescore.json")


if __name__ == "__main__":
    main()
