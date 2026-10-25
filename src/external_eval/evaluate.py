"""Evaluation loop and report writing (torch-free: the predictor is passed in as a function)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .datasets import Item, load_image_rgb
from .mapping import (LabelSpace, apply_policy, confusion_to_rows, metrics_from_confusion,
                      native_confusion)


def evaluate_native(
    items: Sequence[Item],
    load_gt: Callable[[Item], np.ndarray],
    predict_fn: Callable[[Image.Image], np.ndarray],
    n_gt: int,
    n_pred: int,
    load_image: Callable = load_image_rgb,
) -> Tuple[np.ndarray, List[dict]]:
    """Run ``predict_fn`` on every item and pool a native confusion matrix ``N[gt_id, predicted_id]``.

    ``predict_fn`` must return a class-id mask with the SAME height and width as the image.
    Evaluation happens on the image's own pixel grid (like TiSage), not on a resized copy.
    """
    native = np.zeros((n_gt, n_pred), dtype=np.int64)
    rows: List[dict] = []
    for it in items:
        img = load_image(it.image_path)
        gt = load_gt(it)
        pred = np.asarray(predict_fn(img))
        if pred.shape != gt.shape:
            raise ValueError(
                f"{it.id}: prediction {pred.shape} vs ground truth {gt.shape}. "
                "Image and mask sizes must match."
            )
        c = native_confusion(gt, pred, n_gt, n_pred)
        native += c
        rows.append({"id": it.id, "width": int(gt.shape[1]), "height": int(gt.shape[0]),
                     "evaluated_pixels": int(c.sum())})
    return native, rows


def variant_key(gt_policy: str, pred_policy: str) -> str:
    return f"{gt_policy}|{pred_policy}"


def zero_shot_variants(native: np.ndarray, space: LabelSpace) -> Dict[str, dict]:
    """Metrics for every (ground-truth policy x prediction policy) combination of a label space."""
    out: Dict[str, dict] = {}
    for gp in space.gt_map:
        for pp in ("strict", "lenient"):
            conf = apply_policy(native, space, gp, pp)
            m = metrics_from_confusion(conf, space.class_names)
            out[variant_key(gp, pp)] = {"gt_policy": gp, "pred_policy": pp, "metrics": m,
                                         "confusion": conf.tolist()}
    return out


def write_zero_shot_report(out_dir, name: str, native: np.ndarray, native_gt_names: List[str],
                           native_pred_names: List[str], space: LabelSpace, per_image: List[dict],
                           extra: Optional[dict] = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = zero_shot_variants(native, space)
    report = {
        "name": name,
        "label_space": space.name,
        "common_classes": space.class_names,
        "notes": space.notes,
        "native_confusion": native.tolist(),
        "native_gt_classes": native_gt_names,
        "native_pred_classes": native_pred_names,
        "variants": variants,
        "images": len(per_image),
        "extra": extra or {},
        "limitations": [
            "Zero-shot: the model was trained on LUTSeg only; the label sets differ, so numbers depend on the "
            "class mapping and are NOT comparable with models trained on this dataset.",
            "Research prototype; not clinically validated.",
        ],
    }
    (out_dir / f"{name}.json").write_text(json.dumps(report, indent=2))

    with (out_dir / f"{name}_native_confusion.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + [f"pred_{n}" for n in native_pred_names])
        for i, n in enumerate(native_gt_names):
            w.writerow([f"gt_{n}"] + [int(v) for v in native[i]])

    with (out_dir / f"{name}_variants.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "mean_iou", "mean_dice", "tissue_mean_iou", "tissue_mean_dice",
                    "binary_wound_dice", "unmapped_pixels"]
                   + [f"{c}_iou" for c in space.class_names] + [f"{c}_dice" for c in space.class_names])
        for k, v in variants.items():
            m = v["metrics"]
            w.writerow([k, f"{m['mean_iou']:.2f}", f"{m['mean_dice']:.2f}", f"{m['tissue_mean_iou']:.2f}",
                        f"{m['tissue_mean_dice']:.2f}", f"{m['binary_wound_dice']:.2f}", m["unmapped_pixels"]]
                       + [f"{m['classes'][c]['iou']:.2f}" for c in space.class_names]
                       + [f"{m['classes'][c]['dice']:.2f}" for c in space.class_names])

    with (out_dir / f"{name}_per_image.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
        w.writeheader()
        w.writerows(per_image)
    return report


def print_variants(report: dict) -> None:
    names = report["common_classes"]
    print(f"\n=== {report['name']}  ({report['images']} images, label space: {report['label_space']}) ===")
    print(f"{'variant (gt policy | prediction policy)':<42}{'mIoU':>7}{'mDice':>7}  " +
          "  ".join(f"{n[:12]:>12}" for n in names))
    for k, v in report["variants"].items():
        m = v["metrics"]
        print(f"{k:<42}{m['mean_iou']:>7.2f}{m['mean_dice']:>7.2f}  " +
              "  ".join(f"{m['classes'][n]['iou']:>12.2f}" for n in names) + "   (per-class IoU)")
    for n in report["notes"]:
        print("  note:", n)
