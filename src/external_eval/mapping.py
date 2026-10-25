"""Class-space mapping and metrics (numpy only).

Two kinds of evaluation use this module:

1. Zero-shot transfer of the 6-class LUTSeg model to another dataset. The label sets differ, so the
   comparison happens in a COMMON class space. One pass over the images fills a *native* confusion
   matrix ``N[gt_native_id, our_predicted_class]``; ``apply_policy`` then turns it into the common-space
   matrix for each documented policy, so all variants come from the same predictions.
2. A model trained on the target dataset (identity mapping, no policy).

Common-space confusion matrices have shape (C, C + 1): rows = ground-truth common class, columns = the
C predicted common classes plus a last column "unmapped" (the model predicted a class that has no
counterpart in the target label set). Under the STRICT policy an unmapped prediction counts as an error
for the ground-truth class and creates no false positive for any other class. Under the LENIENT policy
unmapped predictions are folded into background.

Metrics follow TiSage (carlosh93/TiSage, ``supervised.py`` and ``method/eval/evaluate_checkpoint.py``):
pixel-pooled confusion over all images, per-class IoU = TP / (TP + FP + FN), Dice = 2TP / (2TP + FP + FN),
means over ALL classes including background (an absent class would count as 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import IGNORE, OUR_CLASSES

OUR_N = len(OUR_CLASSES)  # 6


# ----------------------------------------------------------------------------------------------
# Native confusion
# ----------------------------------------------------------------------------------------------
def native_confusion(gt: np.ndarray, pred: np.ndarray, n_gt: int, n_pred: int = OUR_N,
                     ignore: int = IGNORE) -> np.ndarray:
    """Pixel counts ``N[g, p]`` for one image. ``gt`` ids >= n_gt (incl. 255) are ignored."""
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    if gt.shape != pred.shape:
        raise ValueError(f"gt {gt.shape} and prediction {pred.shape} differ in shape")
    valid = (gt != ignore) & (gt < n_gt)
    if pred.size and pred.max() >= n_pred:
        raise ValueError(f"prediction contains class id {int(pred.max())} >= {n_pred}")
    idx = gt[valid].astype(np.int64) * n_pred + pred[valid].astype(np.int64)
    return np.bincount(idx, minlength=n_gt * n_pred).reshape(n_gt, n_pred).astype(np.int64)


# ----------------------------------------------------------------------------------------------
# Label spaces
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LabelSpace:
    """``gt_map[native_id]`` -> common id or IGNORE. ``pred_map_*[our_class]`` -> common id or -1 (unmapped)."""

    name: str
    class_names: List[str]
    gt_map: Dict[str, List[int]]          # policy name -> list indexed by native gt id
    pred_map_strict: List[int]            # length 6; -1 = unmapped
    pred_map_lenient: List[int]           # length 6; no -1
    notes: List[str] = field(default_factory=list)


def dfutissue_common3() -> LabelSpace:
    """DFUTissue (bg, fibrin, granulation, callus) vs our 6 classes -> common {background, fibrin/slough, granulation}.

    Callus (thickened skin around the ulcer) has no counterpart in our label set. Two ground-truth policies:
    ``callus_ignore`` excludes those pixels; ``callus_background`` treats them as background.
    """
    return LabelSpace(
        name="dfutissue_common3",
        class_names=["background", "fibrin/slough", "granulation"],
        gt_map={
            "callus_ignore": [0, 1, 2, IGNORE],
            "callus_background": [0, 1, 2, 0],
        },
        #                  Bg Epi Slough Gran Necr Other
        pred_map_strict=[0, -1, 1, 2, -1, -1],
        pred_map_lenient=[0, 0, 1, 2, 0, 0],
        notes=[
            "Slough is assumed to correspond to DFUTissue 'fibrin'.",
            "Our Epithelial / Necrotic / Other have no DFUTissue counterpart.",
        ],
    )


def cwdb_common4() -> LabelSpace:
    """ComplexWoundDB (bg, granulation, fibrinoid, necrosis, hematoma) -> {background, slough, granulation, necrotic}.

    Hematoma is excluded (2 of 27 images per expert 1). Epithelial and Other have no counterpart.
    """
    return LabelSpace(
        name="cwdb_common4",
        class_names=["background", "slough/fibrinoid", "granulation", "necrotic"],
        gt_map={"default": [0, 2, 1, 3, IGNORE]},  # native: 0 bg,1 gran,2 fibrinoid,3 necrosis,4 hematoma
        pred_map_strict=[0, -1, 1, 2, 3, -1],
        pred_map_lenient=[0, 0, 1, 2, 3, 0],
        notes=[
            "Slough is assumed to correspond to ComplexWoundDB 'fibrinoid'; Necrotic to 'dry necrosis'.",
            "Hematoma pixels are ignored. Epithelial / Other have no counterpart.",
        ],
    )


def apply_policy(native: np.ndarray, space: LabelSpace, gt_policy: str, pred_policy: str) -> np.ndarray:
    """Turn ``N[gt_native, our_pred]`` into a common-space confusion of shape (C, C + 1)."""
    if gt_policy not in space.gt_map:
        raise ValueError(f"unknown gt policy '{gt_policy}' for {space.name}; options: {list(space.gt_map)}")
    if pred_policy not in ("strict", "lenient"):
        raise ValueError("pred_policy must be 'strict' or 'lenient'")
    gmap = space.gt_map[gt_policy]
    pmap = space.pred_map_strict if pred_policy == "strict" else space.pred_map_lenient
    n_gt, n_pred = native.shape
    if n_gt != len(gmap) or n_pred != len(pmap):
        raise ValueError(f"native confusion {native.shape} does not match mapping ({len(gmap)}, {len(pmap)})")
    c = len(space.class_names)
    out = np.zeros((c, c + 1), dtype=np.int64)
    for g in range(n_gt):
        gc = gmap[g]
        if gc == IGNORE:
            continue
        for p in range(n_pred):
            pc = pmap[p]
            out[gc, c if pc < 0 else pc] += native[g, p]
    return out


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------
def metrics_from_confusion(conf: np.ndarray, class_names: List[str]) -> dict:
    """``conf`` is (C, C) or (C, C + 1) (last column = unmapped predictions, always false for every class)."""
    conf = np.asarray(conf, dtype=np.int64)
    c = len(class_names)
    if conf.shape[0] != c or conf.shape[1] not in (c, c + 1):
        raise ValueError(f"confusion shape {conf.shape} does not fit {c} classes")
    tp = np.diag(conf[:, :c]).astype(np.float64)
    gt_total = conf.sum(axis=1).astype(np.float64)       # includes unmapped column
    pred_total = conf[:, :c].sum(axis=0).astype(np.float64)
    fn = gt_total - tp
    fp = pred_total - tp
    iou_den = tp + fp + fn
    dice_den = 2 * tp + fp + fn
    iou = np.where(iou_den > 0, tp / np.maximum(iou_den, 1), 0.0) * 100.0
    dice = np.where(dice_den > 0, 2 * tp / np.maximum(dice_den, 1), 0.0) * 100.0
    present = gt_total > 0

    # binary "wound vs background": any non-background class (including unmapped predictions) counts as wound
    gt_w = conf[1:, :].sum()
    pred_w = conf[:, 1:].sum()
    inter_w = conf[1:, 1:].sum()
    binary_dice = (2 * inter_w / (gt_w + pred_w) * 100.0) if (gt_w + pred_w) > 0 else 0.0
    binary_iou = (inter_w / (gt_w + pred_w - inter_w) * 100.0) if (gt_w + pred_w - inter_w) > 0 else 0.0

    return {
        "classes": {
            n: {"iou": float(iou[i]), "dice": float(dice[i]), "gt_pixels": int(gt_total[i]),
                "pred_pixels": int(pred_total[i])}
            for i, n in enumerate(class_names)
        },
        "mean_iou": float(iou.mean()),
        "mean_dice": float(dice.mean()),
        "tissue_mean_iou": float(iou[1:].mean()) if c > 1 else None,
        "tissue_mean_dice": float(dice[1:].mean()) if c > 1 else None,
        "mean_iou_present_classes": float(iou[present].mean()) if present.any() else None,
        "binary_wound_dice": float(binary_dice),
        "binary_wound_iou": float(binary_iou),
        "unmapped_pixels": int(conf[:, c].sum()) if conf.shape[1] == c + 1 else 0,
        "total_pixels": int(conf.sum()),
    }


def confusion_to_rows(conf: np.ndarray, class_names: List[str], with_unmapped: bool) -> List[List]:
    header = [""] + [f"pred_{n}" for n in class_names] + (["pred_unmapped"] if with_unmapped else [])
    rows = [header]
    for i, n in enumerate(class_names):
        rows.append([f"gt_{n}"] + [int(v) for v in conf[i]])
    return rows
