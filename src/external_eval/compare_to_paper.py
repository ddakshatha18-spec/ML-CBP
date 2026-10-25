"""Build the comparison table (ours vs the base paper) from the saved result files. No torch needed.

    PYTHONPATH=src python -m external_eval.compare_to_paper

Paper numbers are copied from method/README.md of github.com/carlosh93/TiSage (Table 1, "mIoU / Dice", where Dice is
reported as F1 for DFUTissue Fixed). The paper trains supervised models on all labelled images (DFUTissue Fixed = 94
TrainVal images, evaluated on the 16 test images; LUTSeg full = 111 images, evaluated on the 30 validation images) and
selects its best checkpoint on that evaluation set.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

PAPER_DFU = [  # name, mIoU, Dice  (DFUTissue Fixed)
    ("DeepLabV3+-R50 (paper)", 70.02, 81.12),
    ("DINOv2-DPT (paper)", 68.71, 80.21),
    ("FixMatch (paper, +600 unlabeled)", 68.91, 80.19),
    ("UniMatch-V2 (paper, +600 unlabeled)", 69.94, 80.96),
    ("TiSage (paper, +600 unlabeled)", 72.36, 83.05),
]
PAPER_LUTSEG = [("DeepLabV3+-R50 (paper)", 22.06, 25.61), ("DINOv2-DPT (paper)", 31.37, 39.19)]


def _ms(vals):
    vals = np.asarray(vals, dtype=float)
    return (vals.mean(), vals.std(ddof=1) if len(vals) > 1 else float("nan"), len(vals))


def _fmt(m, s, n):
    return f"{m:.2f} ± {s:.2f} (n={n})" if n > 1 else f"{m:.2f} (n=1)"


def dfu_rows(dfu_dir: Path):
    rows = []
    for protocol, label in (("clean", "Ours, clean protocol (train 78, epoch chosen on val 16, tested on test 16)"),
                            ("paper_style", "Ours, paper protocol (train 94, epoch chosen ON TEST: optimistic)")):
        for init in ("imagenet", "lutseg"):
            files = sorted(glob.glob(str(dfu_dir / f"{protocol}_{init}_s*" / "result.json")))
            files = [f for f in files if not f.endswith("_smoke/result.json") and "_smoke" not in f]
            if not files:
                continue
            res = [json.loads(Path(f).read_text()) for f in files]
            tag = "" if init == "imagenet" else " [init from LUTSeg checkpoint]"
            iou = _fmt(*_ms([r["test_best_epoch"]["mean_iou"] for r in res]))
            dice = _fmt(*_ms([r["test_best_epoch"]["mean_dice"] for r in res]))
            rows.append((label + tag + " - best epoch", iou, dice))
            iou = _fmt(*_ms([r["test_last_epoch"]["mean_iou"] for r in res]))
            dice = _fmt(*_ms([r["test_last_epoch"]["mean_dice"] for r in res]))
            rows.append((label + tag + " - last epoch (not selected)", iou, dice))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dfu-dir", default="outputs/external_eval/dfutissue")
    ap.add_argument("--lutseg-json", default="outputs/external_eval/lutseg_rescore/lutseg_rescore.json")
    ap.add_argument("--zero-shot-dir", default="outputs/external_eval/zero_shot")
    ap.add_argument("--output", default="outputs/external_eval/comparison.md")
    args = ap.parse_args()

    L = ["# Comparison with the base paper (TiSage)", "",
         "Research prototype. Small test sets (16 and 30 images); all numbers preliminary.", "",
         "## DFUTissue (4 classes: background, fibrin, granulation, callus), 16 test images", "",
         "| Model | mIoU | Dice |", "|---|---|---|"]
    for n, a, b in PAPER_DFU:
        L.append(f"| {n} | {a:.2f} | {b:.2f} |")
    rows = dfu_rows(Path(args.dfu_dir))
    for n, a, b in rows:
        L.append(f"| {n} | {a} | {b} |")
    if not rows:
        L.append("| (no DFUTissue training results found yet) | | |")

    L += ["", "## LUTSeg (6 classes incl. background), 30 validation images", "", "| Model | mIoU | Dice |", "|---|---|---|"]
    for n, a, b in PAPER_LUTSEG:
        L.append(f"| {n} | {a:.2f} | {b:.2f} |")
    p = Path(args.lutseg_json)
    if p.is_file():
        data = json.loads(p.read_text())
        for res, label in (("256", "Ours ResNet34 + weights, scored at 256 x 256"),
                           ("native", "Ours ResNet34 + weights, scored at native resolution (TiSage style)")):
            if res in data:
                s = data[res]["_summary"]
                iou = _fmt(s["mean_iou"], s["std_iou"] if s["std_iou"] is not None else float("nan"), s["n_checkpoints"])
                dice = _fmt(s["mean_dice"], s["std_dice"] if s["std_dice"] is not None else float("nan"), s["n_checkpoints"])
                L.append(f"| {label} | {iou} | {dice} |")
    else:
        L.append("| (run eval_lutseg_native first) | | |")

    L += ["", "## Zero-shot (LUTSeg model, no retraining) - NOT comparable with the paper", ""]
    zs = sorted(glob.glob(str(Path(args.zero_shot_dir) / "*.json")))
    zs = [z for z in zs if Path(z).name.startswith(("dfutissue", "cwdb"))]
    if zs:
        L += ["| Dataset | variant | mIoU | Dice | binary wound Dice |", "|---|---|---|---|---|"]
        for z in zs:
            r = json.loads(Path(z).read_text())
            for k, v in r["variants"].items():
                m = v["metrics"]
                L.append(f"| {r['name']} | {k} | {m['mean_iou']:.2f} | {m['mean_dice']:.2f} | {m['binary_wound_dice']:.2f} |")
    else:
        L.append("(no zero-shot results found yet)")

    L += ["", "## What is and is not comparable", "",
          "- Paper models are trained on all labelled training images and the best epoch is picked on the same split that is reported. "
          "Only our 'paper protocol' rows copy that; the 'clean' rows do not, so expect them to be lower.",
          "- Paper and ours both average over all classes including background; classes absent from the data count as 0.",
          "- Our model sees 256 x 256 inputs; the paper's sees native resolution.",
          "- Zero-shot numbers depend on the class mapping (Slough = fibrin/fibrinoid is an assumption) and on ComplexWoundDB's "
          "coarse, partly undocumented expert-4 labels.",
          "- Not clinically validated."]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
