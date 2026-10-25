"""Zero-shot test of the LUTSeg-trained model on DFUTissue and ComplexWoundDB.

    PYTHONPATH=src python -m external_eval.zero_shot --device cpu

No retraining. The label sets differ, so metrics are computed in a common class space (see mapping.py) and
every policy variant is printed and saved. Use --smoke for a 3-image check first.
"""

from __future__ import annotations

import argparse
import time

from . import OUR_CLASSES
from .datasets import (CWDB_CLASSES, DFU_CLASSES, cwdb_items, dfutissue_items, load_cwdb_mask,
                       load_dfutissue_mask)
from .evaluate import evaluate_native, print_variants, write_zero_shot_report
from .mapping import cwdb_common4, dfutissue_common3
from .model_utils import Predictor, load_lutseg_model, pick_device


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/best_model_resnet34_weighted.pth")
    ap.add_argument("--device", default="auto", help="auto | cpu | mps | cuda (cpu = most reproducible)")
    ap.add_argument("--datasets", nargs="+", default=["dfutissue", "cwdb"], choices=["dfutissue", "cwdb"])
    ap.add_argument("--dfu-root", default="external/DFUTissueSegNet/DFUTissue")
    ap.add_argument("--dfu-split", default="test", choices=["test", "val", "train", "trainval", "all"])
    ap.add_argument("--dfu-version", default="Original", choices=["Original", "Padded"])
    ap.add_argument("--cwdb-root", default="external/recogna-datasets/ComplexWoundDB")
    ap.add_argument("--expert", type=int, default=4, choices=[1, 2, 3, 4],
                    help="4 is the expert the dataset authors used as ground truth (experts 1-3 are blended colours)")
    ap.add_argument("--size", type=int, default=256, help="model input size (same as all earlier results)")
    ap.add_argument("--output", default="outputs/external_eval/zero_shot")
    ap.add_argument("--smoke", action="store_true", help="only 3 images per dataset, to check that everything runs")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"device: {device} | checkpoint: {args.checkpoint}")
    model = load_lutseg_model(args.checkpoint, device)
    predict = Predictor(model, device, args.size)

    if "dfutissue" in args.datasets:
        items = dfutissue_items(args.dfu_root, args.dfu_split, args.dfu_version)
        items = items[:3] if args.smoke else items
        t0 = time.time()
        native, per_image = evaluate_native(items, lambda it: load_dfutissue_mask(it.mask_path), predict,
                                            n_gt=len(DFU_CLASSES), n_pred=len(OUR_CLASSES))
        rep = write_zero_shot_report(args.output, f"dfutissue_{args.dfu_split}_{args.dfu_version}", native,
                                     DFU_CLASSES, OUR_CLASSES, dfutissue_common3(), per_image,
                                     {"checkpoint": args.checkpoint, "device": str(device), "size": args.size,
                                      "seconds": round(time.time() - t0, 1)})
        print_variants(rep)

    if "cwdb" in args.datasets:
        items = cwdb_items(args.cwdb_root, args.expert)
        items = items[:3] if args.smoke else items
        for y2 in ("fibrinoid", "ignore"):  # the second yellow of expert 4 is undocumented: measure both ways
            t0 = time.time()
            native, per_image = evaluate_native(items, lambda it, y2=y2: load_cwdb_mask(it.mask_path, yellow2=y2),
                                                predict, n_gt=len(CWDB_CLASSES), n_pred=len(OUR_CLASSES))
            rep = write_zero_shot_report(args.output, f"cwdb_expert{args.expert}_yellow2-{y2}", native,
                                         CWDB_CLASSES, OUR_CLASSES, cwdb_common4(), per_image,
                                         {"checkpoint": args.checkpoint, "device": str(device), "size": args.size,
                                          "yellow2": y2, "expert": args.expert,
                                          "seconds": round(time.time() - t0, 1)})
            print_variants(rep)

    print(f"\nSaved JSON/CSV files in {args.output}/")
    print("Reminder: zero-shot numbers depend on the class mapping and are not comparable with trained models.")


if __name__ == "__main__":
    main()
