"""Train and test the final recipe on DFUTissue (4 classes: background, fibrin, granulation, callus).

    PYTHONPATH=src python -m external_eval.train_dfutissue --protocol clean --seed 0

Recipe = the LUTSeg recipe: smp.Unet(ResNet34, ImageNet encoder), CE + Dice loss, Adam lr 1e-3, batch 2,
30 epochs, 384 x 384 training crops (flip / rotation / brightness-contrast), prediction at 256 x 256, metrics on the
image's own pixel grid. Nothing is tuned on this dataset.

Protocols
  clean        train on the 78 train images, choose the epoch on the 16 val images, report on the 16 test images.
               No test image influences training or epoch choice.  <- headline number
  paper_style  train on the 94 TrainVal images, choose the epoch by best score on the 16 TEST images, which is what
               TiSage's supervised.py does ("best-validation checkpoint", its validation set is the test split).
               The best-epoch test score is therefore OPTIMISTIC; the last-epoch score is reported next to it.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from losses import CombinedLoss  # src/losses.py (needs PYTHONPATH=src)

from .datasets import DFU_CLASSES, dfutissue_items, load_dfutissue_mask, load_image_rgb
from .evaluate import evaluate_native
from .mapping import metrics_from_confusion
from .model_utils import Predictor, build_unet, init_from_lutseg, pick_device

NUM_CLASSES = len(DFU_CLASSES)


class DFUTrainDataset(Dataset):
    """Same augmentation as src/dataset.py (LUTSegDataset), applied to DFUTissue items."""

    def __init__(self, items, size=384, augment=True):
        self.items, self.size, self.augment = items, size, augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        image = load_image_rgb(it.image_path)
        mask = Image.fromarray(load_dfutissue_mask(it.mask_path))
        image = TF.resize(image, (self.size, self.size), interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (self.size, self.size), interpolation=InterpolationMode.NEAREST)
        if self.augment:
            if random.random() < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            if random.random() < 0.2:
                image, mask = TF.vflip(image), TF.vflip(mask)
            if random.random() < 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
                mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=255)
            if random.random() < 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.9, 1.1))
                image = TF.adjust_contrast(image, random.uniform(0.9, 1.1))
        tensor = TF.normalize(TF.to_tensor(image), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return tensor, torch.from_numpy(np.array(mask)).long()


def class_weights_isqrt(items) -> torch.Tensor:
    """Inverse-sqrt pixel frequency, normalised to mean 1 (not capped, not tuned). Off by default."""
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for it in items:
        counts += np.bincount(load_dfutissue_mask(it.mask_path).ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]
    w = 1.0 / np.sqrt(np.maximum(counts / counts.sum(), 1e-8))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def evaluate_items(model, device, items, infer_size):
    """Metrics on the images' own pixel grid (identity label mapping: 4 predicted classes = 4 dataset classes)."""
    native, _ = evaluate_native(items, lambda it: load_dfutissue_mask(it.mask_path),
                                Predictor(model, device, infer_size), NUM_CLASSES, NUM_CLASSES)
    return metrics_from_confusion(native, DFU_CLASSES), native


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", choices=["clean", "paper_style"], default="clean")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dfu-root", default="external/DFUTissueSegNet/DFUTissue")
    ap.add_argument("--init", choices=["imagenet", "lutseg"], default="imagenet",
                    help="imagenet = same start as the LUTSeg recipe; lutseg = fine-tune from the LUTSeg checkpoint (extra experiment)")
    ap.add_argument("--init-checkpoint", default="checkpoints/best_model_resnet34_weighted.pth")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-size", type=int, default=384)
    ap.add_argument("--infer-size", type=int, default=256)
    ap.add_argument("--class-weights", choices=["none", "isqrt"], default="none")
    ap.add_argument("--select-metric", choices=["dice", "iou"], default="dice")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output", default="outputs/external_eval/dfutissue")
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    ap.add_argument("--smoke", action="store_true", help="1 epoch, 4 training images, 4 evaluation images")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    if args.protocol == "clean":
        train_items = dfutissue_items(args.dfu_root, "train")
        select_items = dfutissue_items(args.dfu_root, "val")
        selected_on_test = False
    else:
        train_items = dfutissue_items(args.dfu_root, "trainval")
        select_items = dfutissue_items(args.dfu_root, "test")
        selected_on_test = True
    test_items = dfutissue_items(args.dfu_root, "test")
    epochs = args.epochs
    if args.smoke:
        train_items, select_items, test_items, epochs = train_items[:4], select_items[:4], test_items[:4], 1

    run_name = f"{args.protocol}_{args.init}_s{args.seed}" + ("_smoke" if args.smoke else "")
    out_dir = Path(args.output) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    model = build_unet(NUM_CLASSES, imagenet=(args.init == "imagenet"))
    if args.init == "lutseg":
        init_from_lutseg(model, args.init_checkpoint)
    model = model.to(device)

    weights = class_weights_isqrt(train_items).to(device) if args.class_weights == "isqrt" else None
    criterion = CombinedLoss(num_classes=NUM_CLASSES, class_weights=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(DFUTrainDataset(train_items, args.train_size, True), batch_size=args.batch_size,
                        shuffle=True, num_workers=0, generator=gen)

    print(f"run: {run_name} | device {device} | train {len(train_items)} | select on {len(select_items)} "
          f"({'TEST set - optimistic' if selected_on_test else 'val set'}) | report on {len(test_items)} test")

    key = "mean_dice" if args.select_metric == "dice" else "mean_iou"
    best_score, best_epoch, best_state, history = -1.0, 0, None, []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            total += loss.item()
        model.eval()
        m, _ = evaluate_items(model, device, select_items, args.infer_size)
        history.append({"epoch": epoch, "train_loss": total / len(loader), "select_mean_iou": m["mean_iou"],
                        "select_mean_dice": m["mean_dice"]})
        print(f"epoch {epoch:>2}/{epochs}  loss {total / len(loader):.4f}  select mIoU {m['mean_iou']:.2f}  "
              f"mean Dice {m['mean_dice']:.2f}  ({time.time() - t0:.0f}s)")
        if m[key] > best_score:
            best_score, best_epoch = m[key], epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

    last_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    results = {}
    for label, state in (("best", best_state), ("last", last_state)):
        model.load_state_dict(state)
        model.to(device).eval()
        tm, conf = evaluate_items(model, device, test_items, args.infer_size)
        results[f"test_{label}_epoch"] = tm
        results[f"test_{label}_epoch_confusion"] = conf.tolist()

    result = {
        "run": run_name, "protocol": args.protocol, "seed": args.seed, "init": args.init,
        "selected_on_test": selected_on_test, "best_epoch": best_epoch, "epochs": epochs,
        "train_images": len(train_items), "select_images": len(select_items), "test_images": len(test_items),
        "classes": DFU_CLASSES, "args": vars(args), "device": str(device), **results,
        "notes": [
            "Metrics are percentages, pooled over all test images at each image's own resolution (TiSage-style); "
            "means include background.",
            "paper_style selects the epoch on the test set, so test_best_epoch is optimistic; test_last_epoch is not selected.",
            "Research prototype; 16 test images, so results are preliminary.",
        ],
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    with (out_dir / "history.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    with (out_dir / "class_metrics_test_best.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "iou", "dice", "gt_pixels"])
        for n in DFU_CLASSES:
            c = results["test_best_epoch"]["classes"][n]
            w.writerow([n, f"{c['iou']:.2f}", f"{c['dice']:.2f}", c["gt_pixels"]])
    torch.save({"model_state_dict": best_state, "meta": {"run": run_name, "best_epoch": best_epoch,
                                                         "classes": DFU_CLASSES}},
               Path(args.checkpoint_dir) / f"dfutissue_{run_name}.pth")

    b, l = results["test_best_epoch"], results["test_last_epoch"]
    print("\n" + "=" * 60)
    print(f"{run_name}: best epoch {best_epoch}{' (chosen on TEST - optimistic)' if selected_on_test else ''}")
    print(f"  TEST best-epoch  mIoU {b['mean_iou']:.2f}  Dice {b['mean_dice']:.2f}")
    print(f"  TEST last-epoch  mIoU {l['mean_iou']:.2f}  Dice {l['mean_dice']:.2f}   (not selected)")
    for n in DFU_CLASSES:
        c = b["classes"][n]
        print(f"    {n:<12} IoU {c['iou']:>6.2f}  Dice {c['dice']:>6.2f}")
    print(f"Saved: {out_dir}/ and checkpoints/dfutissue_{run_name}.pth")


if __name__ == "__main__":
    main()
