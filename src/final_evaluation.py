import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import LUTSegDataset
from model import UNet


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = "data/LUTSeg"
CHECKPOINT = "checkpoints/best_model.pth"
OUTPUT_DIR = "outputs/final_evaluation"

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 2
NUM_CLASSES = 6

CLASS_NAMES = [
    "Background",
    "Epithelial",
    "Slough",
    "Granulation",
    "Necrotic",
    "Other"
]

IGNORE_INDEX = 255


# ============================================================
# SETUP
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("FINAL WOUND SEGMENTATION EVALUATION")
print("=" * 60)
print(f"Device: {device}")
print()


# ============================================================
# LOAD DATASET
# ============================================================

val_dataset = LUTSegDataset(
    root_dir=DATASET_ROOT,
    split="val",
    image_size=IMAGE_SIZE,
    augment=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"Validation images: {len(val_dataset)}")


# ============================================================
# LOAD MODEL
# ============================================================

model = UNet(
    in_channels=3,
    num_classes=NUM_CLASSES
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)

# Handle both normal state_dict and checkpoint dictionaries
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

print("Model loaded.")
print()


# ============================================================
# STORAGE
# ============================================================

confusion_matrix = np.zeros(
    (NUM_CLASSES, NUM_CLASSES),
    dtype=np.int64
)

# Binary wound statistics
binary_intersection = 0
binary_prediction = 0
binary_target = 0

# Batch statistics
wound_area_percentages = []

tissue_percentages = {
    "Epithelial": [],
    "Slough": [],
    "Granulation": [],
    "Necrotic": [],
    "Other": []
}


# ============================================================
# EVALUATION
# ============================================================

with torch.no_grad():

    for batch_idx, (images, targets) in enumerate(val_loader):

        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        # ----------------------------------------------------
        # Process each image individually
        # ----------------------------------------------------

        for i in range(images.shape[0]):

            pred = predictions[i]
            target = targets[i]

            # Valid pixels
            valid = target != IGNORE_INDEX

            pred_valid = pred[valid]
            target_valid = target[valid]

            # ------------------------------------------------
            # Multi-class confusion matrix
            # ------------------------------------------------

            for true_class in range(NUM_CLASSES):

                true_pixels = target_valid == true_class

                if true_pixels.sum() == 0:
                    continue

                predicted_for_class = pred_valid[true_pixels]

                for predicted_class in range(NUM_CLASSES):

                    count = (
                        predicted_for_class == predicted_class
                    ).sum().item()

                    confusion_matrix[
                        true_class,
                        predicted_class
                    ] += count

            # ------------------------------------------------
            # Binary wound segmentation
            #
            # Classes 1-5 = wound
            # Class 0 = background
            # ------------------------------------------------

            predicted_wound = pred > 0
            target_wound = target > 0

            predicted_wound = predicted_wound & valid
            target_wound = target_wound & valid

            intersection = (
                predicted_wound & target_wound
            ).sum().item()

            binary_intersection += intersection
            binary_prediction += predicted_wound.sum().item()
            binary_target += target_wound.sum().item()

            # ------------------------------------------------
            # Wound area percentage
            # ------------------------------------------------

            valid_pixel_count = valid.sum().item()

            if valid_pixel_count > 0:

                wound_pixels = predicted_wound.sum().item()

                wound_percentage = (
                    wound_pixels /
                    valid_pixel_count
                ) * 100

                wound_area_percentages.append(
                    wound_percentage
                )

                # --------------------------------------------
                # Tissue percentages inside predicted wound
                # --------------------------------------------

                for class_id, class_name in zip(
                    range(1, NUM_CLASSES),
                    CLASS_NAMES[1:]
                ):

                    class_pixels = (
                        (pred == class_id) &
                        predicted_wound
                    ).sum().item()

                    if wound_pixels > 0:
                        percentage = (
                            class_pixels /
                            wound_pixels
                        ) * 100
                    else:
                        percentage = 0.0

                    tissue_percentages[class_name].append(
                        percentage
                    )

        print(
            f"Processed: "
            f"{min((batch_idx + 1) * BATCH_SIZE, len(val_dataset))}"
            f"/{len(val_dataset)}"
        )


# ============================================================
# BINARY WOUND METRICS
# ============================================================

dice_denominator = (
    binary_prediction +
    binary_target
)

iou_denominator = (
    binary_prediction +
    binary_target -
    binary_intersection
)

if dice_denominator > 0:
    wound_dice = (
        2 * binary_intersection
    ) / dice_denominator
else:
    wound_dice = 0.0

if iou_denominator > 0:
    wound_iou = (
        binary_intersection /
        iou_denominator
    )
else:
    wound_iou = 0.0


# ============================================================
# PER-CLASS DICE AND IOU
# ============================================================

class_metrics = {}

dice_values = []
iou_values = []

for class_id, class_name in enumerate(CLASS_NAMES):

    true_positive = confusion_matrix[
        class_id,
        class_id
    ]

    false_positive = (
        confusion_matrix[:, class_id].sum()
        - true_positive
    )

    false_negative = (
        confusion_matrix[class_id, :].sum()
        - true_positive
    )

    dice_denominator = (
        2 * true_positive
        + false_positive
        + false_negative
    )

    iou_denominator = (
        true_positive
        + false_positive
        + false_negative
    )

    if dice_denominator > 0:
        dice = (
            2 * true_positive
        ) / dice_denominator
    else:
        dice = 0.0

    if iou_denominator > 0:
        iou = (
            true_positive /
            iou_denominator
        )
    else:
        iou = 0.0

    class_metrics[class_name] = {
        "dice": float(dice),
        "iou": float(iou)
    }

    dice_values.append(dice)
    iou_values.append(iou)


mean_dice = np.mean(dice_values)
mean_iou = np.mean(iou_values)


# ============================================================
# BATCH STATISTICS
# ============================================================

average_wound_area = np.mean(
    wound_area_percentages
)

average_tissue_percentages = {}

for class_name, values in tissue_percentages.items():

    average_tissue_percentages[class_name] = (
        float(np.mean(values))
        if values
        else 0.0
    )


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print()
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print()
print("OVERALL WOUND SEGMENTATION")
print("-" * 40)
print(f"Wound Dice : {wound_dice:.4f}")
print(f"Wound IoU  : {wound_iou:.4f}")

print()
print("SIX-CLASS TISSUE SEGMENTATION")
print("-" * 40)

for class_name in CLASS_NAMES:

    dice = class_metrics[class_name]["dice"]
    iou = class_metrics[class_name]["iou"]

    print(
        f"{class_name:<15} "
        f"Dice: {dice:.4f}    "
        f"IoU: {iou:.4f}"
    )

print()
print(f"Mean Dice: {mean_dice:.4f}")
print(f"Mean IoU : {mean_iou:.4f}")

print()
print("AVERAGE PREDICTED WOUND COMPOSITION")
print("-" * 40)

print(
    f"Wound area: "
    f"{average_wound_area:.2f}%"
)

for class_name, percentage in average_tissue_percentages.items():

    print(
        f"{class_name}: "
        f"{percentage:.2f}%"
    )


# ============================================================
# PRINT CONFUSION MATRIX
# ============================================================

print()
print("CONFUSION MATRIX")
print("-" * 40)

print("Rows = Ground Truth")
print("Columns = Prediction")
print()

print(
    f"{'':15}",
    end=""
)

for class_name in CLASS_NAMES:
    print(
        f"{class_name[:10]:>12}",
        end=""
    )

print()

for i, class_name in enumerate(CLASS_NAMES):

    print(
        f"{class_name[:15]:15}",
        end=""
    )

    for j in range(NUM_CLASSES):

        print(
            f"{confusion_matrix[i, j]:>12}",
            end=""
        )

    print()


# ============================================================
# SAVE JSON REPORT
# ============================================================

report = {
    "model": {
        "architecture": "U-Net",
        "checkpoint": CHECKPOINT,
        "image_size": list(IMAGE_SIZE),
        "num_classes": NUM_CLASSES
    },

    "dataset": {
        "name": "LUTSeg",
        "split": "validation",
        "validation_images": len(val_dataset)
    },

    "binary_wound_segmentation": {
        "dice": float(wound_dice),
        "iou": float(wound_iou)
    },

    "tissue_segmentation": {
        "classes": class_metrics,
        "mean_dice": float(mean_dice),
        "mean_iou": float(mean_iou)
    },

    "batch_analysis": {
        "average_wound_area_percent": float(
            average_wound_area
        ),
        "average_tissue_percentages": (
            average_tissue_percentages
        )
    },

    "confusion_matrix": (
        confusion_matrix.tolist()
    ),

    "limitations": [
        "Evaluation uses 30 validation images.",
        "Wound area is reported as percentage of image pixels, not physical area in cm2.",
        "Fine-grained tissue segmentation performs weaker than overall wound localization.",
        "Results are specific to the LUTSeg validation dataset.",
        "This evaluation does not constitute clinical validation."
    ]
}

report_path = os.path.join(
    OUTPUT_DIR,
    "final_evaluation_report.json"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=4
    )


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

confusion_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.csv"
)

with open(
    confusion_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        ",".join(CLASS_NAMES)
        + "\n"
    )

    for row in confusion_matrix:

        f.write(
            ",".join(
                str(value)
                for value in row
            )
            + "\n"
        )


# ============================================================
# SAVE CLASS METRICS CSV
# ============================================================

metrics_path = os.path.join(
    OUTPUT_DIR,
    "class_metrics.csv"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as f:

    f.write("Class,Dice,IoU\n")

    for class_name in CLASS_NAMES:

        dice = class_metrics[class_name]["dice"]
        iou = class_metrics[class_name]["iou"]

        f.write(
            f"{class_name},{dice:.6f},{iou:.6f}\n"
        )


# ============================================================
# DONE
# ============================================================

print()
print("=" * 60)
print("FINAL EVALUATION COMPLETE")
print("=" * 60)

print()
print("Generated files:")

print(
    f"1. {report_path}"
)

print(
    f"2. {confusion_path}"
)

print(
    f"3. {metrics_path}"
)

print()