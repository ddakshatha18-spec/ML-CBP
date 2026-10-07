import torch
from torch.utils.data import DataLoader

from dataset import LUTSegDataset
from model import UNet


DATASET_ROOT = "data/LUTSeg"
CHECKPOINT = "checkpoints/best_model.pth"

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 2
NUM_CLASSES = 6

CLASS_NAMES = [
    "Background",
    "Epithelial",
    "Slough",
    "Granulation",
    "Necrotic",
    "Other",
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# -----------------------------
# Dataset
# -----------------------------

dataset = LUTSegDataset(
    DATASET_ROOT,
    split="val",
    image_size=IMAGE_SIZE
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# -----------------------------
# Model
# -----------------------------

model = UNet(
    in_channels=3,
    num_classes=NUM_CLASSES
).to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("\n========================================")
print("Model Evaluation")
print("========================================")
print(f"Device: {DEVICE}")
print(f"Validation images: {len(dataset)}")
print("========================================\n")


# -----------------------------
# Accumulate confusion matrix
# -----------------------------

confusion_matrix = torch.zeros(
    NUM_CLASSES,
    NUM_CLASSES,
    dtype=torch.long
)

with torch.no_grad():

    for images, masks in loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        valid = masks != 255

        true_pixels = masks[valid].cpu()
        pred_pixels = predictions[valid].cpu()

        for true_class, pred_class in zip(
            true_pixels,
            pred_pixels
        ):
            confusion_matrix[
                true_class,
                pred_class
            ] += 1


# -----------------------------
# Calculate per-class metrics
# -----------------------------

print("Per-class results:\n")

dice_scores = []
iou_scores = []

for class_id in range(NUM_CLASSES):

    true_positive = confusion_matrix[
        class_id,
        class_id
    ].item()

    false_positive = (
        confusion_matrix[:, class_id].sum().item()
        - true_positive
    )

    false_negative = (
        confusion_matrix[class_id, :].sum().item()
        - true_positive
    )

    denominator_dice = (
        2 * true_positive
        + false_positive
        + false_negative
    )

    denominator_iou = (
        true_positive
        + false_positive
        + false_negative
    )

    if denominator_dice == 0:
        dice = 0.0
    else:
        dice = (
            2 * true_positive
            / denominator_dice
        )

    if denominator_iou == 0:
        iou = 0.0
    else:
        iou = (
            true_positive
            / denominator_iou
        )

    dice_scores.append(dice)
    iou_scores.append(iou)

    print(
        f"{CLASS_NAMES[class_id]:<15} "
        f"Dice: {dice:.4f}   "
        f"IoU: {iou:.4f}"
    )


# -----------------------------
# Overall metrics
# -----------------------------

mean_dice = sum(dice_scores) / NUM_CLASSES
mean_iou = sum(iou_scores) / NUM_CLASSES

print("\n========================================")
print("Overall Results")
print("========================================")

print(f"Mean Dice: {mean_dice:.4f}")
print(f"Mean IoU : {mean_iou:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix)

print("\nEvaluation complete!")