import os
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LUTSegDataset
from model import UNet
from losses import CombinedLoss
from metrics import dice_score, iou_score


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_ROOT = "data/LUTSeg"
CHECKPOINT_DIR = "checkpoints"

IMAGE_SIZE = (384, 384)
BATCH_SIZE = 2
EPOCHS = 30
LEARNING_RATE = 1e-3

NUM_CLASSES = 6

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

train_dataset = LUTSegDataset(
    DATASET_ROOT,
    split="train",
    image_size=IMAGE_SIZE,
    augment=True
)

val_dataset = LUTSegDataset(
    DATASET_ROOT,
    split="val",
    image_size=IMAGE_SIZE
)


# --------------------------------------------------
# DataLoaders
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = UNet(
    in_channels=3,
    num_classes=NUM_CLASSES
).to(DEVICE)


# --------------------------------------------------
# Loss and optimizer
# --------------------------------------------------

# Capped inverse-square-root class weights, normalized to mean 1.
class_weights = torch.tensor(
    [0.0815, 0.3712, 0.4980, 0.3763, 1.1845, 3.4885],
    dtype=torch.float32,
    device=DEVICE,
)

criterion = CombinedLoss(
    num_classes=NUM_CLASSES,
    class_weights=class_weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# Training
# --------------------------------------------------

best_dice = 0.0

print("\n========================================")
print("Starting training")
print("========================================")
print(f"Device: {DEVICE}")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print("========================================\n")


for epoch in range(EPOCHS):

    # ------------------------------
    # Training
    # ------------------------------

    model.train()

    train_loss = 0.0

    train_progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]"
    )

    for images, masks in train_progress:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, masks)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        train_progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    train_loss /= len(train_loader)


    # ------------------------------
    # Validation
    # ------------------------------

    model.eval()

    val_loss = 0.0
    val_dice = 0.0
    val_iou = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, masks)

            val_loss += loss.item()

            val_dice += dice_score(
                outputs,
                masks,
                num_classes=NUM_CLASSES
            )

            val_iou += iou_score(
                outputs,
                masks,
                num_classes=NUM_CLASSES
            )

    val_loss /= len(val_loader)
    val_dice /= len(val_loader)
    val_iou /= len(val_loader)


    # ------------------------------
    # Print results
    # ------------------------------

    print()
    print("----------------------------------------")
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Train Loss : {train_loss:.4f}")
    print(f"Val Loss   : {val_loss:.4f}")
    print(f"Val Dice   : {val_dice:.4f}")
    print(f"Val IoU    : {val_iou:.4f}")
    print("----------------------------------------")


    # ------------------------------
    # Save best model
    # ------------------------------

    if val_dice > best_dice:

        best_dice = val_dice

        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            "best_model_weighted.pth"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_dice,
                "val_iou": val_iou,
            },
            checkpoint_path
        )

        print(
            f"✓ New best model saved! "
            f"Dice: {best_dice:.4f}"
        )


print("\n========================================")
print("Training complete!")
print(f"Best validation Dice: {best_dice:.4f}")
print("========================================")