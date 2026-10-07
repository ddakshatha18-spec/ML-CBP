import torch
from torch.utils.data import DataLoader

from dataset import LUTSegDataset
from model import UNet


DATASET_ROOT = "data/LUTSeg"
CHECKPOINT = "checkpoints/best_model.pth"

IMAGE_SIZE = (384, 384)
BATCH_SIZE = 2
NUM_CLASSES = 6

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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

intersection = 0
predicted_pixels = 0
target_pixels = 0

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

        # Classes 1-5 = wound
        target_wound = (masks > 0) & valid
        predicted_wound = (predictions > 0) & valid

        intersection += (
            target_wound & predicted_wound
        ).sum().item()

        predicted_pixels += (
            predicted_wound
        ).sum().item()

        target_pixels += (
            target_wound
        ).sum().item()


dice = (
    2 * intersection
    / (predicted_pixels + target_pixels + 1e-6)
)

iou = (
    intersection
    / (
        predicted_pixels
        + target_pixels
        - intersection
        + 1e-6
    )
)

print("\n========================================")
print("Binary Wound Segmentation Evaluation")
print("========================================")
print(f"Device: {DEVICE}")
print(f"Validation images: {len(dataset)}")
print("----------------------------------------")
print(f"Wound Dice: {dice:.4f}")
print(f"Wound IoU : {iou:.4f}")
print("========================================")