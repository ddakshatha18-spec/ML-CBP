import os

import matplotlib.pyplot as plt
import torch

from dataset import LUTSegDataset
from model import UNet


# -----------------------------
# Configuration
# -----------------------------

DATASET_ROOT = "data/LUTSeg"
CHECKPOINT = "checkpoints/best_model.pth"
OUTPUT_DIR = "outputs/predictions"

IMAGE_SIZE = (256, 256)
NUM_CLASSES = 6

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Load validation dataset
# -----------------------------

dataset = LUTSegDataset(
    DATASET_ROOT,
    split="val",
    image_size=IMAGE_SIZE
)


# -----------------------------
# Load model
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

print(f"Loaded model from: {CHECKPOINT}")
print(f"Device: {DEVICE}")


# -----------------------------
# Class names
# -----------------------------

class_names = [
    "Background",
    "Epithelial",
    "Slough",
    "Granulation",
    "Necrotic",
]


# -----------------------------
# Generate predictions
# -----------------------------

with torch.no_grad():

    for index in range(min(5, len(dataset))):

        image, mask = dataset[index]

        input_image = image.unsqueeze(0).to(DEVICE)

        output = model(input_image)

        prediction = torch.argmax(
            output,
            dim=1
        ).squeeze(0).cpu().numpy()

        # Convert normalized image back for display
        display_image = image.permute(1, 2, 0).numpy()

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        for channel in range(3):
            display_image[:, :, channel] = (
                display_image[:, :, channel] * std[channel]
                + mean[channel]
            )

        display_image = display_image.clip(0, 1)

        ground_truth = mask.numpy()

        # -----------------------------
        # Display
        # -----------------------------

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(display_image)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(ground_truth, vmin=0, vmax=5)
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(prediction, vmin=0, vmax=5)
        plt.title("U-Net Prediction")
        plt.axis("off")

        plt.tight_layout()

        output_path = os.path.join(
            OUTPUT_DIR,
            f"prediction_{index + 1}.png"
        )

        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.show()
        plt.close()

        print(
            f"Saved prediction {index + 1}: {output_path}"
        )

print("\nPrediction complete!")