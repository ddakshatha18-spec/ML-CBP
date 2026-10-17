import json
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torchvision.transforms import functional as TF

from model import UNet


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT = "checkpoints/best_model.pth"

IMAGE_SIZE = (256, 256)
NUM_CLASSES = 6

OUTPUT_DIR = "outputs/wound_analysis"

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


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

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

    print(f"Model loaded from: {CHECKPOINT}")
    print(f"Device: {DEVICE}")

    return model


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image_path):

    image = Image.open(image_path).convert("RGB")

    original_image = image.copy()

    image = TF.resize(
        image,
        IMAGE_SIZE,
        interpolation=TF.InterpolationMode.BILINEAR
    )

    image_tensor = TF.to_tensor(image)

    image_tensor = TF.normalize(
        image_tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    image_tensor = image_tensor.unsqueeze(0)

    return original_image, image_tensor


# ============================================================
# RUN SEGMENTATION
# ============================================================

def predict(model, image_tensor):

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        prediction = torch.argmax(
            outputs,
            dim=1
        )

    prediction = prediction.squeeze(0).cpu().numpy()

    return prediction


# ============================================================
# CALCULATE WOUND / TISSUE MEASUREMENTS
# ============================================================

def calculate_measurements(prediction):

    total_pixels = prediction.size

    # Classes 1-5 represent wound tissue
    wound_mask = prediction > 0

    wound_pixels = np.sum(wound_mask)

    wound_percentage = (
        wound_pixels / total_pixels
    ) * 100

    measurements = {
        "total_pixels": int(total_pixels),
        "wound_pixels": int(wound_pixels),
        "wound_percentage": float(wound_percentage),
    }

    tissue_classes = {
        1: "epithelial",
        2: "slough",
        3: "granulation",
        4: "necrotic",
        5: "other",
    }

    for class_id, class_name in tissue_classes.items():

        class_pixels = np.sum(
            prediction == class_id
        )

        measurements[
            f"{class_name}_pixels"
        ] = int(class_pixels)

        if wound_pixels > 0:

            percentage = (
                class_pixels / wound_pixels
            ) * 100

        else:

            percentage = 0.0

        measurements[
            f"{class_name}_percentage"
        ] = float(percentage)

    return measurements


# ============================================================
# CREATE VISUALIZATION
# ============================================================

def create_visualization(
    original_image,
    prediction,
    measurements,
    output_path
):

    image = np.array(original_image)

    # Resize prediction back to original image size
    prediction_image = Image.fromarray(
        prediction.astype(np.uint8)
    )

    prediction_image = prediction_image.resize(
        original_image.size,
        Image.Resampling.NEAREST
    )

    prediction = np.array(
        prediction_image
    )

    wound_mask = prediction > 0

    # Create simple binary wound visualization
    wound_visual = np.zeros_like(image)

    wound_visual[wound_mask] = [255, 255, 255]

    # Create tissue visualization
    tissue_visual = np.zeros_like(image)

    # Simple fixed visualization colors
    tissue_visual[prediction == 1] = [255, 255, 0]
    tissue_visual[prediction == 2] = [255, 0, 0]
    tissue_visual[prediction == 3] = [0, 255, 0]
    tissue_visual[prediction == 4] = [0, 0, 255]
    tissue_visual[prediction == 5] = [255, 0, 255]

    # Overlay
    overlay = image.copy()

    tissue_mask = prediction > 0

    overlay[tissue_mask] = (
        0.55 * image[tissue_mask]
        + 0.45 * tissue_visual[tissue_mask]
    ).astype(np.uint8)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(18, 5)
    )

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(wound_visual)
    axes[1].set_title("Wound Mask")
    axes[1].axis("off")

    axes[2].imshow(tissue_visual)
    axes[2].set_title("Tissue Segmentation")
    axes[2].axis("off")

    axes[3].imshow(overlay)
    axes[3].set_title("Tissue Overlay")
    axes[3].axis("off")

    text = (
        f"Wound: {measurements['wound_percentage']:.2f}%\n"
        f"Epithelial: {measurements['epithelial_percentage']:.2f}%\n"
        f"Slough: {measurements['slough_percentage']:.2f}%\n"
        f"Granulation: {measurements['granulation_percentage']:.2f}%\n"
        f"Necrotic: {measurements['necrotic_percentage']:.2f}%\n"
        f"Other: {measurements['other_percentage']:.2f}%"
    )

    fig.text(
        0.5,
        0.01,
        text,
        ha="center",
        fontsize=10
    )

    plt.tight_layout(
        rect=[0, 0.08, 1, 1]
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) < 2:

        print(
            "\nUsage:"
            "\npython src/wound_analysis.py "
            "<image_path>\n"
        )

        print(
            "Example:"
            "\npython src/wound_analysis.py "
            "data/LUTSeg/Images/Patient_1/P1_T1.jpeg"
        )

        return

    image_path = sys.argv[1]

    if not os.path.exists(image_path):

        print(
            f"\nERROR: Image not found:\n{image_path}"
        )

        return

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("\n========================================")
    print("WOUND IMAGE ANALYSIS")
    print("========================================")

    model = load_model()

    print("\nLoading image...")

    original_image, image_tensor = preprocess_image(
        image_path
    )

    print(
        f"Original image size: "
        f"{original_image.size}"
    )

    print("\nRunning segmentation...")

    prediction = predict(
        model,
        image_tensor
    )

    print("Segmentation complete.")

    print("\nCalculating measurements...")

    measurements = calculate_measurements(
        prediction
    )

    print("\n========================================")
    print("WOUND ANALYSIS RESULTS")
    print("========================================")

    print(
        f"Wound area: "
        f"{measurements['wound_percentage']:.2f}%"
    )

    print(
        f"Epithelial: "
        f"{measurements['epithelial_percentage']:.2f}%"
    )

    print(
        f"Slough: "
        f"{measurements['slough_percentage']:.2f}%"
    )

    print(
        f"Granulation: "
        f"{measurements['granulation_percentage']:.2f}%"
    )

    print(
        f"Necrotic: "
        f"{measurements['necrotic_percentage']:.2f}%"
    )

    print(
        f"Other: "
        f"{measurements['other_percentage']:.2f}%"
    )

    base_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    visualization_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_analysis.png"
    )

    json_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_metrics.json"
    )

    print("\nCreating visualization...")

    create_visualization(
        original_image,
        prediction,
        measurements,
        visualization_path
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            measurements,
            f,
            indent=4
        )

    print("\n========================================")
    print("Analysis complete!")
    print("========================================")

    print(
        f"Visualization:\n"
        f"{visualization_path}"
    )

    print(
        f"\nMetrics:\n"
        f"{json_path}"
    )


if __name__ == "__main__":
    main()