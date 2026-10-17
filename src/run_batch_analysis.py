import csv
import json
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from model import UNet


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path("data/LUTSeg")
CHECKPOINT = "checkpoints/best_model.pth"

IMAGE_SIZE = (256, 256)
NUM_CLASSES = 6

OUTPUT_DIR = Path("outputs/wound_analysis")

CLASS_NAMES = {
    1: "epithelial",
    2: "slough",
    3: "granulation",
    4: "necrotic",
    5: "other",
}

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

    return model


# ============================================================
# PREPROCESS
# ============================================================

def preprocess_image(image):

    image = TF.resize(
        image,
        IMAGE_SIZE,
        interpolation=TF.InterpolationMode.BILINEAR
    )

    tensor = TF.to_tensor(image)

    tensor = TF.normalize(
        tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    return tensor.unsqueeze(0)


# ============================================================
# PREDICT
# ============================================================

def predict(model, image):

    tensor = preprocess_image(image).to(DEVICE)

    with torch.no_grad():

        output = model(tensor)

        prediction = torch.argmax(
            output,
            dim=1
        )

    return prediction.squeeze(0).cpu().numpy()


# ============================================================
# CALCULATE MEASUREMENTS
# ============================================================

def calculate_measurements(prediction):

    total_pixels = prediction.size

    wound_mask = prediction > 0

    wound_pixels = int(
        wound_mask.sum()
    )

    wound_percentage = (
        wound_pixels / total_pixels
    ) * 100

    results = {
        "wound_pixels": wound_pixels,
        "wound_percentage": wound_percentage,
    }

    for class_id, class_name in CLASS_NAMES.items():

        class_pixels = int(
            (prediction == class_id).sum()
        )

        if wound_pixels > 0:

            percentage = (
                class_pixels / wound_pixels
            ) * 100

        else:

            percentage = 0.0

        results[
            f"{class_name}_pixels"
        ] = class_pixels

        results[
            f"{class_name}_percentage"
        ] = percentage

    return results


# ============================================================
# READ VALIDATION IMAGES
# ============================================================

def get_validation_images():

    split_file = DATASET_ROOT / "val.txt"

    samples = []

    for line in split_file.read_text().splitlines():

        line = line.strip()

        if not line:
            continue

        image_path, _ = line.split()

        samples.append(
            DATASET_ROOT / image_path
        )

    return samples


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n========================================")
    print("BATCH WOUND ANALYSIS")
    print("========================================")

    print(f"Device: {DEVICE}")

    model = load_model()

    print("Model loaded.")

    image_paths = get_validation_images()

    print(
        f"Validation images: "
        f"{len(image_paths)}"
    )

    all_results = []

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        print(
            f"\n[{index}/{len(image_paths)}] "
            f"{image_path.name}"
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        prediction = predict(
            model,
            image
        )

        measurements = calculate_measurements(
            prediction
        )

        result = {
            "image": image_path.name,
            **measurements
        }

        all_results.append(result)

        print(
            f"Wound: "
            f"{measurements['wound_percentage']:.2f}%"
        )

        print(
            f"Granulation: "
            f"{measurements['granulation_percentage']:.2f}%"
        )

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    json_path = (
        OUTPUT_DIR /
        "batch_results.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_results,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    csv_path = (
        OUTPUT_DIR /
        "batch_results.csv"
    )

    fieldnames = list(
        all_results[0].keys()
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            all_results
        )

    # --------------------------------------------------------
    # CALCULATE AVERAGES
    # --------------------------------------------------------

    numeric_fields = [
        "wound_percentage",
        "epithelial_percentage",
        "slough_percentage",
        "granulation_percentage",
        "necrotic_percentage",
        "other_percentage",
    ]

    averages = {}

    for field in numeric_fields:

        values = [
            result[field]
            for result in all_results
        ]

        averages[field] = (
            sum(values) / len(values)
        )

    summary_path = (
        OUTPUT_DIR /
        "batch_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            averages,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print("\n========================================")
    print("BATCH SUMMARY")
    print("========================================")

    print(
        f"Average wound area: "
        f"{averages['wound_percentage']:.2f}%"
    )

    print(
        f"Average epithelial: "
        f"{averages['epithelial_percentage']:.2f}%"
    )

    print(
        f"Average slough: "
        f"{averages['slough_percentage']:.2f}%"
    )

    print(
        f"Average granulation: "
        f"{averages['granulation_percentage']:.2f}%"
    )

    print(
        f"Average necrotic: "
        f"{averages['necrotic_percentage']:.2f}%"
    )

    print(
        f"Average other: "
        f"{averages['other_percentage']:.2f}%"
    )

    print("\n========================================")
    print("Batch analysis complete!")
    print("========================================")

    print(f"\nJSON: {json_path}")
    print(f"CSV : {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()