
from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path("data/LUTSeg")

CHECKS = {
    "Patient_6": [4, 5, 6],
    "Patient_22": [1, 2],
    "Patient_31": [2, 3],
}

CLASS_NAMES = {
    0: "Background",
    1: "Epithelial",
    2: "Slough",
    3: "Granulation",
    4: "Necrotic",
    5: "Other",
    255: "Ignore",
}


def main():
    for patient, visits in CHECKS.items():
        patient_number = patient.split("_")[1]

        for visit in visits:
            image_path = (
                ROOT / "Images" / patient
                / f"P{patient_number}T_{visit}.jpg"
            )
            mask_path = (
                ROOT / "Masks" / patient
                / f"P{patient_number}T_{visit}.png"
            )

            with Image.open(image_path) as image:
                image_size = image.size

            with Image.open(mask_path) as mask:
                labels = np.array(mask.convert("L"))

            print(f"\n{patient}, visit {visit}")
            print(f"Image size (width, height): {image_size}")
            print(
                "Mask size (width, height): "
                f"{(labels.shape[1], labels.shape[0])}"
            )

            values, counts = np.unique(labels, return_counts=True)

            for value, count in zip(values, counts):
                name = CLASS_NAMES.get(int(value), "Unknown")
                print(f"  {int(value):3} {name:12} {int(count):>10} pixels")


if __name__ == "__main__":
    main()
