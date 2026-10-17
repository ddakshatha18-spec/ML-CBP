
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

DATA_ROOT = Path("data/LUTSeg")
OUTPUT_DIR = Path("person2_trajectory_model/outputs/transition_images")

PATIENT_VISITS = {
    "Patient_6": [4, 5, 6],
    "Patient_22": [1, 2],
    "Patient_31": [2, 3],
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for patient_id, visits in PATIENT_VISITS.items():
        figure, axes = plt.subplots(
            len(visits), 2,
            figsize=(10, 4 * len(visits))
        )

        if len(visits) == 1:
            axes = np.array([axes])

        for row_index, visit in enumerate(visits):
            image_path = (
                DATA_ROOT / "Images" / patient_id
                / f"P{patient_id.split('_')[1]}T_{visit}.jpg"
            )

            mask_path = (
                DATA_ROOT / "Masks" / patient_id
                / f"P{patient_id.split('_')[1]}T_{visit}.png"
            )

            if not image_path.exists() or not mask_path.exists():
                print(f"Missing file for {patient_id}, visit {visit}")
                continue

            with Image.open(image_path) as image:
                original = image.convert("RGB")

            with Image.open(mask_path) as mask:
                tissue_mask = np.array(mask.convert("L"))

            axes[row_index, 0].imshow(original)
            axes[row_index, 0].set_title(
                f"{patient_id} — Visit {visit}: Original"
            )
            axes[row_index, 0].axis("off")

            axes[row_index, 1].imshow(tissue_mask, cmap="tab10", vmin=0, vmax=5)
            axes[row_index, 1].set_title(
                f"{patient_id} — Visit {visit}: Tissue mask"
            )
            axes[row_index, 1].axis("off")

        figure.tight_layout()

        output_path = OUTPUT_DIR / f"{patient_id}_comparison.png"
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(figure)

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
