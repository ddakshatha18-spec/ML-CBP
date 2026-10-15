from pathlib import Path
from PIL import Image
import csv
import re

DATA_ROOT = Path("data/LUTSeg")
OUTPUT_FILE = Path(
    "person2_trajectory_model/outputs/trajectory_measurements.csv"
)

TISSUE_CLASSES = {
    1: "epithelial",
    2: "slough",
    3: "granulation",
    4: "necrotic",
    5: "other",
}


def get_visit_number(filename):
    match = re.search(r"T_?(\d+)$", filename.stem)
    if match:
        return int(match.group(1))
    return None


def main():
    image_root = DATA_ROOT / "Images"
    wound_root = DATA_ROOT / "Wound_Masks"
    tissue_root = DATA_ROOT / "Masks"

    if not image_root.exists():
        raise FileNotFoundError(
            f"Dataset image folder not found: {image_root}"
        )

    images = sorted(
        path
        for path in image_root.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    rows = []
    skipped = []

    for image_path in images:
        relative_path = image_path.relative_to(image_root)
        patient_id = relative_path.parts[0]
        visit_number = get_visit_number(image_path)

        wound_path = (wound_root / relative_path).with_suffix(".png")
        tissue_path = (tissue_root / relative_path).with_suffix(".png")

        if visit_number is None:
            skipped.append(f"Unrecognized visit number: {image_path}")
            continue

        if not wound_path.exists() or not tissue_path.exists():
            skipped.append(f"Missing mask for: {image_path}")
            continue

        with Image.open(wound_path) as image:
            wound_mask = image.convert("L")

        with Image.open(tissue_path) as image:
            tissue_mask = image.convert("L")

        if wound_mask.size != tissue_mask.size:
            skipped.append(f"Mask dimensions differ: {image_path}")
            continue

        wound_pixels = sum(
            1 for value in wound_mask.getdata() if value > 0
        )
        total_pixels = wound_mask.width * wound_mask.height

        if total_pixels == 0:
            skipped.append(f"Empty image dimensions: {image_path}")
            continue

        tissue_pixels = {
            class_id: sum(
                1 for value in tissue_mask.getdata()
                if value == class_id
            )
            for class_id in TISSUE_CLASSES
        }

        total_tissue_pixels = sum(tissue_pixels.values())

        row = {
            "patient_id": patient_id,
            "visit_number": visit_number,
            "image_filename": image_path.name,
            "wound_pixels": wound_pixels,
            "image_pixels": total_pixels,
            "wound_area_pct": round(
                wound_pixels / total_pixels * 100, 4
            ),
            "tissue_labeled_pixels": total_tissue_pixels,
        }

        for class_id, name in TISSUE_CLASSES.items():
            count = tissue_pixels[class_id]
            row[f"{name}_pixels"] = count
            row[f"{name}_pct_of_labeled_tissue"] = round(
                count / total_tissue_pixels * 100, 4
            ) if total_tissue_pixels else 0.0

        rows.append(row)

    rows.sort(key=lambda row: (
        row["patient_id"], row["visit_number"]
    ))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        with OUTPUT_FILE.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    print(f"Images found: {len(images)}")
    print(f"Measurements created: {len(rows)}")
    print(f"Images skipped: {len(skipped)}")
    print(f"Output: {OUTPUT_FILE}")

    if skipped:
        print("\nFirst skipped items:")
        for item in skipped[:10]:
            print("-", item)


if __name__ == "__main__":
    main()
