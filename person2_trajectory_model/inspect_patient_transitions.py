
from pathlib import Path
import pandas as pd

INPUT_FILE = Path(
    "person2_trajectory_model/outputs/trajectory_measurements.csv"
)

PATIENT_VISITS = {
    "Patient_6": [4, 5, 6],
    "Patient_22": [1, 2],
    "Patient_31": [2, 3],
}

TISSUES = [
    "epithelial",
    "slough",
    "granulation",
    "necrotic",
    "other",
]


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Measurements file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    for patient_id, visits in PATIENT_VISITS.items():
        print("\n" + "=" * 75)
        print(f"{patient_id}: selected visits")

        selected = df[
            (df["patient_id"] == patient_id)
            & (df["visit_number"].isin(visits))
        ].sort_values("visit_number")

        columns = [
            "patient_id",
            "visit_number",
            "image_filename",
            "image_pixels",
            "wound_pixels",
            "wound_area_pct",
            "tissue_labeled_pixels",
        ]

        for tissue in TISSUES:
            columns.append(f"{tissue}_pixels")
            columns.append(f"{tissue}_pct_of_labeled_tissue")

        available_columns = [
            column for column in columns if column in selected.columns
        ]

        print(selected[available_columns].to_string(index=False))

        if selected.empty:
            print("No matching rows found.")

    print("\nNote: pixel counts are image measurements,")
    print("not calibrated physical wound areas.")


if __name__ == "__main__":
    main()
