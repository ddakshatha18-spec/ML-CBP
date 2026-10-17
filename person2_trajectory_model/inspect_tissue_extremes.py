
from pathlib import Path
import pandas as pd

INPUT_FILE = Path(
    "person2_trajectory_model/outputs/temporal_features.csv"
)

TISSUE_COLUMNS = [
    "epithelial_change_pp",
    "slough_change_pp",
    "granulation_change_pp",
    "necrotic_change_pp",
]


def main():
    df = pd.read_csv(INPUT_FILE)
    df = df[df["has_previous_visit"]].copy()

    for column in TISSUE_COLUMNS:
        print(f"\n{'=' * 65}")
        print(f"Tissue feature: {column}")

        valid = df.dropna(subset=[column])

        print("\nLargest increases:")
        print(
            valid.nlargest(5, column)[
                [
                    "patient_id",
                    "previous_visit",
                    "visit_number",
                    column,
                ]
            ].to_string(index=False)
        )

        print("\nLargest decreases:")
        print(
            valid.nsmallest(5, column)[
                [
                    "patient_id",
                    "previous_visit",
                    "visit_number",
                    column,
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
