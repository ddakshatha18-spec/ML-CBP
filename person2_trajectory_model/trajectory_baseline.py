
from pathlib import Path
import pandas as pd

INPUT_FILE = Path(
    "person2_trajectory_model/outputs/temporal_features.csv"
)

OUTPUT_FILE = Path(
    "person2_trajectory_model/outputs/trajectory_predictions.csv"
)

THRESHOLD_PP = 5.0


def classify_trajectory(row):
    if not row["has_previous_visit"]:
        return "Insufficient data"

    change = row["wound_area_change_pp"]

    if pd.isna(change):
        return "Insufficient data"

    if change < -THRESHOLD_PP:
        return "Area decreased"

    if change > THRESHOLD_PP:
        return "Area increased"

    return "Relatively stable"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df["area_change_category"] = df.apply(
        classify_trajectory,
        axis=1
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Images processed: {len(df)}")
    print(f"Patients processed: {df['patient_id'].nunique()}")
    print(f"Threshold: +/- {THRESHOLD_PP} percentage points")

    print("\nArea change category counts:")
    print(df["area_change_category"].value_counts().to_string())

    print(f"\nPredictions saved to: {OUTPUT_FILE}")

    print("\nSample results:")
    print(
        df[
            [
                "patient_id",
                "visit_number",
                "wound_area_pct",
                "wound_area_change_pp",
                "area_change_category",
            ]
        ].head(15).to_string(index=False)
    )


if __name__ == "__main__":
    main()
