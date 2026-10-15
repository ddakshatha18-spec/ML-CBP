from pathlib import Path
import pandas as pd

INPUT_FILE = Path(
    "person2_trajectory_model/outputs/trajectory_measurements.csv"
)

OUTPUT_FILE = Path(
    "person2_trajectory_model/outputs/temporal_features.csv"
)

AREA_COLUMN = "wound_area_pct"

TISSUE_COLUMNS = [
    "epithelial_pct_of_labeled_tissue",
    "slough_pct_of_labeled_tissue",
    "granulation_pct_of_labeled_tissue",
    "necrotic_pct_of_labeled_tissue",
    "other_pct_of_labeled_tissue",
]


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Run extract_mask_measurements.py first: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df = df.sort_values(
        ["patient_id", "visit_number"]
    ).copy()

    # Compare each visit with the previous visit for that patient.
    grouped = df.groupby("patient_id", sort=False)

    df["previous_visit"] = grouped["visit_number"].shift(1)
    df["visit_interval"] = (
        df["visit_number"] - df["previous_visit"]
    )

    df["previous_wound_area_pct"] = (
        grouped[AREA_COLUMN].shift(1)
    )

    df["wound_area_change_pp"] = (
        df[AREA_COLUMN] - df["previous_wound_area_pct"]
    )

    df["wound_area_relative_change_pct"] = (
        (
            df[AREA_COLUMN] - df["previous_wound_area_pct"]
        )
        / df["previous_wound_area_pct"].where(
            df["previous_wound_area_pct"] != 0
        )
        * 100
    )

    for column in TISSUE_COLUMNS:
        previous = grouped[column].shift(1)
        feature_name = column.replace(
            "_pct_of_labeled_tissue", "_change_pp"
        )
        df[feature_name] = df[column] - previous

    # The first visit for each patient has no previous visit.
    df["has_previous_visit"] = df["previous_visit"].notna()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    comparisons = int(df["has_previous_visit"].sum())
    patients = df["patient_id"].nunique()

    print(f"Patients processed: {patients}")
    print(f"Images processed: {len(df)}")
    print(f"Consecutive-visit comparisons: {comparisons}")
    print(f"Output saved to: {OUTPUT_FILE}")
    print("\nSample rows:")
    print(
        df[
            [
                "patient_id",
                "visit_number",
                AREA_COLUMN,
                "wound_area_change_pp",
                "has_previous_visit",
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
