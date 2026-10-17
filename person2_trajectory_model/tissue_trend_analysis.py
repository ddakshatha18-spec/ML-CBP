
from pathlib import Path
import pandas as pd

INPUT_FILE = Path(
    "person2_trajectory_model/outputs/temporal_features.csv"
)

OUTPUT_FILE = Path(
    "person2_trajectory_model/outputs/tissue_trend_summary.csv"
)

TISSUE_COLUMNS = {
    "epithelial": "epithelial_change_pp",
    "slough": "slough_change_pp",
    "granulation": "granulation_change_pp",
    "necrotic": "necrotic_change_pp",
}


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    # Keep only rows with a previous visit.
    comparisons = df[df["has_previous_visit"]].copy()

    results = []

    for tissue_name, column in TISSUE_COLUMNS.items():
        values = comparisons[column].dropna()

        results.append({
            "tissue": tissue_name,
            "valid_comparisons": len(values),
            "mean_change_pp": round(values.mean(), 4),
            "median_change_pp": round(values.median(), 4),
            "minimum_change_pp": round(values.min(), 4),
            "maximum_change_pp": round(values.max(), 4),
        })

    summary = pd.DataFrame(results)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    summary.to_csv(OUTPUT_FILE, index=False)

    print(f"Patients: {df['patient_id'].nunique()}")
    print(f"Images: {len(df)}")
    print(f"Visit comparisons: {len(comparisons)}")
    print("\nTissue trend summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
