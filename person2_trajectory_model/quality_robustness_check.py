
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

FEATURES_FILE = OUTPUT_DIR / "temporal_features.csv"
FLAGS_FILE = OUTPUT_DIR / "quality_flags.csv"

THRESHOLD = 5


def classify_changes(change, threshold=5):
    categories = pd.Series(
        "Relatively stable",
        index=change.index,
        dtype="object",
    )

    categories[change.isna()] = "Insufficient data"
    categories[change < -threshold] = "Area decreased"
    categories[change > threshold] = "Area increased"

    return categories


def main():
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(FEATURES_FILE)

    if not FLAGS_FILE.exists():
        raise FileNotFoundError(FLAGS_FILE)

    features = pd.read_csv(FEATURES_FILE)
    flags = pd.read_csv(FLAGS_FILE)

    required_features = {
        "patient_id",
        "visit_number",
        "wound_area_change_pp",
    }
    required_flags = {
        "patient_id",
        "visit_number",
        "quality_flags",
    }

    if not required_features.issubset(features.columns):
        raise ValueError(
            f"Missing feature columns: "
            f"{required_features - set(features.columns)}"
        )

    if not required_flags.issubset(flags.columns):
        raise ValueError(
            f"Missing flag columns: "
            f"{required_flags - set(flags.columns)}"
        )

    keys = ["patient_id", "visit_number"]

    if flags.duplicated(keys).any():
        raise ValueError(
            "Duplicate patient/visit keys in quality_flags.csv"
        )

    flags["quality_flags"] = (
        flags["quality_flags"].fillna("").astype(str).str.strip()
    )

    flagged_keys = flags.loc[
        flags["quality_flags"].ne("")
        & flags["quality_flags"].ne("No threshold-based flag"),
        keys,
    ].drop_duplicates()

    data = features.merge(
        flags[keys + ["quality_flags"]],
        on=keys,
        how="left",
        validate="many_to_one",
    )

    data["quality_flags"] = data["quality_flags"].fillna("")
    data["wound_area_change_pp"] = pd.to_numeric(
        data["wound_area_change_pp"],
        errors="coerce",
    )

    data["is_flagged"] = (
        data["quality_flags"].ne("")
        & data["quality_flags"].ne("No threshold-based flag")
    )

    data["baseline_category"] = classify_changes(
        data["wound_area_change_pp"],
        THRESHOLD,
    )

    # Keep only records that have a previous visit and a measurable change.
    comparable = data[
        data["wound_area_change_pp"].notna()
    ].copy()

    included = comparable[~comparable["is_flagged"]].copy()

    print("QUALITY-FLAG ROBUSTNESS CHECK")
    print("============================")
    print(f"Total image records: {len(data)}")
    print(f"Records with measurable changes: {len(comparable)}")
    print(f"Flagged comparable records: {comparable['is_flagged'].sum()}")
    print(f"Unflagged comparable records: {len(included)}")
    print(f"Threshold: +/- {THRESHOLD} percentage points")

    print("\nAll comparable records:")
    print(
        comparable["baseline_category"]
        .value_counts()
        .reindex(
            [
                "Area decreased",
                "Relatively stable",
                "Area increased",
                "Insufficient data",
            ],
            fill_value=0,
        )
        .to_string()
    )

    print("\nUnflagged comparable records only:")
    print(
        included["baseline_category"]
        .value_counts()
        .reindex(
            [
                "Area decreased",
                "Relatively stable",
                "Area increased",
                "Insufficient data",
            ],
            fill_value=0,
        )
        .to_string()
    )

    # Save a separate report; never overwrite the original features.
    report = pd.DataFrame(
        {
            "analysis": [
                "All comparable records",
                "Unflagged comparable records",
            ],
            "record_count": [
                len(comparable),
                len(included),
            ],
        }
    )

    for category in [
        "Area decreased",
        "Relatively stable",
        "Area increased",
        "Insufficient data",
    ]:
        report[category] = [
            (comparable["baseline_category"] == category).sum(),
            (included["baseline_category"] == category).sum(),
        ]

    output_file = OUTPUT_DIR / "quality_robustness_summary.csv"
    report.to_csv(output_file, index=False)

    print(f"\nSeparate summary saved to: {output_file}")
    print("Original analysis files were not changed.")


if __name__ == "__main__":
    main()
