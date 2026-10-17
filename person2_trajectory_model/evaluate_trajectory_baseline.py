
from pathlib import Path

import pandas as pd


BASE_DIR = Path("person2_trajectory_model")
INPUT_FILE = BASE_DIR / "outputs" / "temporal_features.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "baseline_evaluation"

THRESHOLDS = [3.0, 5.0, 10.0]


def classify_change(change, has_previous, threshold):
    """Apply the rule-based area-change classification."""
    if not has_previous or pd.isna(change):
        return "Insufficient data"

    if change < -threshold:
        return "Area decreased"

    if change > threshold:
        return "Area increased"

    return "Relatively stable"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required = {
        "patient_id",
        "visit_number",
        "wound_area_pct",
        "wound_area_change_pp",
        "has_previous_visit",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Normalize IDs and numeric fields.
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["visit_number"] = pd.to_numeric(
        df["visit_number"], errors="coerce"
    )
    df["wound_area_pct"] = pd.to_numeric(
        df["wound_area_pct"], errors="coerce"
    )
    df["wound_area_change_pp"] = pd.to_numeric(
        df["wound_area_change_pp"], errors="coerce"
    )

    # Parse booleans safely, including CSV strings.
    if df["has_previous_visit"].dtype != bool:
        previous = (
            df["has_previous_visit"]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        invalid = ~previous.isin(["true", "false", "1", "0"])
        if invalid.any():
            raise ValueError(
                "Unexpected has_previous_visit values: "
                f"{previous[invalid].unique().tolist()}"
            )
        df["has_previous_visit"] = previous.isin(["true", "1"])

    if df["patient_id"].eq("").any():
        raise ValueError("Some rows have an empty patient_id.")

    if df.duplicated(["patient_id", "visit_number"]).any():
        raise ValueError("Duplicate patient_id / visit_number rows found.")

    df = df.sort_values(["patient_id", "visit_number"]).copy()

    # Only comparable rows are included in threshold comparisons.
    comparable = (
        df["has_previous_visit"]
        & df["wound_area_change_pp"].notna()
    )

    df["is_comparable"] = comparable

    # Generate classifications at each threshold.
    for threshold in THRESHOLDS:
        label = f"category_{int(threshold)}pp"
        df[label] = [
            classify_change(change, has_previous, threshold)
            for change, has_previous in zip(
                df["wound_area_change_pp"],
                df["has_previous_visit"],
            )
        ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Row-level classifications at all thresholds.
    df.to_csv(OUTPUT_DIR / "threshold_classifications.csv", index=False)

    # 2. Category counts and percentages for every threshold.
    category_rows = []
    for threshold in THRESHOLDS:
        col = f"category_{int(threshold)}pp"
        counts = df[col].value_counts()

        for category in [
            "Area decreased",
            "Relatively stable",
            "Area increased",
            "Insufficient data",
        ]:
            count = int(counts.get(category, 0))
            category_rows.append({
                "threshold_pp": threshold,
                "category": category,
                "count": count,
                "percent_of_all_images": round(100 * count / len(df), 2),
                "percent_of_comparable_visits": (
                    round(100 * count / comparable.sum(), 2)
                    if comparable.sum() and category != "Insufficient data"
                    else None
                ),
            })

    pd.DataFrame(category_rows).to_csv(
        OUTPUT_DIR / "threshold_sensitivity.csv", index=False
    )

    # 3. Patient-level summary at the primary +/-5 pp threshold.
    patient_rows = []
    for patient_id, group in df.groupby("patient_id", sort=True):
        group = group.sort_values("visit_number")
        valid_area = group["wound_area_pct"].dropna()
        valid_changes = group.loc[
            group["is_comparable"], "wound_area_change_pp"
        ]
        labels = group["category_5pp"]

        patient_rows.append({
            "patient_id": patient_id,
            "recorded_visits": len(group),
            "comparable_transitions": int(group["is_comparable"].sum()),
            "first_visit_number": group["visit_number"].iloc[0],
            "last_visit_number": group["visit_number"].iloc[-1],
            "first_recorded_area_pct": (
                valid_area.iloc[0] if len(valid_area) else None
            ),
            "last_recorded_area_pct": (
                valid_area.iloc[-1] if len(valid_area) else None
            ),
            "area_min_pct": valid_area.min() if len(valid_area) else None,
            "area_max_pct": valid_area.max() if len(valid_area) else None,
            "area_range_pp": (
                valid_area.max() - valid_area.min()
                if len(valid_area) else None
            ),
            "mean_area_change_pp": (
                valid_changes.mean() if len(valid_changes) else None
            ),
            "decreased_transitions": int((labels == "Area decreased").sum()),
            "stable_transitions": int((labels == "Relatively stable").sum()),
            "increased_transitions": int((labels == "Area increased").sum()),
        })

    patient_summary = pd.DataFrame(patient_rows)
    patient_summary.to_csv(
        OUTPUT_DIR / "patient_level_summary.csv", index=False
    )

    # 4. Agreement between thresholds on the same comparable transitions.
    agreement_rows = []
    for low, high in [(3.0, 5.0), (5.0, 10.0), (3.0, 10.0)]:
        low_col = f"category_{int(low)}pp"
        high_col = f"category_{int(high)}pp"
        a = df.loc[comparable, low_col]
        b = df.loc[comparable, high_col]

        agreement_rows.append({
            "threshold_a_pp": low,
            "threshold_b_pp": high,
            "comparable_transitions": len(a),
            "same_category_count": int((a == b).sum()),
            "agreement_percent": (
                round(100 * (a == b).mean(), 2) if len(a) else None
            ),
        })

    pd.DataFrame(agreement_rows).to_csv(
        OUTPUT_DIR / "threshold_agreement.csv", index=False
    )

    # 5. Summary text for the terminal.
    print("\nBASELINE EVALUATION COMPLETE")
    print(f"Images: {len(df)}")
    print(f"Patients: {df['patient_id'].nunique()}")
    print(f"Comparable transitions: {int(comparable.sum())}")
    print(f"First visits / unavailable comparisons: {int((~comparable).sum())}")

    print("\nCategory counts by threshold:")
    for threshold in THRESHOLDS:
        col = f"category_{int(threshold)}pp"
        print(f"\n+/- {threshold:g} pp")
        print(df.loc[comparable, col].value_counts().to_string())

    print("\nThreshold agreement:")
    print(pd.DataFrame(agreement_rows).to_string(index=False))

    print(f"\nResults saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
