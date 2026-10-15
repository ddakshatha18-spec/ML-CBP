
from pathlib import Path
import pandas as pd

INPUT_FILE = Path(
    "person2_trajectory_model/outputs/trajectory_predictions.csv"
)

OUTPUT_FILE = Path(
    "person2_trajectory_model/outputs/patient_summary.csv"
)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"File not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df = df.sort_values(
        ["patient_id", "visit_number"]
    )

    summaries = []

    for patient_id, group in df.groupby("patient_id"):
        first = group.iloc[0]
        last = group.iloc[-1]

        summaries.append({
            "patient_id": patient_id,
            "number_of_visits": len(group),
            "first_visit": int(first["visit_number"]),
            "last_visit": int(last["visit_number"]),
            "first_wound_area_pct": first["wound_area_pct"],
            "last_wound_area_pct": last["wound_area_pct"],
            "overall_area_change_pp": round(
                last["wound_area_pct"]
                - first["wound_area_pct"],
                4
            ),
            "area_decreased_comparisons": int(
                (group["area_change_category"] == "Area decreased").sum()
            ),
            "area_increased_comparisons": int(
                (group["area_change_category"] == "Area increased").sum()
            ),
            "relatively_stable_comparisons": int(
                (group["area_change_category"] == "Relatively stable").sum()
            ),
        })

    summary = pd.DataFrame(summaries)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    summary.to_csv(OUTPUT_FILE, index=False)

    print(f"Patients summarized: {len(summary)}")
    print(f"Output saved to: {OUTPUT_FILE}")
    print("\nFirst 10 patient summaries:")
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
