
from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("person2_trajectory_model/outputs")
INPUT_FILE = OUTPUT_DIR / "quality_flags.csv"
REVIEW_FILE = OUTPUT_DIR / "quality_flag_review.csv"

df = pd.read_csv(INPUT_FILE)

# Keep records that have at least one threshold-based quality flag.
flagged = df.loc[
    df["quality_flags"].fillna("No threshold-based flag")
    != "No threshold-based flag"
].copy()

# Identify flag types independently, since multiple flags can overlap.
flags = flagged["quality_flags"].fillna("")

flagged["area_change_flag"] = flags.str.contains(
    "Large wound-area change", regex=False
)
flagged["tissue_change_flag"] = flags.str.contains(
    "Large tissue-composition change", regex=False
)
flagged["small_region_flag"] = flags.str.contains(
    "Very small wound region", regex=False
)

# Keep the transition context and tissue measurements/changes.
columns = [
    "patient_id",
    "visit_number",
    "previous_visit",
    "has_previous_visit",
    "image_filename",
    "wound_area_pct",
    "previous_wound_area_pct",
    "wound_area_change_pp",
    "wound_area_relative_change_pct",
    "epithelial_pct_of_labeled_tissue",
    "epithelial_change_pp",
    "slough_pct_of_labeled_tissue",
    "slough_change_pp",
    "granulation_pct_of_labeled_tissue",
    "granulation_change_pp",
    "necrotic_pct_of_labeled_tissue",
    "necrotic_change_pp",
    "other_pct_of_labeled_tissue",
    "other_change_pp",
    "area_change_flag",
    "tissue_change_flag",
    "small_region_flag",
    "quality_flags",
]

columns = [column for column in columns if column in flagged.columns]
review = flagged[columns].copy()

# Sort for easier patient-by-patient review.
sort_columns = [
    column for column in ["patient_id", "visit_number"]
    if column in review.columns
]
if sort_columns:
    review = review.sort_values(sort_columns)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
review.to_csv(REVIEW_FILE, index=False)

print("QUALITY FLAG REVIEW")
print(f"Total records: {len(df)}")
print(f"Flagged records: {len(review)}")
print(f"Unflagged records: {len(df) - len(review)}")
print()
print("Flag-type counts (overlap is possible):")
for column, label in [
    ("area_change_flag", "Large wound-area change"),
    ("tissue_change_flag", "Large tissue-composition change"),
    ("small_region_flag", "Very small wound region"),
]:
    if column in review.columns:
        print(f"  {label}: {int(review[column].sum())}")

if "has_previous_visit" in review.columns:
    has_previous = review["has_previous_visit"].astype(str).str.lower().isin(
        ["true", "1"]
    )
    print(f"Flagged records with a previous visit: {int(has_previous.sum())}")
    print(f"Flagged records without a previous visit: {int((~has_previous).sum())}")

print()
print(f"Review table saved to: {REVIEW_FILE}")
print()
if not review.empty:
    display_columns = [
        column for column in [
            "patient_id",
            "visit_number",
            "previous_visit",
            "wound_area_pct",
            "wound_area_change_pp",
            "epithelial_change_pp",
            "slough_change_pp",
            "granulation_change_pp",
            "necrotic_change_pp",
            "quality_flags",
        ]
        if column in review.columns
    ]
    print(review[display_columns].to_string(index=False))
else:
    print("No threshold-based quality flags were found.")
