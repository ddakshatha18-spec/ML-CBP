
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

features = pd.read_csv(OUTPUT_DIR / "temporal_features.csv")
flags = pd.read_csv(OUTPUT_DIR / "quality_flags.csv")

keys = ["patient_id", "visit_number"]

data = features.merge(
    flags[keys + ["quality_flags"]],
    on=keys,
    how="left",
    validate="one_to_one",
)

data["quality_flags"] = (
    data["quality_flags"].fillna("").astype(str).str.strip()
)

change = pd.to_numeric(
    data["wound_area_change_pp"], errors="coerce"
)

data["category"] = "Relatively stable"
data.loc[change.isna(), "category"] = "Insufficient data"
data.loc[change < -5, "category"] = "Area decreased"
data.loc[change > 5, "category"] = "Area increased"

data["tissue_flag"] = data["quality_flags"].str.contains(
    "Large tissue-composition change", regex=False
)
data["small_region_flag"] = data["quality_flags"].str.contains(
    "Very small wound region", regex=False
)

# Compare records with measurable changes.
comparable = data[change.notna()].copy()

# Exclude only tissue-composition and small-region flags.
exclude = comparable["tissue_flag"] | comparable["small_region_flag"]
filtered = comparable[~exclude].copy()

categories = [
    "Area decreased",
    "Relatively stable",
    "Area increased",
]

print("TISSUE/SMALL-REGION ROBUSTNESS CHECK")
print("====================================")
print("Comparable records:", len(comparable))
print("Excluded records:", int(exclude.sum()))
print("Remaining records:", len(filtered))

print("\nAll comparable records:")
print(
    comparable["category"]
    .value_counts()
    .reindex(categories, fill_value=0)
    .to_string()
)

print("\nAfter excluding tissue/small-region flags:")
print(
    filtered["category"]
    .value_counts()
    .reindex(categories, fill_value=0)
    .to_string()
)

report = pd.DataFrame({
    "analysis": [
        "All comparable records",
        "Exclude tissue/small-region flags",
    ],
    "record_count": [len(comparable), len(filtered)],
})

for category in categories:
    report[category] = [
        int((comparable["category"] == category).sum()),
        int((filtered["category"] == category).sum()),
    ]

output_file = OUTPUT_DIR / "tissue_quality_robustness_summary.csv"
report.to_csv(output_file, index=False)

print(f"\nSaved separate report: {output_file}")
print("Original analysis files were not changed.")
