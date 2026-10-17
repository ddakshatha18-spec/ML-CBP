
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
    data["wound_area_change_pp"],
    errors="coerce",
)

data["baseline_category"] = "Relatively stable"
data.loc[change.isna(), "baseline_category"] = "Insufficient data"
data.loc[change < -5, "baseline_category"] = "Area decreased"
data.loc[change > 5, "baseline_category"] = "Area increased"

data["area_flag"] = data["quality_flags"].str.contains(
    "Large wound-area change", regex=False
)
data["tissue_flag"] = data["quality_flags"].str.contains(
    "Large tissue-composition change", regex=False
)
data["small_region_flag"] = data["quality_flags"].str.contains(
    "Very small wound region", regex=False
)

comparable = data[change.notna()].copy()

print("FLAG OVERLAP CHECK")
print("==================")
print("Comparable records:", len(comparable))
print("Area-change flags:", int(comparable["area_flag"].sum()))
print("Tissue-composition flags:", int(comparable["tissue_flag"].sum()))
print("Small-region flags:", int(comparable["small_region_flag"].sum()))

print("\nBaseline category by area-change flag:")
print(pd.crosstab(
    comparable["area_flag"],
    comparable["baseline_category"],
).to_string())

print("\nBaseline category by any quality flag:")
comparable["any_flag"] = (
    comparable["quality_flags"].ne("")
    & comparable["quality_flags"].ne("No threshold-based flag")
)
print(pd.crosstab(
    comparable["any_flag"],
    comparable["baseline_category"],
).to_string())

print("\nRecords classified as increased/decreased but without area flags:")
exceptions = comparable[
    comparable["baseline_category"].isin(
        ["Area decreased", "Area increased"]
    )
    & ~comparable["area_flag"]
]
print("Count:", len(exceptions))
print(exceptions[
    ["patient_id", "visit_number", "wound_area_change_pp",
     "quality_flags", "baseline_category"]
].to_string(index=False))
