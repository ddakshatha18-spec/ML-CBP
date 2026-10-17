
import pandas as pd

INPUT_FILE = (
    "person2_trajectory_model/outputs/temporal_features.csv"
)

df = pd.read_csv(INPUT_FILE)

change_column = "wound_area_change_pp"

if change_column not in df.columns:
    raise ValueError(f"Missing column: {change_column}")

change = pd.to_numeric(df[change_column], errors="coerce")

print("THRESHOLD SENSITIVITY ANALYSIS")
print("==============================")
print("Total image records:", len(df))
print()

for threshold in [3, 5, 10]:
    categories = pd.Series(
        "Relatively stable",
        index=df.index,
        dtype="object",
    )

    categories[change.isna()] = "Insufficient data"
    categories[change < -threshold] = "Area decreased"
    categories[change > threshold] = "Area increased"

    print(f"Threshold: +/- {threshold} percentage points")
    print(categories.value_counts().to_string())
    print()

print("Analysis complete. Original files were not changed.")
