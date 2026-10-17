
import os
import pandas as pd

INPUT_FILE = "person2_trajectory_model/outputs/temporal_features.csv"
OUTPUT_FILE = "person2_trajectory_model/outputs/quality_flags.csv"

df = pd.read_csv(INPUT_FILE)

df["quality_flags"] = ""

for index, row in df.iterrows():
    flags = []

    # Flag unusually large wound-area changes
    change = row.get("wound_area_change_pp")

    if pd.notna(change) and abs(change) > 5:
        flags.append("Large wound-area change")

    # Flag unusually large tissue-composition changes
    tissue_columns = [
        "epithelial_change_pp",
        "slough_change_pp",
        "granulation_change_pp",
        "necrotic_change_pp",
    ]

    for column in tissue_columns:
        if column in df.columns:
            value = row.get(column)

            if pd.notna(value) and abs(value) > 50:
                flags.append("Large tissue-composition change")
                break

    # Flag very small wound regions for manual review
    area = row.get("wound_area_pct")

    if pd.notna(area) and area < 0.2:
        flags.append("Very small wound region")

    if not flags:
        flags.append("No threshold-based flag")

    df.at[index, "quality_flags"] = "; ".join(flags)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("Quality checks completed.")
print("Rows checked:", len(df))
print("\nQuality flag counts:")
print(df["quality_flags"].value_counts().to_string())
print("\nSaved to:", OUTPUT_FILE)
