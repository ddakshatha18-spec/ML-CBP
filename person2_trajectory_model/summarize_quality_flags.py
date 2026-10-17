
import pandas as pd

file_path = "person2_trajectory_model/outputs/quality_flags.csv"
df = pd.read_csv(file_path)

print("QUALITY FLAG SUMMARY")
print("====================")

flag_types = [
    "Large wound-area change",
    "Large tissue-composition change",
    "Very small wound region",
]

for flag in flag_types:
    count = df["quality_flags"].str.contains(
        flag, regex=False
    ).sum()

    print(f"{flag}: {count}")

print("\nFLAGGED RECORDS BY PATIENT")
print("==========================")

flagged = df[
    df["quality_flags"] != "No threshold-based flag"
]

print(
    flagged.groupby("patient_id")
    .size()
    .sort_values(ascending=False)
    .to_string()
)
