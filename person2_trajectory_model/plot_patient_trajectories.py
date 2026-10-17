
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
def is_flagged(series):
    values = series.fillna("").astype(str).str.strip()
    return (
        values.ne("")
        & values.ne("No threshold-based flag")
    )

# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

TEMPORAL_CSV = OUTPUT_DIR / "temporal_features.csv"
QUALITY_CSV = OUTPUT_DIR / "quality_flags.csv"

PATIENTS_TO_PLOT = ["Patient_20", "Patient_6", "Patient_22"]


# Candidate names support common variations in column naming.
TISSUE_CANDIDATES = {
    "Epithelial tissue": [
        "epithelial_pct",
        "epithelial_percentage",
        "epithelial_tissue_pct",
        "epithelial_pct_of_labeled_tissue",
    ],
    "Slough": [
        "slough_pct",
        "slough_percentage",
        "slough_pct_of_labeled_tissue",
    ],
    "Granulation tissue": [
        "granulation_pct",
        "granulation_percentage",
        "granulation_tissue_pct",
        "granulation_pct_of_labeled_tissue",
    ],
    "Necrotic tissue": [
        "necrotic_pct",
        "necrotic_percentage",
        "necrotic_tissue_pct",
        "necrotic_pct_of_labeled_tissue",
    ],
    "Other tissue": [
        "other_pct",
        "other_percentage",
        "other_tissue_pct",
        "other_pct_of_labeled_tissue",
    ],
}


def find_column(df, candidates):
    """Return the first matching column name, ignoring case."""
    lookup = {col.lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def load_data():
    """Load temporal measurements and quality flags."""
    if not TEMPORAL_CSV.exists():
        raise FileNotFoundError(
            f"Missing file: {TEMPORAL_CSV}\n"
            "Run temporal_features.py first."
        )

    if not QUALITY_CSV.exists():
        raise FileNotFoundError(
            f"Missing file: {QUALITY_CSV}\n"
            "Run quality_flags.py first."
        )

    temporal = pd.read_csv(TEMPORAL_CSV)
    flags = pd.read_csv(QUALITY_CSV)

    required = {"patient_id", "visit_number", "wound_area_pct"}
    missing = required - set(temporal.columns)

    if missing:
        raise ValueError(
            f"Missing required columns in temporal_features.csv: "
            f"{sorted(missing)}"
        )

    flag_required = {"patient_id", "visit_number", "quality_flags"}
    missing_flags = flag_required - set(flags.columns)

    if missing_flags:
        raise ValueError(
            f"Missing required columns in quality_flags.csv: "
            f"{sorted(missing_flags)}"
        )

    # Use only the flag fields needed for plotting.
    flags = flags[
        ["patient_id", "visit_number", "quality_flags"]
    ].drop_duplicates(subset=["patient_id", "visit_number"])

    # Keep every temporal record, including records without flags.
    data = temporal.merge(
        flags,
        on=["patient_id", "visit_number"],
        how="left",
        validate="many_to_one",
    )

    data["quality_flags"] = data["quality_flags"].fillna("")
    data["visit_number"] = pd.to_numeric(
        data["visit_number"], errors="coerce"
    )
    data["wound_area_pct"] = pd.to_numeric(
        data["wound_area_pct"], errors="coerce"
    )

    data = data.dropna(
        subset=["patient_id", "visit_number", "wound_area_pct"]
    )

    return data


def plot_wound_area(patient_data, patient_id):
    """Plot wound-area percentage and highlight flagged visits."""
    patient_data = patient_data.sort_values("visit_number")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        patient_data["visit_number"],
        patient_data["wound_area_pct"],
        marker="o",
        linewidth=2,
        label="Wound area (%)",
    )

    flagged = patient_data[is_flagged(patient_data["quality_flags"])]

    if not flagged.empty:
        ax.scatter(
            flagged["visit_number"],
            flagged["wound_area_pct"],
            marker="x",
            s=100,
            label="Quality-flagged visit",
        )

        # Add a short annotation for each flagged point.
        for _, row in flagged.iterrows():
            ax.annotate(
                "Flagged",
                (
                    row["visit_number"],
                    row["wound_area_pct"],
                ),
                xytext=(4, 8),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_title(f"{patient_id}: Wound-area trajectory")
    ax.set_xlabel("Recorded visit number")
    ax.set_ylabel("Wound area (% of image)")
    ax.set_xticks(patient_data["visit_number"].unique())
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = OUTPUT_DIR / f"{patient_id}_wound_area.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_tissue_composition(patient_data, patient_id):
    """Plot tissue percentages when the required columns exist."""
    tissue_columns = {}

    for tissue_name, candidates in TISSUE_CANDIDATES.items():
        column = find_column(patient_data, candidates)

        if column is not None:
            tissue_columns[tissue_name] = column

    if not tissue_columns:
        print(
            f"Skipped tissue plot for {patient_id}: "
            "no matching tissue-percentage columns found."
        )
        return

    patient_data = patient_data.sort_values("visit_number")

    fig, ax = plt.subplots(figsize=(10, 6))

    for tissue_name, column in tissue_columns.items():
        values = pd.to_numeric(
            patient_data[column], errors="coerce"
        )

        ax.plot(
            patient_data["visit_number"],
            values,
            marker="o",
            linewidth=2,
            label=tissue_name,
        )

    flagged = patient_data[is_flagged(patient_data["quality_flags"])]

    # Mark flagged visits with vertical dotted lines.
    for visit in flagged["visit_number"].unique():
        ax.axvline(
            x=visit,
            linestyle=":",
            alpha=0.4,
        )

    ax.set_title(f"{patient_id}: Tissue composition by visit")
    ax.set_xlabel("Recorded visit number")
    ax.set_ylabel("Tissue composition (%)")
    ax.set_xticks(patient_data["visit_number"].unique())
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = OUTPUT_DIR / f"{patient_id}_tissue_composition.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()

    print(f"Loaded records: {len(data)}")
    print(
        "Patients in dataset:",
        data["patient_id"].nunique(),
    )
    print()

    for patient_id in PATIENTS_TO_PLOT:
        patient_data = data[
            data["patient_id"].astype(str) == patient_id
        ].copy()

        if patient_data.empty:
            print(f"No records found for {patient_id}; skipping.")
            continue

        print(
            f"{patient_id}: {len(patient_data)} visits, "
            f"{(is_flagged(patient_data["quality_flags"]).sum())} "
            "flagged visits"
        )

        plot_wound_area(patient_data, patient_id)
        plot_tissue_composition(patient_data, patient_id)
        print()

    print("Trajectory plotting complete.")


if __name__ == "__main__":
    main()
