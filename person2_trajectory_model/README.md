# Wound Healing Trajectory Analysis

## Overview

This module analyzes longitudinal wound-image measurements from the LUTSeg dataset to study changes in wound area and tissue composition across recorded patient visits.

The objective is to move beyond individual-image analysis by comparing measurements across visits and summarizing observed wound-area trends for each patient.

This module is an experimental, mask-based analysis baseline. It does not provide clinically validated healing predictions.

## Objectives

- Extract wound-area and tissue-composition measurements from segmentation masks.
- Calculate changes between consecutive recorded visits for each patient.
- Categorize observed wound-area changes using an experimental threshold.
- Generate patient-level summaries across recorded visits.
- Visualize wound-area and tissue-composition trajectories.
- Identify records that may warrant manual quality review.
- Examine how baseline categories change under different thresholds and record-exclusion rules.
- Summarize descriptive changes in tissue composition across recorded visits.

## Dataset

The module uses the LUTSeg dataset, which contains wound images and corresponding segmentation masks.

The current local analysis processed:

- **141 images**
- **39 pseudonymized patients**
- **111 training images**
- **30 validation images**
- **102 consecutive-visit comparisons**

Measurements are calculated from annotated segmentation masks. Wound area is expressed as a percentage of total image pixels, not as physical wound area in square centimetres.

The dataset's recorded visit identifiers do not establish the elapsed time between visits in days. Visit intervals must therefore be interpreted as recorded visit-number differences, not as verified time intervals.

## Tissue Classes

The segmentation masks use the following class mapping:

| Class ID | Tissue category |
|---:|---|
| 0 | Background |
| 1 | Epithelial |
| 2 | Slough |
| 3 | Granulation |
| 4 | Necrotic |
| 5 | Other |

Tissue-composition percentages describe the distribution of labeled tissue according to the implementation's measurement definitions. They should not be interpreted as absolute tissue volumes or physical areas.

## Module Components

### 1. `extract_mask_measurements.py`

Extracts wound-area and tissue-composition measurements from segmentation masks.

Measurements include:

- Wound pixel count and wound area percentage.
- Epithelial tissue pixel count and percentage.
- Slough tissue pixel count and percentage.
- Granulation tissue pixel count and percentage.
- Necrotic tissue pixel count and percentage.
- Other tissue pixel count and percentage.

Output:

`outputs/trajectory_measurements.csv`

### 2. `temporal_features.py`

Compares each patient's measurements with the previous recorded visit.

Features include:

- Previous visit number.
- Visit interval in recorded visit numbers.
- Wound-area change in percentage points.
- Relative wound-area change percentage.
- Changes in tissue-composition percentages.
- Indicator for whether a previous visit exists.

Output:

`outputs/temporal_features.csv`

### 3. `trajectory_baseline.py`

Applies an experimental threshold to consecutive-visit wound-area changes.

The current threshold is ±5 percentage points.

| Category | Rule |
|---|---|
| Area decreased | Change is below −5 percentage points |
| Area increased | Change is above +5 percentage points |
| Relatively stable | Change is between −5 and +5 percentage points, inclusive |
| Insufficient data | No previous visit or missing comparison value |

The threshold is an experimental setting, not a clinically validated decision boundary. The labels describe measured image-area changes and are not validated clinical outcome labels.

Output:

`outputs/trajectory_predictions.csv`

### 4. `patient_summary.py`

Summarizes each patient's first and last recorded wound-area percentages and counts the observed area-decrease, area-increase, and relatively stable comparisons.

Output:

`outputs/patient_summary.csv`

### 5. `evaluate_trajectory_baseline.py`

Evaluates the rule-based baseline and produces summary tables for its visit-level categories.

Integrity checks cover:

- Whether each patient's first recorded visit is marked as having no previous visit.
- Whether first-visit wound-area changes are missing as expected.
- Whether duplicate patient/visit keys exist.

Outputs are saved under:

`outputs/baseline_evaluation/`

These checks assess data and evaluation consistency. They do not measure clinical prediction accuracy.

### 6. `threshold_sensitivity.py`

Examines how the distribution of baseline categories changes when different wound-area-change thresholds are used.

The current analysis compares thresholds of ±3, ±5, and ±10 percentage points.

Outputs are saved under:

`outputs/baseline_evaluation/`

Threshold sensitivity indicates how strongly the results depend on the selected boundary. It does not establish which threshold is clinically appropriate.

### 7. `plot_patient_trajectories.py`

Generates patient-level visualizations of wound-area percentages and tissue composition across recorded visits.

The current analysis generated plots for Patients 20, 6, and 22. These are examples selected for trajectory inspection and should not be considered a representative sample of all patients.

Plots are saved under:

`outputs/`

### 8. `quality_flags.py`

Applies heuristic flags to records that may warrant closer inspection.

The current rules include:

- **Large wound-area change:** absolute wound-area change greater than 5 percentage points.
- **Large tissue-composition change:** an absolute change greater than 50 percentage points in a monitored tissue category.
- **Very small wound region:** wound area below 0.2% of image pixels.

The flags identify review candidates; they do not confirm that a measurement or annotation is incorrect.

Output:

`outputs/quality_flags.csv`

### 9. `review_quality_flags.py`

Creates a review table containing flagged records, previous-visit information, wound-area measurements, tissue percentages, and relevant changes.

Output:

`outputs/quality_flag_review.csv`

The review table is intended to help inspect potentially unusual records before deciding how they should be handled in subsequent analyses.

### 10. Quality robustness scripts

`quality_robustness_check.py` examines baseline categories after excluding records with threshold-based quality flags.

`tissue_quality_robustness.py` examines category counts after excluding records flagged for tissue-composition changes or very small wound regions.

These are exploratory sensitivity checks. Excluding records based on heuristic rules does not prove that the remaining records are more accurate or reliable.

**Important interpretation note:** the large-wound-area-change flag uses the same 5-percentage-point boundary as the baseline categories. Agreement between this flag and the baseline category is therefore not independent evidence of data quality.

### 11. `tissue_trend_analysis.py`

Summarizes changes in tissue-composition percentages between consecutive recorded visits.

Output:

`outputs/tissue_trend_summary.csv`

The results are descriptive summaries of percentage-point changes. Large changes may reflect annotation differences, mask composition, or changes in the labeled-tissue denominator. They should not automatically be interpreted as biological or clinical improvement or deterioration.

### 12. Unit tests

The `tests/` directory contains unit tests for the baseline classification rules, including:

- Decreases and increases.
- Exact threshold boundaries.
- Small changes within the stable range.
- Missing comparison values.
- First visits without previous-visit data.

Run the tests from the repository root:

```bash
python -m unittest discover -s person2_trajectory_model/tests -v
```

The tests check the implemented classification behavior. They do not validate the threshold clinically or establish predictive performance.

## Execution

Run the following commands from the repository root after installing the required Python packages and downloading the dataset into `data/LUTSeg/`.

### Core measurement and baseline pipeline

```bash
python person2_trajectory_model/extract_mask_measurements.py
python person2_trajectory_model/temporal_features.py
python person2_trajectory_model/trajectory_baseline.py
python person2_trajectory_model/patient_summary.py
```

### Baseline evaluation and threshold sensitivity

```bash
python person2_trajectory_model/evaluate_trajectory_baseline.py
python person2_trajectory_model/threshold_sensitivity.py
```

### Quality review and robustness analysis

```bash
python person2_trajectory_model/quality_flags.py
python person2_trajectory_model/review_quality_flags.py
python person2_trajectory_model/quality_robustness_check.py
python person2_trajectory_model/tissue_quality_robustness.py
```

### Tissue trends and visualizations

```bash
python person2_trajectory_model/tissue_trend_analysis.py
python person2_trajectory_model/plot_patient_trajectories.py
```

### Unit tests

```bash
python -m unittest discover -s person2_trajectory_model/tests -v
```

Run scripts in an order that ensures their input CSV files have already been generated. The scripts use pandas for tabular analysis; measurement extraction also uses Pillow. Install any additional dependencies required by the visualization scripts in your Python environment.

Generated CSV files, review tables, and plots are written under `person2_trajectory_model/outputs/`. The dataset and generated outputs are expected to remain local and are not included in the source-code commit.

## Current Analysis Results

### Baseline category distribution

The current baseline processed 141 images from 39 patients. Of these, 102 records had a preceding-visit comparison and 39 were first recorded visits or otherwise lacked a usable comparison.

Using the experimental ±5 percentage-point threshold, the visit-level categories were:

| Category | Count |
|---|---:|
| Area decreased | 35 |
| Area increased | 13 |
| Relatively stable | 54 |
| Insufficient data | 39 |
| **Total** | **141** |

The counts describe observed image-area change categories, not validated clinical outcomes.

### Threshold sensitivity

The category distribution changes when the threshold changes.

| Threshold | Area decreased | Relatively stable | Area increased |
|---|---:|---:|---:|
| ±3 percentage points | 41 | 39 | 22 |
| ±5 percentage points | 35 | 54 | 13 |
| ±10 percentage points | 15 | 82 | 5 |

These counts cover the 102 comparable visit transitions. At the wider threshold, more transitions are categorized as relatively stable.

This demonstrates that category counts depend on the selected threshold. It does not establish prediction accuracy or identify a clinically appropriate threshold.

### Quality-flag review

The current quality review identified 58 flagged records and 83 records without a threshold-based flag, out of 141 records.

| Flag type | Records |
|---|---:|
| Large wound-area change | 48 |
| Large tissue-composition change | 16 |
| Very small wound region | 2 |

Flag types can overlap, so the counts by flag type do not sum to the number of unique flagged records.

These flags are heuristic screening criteria, not verified errors. Flagged records should be reviewed before deciding whether to exclude them from further analysis.

### Tissue-composition changes

Across 102 consecutive-visit comparisons, the mean changes in tissue-composition percentage were:

| Tissue category | Mean change (percentage points) |
|---|---:|
| Epithelial | +4.61 |
| Slough | +0.59 |
| Granulation | −4.78 |
| Necrotic | −0.87 |

These are descriptive averages across recorded comparisons. They do not establish a clinical healing trend, and individual changes can differ substantially from the mean.

Tissue percentages are compositional measurements. A change in one category's percentage may partly reflect changes in the proportions of other labeled categories rather than an equivalent change in absolute tissue area.

## Limitations

- **Image-based area:** Wound area is measured as a proportion of image pixels, not physical area. Camera distance, framing, perspective, and image conditions may affect comparisons.
- **Unverified time intervals:** Recorded visit numbers do not establish elapsed time in days. Changes between consecutive recorded visits may span different amounts of time.
- **Annotated masks:** The current analysis uses annotated segmentation masks rather than independently generated predictions from a trained end-to-end model.
- **Experimental threshold:** The ±5 percentage-point threshold is not clinically validated. The category distribution is sensitive to the selected threshold.
- **No validated trajectory labels:** The current analysis does not have validated clinical outcome labels for the proposed trajectory categories.
- **Heuristic quality flags:** Flags identify records for review; they do not establish that the corresponding masks or measurements are wrong.
- **Exploratory robustness checks:** Excluding flagged records does not guarantee improved data quality and can change the analyzed sample.
- **Tissue composition:** Tissue percentages describe relative composition and should not be interpreted as absolute tissue amounts.
- **Limited dataset:** Results from the current 141 images and 39 patients do not establish generalization to other datasets or clinical settings.
- **No clinical performance claim:** The current results do not establish predictive accuracy, clinical effectiveness, diagnostic performance, or suitability for clinical decision-making.

## Future Work

- Validate mask alignment, label conventions, and measurement calculations.
- Inspect flagged masks and unusual patient trajectories.
- Analyze tissue-composition trends alongside wound-area changes.
- Examine patient-level trajectories and variable visit intervals.
- Establish defensible evaluation targets and a patient-level evaluation protocol.
- Separate exploratory threshold selection from final evaluation.
- Evaluate any future trained model on appropriate independent data before making performance claims.

## Disclaimer

This module is intended for research and educational purposes only. It is not a medical diagnostic tool and must not be used to make clinical decisions.