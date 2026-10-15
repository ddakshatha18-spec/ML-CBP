# Wound Healing Trajectory Analysis

## Overview

This module analyzes longitudinal wound-image measurements from the LUTSeg dataset to study changes in wound area and tissue composition across recorded patient visits.

The objective is to move beyond individual-image analysis by comparing measurements across visits and summarizing observed wound-area trends for each patient.

This module is an experimental, mask-based analysis baseline. It does not provide clinically validated healing predictions.

## Objectives

- Extract wound-area and tissue-composition measurements from segmentation masks.
- Calculate changes between consecutive visits for each patient.
- Categorize observed wound-area changes using an experimental threshold.
- Generate patient-level summaries across recorded visits.

## Dataset

The module uses the LUTSeg dataset, which contains longitudinal wound images and corresponding segmentation masks.

The current local analysis processed 141 images from 39 pseudonymized patients.

Measurements are calculated from the dataset's annotated masks. Wound area is expressed as a percentage of total image pixels, not as physical wound area in square centimetres.

## Module Components

### 1. `extract_mask_measurements.py`

Extracts measurements from wound and tissue segmentation masks.

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

Applies a configurable experimental threshold to wound-area changes.

The current threshold is ±5 percentage points.

| Category | Rule |
|---|---|
| Area decreased | Change is below −5 percentage points |
| Area increased | Change is above +5 percentage points |
| Relatively stable | Change is between −5 and +5 percentage points, inclusive |
| Insufficient data | No previous visit or missing comparison value |

The threshold is an experimental setting, not a clinically validated decision boundary.

Output:

`outputs/trajectory_predictions.csv`

### 4. `patient_summary.py`

Summarizes the first and last recorded wound-area percentages for each patient and counts the observed area-decrease, area-increase, and relatively stable comparisons.

Output:

`outputs/patient_summary.csv`

## Execution

Run these commands from the repository root after installing the required Python packages and downloading the dataset into `data/LUTSeg/`.

```bash
python person2_trajectory_model/extract_mask_measurements.py
python person2_trajectory_model/temporal_features.py
python person2_trajectory_model/trajectory_baseline.py
python person2_trajectory_model/patient_summary.py
```

The temporal feature, baseline, and summary scripts use pandas. The measurement extraction script uses Pillow.

The dataset and generated CSV files are expected to remain local and are not included in this module's source-code commit.

## Current Analysis Results

The current baseline processed:

- 141 images.
- 39 patients.
- 102 consecutive-visit comparisons.

Using the experimental ±5 percentage-point threshold, the visit-level categories were:

| Category | Count |
|---|---:|
| Area decreased | 35 |
| Area increased | 13 |
| Relatively stable | 54 |
| Insufficient data | 39 |
| **Total** | **141** |

These are observed image-area change categories, not validated clinical outcome labels.

## Limitations

- Wound area is measured as a proportion of image pixels; camera distance, framing, and image conditions may affect comparisons.
- Recorded visit numbers do not establish the elapsed time between visits in days.
- The current analysis uses annotated segmentation masks rather than independently generated predictions from a deployed end-to-end system.
- The ±5 percentage-point threshold is experimental and has not been clinically validated.
- LUTSeg does not provide validated trajectory labels for this module's proposed categories.
- The current results do not establish predictive accuracy, clinical effectiveness, or generalization to other datasets.

## Future Work

- Validate mask alignment and measurement calculations.
- Analyze tissue-composition trends alongside wound-area changes.
- Examine patient-level trajectories and the effects of variable visit intervals.
- Establish defensible evaluation targets and a patient-level evaluation protocol.
- Evaluate the method on independent data before making clinical performance claims.

## Disclaimer

This module is intended for research and educational purposes. It is not a medical diagnostic tool and must not be used to make clinical decisions.