# ML-CBP Data Contract

## Pipeline

Dataset → Person 1 preprocessing/segmentation → Person 3 dataset/features/evaluation → Person 2 trajectory/model → Person 4 integration/API/UI.

## Identity

Every record uses:

- `patient_id`
- `wound_id`
- `visit_id`

An image filename is not a primary identity.

The same wound across visits must retain the same `wound_id`.

## Longitudinal ordering

Visits belonging to the same wound are grouped and ordered by `day_from_baseline`.

Person 3 produces `TrajectoryInputDTO` for Person 2.

## Leakage prevention

Individual images/visits from the same wound or patient must not be randomly distributed across train/test.

The exact patient- versus wound-level strategy will be documented after the final dataset schema is confirmed.

## Missing values

`None` means unavailable/not measured.

`0.0` means measured and absent.

Person 3 must never invent missing labels, dates, or measurements.

## Proposed trajectory classes

The project contract proposes:

1. Rapid Healing
2. Normal Healing
3. Slow Healing
4. Stalled
5. Regression

These are project-level classes. They are not assumed to be ground-truth labels in any candidate dataset.

A labeling methodology must therefore be defined before supervised trajectory evaluation.

## Dataset handling

Raw datasets, model weights, and generated artifacts must not be committed to Git.

Dataset paths should remain configurable.

## Current status

This commit provides shared DTOs and Person 3 scaffolding.

Dataset-specific adapters will be added only after the team confirms the final dataset and its actual metadata/label schema.
