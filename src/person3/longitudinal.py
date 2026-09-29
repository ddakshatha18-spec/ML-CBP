"""Utilities for building wound-level longitudinal trajectories."""

from collections import defaultdict
from typing import Dict, Iterable, List

from src.dto import TrajectoryInputDTO, VisitFeatureDTO


def build_trajectories(
    visits: Iterable[VisitFeatureDTO],
) -> List[TrajectoryInputDTO]:
    """Group visits by patient/wound and order them chronologically."""

    grouped: Dict[tuple, List[VisitFeatureDTO]] = defaultdict(list)

    for visit in visits:
        grouped[(visit.patient_id, visit.wound_id)].append(visit)

    trajectories = []

    for (patient_id, wound_id), wound_visits in sorted(grouped.items()):
        ordered = sorted(
            wound_visits,
            key=lambda item: item.day_from_baseline,
        )

        areas = [
            visit.wound_area
            for visit in ordered
            if visit.wound_area is not None
        ]

        trajectories.append(
            TrajectoryInputDTO(
                patient_id=patient_id,
                wound_id=wound_id,
                visits=ordered,
                sequence_length=len(ordered),
                baseline_area=areas[0] if areas else None,
                final_area=areas[-1] if areas else None,
                observation_duration_days=(
                    ordered[-1].day_from_baseline
                    - ordered[0].day_from_baseline
                ),
            )
        )

    return trajectories
