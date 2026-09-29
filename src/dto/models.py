"""Shared DTO contracts for the ML-CBP wound-healing pipeline."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ImageDTO:
    patient_id: str
    wound_id: str
    visit_id: str
    image_path: str
    capture_date: Optional[str] = None
    day_from_baseline: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    dataset_name: Optional[str] = None


@dataclass(frozen=True)
class SegmentationResultDTO:
    patient_id: str
    wound_id: str
    visit_id: str
    image_path: str
    mask_path: Optional[str]
    wound_area_pixels: Optional[float]
    image_width: Optional[int]
    image_height: Optional[int]
    wound_area_ratio: Optional[float]
    segmentation_confidence: Optional[float]
    processing_status: str


@dataclass(frozen=True)
class TissueFeatureDTO:
    patient_id: str
    wound_id: str
    visit_id: str
    fibrin_ratio: Optional[float] = None
    granulation_ratio: Optional[float] = None
    necrotic_ratio: Optional[float] = None
    epithelial_ratio: Optional[float] = None
    tissue_classification_confidence: Optional[float] = None


@dataclass(frozen=True)
class VisitFeatureDTO:
    patient_id: str
    wound_id: str
    visit_id: str
    day_from_baseline: float
    wound_area: Optional[float] = None
    wound_area_ratio: Optional[float] = None
    fibrin_ratio: Optional[float] = None
    granulation_ratio: Optional[float] = None
    necrotic_ratio: Optional[float] = None
    epithelial_ratio: Optional[float] = None


@dataclass(frozen=True)
class TemporalFeatureDTO:
    patient_id: str
    wound_id: str
    area_change: Optional[float] = None
    area_change_percent: Optional[float] = None
    area_change_rate: Optional[float] = None
    granulation_change: Optional[float] = None
    fibrin_change: Optional[float] = None
    necrotic_change: Optional[float] = None
    epithelial_change: Optional[float] = None


@dataclass(frozen=True)
class TrajectoryInputDTO:
    patient_id: str
    wound_id: str
    visits: List[VisitFeatureDTO] = field(default_factory=list)
    sequence_length: int = 0
    baseline_area: Optional[float] = None
    final_area: Optional[float] = None
    observation_duration_days: Optional[float] = None

    def __post_init__(self) -> None:
        if self.sequence_length != len(self.visits):
            raise ValueError("sequence_length must equal len(visits)")

        if any(
            self.visits[i].day_from_baseline
            > self.visits[i + 1].day_from_baseline
            for i in range(len(self.visits) - 1)
        ):
            raise ValueError("visits must be ordered by day_from_baseline")


@dataclass(frozen=True)
class TrajectoryPredictionDTO:
    patient_id: str
    wound_id: str
    predicted_class: Optional[str]
    probabilities: Dict[str, float]
    model_version: str
    confidence: Optional[float]
    prediction_status: str


@dataclass(frozen=True)
class EvaluationResultDTO:
    dataset_name: str
    split_name: str
    num_samples: int
    accuracy: Optional[float]
    precision_macro: Optional[float]
    recall_macro: Optional[float]
    f1_macro: Optional[float]
    confusion_matrix_path: Optional[str]
    class_distribution: Dict[str, int]
