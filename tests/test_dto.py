from src.dto import TrajectoryInputDTO, VisitFeatureDTO


def test_trajectory_requires_chronological_visits():
    visits = [
        VisitFeatureDTO("p1", "w1", "v2", 10),
        VisitFeatureDTO("p1", "w1", "v1", 0),
    ]

    try:
        TrajectoryInputDTO("p1", "w1", visits, 2)
    except ValueError as exc:
        assert "ordered" in str(exc)
    else:
        raise AssertionError("Expected chronological-order validation")


def test_trajectory_sequence_length_matches_visits():
    visits = [
        VisitFeatureDTO("p1", "w1", "v1", 0)
    ]

    try:
        TrajectoryInputDTO("p1", "w1", visits, 2)
    except ValueError as exc:
        assert "sequence_length" in str(exc)
    else:
        raise AssertionError("Expected sequence-length validation")
