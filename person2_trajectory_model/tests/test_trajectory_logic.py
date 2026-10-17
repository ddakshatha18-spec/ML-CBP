
import unittest
import pandas as pd

from person2_trajectory_model.trajectory_baseline import classify_trajectory


def make_row(has_previous_visit=True, change=0.0):
    return pd.Series({
        "has_previous_visit": has_previous_visit,
        "wound_area_change_pp": change,
    })


class TestTrajectoryLogic(unittest.TestCase):
    def test_decreased_area(self):
        row = make_row(change=-5.1)
        self.assertEqual(classify_trajectory(row), "Area decreased")

    def test_increased_area(self):
        row = make_row(change=5.1)
        self.assertEqual(classify_trajectory(row), "Area increased")

    def test_exact_threshold_is_stable(self):
        self.assertEqual(
            classify_trajectory(make_row(change=-5.0)),
            "Relatively stable",
        )
        self.assertEqual(
            classify_trajectory(make_row(change=5.0)),
            "Relatively stable",
        )

    def test_small_change_is_stable(self):
        for change in [-2.0, 0.0, 3.5]:
            with self.subTest(change=change):
                self.assertEqual(
                    classify_trajectory(make_row(change=change)),
                    "Relatively stable",
                )

    def test_missing_change_is_insufficient(self):
        row = make_row(change=float("nan"))
        self.assertEqual(
            classify_trajectory(row),
            "Insufficient data",
        )

    def test_no_previous_visit_is_insufficient(self):
        row = make_row(has_previous_visit=False, change=-8.0)
        self.assertEqual(
            classify_trajectory(row),
            "Insufficient data",
        )


if __name__ == "__main__":
    unittest.main()
