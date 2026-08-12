"""Reporting-target tests shared by full-history and one-lag models."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import polars as pl

BRANCH_ROOT = Path(__file__).resolve().parents[1]
if str(BRANCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BRANCH_ROOT))

from mcw.design import Specification
from mcw.estimands import (
    CoefficientLayout,
    RowGradient,
    RowVector,
    TargetPopulation,
)
from mcw.pipeline import (
    _average_current_contrast,
    _block_view,
    _current_row_gradient,
    _fixed_observed_mean_metric,
    _proportional_outcome_gradient,
)


class ReportingTargetTest(unittest.TestCase):
    @staticmethod
    def _specification(treatment: str, transform: str) -> Specification:
        return Specification(
            specification_id=f"{treatment}__{transform}",
            stage="test",
            treatment=treatment,
            history="full",
            fixed_effects="county_year",
            moderator_set="none",
            cluster="aewr_region",
            treatment_transform=transform,
        )

    def test_observed_mean_metric_follows_exact_treatment_coordinate(self) -> None:
        rows = ("a", "b")
        layout = CoefficientLayout(("effect",), ("applications",))
        row_gradient = RowGradient(
            "current",
            rows,
            np.array([[2.0], [4.0]]),
            layout.coefficient_names,
        )
        observed = RowVector("applications", rows, np.array([10.0, 30.0]))
        target = TargetPopulation.all_rows(rows)
        cases = (
            ("aewr_log_level", "continuous_raw", True),
            ("aewr_log_change", "continuous_raw", True),
            ("aewr_log_level", "binary_median", False),
            ("aewr_dollar_level", "continuous_raw", False),
            ("exposure_log_f0809", "continuous_raw", False),
        )
        for treatment, transform, is_elasticity in cases:
            with self.subTest(treatment=treatment, transform=transform):
                metric = _fixed_observed_mean_metric(
                    self._specification(treatment, transform),
                    outcome="applications",
                    row_gradient=row_gradient,
                    observed_outcome=observed,
                    target=target,
                    layout=layout,
                )
                self.assertAlmostEqual(metric.values[0, 0], 15.0)
                if is_elasticity:
                    self.assertEqual(
                        metric.name, "applications_elasticity_at_observed_mean"
                    )
                    self.assertEqual(metric.kind, "fixed_observed_mean_elasticity")
                else:
                    self.assertEqual(
                        metric.name,
                        "applications_percent_of_observed_mean_per_treatment_unit",
                    )
                    self.assertEqual(
                        metric.kind,
                        "fixed_observed_mean_percent_per_treatment_unit",
                    )

    def test_proportional_cross_outcome_gradient_factorization(self) -> None:
        base = np.array([1.0, -2.0, 0.5])
        gradient = np.column_stack((base, np.zeros(3), -3.0 * base))
        result = _proportional_outcome_gradient(gradient)
        self.assertIsNotNone(result)
        assert result is not None
        direction, active, scales = result
        np.testing.assert_array_equal(active, np.array([0, 2]))
        np.testing.assert_allclose(
            direction[:, None] * scales[None, :], gradient[:, active]
        )
        self.assertIsNone(
            _proportional_outcome_gradient(
                np.column_stack((base, np.array([1.0, 0.0, 0.0])))
            )
        )

    def test_cross_outcome_block_view_preserves_outcome_major_layout(self) -> None:
        covariance = np.arange(36.0).reshape(6, 6)
        blocks = _block_view(covariance, coefficient_count=2, outcomes=3)
        self.assertTrue(np.shares_memory(blocks, covariance))
        np.testing.assert_array_equal(blocks[1, 2], covariance[2:4, 4:6])

    def test_coefficient_and_row_gradients_share_2013_2022_target(self) -> None:
        names = (
            "effect_y2012_h2011",
            "effect_y2012_h2012",
            "effect_y2013_h2013",
            "effect_y2013_h2013__x__mc_baseline_bite_z",
            "effect_y2022_h2022",
            "nuisance",
        )
        frame = pl.DataFrame(
            {
                "county_fips": [
                    "01001",
                    "01001",
                    "01001",
                    "01003",
                    "01003",
                    "01003",
                ],
                "aewr_region_id": ["01"] * 6,
                "year": [2012, 2013, 2022, 2012, 2013, 2022],
                "mc_baseline_bite_z": [1.0, 1.0, 1.0, 3.0, 3.0, 3.0],
            }
        )
        contrast = _average_current_contrast(frame, names, causal_count=5)
        np.testing.assert_array_equal(
            contrast,
            np.array([0.0, 0.0, 0.5, 0.0, 0.5, 0.0]),
        )

        gradient = _current_row_gradient(frame, names, causal_count=5).values
        np.testing.assert_array_equal(gradient[0], np.zeros(6))
        np.testing.assert_array_equal(gradient[1], np.array([0, 0, 1, -1, 0, 0]))
        np.testing.assert_array_equal(gradient[2], np.array([0, 0, 0, 0, 1, 0]))
        np.testing.assert_array_equal(gradient[4], np.array([0, 0, 1, 1, 0, 0]))


if __name__ == "__main__":
    unittest.main()
