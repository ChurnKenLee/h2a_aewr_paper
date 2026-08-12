from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import polars as pl

BRANCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BRANCH_ROOT))

from mcw.estimands import (
    CoefficientLayout,
    CombinedOutcomeTerm,
    CommonCoefficientMatrix,
    RowGradient,
    RowVector,
    TargetPopulation,
    apply_delta_method,
    average_unit_ratio_derivative,
    direct_combined_outcome_oracle,
    fixed_observed_mean_elasticity,
    fixed_observed_mean_percent_per_treatment_unit,
    hours_per_position_derivative,
    linear_primitive_effect,
    per_baseline_worker_effect,
    positions_per_application_derivative,
)


class EstimandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = ("01001:2019", "01003:2019", "01005:2019", "01007:2019")
        self.layout = CoefficientLayout(
            coefficient_names=("current_dose", "bite_interaction"),
            outcome_names=(
                "applications",
                "certified_positions",
                "certified_hours",
            ),
        )
        self.coefficients = CommonCoefficientMatrix(
            values=np.array(
                [
                    [-0.4, -2.0, -30.0],
                    [-0.1, -0.5, -8.0],
                ]
            ),
            layout=self.layout,
        )
        self.application_gradient = RowGradient(
            "application_effect_rows",
            self.rows,
            np.array([[1.0, 0.0], [1.0, 1.0], [0.5, -1.0], [2.0, 0.5]]),
            self.layout.coefficient_names,
        )
        self.position_gradient = RowGradient(
            "position_effect_rows",
            self.rows,
            np.array([[1.0, 0.5], [0.5, 1.0], [1.5, -0.5], [1.0, 2.0]]),
            self.layout.coefficient_names,
        )
        self.hour_gradient = RowGradient(
            "hour_effect_rows",
            self.rows,
            np.array([[0.5, 1.0], [1.0, 0.5], [2.0, -0.5], [1.5, 1.0]]),
            self.layout.coefficient_names,
        )
        self.applications = RowVector(
            "observed_applications", self.rows, np.array([1.0, 4.0, 2.0, 8.0])
        )
        self.positions = RowVector(
            "observed_positions", self.rows, np.array([2.0, 5.0, 8.0, 12.0])
        )
        self.hours = RowVector(
            "observed_hours", self.rows, np.array([20.0, 70.0, 80.0, 300.0])
        )
        self.target = TargetPopulation(
            name="declared_target",
            row_ids=self.rows,
            include=np.array([True, True, False, True]),
            weights=np.array([1.0, 2.0, 99.0, 0.5]),
        )

    def _finite_difference(self, function, values, epsilon=1e-6):
        derivative = np.empty_like(values)
        for index in np.ndindex(values.shape):
            upper = values.copy()
            lower = values.copy()
            upper[index] += epsilon
            lower[index] -= epsilon
            derivative[index] = (function(upper) - function(lower)) / (2 * epsilon)
        return derivative

    def test_linear_primitive_gradient_and_direct_oracle(self) -> None:
        gradient = linear_primitive_effect(
            name="application_ame",
            outcome="applications",
            row_gradient=self.application_gradient,
            target=self.target,
            layout=self.layout,
        )
        direct = direct_combined_outcome_oracle(
            self.coefficients,
            target=self.target,
            terms=(
                CombinedOutcomeTerm(
                    outcome="applications",
                    row_gradient=self.application_gradient,
                ),
            ),
        )
        self.assertAlmostEqual(gradient.evaluate(self.coefficients), direct)
        self.assertTrue(np.all(gradient.values[:, 1:] == 0.0))

    def test_ratio_of_aggregates_gradient_matches_finite_difference(self) -> None:
        gradient = positions_per_application_derivative(
            positions_row_gradient=self.position_gradient,
            applications_row_gradient=self.application_gradient,
            observed_positions=self.positions,
            observed_applications=self.applications,
            target=self.target,
            layout=self.layout,
        )
        observed_positions = self.target.weighted_sum(self.positions)
        observed_applications = self.target.weighted_sum(self.applications)
        position_effect = self.target.aggregate_gradient(
            self.position_gradient, "weighted_sum"
        )
        application_effect = self.target.aggregate_gradient(
            self.application_gradient, "weighted_sum"
        )

        def direct(values):
            numerator = position_effect @ values[:, 1]
            denominator = application_effect @ values[:, 0]
            return (
                numerator / observed_applications
                - observed_positions * denominator / observed_applications**2
            )

        numerical = self._finite_difference(direct, self.coefficients.values.copy())
        np.testing.assert_allclose(gradient.values, numerical, rtol=1e-8, atol=1e-8)

        direct_oracle = direct_combined_outcome_oracle(
            self.coefficients,
            target=self.target,
            terms=(
                CombinedOutcomeTerm(
                    outcome="certified_positions",
                    row_gradient=self.position_gradient,
                    multiplier=1.0 / observed_applications,
                    aggregation="weighted_sum",
                ),
                CombinedOutcomeTerm(
                    outcome="applications",
                    row_gradient=self.application_gradient,
                    multiplier=-observed_positions / observed_applications**2,
                    aggregation="weighted_sum",
                ),
            ),
        )
        self.assertAlmostEqual(gradient.evaluate(self.coefficients), direct_oracle)

    def test_ratio_of_aggregates_and_average_unit_ratio_are_distinct(self) -> None:
        ratio_of_aggregates = positions_per_application_derivative(
            positions_row_gradient=self.position_gradient,
            applications_row_gradient=self.application_gradient,
            observed_positions=self.positions,
            observed_applications=self.applications,
            target=self.target,
            layout=self.layout,
        )
        average_ratio = average_unit_ratio_derivative(
            name="average_county_positions_per_application_derivative",
            numerator_outcome="certified_positions",
            denominator_outcome="applications",
            numerator_row_gradient=self.position_gradient,
            denominator_row_gradient=self.application_gradient,
            observed_numerator=self.positions,
            observed_denominator=self.applications,
            target=self.target,
            layout=self.layout,
        )
        self.assertEqual(ratio_of_aggregates.kind, "ratio_of_aggregates_derivative")
        self.assertEqual(average_ratio.kind, "average_unit_ratio_derivative")
        self.assertFalse(np.allclose(ratio_of_aggregates.values, average_ratio.values))

    def test_hours_per_position_wrapper_uses_both_outcome_blocks(self) -> None:
        gradient = hours_per_position_derivative(
            hours_row_gradient=self.hour_gradient,
            positions_row_gradient=self.position_gradient,
            observed_hours=self.hours,
            observed_positions=self.positions,
            target=self.target,
            layout=self.layout,
        )
        self.assertTrue(np.all(gradient.values[:, 0] == 0.0))
        self.assertTrue(np.any(gradient.values[:, 1] != 0.0))
        self.assertTrue(np.any(gradient.values[:, 2] != 0.0))

    def test_per_worker_and_elasticity_rescaling(self) -> None:
        workers = RowVector(
            "baseline_workers", self.rows, np.array([10.0, 20.0, 30.0, 40.0])
        )
        rate = per_baseline_worker_effect(
            name="positions_per_1000_baseline_workers",
            outcome="certified_positions",
            row_gradient=self.position_gradient,
            baseline_workers=workers,
            target=self.target,
            layout=self.layout,
            scale=1000.0,
        )
        scaled_workers = RowVector(
            "baseline_workers_x10", self.rows, workers.values * 10.0
        )
        scaled_rate = per_baseline_worker_effect(
            name="positions_per_1000_baseline_workers_x10",
            outcome="certified_positions",
            row_gradient=self.position_gradient,
            baseline_workers=scaled_workers,
            target=self.target,
            layout=self.layout,
            scale=1000.0,
        )
        np.testing.assert_allclose(scaled_rate.values, rate.values / 10.0)

        elasticity = fixed_observed_mean_elasticity(
            name="positions_elasticity",
            outcome="certified_positions",
            row_gradient=self.position_gradient,
            observed_outcome=self.positions,
            target=self.target,
            layout=self.layout,
        )
        scaled_gradient = RowGradient(
            "position_effect_rows_x7",
            self.rows,
            self.position_gradient.values * 7.0,
            self.layout.coefficient_names,
        )
        scaled_positions = RowVector(
            "positions_x7", self.rows, self.positions.values * 7.0
        )
        scaled_elasticity = fixed_observed_mean_elasticity(
            name="positions_elasticity_x7",
            outcome="certified_positions",
            row_gradient=scaled_gradient,
            observed_outcome=scaled_positions,
            target=self.target,
            layout=self.layout,
        )
        np.testing.assert_allclose(scaled_elasticity.values, elasticity.values)

        percent_per_unit = fixed_observed_mean_percent_per_treatment_unit(
            name="positions_percent_of_mean_per_dollar",
            outcome="certified_positions",
            row_gradient=self.position_gradient,
            observed_outcome=self.positions,
            target=self.target,
            layout=self.layout,
        )
        self.assertEqual(
            percent_per_unit.kind,
            "fixed_observed_mean_percent_per_treatment_unit",
        )
        np.testing.assert_allclose(percent_per_unit.values, elasticity.values)
        self.assertNotEqual(percent_per_unit.kind, elasticity.kind)

    def test_denominator_must_have_exact_same_rows_and_order(self) -> None:
        reordered = RowVector(
            "reordered_positions",
            tuple(reversed(self.rows)),
            self.positions.values[::-1],
        )
        with self.assertRaisesRegex(ValueError, "exact rows in the same order"):
            fixed_observed_mean_elasticity(
                name="invalid",
                outcome="certified_positions",
                row_gradient=self.position_gradient,
                observed_outcome=reordered,
                target=self.target,
                layout=self.layout,
            )

        active = TargetPopulation(
            "positive_applications",
            self.rows,
            np.array([True, False, False, True]),
            np.ones(4),
        )
        active_mean = active.weighted_mean(self.positions)
        all_mean = TargetPopulation.all_rows(self.rows).weighted_mean(self.positions)
        self.assertNotEqual(active_mean, all_mean)

    def test_delta_uses_cross_outcome_covariance_blocks(self) -> None:
        gradient = positions_per_application_derivative(
            positions_row_gradient=self.position_gradient,
            applications_row_gradient=self.application_gradient,
            observed_positions=self.positions,
            observed_applications=self.applications,
            target=self.target,
            layout=self.layout,
        )
        rng = np.random.default_rng(4321)
        j = len(self.layout.outcome_names)
        k = len(self.layout.coefficient_names)
        loading = rng.normal(size=(j * k, j * k))
        full_covariance = loading @ loading.T / 100.0
        blocks = np.empty((j, j, k, k))
        for left in range(j):
            for right in range(j):
                blocks[left, right] = full_covariance[
                    left * k : (left + 1) * k,
                    right * k : (right + 1) * k,
                ]
        result = apply_delta_method(gradient, self.coefficients, blocks)
        outcome_major_gradient = gradient.values.T.reshape(-1)
        expected = float(
            outcome_major_gradient @ full_covariance @ outcome_major_gradient
        )
        self.assertAlmostEqual(result.variance, expected)
        self.assertAlmostEqual(result.estimate, gradient.evaluate(self.coefficients))

    def test_polars_constructors_preserve_row_alignment(self) -> None:
        frame = pl.DataFrame(
            {
                "row": list(self.rows),
                "keep": [True, True, False, True],
                "weight": [1.0, 2.0, 9.0, 0.5],
                "observed": [1.0, 4.0, 2.0, 8.0],
                "current_dose": [1.0, 1.0, 0.5, 2.0],
                "bite_interaction": [0.0, 1.0, -1.0, 0.5],
            }
        )
        target = TargetPopulation.from_polars(
            frame,
            row_id_column="row",
            include_column="keep",
            weight_column="weight",
        )
        observed = RowVector.from_polars(
            frame, row_id_column="row", value_column="observed"
        )
        gradient = RowGradient.from_polars(
            frame,
            row_id_column="row",
            coefficient_columns=self.layout.coefficient_names,
            name="polars_gradient",
        )
        self.assertEqual(target.observations, 3)
        self.assertAlmostEqual(target.weighted_sum(observed), 13.0)
        np.testing.assert_allclose(
            target.aggregate_gradient(gradient, "weighted_sum"),
            np.array([4.0, 2.25]),
        )


if __name__ == "__main__":
    unittest.main()
