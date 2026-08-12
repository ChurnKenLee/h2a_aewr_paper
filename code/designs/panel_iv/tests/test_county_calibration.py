"""Synthetic checks for the county-level FLS calibration."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).parents[1] / "04_recover_fls_geography.py"
SPEC = importlib.util.spec_from_file_location("panel_iv_county_recovery", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {MODULE_PATH}")
RECOVERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECOVERY)


class CompositionMomentTests(unittest.TestCase):
    def test_equal_rates_are_valid_and_inactive(self) -> None:
        residual = RECOVERY.composition_residual(
            14.25,
            14.25,
            14.25,
            np.array([100.0, 20.0, 5.0]),
            np.array([10.0, 40.0, 8.0]),
        )
        np.testing.assert_array_equal(residual, np.zeros(3))
        standardized = RECOVERY.standardize_moment(
            residual,
            target=0.0,
            prior=np.array([0.2, 0.3, 0.5]),
        )
        self.assertFalse(standardized["active"])
        self.assertEqual(
            standardized["status"], "inactive_zero_prior_variation"
        )

    def test_rounded_near_equal_rates_use_the_undivided_residual(self) -> None:
        crop = np.array([100.0, 50.0, 15.0])
        animal = np.array([20.0, 80.0, 35.0])
        residual = RECOVERY.composition_residual(
            14.20,
            13.70,
            14.00,
            crop,
            animal,
        )
        expected = (14.20 - 14.00) * crop + (13.70 - 14.00) * animal
        np.testing.assert_allclose(residual, expected, rtol=0, atol=1e-12)
        self.assertTrue(np.all(np.isfinite(residual)))

        nearly_equal = RECOVERY.composition_residual(
            12.34,
            12.35,
            12.34,
            crop,
            animal,
        )
        np.testing.assert_allclose(nearly_equal, 0.01 * animal, atol=1e-12)
        standardized = RECOVERY.standardize_moment(
            nearly_equal,
            target=0.0,
            prior=np.array([0.5, 0.3, 0.2]),
        )
        self.assertTrue(standardized["active"])


class SeasonalAndSolverTests(unittest.TestCase):
    def test_four_quarters_have_three_independent_contrasts(self) -> None:
        basis = RECOVERY.helmert_basis(4)
        self.assertEqual(basis.shape, (4, 3))
        np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.ones(4) @ basis, np.zeros(3), atol=1e-12)
        self.assertEqual(np.linalg.matrix_rank(basis), 3)

    def test_soft_entropy_weights_sum_to_one(self) -> None:
        prior = np.array([0.2, 0.3, 0.5])
        design = np.array([[-1.0, 0.5], [0.0, -1.0], [1.0, 0.4]])
        target = prior @ design
        solution = RECOVERY.solve_soft_entropy_batch(prior, design, target)
        self.assertTrue(bool(solution["success"][0]))
        np.testing.assert_allclose(solution["weights"].sum(axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(solution["weights"][0], prior, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
