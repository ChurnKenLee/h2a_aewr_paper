"""Dense-design oracles for the version-4 FWL implementation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

BRANCH_ROOT = Path(__file__).resolve().parents[1]
if str(BRANCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BRANCH_ROOT))

from mcw.fwl import NestedFixedEffectProjector, NoFixedEffectProjector, fit_common_ols


def _fixture() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(20260812)
    parents = np.repeat(["a", "b"], [4, 3])
    units = np.array([f"u{i}" for i in range(parents.size)])
    years = np.array(["2011", "2012", "2013", "2014"])
    row_unit = np.repeat(units, years.size)
    row_parent = np.repeat(parents, years.size)
    row_year = np.tile(years, units.size)
    n = row_unit.size
    causal = rng.normal(size=(n, 2))
    nuisance = rng.normal(size=(n, 3))
    unit_effect = rng.normal(size=(units.size, 2))
    parent_year_effect = rng.normal(size=(2, years.size, 2))
    beta = np.array([[0.7, -0.1], [-0.3, 0.5], [0.2, 0.1], [0.4, -0.2], [-0.1, 0.3]])
    unit_code = np.repeat(np.arange(units.size), years.size)
    parent_code = np.repeat(np.repeat(np.arange(2), [4, 3]), years.size)
    year_code = np.tile(np.arange(years.size), units.size)
    x = np.column_stack((causal, nuisance))
    outcomes = (
        x @ beta
        + unit_effect[unit_code]
        + parent_year_effect[parent_code, year_code]
        + rng.normal(scale=0.1, size=(n, 2))
    )
    return row_unit, row_year, row_parent, causal, nuisance, outcomes


def _dense_fixed_effects(
    unit: np.ndarray, year: np.ndarray, parent: np.ndarray
) -> np.ndarray:
    unit_levels = np.unique(unit)
    year_levels = np.unique(year)
    parent_levels = np.unique(parent)
    blocks = [(unit[:, None] == unit_levels).astype(float)]
    for parent_level in parent_levels:
        for year_level in year_levels[1:]:
            blocks.append(
                ((parent == parent_level) & (year == year_level)).astype(float)[:, None]
            )
    return np.column_stack(blocks)


class FWLTest(unittest.TestCase):
    def test_pooled_explicit_nuisance_matches_dense_ols_and_leverage(self) -> None:
        rng = np.random.default_rng(20260813)
        groups = np.repeat(np.array(["north", "south", "west"]), 20)
        years = np.tile(np.repeat(np.arange(2011, 2016), 4), 3)
        n_rows = groups.size
        causal = rng.normal(size=(n_rows, 2))
        group_levels = np.unique(groups)
        year_levels = np.unique(years)
        nuisance = np.column_stack(
            (
                np.ones(n_rows),
                (groups[:, None] == group_levels).astype(float),
                (years[:, None] == year_levels).astype(float),
                rng.normal(size=(n_rows, 2)),
            )
        )
        nuisance_names = (
            "intercept",
            *(f"group_{group}" for group in group_levels),
            *(f"year_{year}" for year in year_levels),
            "projection_1",
            "projection_2",
        )
        outcome_loading = rng.normal(size=(causal.shape[1] + nuisance.shape[1], 2))
        outcomes = np.column_stack((causal, nuisance)) @ outcome_loading + rng.normal(
            scale=0.1, size=(n_rows, 2)
        )

        projector = NoFixedEffectProjector.from_row_count(n_rows)
        fit = fit_common_ols(
            projector,
            causal,
            nuisance,
            outcomes,
            ("dose_1", "dose_2"),
            nuisance_names,
            ("outcome_1", "outcome_2"),
        )

        dense_design = fit.raw_design
        dense_beta, *_ = np.linalg.lstsq(dense_design, outcomes, rcond=None)
        dense_residual = outcomes - dense_design @ dense_beta
        dense_bread = np.linalg.inv(dense_design.T @ dense_design)
        dense_leverage = np.einsum(
            "ij,ij->i", dense_design @ dense_bread, dense_design, optimize=True
        )

        np.testing.assert_allclose(fit.coefficient, dense_beta, atol=2e-11)
        np.testing.assert_allclose(fit.residual, dense_residual, atol=2e-11)
        np.testing.assert_allclose(fit.fitted, outcomes - dense_residual, atol=2e-11)
        np.testing.assert_allclose(fit.bread, dense_bread, atol=2e-11)
        np.testing.assert_allclose(fit.leverage, dense_leverage, atol=2e-11)
        self.assertEqual(fit.fixed_effect_rank, 0)
        self.assertEqual(fit.model_rank, dense_design.shape[1])
        self.assertEqual(fit.residual_df, n_rows - dense_design.shape[1])
        self.assertEqual(len(fit.dropped_nuisance_names), 2)
        self.assertAlmostEqual(float(fit.leverage.sum()), fit.model_rank)
        self.assertLess(fit.solve_relative_residual, 1e-12)

    def test_pooled_projector_is_identity_and_checks_rows(self) -> None:
        projector = NoFixedEffectProjector(4)
        vector = np.arange(4.0)
        matrix = np.column_stack((vector, vector**2))
        np.testing.assert_array_equal(projector.within(vector), vector)
        np.testing.assert_array_equal(projector.within(matrix), matrix)
        np.testing.assert_array_equal(projector.leverage_diagonal(), np.zeros(4))
        self.assertEqual(projector.rank, 0)
        with self.assertRaisesRegex(ValueError, "row count"):
            projector.within(np.arange(3.0))

    def test_nested_fwl_matches_dense_ols_and_full_leverage(self) -> None:
        unit, year, parent, causal, nuisance, outcomes = _fixture()
        projector = NestedFixedEffectProjector.from_arrays(unit, year, parent)
        fit = fit_common_ols(
            projector,
            causal,
            nuisance,
            outcomes,
            ("d1", "d2"),
            ("z1", "z2", "z3"),
            ("y1", "y2"),
        )
        fixed_effects = _dense_fixed_effects(unit, year, parent)
        dense_design = np.column_stack((fixed_effects, fit.raw_design))
        dense_beta, *_ = np.linalg.lstsq(dense_design, outcomes, rcond=None)
        dense_residual = outcomes - dense_design @ dense_beta
        dense_bread = np.linalg.inv(dense_design.T @ dense_design)
        dense_leverage = np.einsum(
            "ij,ij->i", dense_design @ dense_bread, dense_design, optimize=True
        )

        np.testing.assert_allclose(
            fit.coefficient, dense_beta[-fit.coefficient.shape[0] :], atol=2e-11
        )
        np.testing.assert_allclose(fit.residual, dense_residual, atol=2e-11)
        np.testing.assert_allclose(fit.leverage, dense_leverage, atol=2e-11)
        self.assertEqual(fit.fixed_effect_rank, fixed_effects.shape[1])
        self.assertLess(fit.solve_relative_residual, 1e-12)

    def test_projector_annihilates_both_fixed_effect_sets(self) -> None:
        unit, year, parent, causal, *_ = _fixture()
        projector = NestedFixedEffectProjector.from_arrays(unit, year, parent)
        within = projector.within(causal)
        fixed_effects = _dense_fixed_effects(unit, year, parent)
        np.testing.assert_allclose(fixed_effects.T @ within, 0.0, atol=1e-12)

    def test_causal_first_contract_never_drops_duplicate_causal_term(self) -> None:
        unit, year, parent, causal, nuisance, outcomes = _fixture()
        projector = NestedFixedEffectProjector.from_arrays(unit, year, parent)
        duplicate = np.column_stack((causal[:, 0], causal[:, 0]))
        with self.assertRaisesRegex(ValueError, "causal block has rank"):
            fit_common_ols(
                projector,
                duplicate,
                nuisance,
                outcomes,
                ("d1", "d1_duplicate"),
                ("z1", "z2", "z3"),
                ("y1", "y2"),
            )

    def test_time_invariant_causal_coordinate_is_not_fit_from_roundoff(self) -> None:
        unit, year, parent, _, nuisance, outcomes = _fixture()
        projector = NestedFixedEffectProjector.from_arrays(unit, year, parent)
        unit_level_dose = np.repeat(
            np.linspace(-1.0, 1.0, np.unique(unit).size),
            np.unique(year).size,
        )[:, None]
        with self.assertRaisesRegex(ValueError, "numerically absorbed"):
            fit_common_ols(
                projector,
                unit_level_dose,
                nuisance,
                outcomes,
                ("time_invariant_dose",),
                ("z1", "z2", "z3"),
                ("y1", "y2"),
            )


if __name__ == "__main__":
    unittest.main()
