"""Identification-boundary tests for the version-4 history block."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import polars as pl

BRANCH_ROOT = Path(__file__).resolve().parents[1]
if str(BRANCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BRANCH_ROOT))

from mcw.design import ANALYSIS_YEARS, TREATMENT_HISTORY_YEARS, Specification
from mcw.fwl import NestedFixedEffectProjector
from mcw.model import build_causal_matrix, causal_moderator_values


def _history_frame() -> pl.DataFrame:
    rng = np.random.default_rng(20260812)
    region_paths = rng.normal(size=(17, len(TREATMENT_HISTORY_YEARS)))
    county_moderators = rng.normal(size=34)
    rows: list[dict[str, object]] = []
    for county_index in range(34):
        region_index = county_index % 17
        for outcome_year in ANALYSIS_YEARS:
            row: dict[str, object] = {
                "county_fips": f"{county_index:05d}",
                "aewr_region_id": f"{region_index:02d}",
                "year": outcome_year,
                "mc_baseline_bite_z": county_moderators[county_index],
            }
            for history_index, history_year in enumerate(TREATMENT_HISTORY_YEARS):
                row[f"mc_aewr_log_level_{history_year}"] = region_paths[
                    region_index, history_index
                ]
            rows.append(row)
    return pl.DataFrame(rows)


def _spec(history: str, moderator_set: str = "none") -> Specification:
    return Specification(
        specification_id=f"history_{history}",
        stage="test",
        treatment="aewr_log_level",
        history=history,  # type: ignore[arg-type]
        fixed_effects="county_year",
        moderator_set=moderator_set,
        cluster="aewr_region",
    )


def _pooled_spec(history: str, moderator_set: str = "none") -> Specification:
    return Specification(
        specification_id=f"pooled_history_{history}",
        stage="test",
        treatment="aewr_log_level",
        history=history,  # type: ignore[arg-type]
        fixed_effects="pooled_wmc",
        moderator_set=moderator_set,
        cluster="aewr_region",
    )


def _projector(frame: pl.DataFrame) -> NestedFixedEffectProjector:
    return NestedFixedEffectProjector.from_arrays(
        frame["county_fips"].to_numpy(),
        frame["year"].to_numpy(),
    )


def _unrestricted_full_history(
    frame: pl.DataFrame,
) -> tuple[np.ndarray, dict[int, list[int]]]:
    year = frame["year"].to_numpy()
    columns: list[np.ndarray] = []
    boundary_indices = {2011: [], 2012: []}
    for outcome_year in ANALYSIS_YEARS:
        outcome_cell = year == outcome_year
        for history_year in range(2011, outcome_year + 1):
            if history_year in boundary_indices:
                boundary_indices[history_year].append(len(columns))
            columns.append(
                outcome_cell * frame[f"mc_aewr_log_level_{history_year}"].to_numpy()
            )
    return np.column_stack(columns), boundary_indices


class CausalHistoryTest(unittest.TestCase):
    def test_pooled_history_keeps_every_level_coordinate(self) -> None:
        frame = _history_frame()
        full, full_names, full_metadata = build_causal_matrix(
            frame, _pooled_spec("full")
        )
        one_lag, one_lag_names, _ = build_causal_matrix(frame, _pooled_spec("one_lag"))
        self.assertEqual(full.shape[1], 77)
        self.assertEqual(one_lag.shape[1], 22)
        self.assertIn("effect_y2012_h2011", full_names)
        self.assertIn("effect_y2012_h2012", full_names)
        self.assertFalse(any("effect_difference" in name for name in full_names))
        self.assertFalse(any("effect_difference" in name for name in one_lag_names))
        self.assertTrue(
            all(item["identification"] == "level_pooled_wmc" for item in full_metadata)
        )
        self.assertEqual(np.linalg.matrix_rank(full), 77)
        self.assertEqual(np.linalg.matrix_rank(one_lag), 22)

    def test_causal_moderators_are_centered_within_region(self) -> None:
        frame = _history_frame()
        values = causal_moderator_values(frame, "mc_baseline_bite_z")
        check = (
            pl.DataFrame(
                {
                    "region": frame["aewr_region_id"],
                    "value": values,
                }
            )
            .group_by("region")
            .agg(pl.col("value").mean())
        )
        np.testing.assert_allclose(check["value"].to_numpy(), 0.0, atol=1e-12)

    def test_full_history_uses_two_explicit_boundary_references(self) -> None:
        frame = _history_frame()
        matrix, names, metadata = build_causal_matrix(frame, _spec("full"))
        self.assertEqual(matrix.shape[1], 75)
        self.assertEqual(len(names), len(metadata))
        self.assertNotIn("effect_y2012_h2011", names)
        self.assertNotIn("effect_y2012_h2012", names)
        self.assertIn("effect_difference_y2013_vs_y2012_h2011", names)
        self.assertIn("effect_difference_y2013_vs_y2012_h2012", names)
        self.assertIn("effect_y2013_h2013", names)
        first_history = [item for item in metadata if int(item["history_year"]) == 2011]
        second_history = [
            item for item in metadata if int(item["history_year"]) == 2012
        ]
        self.assertEqual(len(first_history), 10)
        self.assertEqual(len(second_history), 10)
        self.assertTrue(
            all(
                item["identification"] == "difference_from_first_outcome_year"
                and item["reference_year"] == 2012
                for item in first_history + second_history
            )
        )
        self.assertEqual(np.linalg.matrix_rank(_projector(frame).within(matrix)), 75)

    def test_one_lag_keeps_all_identified_current_and_previous_cells(self) -> None:
        frame = _history_frame()
        matrix, names, _ = build_causal_matrix(frame, _spec("one_lag"))
        self.assertEqual(matrix.shape[1], 22)
        self.assertIn("effect_y2012_h2011", names)
        self.assertIn("effect_y2012_h2012", names)
        self.assertFalse(any("effect_difference" in name for name in names))
        self.assertIn("effect_y2022_h2021", names)
        self.assertIn("effect_y2022_h2022", names)
        self.assertTrue(np.all(np.isfinite(matrix)))
        self.assertEqual(np.linalg.matrix_rank(_projector(frame).within(matrix)), 22)

    def test_full_history_has_two_known_nulls_and_reference_invariant_space(
        self,
    ) -> None:
        frame = _history_frame()
        projector = _projector(frame)
        unrestricted, boundary_indices = _unrestricted_full_history(frame)
        unrestricted_within = projector.within(unrestricted)
        self.assertEqual(unrestricted.shape[1], 77)
        self.assertEqual(np.linalg.matrix_rank(unrestricted_within), 75)
        for indices in boundary_indices.values():
            np.testing.assert_allclose(
                unrestricted_within[:, indices].sum(axis=1),
                0.0,
                atol=1e-12,
            )

        first_reference_columns = [indices[0] for indices in boundary_indices.values()]
        last_reference_columns = [indices[-1] for indices in boundary_indices.values()]
        first_reference = np.delete(unrestricted_within, first_reference_columns, 1)
        last_reference = np.delete(unrestricted_within, last_reference_columns, 1)
        self.assertEqual(np.linalg.matrix_rank(first_reference), 75)
        self.assertEqual(np.linalg.matrix_rank(last_reference), 75)
        first_q, _ = np.linalg.qr(first_reference)
        last_q, _ = np.linalg.qr(last_reference)
        principal_cosines = np.linalg.svd(first_q.T @ last_q, compute_uv=False)
        np.testing.assert_allclose(principal_cosines, 1.0, atol=1e-10)

    def test_reference_normalization_is_full_rank_with_moderator_block(self) -> None:
        frame = _history_frame()
        matrix, names, metadata = build_causal_matrix(frame, _spec("full", "bite"))
        self.assertEqual(matrix.shape[1], 150)
        self.assertEqual(len(names), len(metadata))
        self.assertEqual(np.linalg.matrix_rank(_projector(frame).within(matrix)), 150)
        interaction_metadata = [item for item in metadata if item["moderator"]]
        self.assertTrue(
            all(
                item["moderator_transform"] == "within_aewr_region_deviation"
                for item in interaction_metadata
            )
        )


if __name__ == "__main__":
    unittest.main()
