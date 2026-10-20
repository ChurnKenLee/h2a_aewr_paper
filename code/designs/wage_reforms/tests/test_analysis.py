"""Check counting, zero retention, fixed assignment, and cluster inference."""

import sys
import unittest

import numpy as np
import polars as pl

from h2a.paths import CODE

sys.path.insert(0, str(CODE / "designs" / "wage_reforms"))
from analysis import (
    absorb,
    classify_baseline,
    clustered_ols,
    cohort_panel,
    fixed_effect_rank,
    select_positive_cases,
)


def source_fixture():
    return pl.DataFrame([
        {"employer_id_balanced": "a", "fiscal_year": 2011, "case_number": "a11",
         "strict_herder": True, "hourly": False, "herder_title": True, "sheep_title": True,
         "worksite_state": "UT", "valid_state": True, "case_states": ["UT"],
         "case_state_missing": False, "aewr_region_id": "2", "offered_wage": 750.0,
         "employer_is_worksite_business": True},
        {"employer_id_balanced": "b", "fiscal_year": 2011, "case_number": "b11",
         "strict_herder": False, "hourly": True, "herder_title": False, "sheep_title": False,
         "worksite_state": "UT", "valid_state": True, "case_states": ["UT"],
         "case_state_missing": False, "aewr_region_id": "2", "offered_wage": 10.0,
         "employer_is_worksite_business": True},
        # A changes occupation and state, but must keep its baseline assignment.
        {"employer_id_balanced": "a", "fiscal_year": 2012, "case_number": "a12",
         "strict_herder": False, "hourly": True, "herder_title": False, "sheep_title": False,
         "worksite_state": "CA", "valid_state": True, "case_states": ["CA"],
         "case_state_missing": False, "aewr_region_id": "5", "offered_wage": 12.0,
         "employer_is_worksite_business": True},
    ]).with_columns(pl.col("fiscal_year").cast(pl.Int32))


class CountingTests(unittest.TestCase):
    def test_master_total_is_not_repeated_on_members(self):
        rows = pl.DataFrame({"fiscal_year": [2014, 2014, 2014], "case_number": ["a", "a", "b"],
                             "is_master": [True, False, False], "positive_case_rows": [2, 2, 1],
                             "certified": [100.0, 60.0, 20.0]})
        self.assertEqual(select_positive_cases(rows)["certified"].sum(), 120.0)

    def test_duplicate_or_missing_master_fails(self):
        for master in [[True, True], [False, False]]:
            rows = pl.DataFrame({"fiscal_year": [2014, 2014], "case_number": ["a", "a"],
                                 "is_master": master, "positive_case_rows": [2, 2]})
            with self.assertRaises(ValueError):
                select_positive_cases(rows)

    def test_future_changes_do_not_reassign_or_drop_baseline_firms(self):
        rows = source_fixture()
        baseline = classify_baseline(rows, 2011)
        self.assertEqual(baseline.filter(pl.col("employer_id_balanced") == "a")["baseline_state"][0], "UT")
        config = {"cohorts": [{"name": "herder", "baseline_year": 2011,
                               "years": [2011, 2012], "reference_year": 2011}]}
        panel = cohort_panel(rows, config)
        self.assertEqual(panel.height, 4)
        disappeared = panel.filter((pl.col("employer_id_balanced") == "b") & (pl.col("fiscal_year") == 2012))
        self.assertEqual(disappeared["certified_participation"][0], 0)
        self.assertIsNone(disappeared["log_wage"][0])
        changed = panel.filter((pl.col("employer_id_balanced") == "a") & (pl.col("fiscal_year") == 2012))
        self.assertEqual(changed["baseline_type"][0], "herder")
        self.assertEqual(changed["baseline_state"][0], "UT")
        self.assertEqual(changed["certified_participation"][0], 1)
        self.assertEqual(changed["same_type_participation"][0], 0)


class InferenceTests(unittest.TestCase):
    def test_absorbed_crv1_equals_full_dummy_regression(self):
        rng = np.random.default_rng(20150116)
        n = 120
        state = np.repeat(np.arange(6), 20)
        title = np.tile(np.arange(10), 12)
        x = rng.normal(size=(n, 3))
        y = x @ np.array([0.3, -0.2, 0.7]) + state + title + rng.normal(size=n)
        fit = clustered_ols(y, x, [state, title], state)
        full = np.column_stack([x, np.eye(6)[state], np.eye(10)[title]])
        beta = np.linalg.lstsq(full, y, rcond=None)[0]
        residual = y - full @ beta
        bread = np.linalg.pinv(full.T @ full)
        scores = np.stack([full[state == c].T @ residual[state == c] for c in range(6)])
        rank = np.linalg.matrix_rank(full)
        covariance = 6 / 5 * (n - 1) / (n - rank) * bread @ scores.T @ scores @ bread
        self.assertAlmostEqual(fit["coefficient"], beta[0], places=9)
        self.assertAlmostEqual(fit["std_error"], np.sqrt(covariance[0, 0]), places=9)
        self.assertEqual(fit["design_rank"], rank)
        np.testing.assert_allclose(absorb(full, [state, title])[:, 3:], 0, atol=1e-9)

    def test_disconnected_fixed_effect_rank(self):
        state = np.array([0, 0, 1, 1, 2, 2])
        title = np.array([0, 1, 1, 2, 3, 4])
        full = np.column_stack([np.eye(3)[state], np.eye(5)[title]])
        self.assertEqual(fixed_effect_rank([state, title]), np.linalg.matrix_rank(full))

    def test_absorbed_policy_does_not_produce_coefficient(self):
        state = np.repeat(np.arange(4), 10)
        result = clustered_ols(np.arange(40).astype(float), state.reshape(-1, 1).astype(float), [state], state)
        self.assertEqual(result["status"], "rank_or_cluster_failure")


if __name__ == "__main__":
    unittest.main()
