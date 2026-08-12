"""Resource guard tests for declared dense MCW designs."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

BRANCH_ROOT = Path(__file__).resolve().parents[1]
if str(BRANCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BRANCH_ROOT))

from mcw.resources import guard_dense_allocation, guard_fit_working_set, resource_budget


class ResourceGuardTest(unittest.TestCase):
    def test_budget_matches_declared_copy_formula(self) -> None:
        budget = resource_budget(1_000, 100)
        self.assertAlmostEqual(
            budget.estimated_peak_gib,
            4 * budget.dense_gib + 3 * budget.gram_gib,
        )

    def test_environment_limits_fail_before_dense_work(self) -> None:
        with (
            patch.dict(
                os.environ,
                {"MC_SPEC_MAX_DENSE_GIB": "0.000001"},
            ),
            self.assertRaisesRegex(MemoryError, "MC_SPEC_MAX_DENSE_GIB"),
        ):
            guard_dense_allocation(1_000, 100, label="test matrix")
        with (
            patch.dict(
                os.environ,
                {
                    "MC_SPEC_MAX_DENSE_GIB": "1",
                    "MC_SPEC_MAX_PEAK_GIB": "0.000001",
                },
            ),
            self.assertRaisesRegex(MemoryError, "MC_SPEC_MAX_PEAK_GIB"),
        ):
            guard_fit_working_set(1_000, 100)


if __name__ == "__main__":
    unittest.main()
