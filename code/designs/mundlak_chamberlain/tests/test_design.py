"""Executable registry-contract tests for MCW version 4."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

BRANCH_ROOT = Path(__file__).resolve().parents[1]
if str(BRANCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BRANCH_ROOT))

from mcw.design import compact_specifications, exhaustive_specifications


class DesignRegistryTest(unittest.TestCase):
    def test_exhaustive_axes_include_pooled_regional_treatments(self) -> None:
        exhaustive = exhaustive_specifications()
        combinations = {
            (
                spec.treatment,
                spec.history,
                spec.fixed_effects,
                spec.moderator_set,
                spec.cluster,
                spec.treatment_transform,
            )
            for spec in exhaustive
        }
        for compact in compact_specifications():
            combination = (
                compact.treatment,
                compact.history,
                compact.fixed_effects,
                compact.moderator_set,
                compact.cluster,
                compact.treatment_transform,
            )
            self.assertIn(combination, combinations)

        self.assertTrue(
            any(
                spec.treatment == "aewr_log_level"
                and spec.fixed_effects == "pooled_wmc"
                for spec in exhaustive
            )
        )


if __name__ == "__main__":
    unittest.main()
