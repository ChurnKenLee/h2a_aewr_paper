"""Build the supported version-4 MCW panel with DuckDB and Polars."""

from __future__ import annotations

import sys
from pathlib import Path

BRANCH_DIR = Path(__file__).resolve().parent
if str(BRANCH_DIR) not in sys.path:
    sys.path.insert(0, str(BRANCH_DIR))

from mcw.build import write_mcw_panel
from mcw.clusters import add_cluster_partitions
from mcw.io import ANALYSIS_PANEL, SOURCE_PANEL, atomic_write_frame


def main() -> None:
    panel = write_mcw_panel(SOURCE_PANEL, ANALYSIS_PANEL)
    panel = add_cluster_partitions(panel)
    atomic_write_frame(panel, ANALYSIS_PANEL)
    eligible = panel.filter(panel["mc_baseline_farm_employment"] > 0)
    print(
        "Built MCW v4 panel: "
        f"{panel.height:,} rows, {panel['county_fips'].n_unique():,} counties; "
        f"{eligible['county_fips'].n_unique():,} positive-baseline-employment counties."
    )


if __name__ == "__main__":
    main()
