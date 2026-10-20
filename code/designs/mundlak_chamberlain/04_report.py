"""Report version-4 constructed estimands and analytic covariance comparators."""

from __future__ import annotations

import sys
from pathlib import Path

from h2a.paths import TABLES

BRANCH_DIR = Path(__file__).resolve().parent
if str(BRANCH_DIR) not in sys.path:
    sys.path.insert(0, str(BRANCH_DIR))

from mcw.pipeline import report_registry

OBSOLETE_V1_EMPLOYER_OUTPUTS = (
    "mc_ame_grid.csv",
    "mc_average_marginal_effects.csv",
    "mc_ccv_diagnostics.csv",
    "mc_collinear_terms.csv",
    "mc_finite_dose_effects.csv",
    "mc_heterogeneity_effects.csv",
    "mc_lead_placebo_effects.csv",
    "mc_model_diagnostics.csv",
    "mc_parameter_estimates.csv",
    "mc_resolved_history_audit.csv",
    "mc_specification_model_diagnostics.csv",
    "mc_specification_run_progress.csv",
    "mc_validation_summary.csv",
    "mc_year_effects.csv",
    "table_mc_ccv_coefficients.html",
    "table_mc_ccv_coefficients.tex",
    "table_mc_dynamic_effects.html",
    "table_mc_dynamic_effects.tex",
    "table_mc_lead_placebos.html",
    "table_mc_lead_placebos.tex",
)


def remove_obsolete_v1_employer_outputs() -> None:
    for filename in OBSOLETE_V1_EMPLOYER_OUTPUTS:
        (TABLES / filename).unlink(missing_ok=True)


if __name__ == "__main__":
    remove_obsolete_v1_employer_outputs()
    report_registry()
