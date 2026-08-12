"""Executable version-4 design contract for the MCW branch.

Version 4 replaces the former three-lag polynomial specification program.  It
keeps every declared treatment-history coordinate linear and separate, fits
only OLS models, and constructs ratios and elasticities after fitting the six
primitive outcomes on one common design.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Final, Literal

import polars as pl

# docs-ground:start mundlak-design-contract
DESIGN_VERSION: Final = "4.0.0"
DEFAULT_STAGE: Final = "compact"
BASELINE_YEARS: Final = (2008, 2009, 2010)
FROZEN_WINDOWS: Final = {
    "f0809": (2008, 2009),
    "f0810": (2008, 2009, 2010),
}
TREATMENT_HISTORY_YEARS: Final = tuple(range(2011, 2023))
ANALYSIS_YEARS: Final = tuple(range(2012, 2023))
MODEL_CLASS: Final = "pooled_ols_identity_link"

# A calendar boundary is needed to separate the frozen pre-period from the
# treatment history.  It is not an outcome denominator, eligibility year, or
# omitted causal reference category.
HISTORY_BOUNDARY_YEAR: Final = 2010
# The user-approved outcome window starts in 2012; 2011 survives only as the
# first clean treatment-history coordinate. In the full-history model, county
# fixed effects absorb the common levels of both 2011 and 2012 history paths,
# because both are present in every outcome-year cell. Outcome-year 2012 is
# therefore their explicit reference cell. The one-lag model retains all its
# cells because each path becomes inactive again. These are rank
# normalizations of the identified column space, not lag profiles.
FULL_HISTORY_REFERENCE_OUTCOME_YEAR: Final = 2012
FULL_HISTORY_REFERENCE_PATH_YEARS: Final = tuple(
    year
    for year in TREATMENT_HISTORY_YEARS
    if year <= FULL_HISTORY_REFERENCE_OUTCOME_YEAR
)
# A current effect in the full model is first identified in 2013; use the same
# target years for the one-lag comparison.
CURRENT_EFFECT_REPORTING_YEARS: Final = tuple(range(2013, 2023))

PRIMITIVE_OUTCOMES: Final = {
    "applications": "mc_y_applications",
    "employers": "mc_y_employers_balanced",
    "requested_positions": "mc_y_requested_positions",
    "certified_positions": "mc_y_certified_positions",
    "certified_hours": "mc_y_certified_hours",
    "any_application": "mc_y_any_application",
}

CONSTRUCTED_OUTCOMES: Final = (
    "applications_per_1000_baseline_farm_workers",
    "requested_positions_per_1000_baseline_farm_workers",
    "certified_positions_per_1000_baseline_farm_workers",
    "certified_hours_per_baseline_farm_worker",
    "positions_per_application_ratio_of_aggregates",
    "hours_per_position_ratio_of_aggregates",
    "log_aewr_elasticity_at_observed_mean",
    "percent_of_observed_mean_per_treatment_unit",
)

# These are pre-period means only.  Separate 2008/2009/2010 coordinates and
# imposed linear trends are deliberately absent.  The estimator constructs
# hierarchical components and interacts them with categorical year contrasts.
BASELINE_VARIABLES: Final = {
    "bite": "derived 2008-2010 mean of real AEWR minus county p25 wage",
    "h2a_applications": "nbr_applications_start_year",
    "h2a_certified_positions": "nbr_workers_certified_start_year",
    "log_population": "ln_pop_census",
    "farm_employment_share": "farm_emp_share",
    "employment_population_ratio": "emp_pop_ratio",
    "crop_income_share": "share_farm_crop_cashandinc",
    "hired_labor_cost_share": "share_farm_laborexp_prodexp",
    "low_wage": "wage_p25",
    "animal_income_share": "share_farm_animal_cashandinc",
    "production_expense_share": "share_farm_prodexp_cashandinc",
    "median_wage": "wage_p50",
    "cropland": "census_cropland_2007",
}

MODERATOR_SETS: Final = {
    "none": (),
    "bite": ("mc_baseline_bite_z",),
    "predetermined": (
        "mc_baseline_bite_z",
        "mc_baseline_farm_employment_share_z",
        "mc_baseline_crop_income_share_z",
        "mc_baseline_hired_labor_cost_share_z",
    ),
    "maximal_predetermined": (
        "mc_baseline_bite_z",
        "mc_baseline_h2a_applications_z",
        "mc_baseline_h2a_certified_positions_z",
        "mc_baseline_log_population_z",
        "mc_baseline_farm_employment_share_z",
        "mc_baseline_employment_population_ratio_z",
        "mc_baseline_crop_income_share_z",
        "mc_baseline_hired_labor_cost_share_z",
        "mc_baseline_low_wage_z",
        "mc_baseline_animal_income_share_z",
        "mc_baseline_production_expense_share_z",
        "mc_baseline_median_wage_z",
        "mc_baseline_cropland_z",
    ),
}

# The frozen-distribution measures are quantile approximations because the
# shared panel contains five wage quantiles, not the underlying micro wage
# distribution.  Their names make that limitation persistent in artifacts.
TREATMENT_DEFINITIONS: Final = {
    "aewr_log_level": {
        "column_prefix": "mc_aewr_log_level_",
        "unit": "one log percentage point relative to 2010",
        "lower_geography": False,
        "status": "core_benchmark",
    },
    "aewr_dollar_level": {
        "column_prefix": "mc_aewr_dollar_level_",
        "unit": "one real dollar relative to 2010",
        "lower_geography": False,
        "status": "core_benchmark",
    },
    "aewr_log_change": {
        "column_prefix": "mc_aewr_log_change_",
        "unit": "one annual log percentage point",
        "lower_geography": False,
        "status": "sensitivity",
    },
    "bite_f0809": {
        "column_prefix": "mc_bite_f0809_",
        "unit": "one real dollar-hour of approximate frozen bite",
        "lower_geography": True,
        "status": "candidate",
    },
    "bite_f0810": {
        "column_prefix": "mc_bite_f0810_",
        "unit": "one real dollar-hour of approximate frozen bite",
        "lower_geography": True,
        "status": "candidate",
    },
    "exposure_log_f0809": {
        "column_prefix": "mc_exposure_log_f0809_",
        "unit": "frozen fraction affected times one log point",
        "lower_geography": True,
        "status": "candidate",
    },
    "exposure_log_f0810": {
        "column_prefix": "mc_exposure_log_f0810_",
        "unit": "frozen fraction affected times one log point",
        "lower_geography": True,
        "status": "candidate",
    },
}

# Only these raw coordinates equal 100 times a log-AEWR change (up to a
# predetermined additive constant). Other treatment families and transforms
# require a separate chain rule before they can be called AEWR elasticities.
DIRECT_LOG_AEWR_TREATMENTS: Final = frozenset({"aewr_log_level", "aewr_log_change"})

FIXED_EFFECT_SETS: Final = {
    # The later archive motivates this pooled candidate after provisionally
    # loosening unit absorption. It uses an explicit Chamberlain--Mundlak
    # projection: region and calendar effects plus rich interactions with the
    # baseline hierarchy. County-FE alternatives remain declared sensitivities.
    "pooled_wmc": ("explicit_region", "explicit_year"),
    "county_year": ("county_fips", "year"),
    "county_region_year": ("county_fips", "aewr_region_id", "year"),
    "county_state_year": ("county_fips", "state_fips", "year"),
}

# The registry is a sensitivity menu, not a significance-selection device.
# AEWR region is the natural assignment cluster for regional AEWR paths; the
# archive did not resolve one primary dependence partition for lower-geography
# bite specifications.
CLUSTER_DEFINITIONS: Final = {
    "aewr_region": "aewr_region_id",
    "state": "state_fips",
    "county": "county_fips",
    "cz": "cz_id",
    "cz_region": "mc_cluster_cz_region",
    "market": "mc_market_id",
    "region_year": "mc_cluster_region_year",
    "state_year": "mc_cluster_state_year",
    "year": "mc_cluster_year",
    "agro2": "mc_cluster_agro2",
    "agro3": "mc_cluster_agro3",
    "agro5": "mc_cluster_agro5",
    "exposure_decile": "mc_cluster_exposure_decile",
    "exposure_decile_region": "mc_cluster_exposure_decile_region",
}

INFERENCE_METHODS: Final = (
    "hc3_full_model_leverage",
    "cr0_cluster_sandwich",
    "cr1_cluster_sandwich",
    "ccv_hc3_scalar_mixture_experimental",
    "ccv_hc3_cr1_scalar_mixture_experimental",
    "hc3_full_model_leverage_joint_delta",
    "cr0_cluster_sandwich_joint_delta",
    "cr1_cluster_sandwich_joint_delta",
    "ccv_hc3_scalar_mixture_experimental_delta",
    "ccv_hc3_cr1_scalar_mixture_experimental_delta",
)
INFERENCE_DIAGNOSTIC_ORACLES: Final = (
    "cv3_direct_refit_diagnostic",
    "cr2_dense_small_design_oracle",
)
CCV_Q: Final = 1.0
CCV_STATUS: Final = "experimental_not_lean_validated"
ALLOW_RANDOMIZATION_INFERENCE: Final = False
ALLOW_BOOTSTRAP: Final = False
ALLOW_NONLINEAR_MODELS: Final = False
ALLOW_POLYNOMIAL_TREATMENT_TERMS: Final = False
ALLOW_DIMENSION_REDUCING_LAG_PROFILES: Final = False
# docs-ground:end mundlak-design-contract


History = Literal["full", "one_lag"]


@dataclass(frozen=True, slots=True)
class Specification:
    """One auditable model-and-inference specification."""

    specification_id: str
    stage: str
    treatment: str
    history: History
    fixed_effects: str
    moderator_set: str
    cluster: str
    treatment_transform: str = "continuous_raw"
    interpretation_status: str = "candidate"

    def validate(self) -> None:
        if self.treatment not in TREATMENT_DEFINITIONS:
            raise ValueError(f"Unknown treatment: {self.treatment}")
        if self.fixed_effects not in FIXED_EFFECT_SETS:
            raise ValueError(f"Unknown fixed-effect set: {self.fixed_effects}")
        if self.moderator_set not in MODERATOR_SETS:
            raise ValueError(f"Unknown moderator set: {self.moderator_set}")
        if self.cluster not in CLUSTER_DEFINITIONS:
            raise ValueError(f"Unknown cluster: {self.cluster}")
        if self.history not in {"full", "one_lag"}:
            raise ValueError(f"Unknown history rule: {self.history}")
        lower_geo = bool(TREATMENT_DEFINITIONS[self.treatment]["lower_geography"])
        if self.fixed_effects not in {"pooled_wmc", "county_year"} and not lower_geo:
            raise ValueError(
                "Region-level treatment is absorbed by region-year or state-year "
                "fixed effects; use a declared lower-geography treatment."
            )


def _make_spec(
    treatment: str,
    history: History,
    fixed_effects: str,
    moderator_set: str,
    cluster: str,
    *,
    stage: str,
    treatment_transform: str = "continuous_raw",
    interpretation_status: str = "candidate",
) -> Specification:
    parts = (
        treatment,
        history,
        fixed_effects,
        moderator_set,
        cluster,
        treatment_transform,
    )
    spec = Specification(
        specification_id="__".join(parts),
        stage=stage,
        treatment=treatment,
        history=history,
        fixed_effects=fixed_effects,
        moderator_set=moderator_set,
        cluster=cluster,
        treatment_transform=treatment_transform,
        interpretation_status=interpretation_status,
    )
    spec.validate()
    return spec


def compact_specifications() -> tuple[Specification, ...]:
    """Return a bounded queue spanning the archive's non-nested choices."""

    return (
        _make_spec(
            "aewr_log_level",
            "full",
            "pooled_wmc",
            "maximal_predetermined",
            "aewr_region",
            stage="compact",
            interpretation_status="pooled_rich_projection_candidate",
        ),
        _make_spec(
            "aewr_log_level",
            "one_lag",
            "pooled_wmc",
            "maximal_predetermined",
            "aewr_region",
            stage="compact",
            interpretation_status="pooled_rich_projection_one_lag",
        ),
        _make_spec(
            "aewr_dollar_level",
            "full",
            "pooled_wmc",
            "maximal_predetermined",
            "aewr_region",
            stage="compact",
            interpretation_status="pooled_dollar_unit_sensitivity",
        ),
        _make_spec(
            "aewr_log_level",
            "full",
            "county_year",
            "predetermined",
            "aewr_region",
            stage="compact",
            interpretation_status="county_fe_sensitivity",
        ),
        _make_spec(
            "aewr_log_level",
            "one_lag",
            "county_year",
            "predetermined",
            "aewr_region",
            stage="compact",
            interpretation_status="one_lag_benchmark",
        ),
        _make_spec(
            "aewr_dollar_level",
            "full",
            "county_year",
            "bite",
            "aewr_region",
            stage="compact",
            interpretation_status="unit_sensitivity",
        ),
        _make_spec(
            "bite_f0809",
            "full",
            "county_state_year",
            "bite",
            "aewr_region",
            stage="compact",
            interpretation_status="lower_geography_candidate",
        ),
        _make_spec(
            "bite_f0810",
            "full",
            "county_state_year",
            "bite",
            "aewr_region",
            stage="compact",
            interpretation_status="baseline_window_sensitivity",
        ),
        _make_spec(
            "exposure_log_f0809",
            "full",
            "county_year",
            "predetermined",
            "aewr_region",
            stage="compact",
            interpretation_status="lower_geography_full_history_candidate",
        ),
        _make_spec(
            "exposure_log_f0809",
            "one_lag",
            "county_region_year",
            "predetermined",
            "aewr_region",
            stage="compact",
            interpretation_status="one_lag_benchmark",
        ),
        _make_spec(
            "exposure_log_f0810",
            "full",
            "county_region_year",
            "predetermined",
            "aewr_region",
            stage="compact",
            interpretation_status="baseline_window_support_sensitivity",
        ),
        _make_spec(
            "exposure_log_f0810",
            "one_lag",
            "county_region_year",
            "predetermined",
            "aewr_region",
            stage="compact",
            interpretation_status="baseline_window_one_lag_sensitivity",
        ),
    )


def exhaustive_specifications() -> tuple[Specification, ...]:
    """Compile the opt-in menu without silently fitting invalid FE pairs."""

    specs: list[Specification] = []
    for treatment, history, fixed_effects, moderators, cluster in product(
        TREATMENT_DEFINITIONS,
        ("full", "one_lag"),
        FIXED_EFFECT_SETS,
        MODERATOR_SETS,
        CLUSTER_DEFINITIONS,
    ):
        lower_geo = bool(TREATMENT_DEFINITIONS[treatment]["lower_geography"])
        if fixed_effects not in {"pooled_wmc", "county_year"} and not lower_geo:
            continue
        transforms = ["continuous_raw"]
        if lower_geo:
            transforms.extend(
                (
                    "continuous_within_region_z",
                    "binary_median",
                    "binary_upper_quartile",
                    "within_region_rank",
                )
            )
        for treatment_transform in transforms:
            specs.append(
                _make_spec(
                    treatment,
                    history,
                    fixed_effects,
                    moderators,
                    cluster,
                    stage="exhaustive",
                    treatment_transform=treatment_transform,
                    interpretation_status=(
                        "candidate"
                        if treatment_transform.startswith("continuous")
                        else "binary_or_rank_sensitivity"
                    ),
                )
            )
    return tuple(specs)


def specification_registry(stage: str = DEFAULT_STAGE) -> pl.DataFrame:
    """Return a machine-readable registry for a bounded or opt-in queue."""

    if stage == "compact":
        specs = compact_specifications()
    elif stage == "exhaustive":
        specs = exhaustive_specifications()
    else:
        raise ValueError("MC_SPEC_STAGE must be 'compact' or 'exhaustive'.")
    rows = []
    for spec in specs:
        row = asdict(spec)
        definition = TREATMENT_DEFINITIONS[spec.treatment]
        row.update(
            treatment_unit=definition["unit"],
            lower_geography=definition["lower_geography"],
            design_version=DESIGN_VERSION,
            model_class=MODEL_CLASS,
            ccv_status=CCV_STATUS,
        )
        rows.append(row)
    return pl.DataFrame(rows)
