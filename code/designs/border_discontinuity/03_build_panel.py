import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    # ruff: noqa: N999
    """Build the unique county-year panel for the AEWR-border ring experiment."""

    import json
    import os
    import re
    import tempfile
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from h2a.geography import harmonize_county_fips_2010, normalize_geo_code
    from h2a.paths import INTERMEDIATE, PROCESSED, ROOT

    DESIGN_DIR = ROOT / "code" / "designs" / "border_discontinuity"
    MEMBERSHIP_PATH = (
        INTERMEDIATE / "border_discontinuity_geography_memberships.parquet"
    )
    OUTPUT_PATH = PROCESSED / "border_discontinuity_panel.parquet"

    def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
        try:
            pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


    def standardize(values: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        scale = numeric.std(ddof=1)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"cannot standardize baseline variable {values.name}")
        return (numeric - numeric.mean()) / scale

    def main() -> None:
        config = json.loads((DESIGN_DIR / "design.json").read_text())
        outcome = str(config["outcome"])
        treatment = str(config["treatment"])
        controls = [str(value) for value in config["baseline_control_columns"]]
        diagnostic_margins = [
            str(value) for value in config["diagnostic_margin_columns"].values()
        ]
        employer_linkage_columns = [
            "employer_linkage_unit",
            "employer_linkage_version",
            "employer_linkage_input_max_year",
            "employer_linkage_review_status",
            "employer_linkage_review_report_hash",
        ]
        required = [
            "county_fips",
            "countyname",
            "year",
            "state_fips",
            "state",
            "state_abbrev",
            "aewr_region_id",
            outcome,
            treatment,
            "aewr_reference_status",
            "aewr_nonstatutory_application_wage_2009_ppi",
            *controls,
            *diagnostic_margins,
            *employer_linkage_columns,
        ]
        panel = pq.read_table(
            PROCESSED / "county_year_panel.parquet", columns=required
        ).to_pandas()
        missing = sorted(set(required).difference(panel.columns))
        if missing:
            raise ValueError(f"shared panel is missing: {', '.join(missing)}")

        linkage = panel[employer_linkage_columns].drop_duplicates()
        expected_linkage = config["employer_linkage"]
        if (
            len(linkage) != 1
            or linkage.iloc[0]["employer_linkage_unit"]
            != expected_linkage["unit"]
            or linkage.iloc[0]["employer_linkage_version"]
            != expected_linkage["version"]
            or int(linkage.iloc[0]["employer_linkage_input_max_year"])
            != int(expected_linkage["input_max_year"])
            or linkage.iloc[0]["employer_linkage_review_status"]
            != expected_linkage["review_status"]
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(linkage.iloc[0]["employer_linkage_review_report_hash"]),
            )
            is None
        ):
            raise ValueError(
                "border design requires the reviewed FY2008--2024 "
                "worksite-business v2 linkage"
            )

        panel["county_fips"] = panel["county_fips"].map(
            harmonize_county_fips_2010
        ).astype("string")
        panel["aewr_region_id"] = panel["aewr_region_id"].map(
            lambda value: normalize_geo_code(value, "aewr_region_id")
        ).astype("string")
        years = pd.to_numeric(panel["year"], errors="coerce")
        if years.isna().any() or not np.allclose(years, np.round(years)):
            raise ValueError("shared-panel years must be finite integers")
        panel["year"] = years.astype("int16")
        panel = panel.loc[
            panel["year"].between(
                int(config["placebo_start_year"]), int(config["policy_end_year"])
            )
        ].copy()
        if panel.empty or panel.duplicated(["county_fips", "year"]).any():
            raise ValueError("border panel must have unique, nonempty county-year keys")

        # Explicitly quarantined source comparator for stage04's historical v5
        # audit. The treatment remains canonical, including missing 2009 AEWR.
        panel["v5_aewr_ppi_comparator"] = panel[treatment].where(
            panel["year"].ne(2009), panel["aewr_nonstatutory_application_wage_2009_ppi"]
        )

        baseline_columns = [*controls, outcome, treatment]
        baseline = (
            panel.loc[
                panel["year"].between(
                    int(config["feature_start_year"]),
                    int(config["feature_end_year"]),
                )
            ]
            .groupby("county_fips", observed=True)[baseline_columns]
            .mean()
        )
        baseline_names: dict[str, str] = {}
        for column in baseline_columns:
            name = f"baseline_{column}_2008_2010_z"
            baseline[name] = standardize(baseline[column])
            baseline_names[column] = name
        baseline = baseline[list(baseline_names.values())].reset_index()
        panel = panel.merge(
            baseline, on="county_fips", how="left", validate="many_to_one"
        )

        # A fixed preanalysis employment weight is distinct from the unchanged
        # fixed-2011 outcome denominator. The feature producer owns this measure.
        employment = pq.read_table(
            INTERMEDIATE / "border_discontinuity_county_features.parquet",
            columns=["county_fips", "mean_farm_employment_2008_2010"],
        ).to_pandas().rename(
            columns={"mean_farm_employment_2008_2010": "border_baseline_farm_employment"}
        )
        if employment["county_fips"].duplicated().any():
            raise ValueError("baseline employment weights must be county unique")
        panel = panel.merge(employment, on="county_fips", how="left", validate="many_to_one")

        memberships = pq.read_table(MEMBERSHIP_PATH).to_pandas()
        disjoint = memberships.loc[memberships["membership_mode"].eq("disjoint")]
        if disjoint.duplicated(["spec_id", "county_fips"]).any():
            raise ValueError("disjoint memberships are not unique by specification-county")
        assignment = disjoint.pivot(
            index="county_fips", columns="spec_id", values="stratum_id"
        ).add_prefix("stratum__")
        assignment.columns.name = None
        panel = panel.merge(
            assignment.reset_index(),
            on="county_fips",
            how="left",
            validate="many_to_one",
        )
        panel["year_centered"] = panel["year"] - int(config["policy_start_year"])
        panel["border_design_id"] = str(config["design_id"])
        panel["border_feature_start_year"] = int(config["feature_start_year"])
        panel["border_feature_end_year"] = int(config["feature_end_year"])
        panel = panel.sort_values(["county_fips", "year"], kind="stable").reset_index(
            drop=True
        )
        write_parquet_atomic(panel, OUTPUT_PATH)
        print(
            f"Wrote {len(panel):,} county-years for {panel['county_fips'].nunique():,} "
            "counties."
        )

    return (main,)


@app.cell
def _(main):
    main()


if __name__ == "__main__":
    app.run()
