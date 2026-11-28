import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    """Validate the shared calendar and report source coverage without filtering it."""
    import hashlib
    import json
    import re

    import numpy as np
    import pandas as pd

    from h2a.geography import normalize_geo_code
    from h2a.paths import CODE, INTERMEDIATE, OUTPUTS, PROCESSED, RAW, ROOT

    def main():
        paths_source = (CODE / "paths.R").read_text()
        first, last = [
            int(re.search(rf"SHARED_PANEL_{bound}_YEAR <- (\d+)L", paths_source)[1])
            for bound in ("START", "END")
        ]
        panel_path = PROCESSED / "county_year_panel.parquet"
        panel = pd.read_parquet(panel_path)
        keys = ["county_fips", "year"]
        if panel.empty or panel.duplicated(keys).any():
            raise ValueError("Shared panel must have nonempty, unique county-year keys")
        if not panel.county_fips.map(
            lambda value: isinstance(value, str)
            and value == normalize_geo_code(value, "county_fips")
        ).all():
            raise ValueError("Shared county keys must be canonical strings")
        if set(panel.year) != set(range(first, last + 1)) or not (
            panel.groupby("county_fips").size() == last - first + 1
        ).all():
            raise ValueError("Each county must have the complete declared calendar")

        aggregate_path = INTERMEDIATE / "h2a_aggregated.parquet"
        aggregate = pd.read_parquet(aggregate_path)
        certified = "nbr_workers_certified_start_year"
        aligned = panel[keys + [certified]].merge(
            aggregate[keys + [certified]], on=keys, how="left",
            suffixes=("_panel", "_source"), validate="one_to_one",
        )
        if not np.allclose(
            aligned[f"{certified}_panel"],
            aligned[f"{certified}_source"].fillna(0), rtol=1e-12, atol=1e-9,
        ):
            raise ValueError("Shared certifications disagree with the owning aggregate")

        tables = OUTPUTS / "tables"
        reports = OUTPUTS / "reports"
        tables.mkdir(parents=True, exist_ok=True)
        reports.mkdir(parents=True, exist_ok=True)
        coverage = []
        for year, rows in panel.groupby("year", sort=True):
            for variable in panel.select_dtypes(include="number").columns:
                if variable == "year":
                    continue
                coverage.append({
                    "year": int(year), "variable": variable,
                    "county_rows": len(rows),
                    "observed_counties": int(rows[variable].notna().sum()),
                    "finite_counties": int(np.isfinite(rows[variable]).sum()),
                })
        pd.DataFrame(coverage).to_csv(tables / "shared_panel_source_coverage.csv", index=False)

        # These are allocated positions before geographic splitting, not workers.
        allocation_path = INTERMEDIATE / "h2a_allocated_worksites.parquet"
        allocation = pd.read_parquet(allocation_path, columns=[
            "allocation_id", "case_number", "fiscal_year", "county_fips_list",
            "period_begin_date", "soc_code", "adjusted_nbr_workers_certified",
        ])
        if allocation.allocation_id.duplicated().any():
            raise ValueError("Worksite allocation IDs must be unique")
        allocation["start_year"] = pd.to_datetime(
            allocation.period_begin_date, errors="coerce"
        ).dt.year
        allocation = allocation[allocation.start_year.between(2023, last)].copy()
        soc = allocation.soc_code.fillna("").str.replace("-", "", regex=False).str[:6]
        big_six = {"452041", "452091", "452092", "452093", "452099", "537064"}
        allocation["occupation_group"] = np.where(
            soc.eq(""), "missing_soc",
            np.where(soc.isin(big_six), "field_livestock_six_soc", "other_soc"),
        )
        allocation["county_resolved"] = allocation.county_fips_list.fillna("").ne("")
        cohorts = allocation.groupby(
            ["start_year", "fiscal_year", "occupation_group", "county_resolved"],
            dropna=False,
        ).agg(
            allocated_rows=("allocation_id", "size"),
            cases=("case_number", "nunique"),
            certified_positions=("adjusted_nbr_workers_certified", "sum"),
        ).reset_index()
        cohorts.to_csv(tables / "shared_panel_h2a_source_cohorts.csv", index=False)

        source_path = INTERMEDIATE / "h2a_disclosure_records.parquet"
        source = pd.read_parquet(source_path, columns=[
            "fiscal_year", "case_number", "period_begin_date", "decision_date",
        ])
        if int(source.fiscal_year.max()) != 2026:
            raise ValueError("Review the declared FY2026 Q4 source vintage before publication")
        source["start_year"] = pd.to_datetime(source.period_begin_date, errors="coerce").dt.year
        newest = source[source.fiscal_year == source.fiscal_year.max()]
        latest_starts = newest[newest.start_year == last]
        input_paths = [panel_path, aggregate_path, allocation_path, source_path,
                       RAW / "fred" / "WPU01.csv"]
        input_paths.extend([
            RAW / "h2a" / "H-2A_Disclosure_Data_FY2026_Q4.xlsx",
            RAW / "h2a" / "H-2A_Addendum_B_Employment_FY2026_Q4.xlsx",
        ])
        fingerprints = {}
        for path in input_paths:
            with path.open("rb") as stream:
                fingerprints[str(path.relative_to(ROOT))] = hashlib.file_digest(stream, "sha256").hexdigest()
        report = {
            "status": "passed", "calendar": [first, last],
            "rows": len(panel), "counties": panel.county_fips.nunique(),
            "source_vintage": "FY2008-FY2026 Q4",
            "latest_fiscal_year": int(newest.fiscal_year.max()),
            "latest_decision_date": str(pd.to_datetime(newest.decision_date).max().date()),
            "latest_fiscal_year_cases_starting_in_panel_end_year": latest_starts.case_number.nunique(),
            "employer_linkage": panel[[c for c in panel if c.startswith("employer_linkage_")]].drop_duplicates().to_dict("records"),
            "notes": [
                "FY2026 Q4 completes the fiscal year; FY2027 decisions may still revise 2025 starts.",
                "Occupation groups describe recorded SOCs, not legal wage-floor assignment; cases can span groups.",
                "Job-order submission dates are not available here to classify the October 2025 rule cohort.",
                "Missing later-year controls and other outcomes remain missing; coverage does not select any design sample.",
            ],
            "input_sha256": fingerprints,
        }
        (reports / "shared_panel_validation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Shared panel validated: {len(panel):,} county-years, {first}-{last}")

    main()


if __name__ == "__main__":
    app.run()
