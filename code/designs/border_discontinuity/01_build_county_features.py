import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    # ruff: noqa: N999
    """Build branch-owned pre-period agro-climatic county features."""

    import json
    import os
    import tempfile
    from pathlib import Path

    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from h2a.paths import INTERMEDIATE, PROCESSED, RAW, ROOT
    from h2a.similarity_clustering import build_county_features, normalize_counties

    DESIGN_DIR = ROOT / "code" / "designs" / "border_discontinuity"
    OUTPUT_PATH = INTERMEDIATE / "border_discontinuity_county_features.parquet"

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


    def build_agricultural_support(
        panel: pd.DataFrame,
        counties: pd.Index,
        start_year: int,
        end_year: int,
        cropland_year: int,
    ) -> pd.DataFrame:
        if cropland_year != 2007 or (start_year, end_year) != (2008, 2010):
            raise ValueError(
                "Agricultural support must use 2007 cropland and 2008--2010 "
                "farm employment."
            )
        if panel.duplicated(["county_fips", "year"]).any():
            raise ValueError("shared county panel must be unique by county-year")
        cropland_variation = panel.groupby("county_fips")[
            "census_cropland_2007"
        ].nunique(dropna=True)
        if cropland_variation.gt(1).any():
            raise ValueError("2007 cropland must be time invariant by county")
        cropland = (
            panel.groupby("county_fips", as_index=False)["census_cropland_2007"]
            .first()
            .rename(columns={"census_cropland_2007": "cropland_acres_2007"})
        )
        employment = (
            panel.loc[panel["year"].between(start_year, end_year)]
            .groupby("county_fips", as_index=False)["emp_farm"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": "mean_farm_employment_2008_2010",
                    "count": "farm_employment_years_observed",
                }
            )
        )
        shape_path = (
            f"/vsizip/{RAW / 'county_shapefile' / 'tl_2010_us_county10.zip'}"
            "/tl_2010_us_county10.shp"
        )
        land_area = gpd.read_file(
            shape_path,
            columns=["GEOID10", "ALAND10"],
            ignore_geometry=True,
        ).rename(
            columns={
                "GEOID10": "county_fips",
                "ALAND10": "land_area_square_meters_2010",
            }
        )
        land_area["county_fips"] = normalize_counties(land_area["county_fips"])
        land_area = land_area.groupby("county_fips", as_index=False)[
            "land_area_square_meters_2010"
        ].sum()
        land_area["land_area_acres_2010"] = (
            land_area["land_area_square_meters_2010"] / 4046.8564224
        )

        support = pd.DataFrame({"county_fips": counties.astype("string")})
        support = support.merge(cropland, on="county_fips", how="left", validate="1:1")
        support = support.merge(employment, on="county_fips", how="left", validate="1:1")
        support = support.merge(
            land_area[["county_fips", "land_area_acres_2010"]],
            on="county_fips",
            how="left",
            validate="1:1",
        )
        support_values = support[
            [
                "cropland_acres_2007",
                "mean_farm_employment_2008_2010",
                "land_area_acres_2010",
            ]
        ]
        if (
            support["land_area_acres_2010"].isna().any()
            or not np.isfinite(support["land_area_acres_2010"]).all()
            or support["land_area_acres_2010"].le(0).any()
            or support_values.lt(0).any().any()
            or support["farm_employment_years_observed"].fillna(0).gt(3).any()
        ):
            raise ValueError("agricultural support contains invalid values")
        support["farm_employment_years_observed"] = (
            support["farm_employment_years_observed"].fillna(0).astype("int8")
        )
        return support


    def main() -> None:
        config = json.loads((DESIGN_DIR / "design.json").read_text())
        start_year = int(config["feature_start_year"])
        end_year = int(config["feature_end_year"])
        agricultural_filter = config["agricultural_county_pair_filter"]

        panel = pq.read_table(
            PROCESSED / "county_year_panel.parquet",
            columns=[
                "county_fips",
                "year",
                "census_cropland_2007",
                "emp_farm",
            ],
        ).to_pandas()
        panel["county_fips"] = normalize_counties(panel["county_fips"])
        counties = pd.Index(sorted(panel["county_fips"].unique()), name="county_fips")
        agricultural_support = build_agricultural_support(
            panel,
            counties,
            int(agricultural_filter["farm_employment_start_year"]),
            int(agricultural_filter["farm_employment_end_year"]),
            int(agricultural_filter["cropland_year"]),
        )

        crop = pq.read_table(
            INTERMEDIATE / "croplandcros_county_crop_type_acres.parquet"
        ).to_pandas()
        climate = pq.read_table(
            INTERMEDIATE / "county_h2a_prediction_climate_basis_annual.parquet"
        ).to_pandas()
        soil = pq.read_table(
            INTERMEDIATE / "county_h2a_prediction_gnatsgo_soil_cells.parquet"
        ).to_pandas()

        raw, scaled, missing_count = build_county_features(
            crop, climate, soil, counties, start_year, end_year
        )
        output = pd.DataFrame(
            {
                "county_fips": counties.astype("string"),
                "feature_start_year": start_year,
                "feature_end_year": end_year,
                "raw_feature_missing_count": missing_count.to_numpy(),
                "raw_feature_complete": missing_count.eq(0).to_numpy(),
            }
        )
        output = output.merge(
            agricultural_support,
            on="county_fips",
            how="left",
            validate="1:1",
        )
        output = pd.concat(
            [
                output,
                raw.reset_index(drop=True).add_prefix("raw__"),
                scaled.reset_index(drop=True).add_prefix("scaled__"),
            ],
            axis=1,
        )
        write_parquet_atomic(output, OUTPUT_PATH)
        print(
            f"Wrote {len(output):,} counties and {scaled.shape[1]:,} scaled features "
            f"for {start_year}--{end_year}."
        )

    return (main,)


@app.cell
def _(main):
    main()


if __name__ == "__main__":
    app.run()
