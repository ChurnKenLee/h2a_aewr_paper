import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")


@app.cell
def _():
    import ssl
    import urllib.error
    import urllib.request
    from pathlib import Path

    import numpy as np
    import polars as pl
    import pyprojroot
    from sklearn.preprocessing import SplineTransformer

    return Path, SplineTransformer, np, pl, pyprojroot, ssl, urllib


@app.cell
def _(pyprojroot):
    root_path = pyprojroot.find_root(criterion="pyproject.toml")

    CACHE_DIR = root_path / "data" / "epinoaa_nclimgrid"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    OUT_DIR = root_path / "binaries"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE = OUT_DIR / "county_h2a_prediction_climate_basis_annual.parquet"
    return CACHE_DIR, OUTPUT_FILE


@app.cell
def _():
    START_YEAR = 2000
    END_YEAR = 2025

    NORMAL_START_YEAR = 2000
    NORMAL_END_YEAR = 2011

    WET_DAY_MM = 1.0

    # Smooth climate-basis design.
    TEMP_SPLINE_VAR = "tavg"
    TEMP_N_KNOTS = 7
    PRCP_N_KNOTS = 5
    SPLINE_DEGREE = 3
    SPLINE_EXTRAPOLATION = "constant"
    SEASON_HARMONICS = 3
    return (
        END_YEAR,
        NORMAL_END_YEAR,
        NORMAL_START_YEAR,
        PRCP_N_KNOTS,
        SEASON_HARMONICS,
        SPLINE_DEGREE,
        SPLINE_EXTRAPOLATION,
        START_YEAR,
        TEMP_N_KNOTS,
        TEMP_SPLINE_VAR,
        WET_DAY_MM,
    )


@app.function
def get_parquet_url(year: int, month: int) -> str:
    ym = f"{year:04d}{month:02d}"
    file_name = f"{ym}.parquet"
    return f"https://noaa-nclimgrid-daily-pds.s3.amazonaws.com/EpiNOAA/v1-0-0/parquet/cty/YEAR={year}/STATUS=scaled/{file_name}"


@app.cell
def _(CACHE_DIR, Path, ssl, urllib):
    def download_monthly_parquet(year: int, month: int) -> Path | None:
        ym_str = f"{year}{month:02d}"
        cache_file = CACHE_DIR / f"{ym_str}.parquet"

        if cache_file.exists():
            print(f"{ym_str}.parquet already cached")
            return cache_file

        url = get_parquet_url(year, month)
        ctx = ssl._create_unverified_context()

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30, context=ctx) as response:
                data = response.read()

                if not data.startswith(b"PAR1"):
                    print(f"ERROR: URL {url} did not return a Parquet file.")
                    preview = data[:250].decode("utf-8", errors="ignore").replace("\n", " ")
                    print(f"Preview of downloaded content: {preview}...\n")
                    return None

                with open(cache_file, "wb") as f_out:
                    f_out.write(data)

            return cache_file

        except urllib.error.URLError as e:
            print(f"Failed to download {year}-{month:02d}: {e}")
            return None

    return (download_monthly_parquet,)


@app.cell
def _(END_YEAR, START_YEAR, download_monthly_parquet):
    print(f"Downloading EpiNOAA monthly county parquets ({START_YEAR}-{END_YEAR})...")
    for _yr in range(START_YEAR, END_YEAR + 1):
        for _mo in range(1, 13):
            download_monthly_parquet(_yr, _mo)

    print("Downloads complete.")
    return


@app.cell
def _(SEASON_HARMONICS):
    season_cols = []
    for h in range(1, SEASON_HARMONICS + 1):
        season_cols.extend([f"season_sin_{h}", f"season_cos_{h}"])
    return (season_cols,)


@app.cell
def _(
    CACHE_DIR,
    NORMAL_END_YEAR,
    NORMAL_START_YEAR,
    PRCP_N_KNOTS,
    SPLINE_DEGREE,
    SPLINE_EXTRAPOLATION,
    SplineTransformer,
    TEMP_N_KNOTS,
    TEMP_SPLINE_VAR,
    WET_DAY_MM,
    np,
    pl,
):
    def read_normal_period_values(cache_file):
        year = int(cache_file.stem[:4])
        if year < NORMAL_START_YEAR or year > NORMAL_END_YEAR:
            return None, None

        chunk = (
            pl.read_parquet(
                cache_file,
                columns=["date", TEMP_SPLINE_VAR, "prcp"],
            )
            .select(
                [
                    pl.col("date"),
                    pl.col(TEMP_SPLINE_VAR).str.strip_chars().cast(pl.Float32),
                    pl.col("prcp").str.strip_chars().cast(pl.Float32),
                ]
            )
            .with_columns(pl.col("date").dt.year().alias("year"))
            .filter(
                (pl.col("year") >= NORMAL_START_YEAR) & (pl.col("year") <= NORMAL_END_YEAR)
            )
        )

        temp_part = (
            chunk.filter(pl.col(TEMP_SPLINE_VAR).is_not_null())
            .select(TEMP_SPLINE_VAR)
            .to_numpy()
            .astype(np.float32)
        )
        prcp_part = (
            chunk.filter(pl.col("prcp") >= WET_DAY_MM)
            .select(pl.col("prcp").log1p().alias("log_prcp"))
            .to_numpy()
            .astype(np.float32)
        )
        return temp_part, prcp_part


    temp_parts = []
    prcp_parts = []
    for _cache_file in sorted(CACHE_DIR.glob("*.parquet")):
        temp_part, prcp_part = read_normal_period_values(_cache_file)
        if temp_part is not None and temp_part.size:
            temp_parts.append(temp_part)
        if prcp_part is not None and prcp_part.size:
            prcp_parts.append(prcp_part)

    temp_fit = np.vstack(temp_parts)
    prcp_fit = np.vstack(prcp_parts)
    temp_fill_value = float(temp_fit.mean())

    temp_spline = SplineTransformer(
        n_knots=TEMP_N_KNOTS,
        degree=SPLINE_DEGREE,
        include_bias=False,
        knots="quantile",
        extrapolation=SPLINE_EXTRAPOLATION,
    ).fit(temp_fit)

    prcp_spline = SplineTransformer(
        n_knots=PRCP_N_KNOTS,
        degree=SPLINE_DEGREE,
        include_bias=False,
        knots="quantile",
        extrapolation=SPLINE_EXTRAPOLATION,
    ).fit(prcp_fit)
    return prcp_spline, temp_fill_value, temp_spline


@app.cell
def _(
    CACHE_DIR,
    END_YEAR,
    NORMAL_END_YEAR,
    NORMAL_START_YEAR,
    START_YEAR,
    TEMP_SPLINE_VAR,
    WET_DAY_MM,
    np,
    pl,
    prcp_spline,
    season_cols,
    temp_fill_value,
    temp_spline,
):
    temp_basis_cols = [
        f"temp_{TEMP_SPLINE_VAR}_b{k:02d}" for k in range(temp_spline.n_features_out_)
    ]
    prcp_basis_cols = [f"prcp_log1p_b{k:02d}" for k in range(prcp_spline.n_features_out_)]

    climate_basis_cols = []
    for c in temp_basis_cols + prcp_basis_cols:
        climate_basis_cols.append(f"cb_{c}")
    for c in temp_basis_cols:
        for s in season_cols:
            climate_basis_cols.append(f"cb_{c}_x_{s}")
    for c in prcp_basis_cols:
        for s in season_cols:
            climate_basis_cols.append(f"cb_{c}_x_{s}")


    def build_monthly_basis_sums(cache_file):
        year = int(cache_file.stem[:4])
        if year < START_YEAR or year > END_YEAR:
            return None

        chunk = (
            pl.read_parquet(
                cache_file,
                columns=["fips", "date", TEMP_SPLINE_VAR, "prcp"],
            )
            .select(
                [
                    pl.col("fips"),
                    pl.col("date"),
                    pl.col(TEMP_SPLINE_VAR).str.strip_chars().cast(pl.Float32),
                    pl.col("prcp").str.strip_chars().cast(pl.Float32),
                ]
            )
            .with_columns(
                [
                    pl.col("date").dt.year().alias("year"),
                    pl.col("date").dt.ordinal_day().alias("doy"),
                    pl.when(pl.col("prcp").is_not_null() & (pl.col("prcp") > 0))
                    .then(pl.col("prcp"))
                    .otherwise(0.0)
                    .alias("prcp0"),
                ]
            )
            .filter((pl.col("year") >= START_YEAR) & (pl.col("year") <= END_YEAR))
        )
        if chunk.is_empty():
            return None

        temp_basis = temp_spline.transform(
            chunk.select(pl.col(TEMP_SPLINE_VAR).fill_null(temp_fill_value)).to_numpy()
        ).astype(np.float32)

        log_prcp = chunk.select(pl.col("prcp0").log1p().alias("log_prcp"))
        wet_mask = (chunk["prcp0"].to_numpy() >= WET_DAY_MM).astype(np.float32)[:, None]
        prcp_basis = (
            prcp_spline.transform(log_prcp.to_numpy()).astype(np.float32) * wet_mask
        )

        season_values = []
        doy = chunk["doy"].to_numpy()
        for h in range(1, (len(season_cols) // 2) + 1):
            angle = doy * 2 * h * np.pi / 365.25
            season_values.append(np.sin(angle).astype(np.float32))
            season_values.append(np.cos(angle).astype(np.float32))
        season_basis = np.column_stack(season_values)

        arrays = {}
        for j, c in enumerate(temp_basis_cols):
            arrays[f"cb_{c}"] = temp_basis[:, j]
        for j, c in enumerate(prcp_basis_cols):
            arrays[f"cb_{c}"] = prcp_basis[:, j]

        for j, c in enumerate(temp_basis_cols):
            for s_idx, s in enumerate(season_cols):
                arrays[f"cb_{c}_x_{s}"] = temp_basis[:, j] * season_basis[:, s_idx]

        for j, c in enumerate(prcp_basis_cols):
            for s_idx, s in enumerate(season_cols):
                arrays[f"cb_{c}_x_{s}"] = prcp_basis[:, j] * season_basis[:, s_idx]

        basis_chunk = pl.DataFrame(
            {
                "fips": chunk["fips"],
                "year": chunk["year"],
                **arrays,
            }
        )

        return basis_chunk.group_by(["fips", "year"]).agg(
            [pl.col(c).sum().alias(c) for c in climate_basis_cols]
            + [pl.len().alias("_n_cb_days")]
        )


    annual_parts = []
    for _cache_file in sorted(CACHE_DIR.glob("*.parquet")):
        part = build_monthly_basis_sums(_cache_file)
        if part is not None:
            annual_parts.append(part)

    climate_basis_sums = (
        pl.concat(annual_parts, how="vertical")
        .group_by(["fips", "year"])
        .agg(
            [pl.col(c).sum().alias(c) for c in climate_basis_cols]
            + [pl.col("_n_cb_days").sum().alias("n_climate_days")]
        )
    )

    climate_basis_annual = climate_basis_sums.with_columns(
        [
            (pl.col(c) / pl.col("n_climate_days")).cast(pl.Float32).alias(c)
            for c in climate_basis_cols
        ]
    )

    climate_normal_cols = [f"normal_{c}" for c in climate_basis_cols]
    climate_basis_normals = (
        climate_basis_annual.filter(
            (pl.col("year") >= NORMAL_START_YEAR) & (pl.col("year") <= NORMAL_END_YEAR)
        )
        .group_by("fips")
        .agg([pl.col(c).mean().alias(f"normal_{c}") for c in climate_basis_cols])
    )

    climate_basis_annual = climate_basis_annual.join(
        climate_basis_normals, on="fips", how="left"
    )
    return climate_basis_annual, climate_basis_cols, climate_normal_cols


@app.cell
def _(
    END_YEAR,
    OUTPUT_FILE,
    START_YEAR,
    climate_basis_annual,
    climate_basis_cols,
    climate_normal_cols,
    pl,
):
    select_cols = (
        ["year", "fips", "state_fips", "county_fips", "n_climate_days"]
        + climate_basis_cols
        + climate_normal_cols
    )

    final_df = (
        climate_basis_annual.with_columns(
            [
                pl.col("fips").str.slice(0, 2).alias("state_fips"),
                pl.col("fips").str.slice(2, 3).alias("county_fips"),
            ]
        )
        .select(select_cols)
        .sort(["fips", "year"])
        .filter((pl.col("year") >= START_YEAR) & (pl.col("year") <= END_YEAR))
    )

    final_df.write_parquet(OUTPUT_FILE)

    print(
        f"\nDone! {final_df.height} rows | "
        f"{final_df.select('fips').n_unique()} unique counties | "
        f"years {START_YEAR}-{END_YEAR}"
    )
    print(f"Annual basis columns: {len(climate_basis_cols)}")
    print(f"Normal basis columns: {len(climate_normal_cols)}")
    print(f"Saved to: {OUTPUT_FILE}")
    return


if __name__ == "__main__":
    app.run()
