import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import pyprojroot
    import urllib.request
    import urllib.error
    import polars as pl
    import us
    import ssl

    return Path, pl, pyprojroot, ssl, urllib, us


@app.cell
def _(Path, pyprojroot):
    root_path = pyprojroot.find_root(criterion='pyproject.toml')
    binary_path = root_path / 'binaries'

    CACHE_DIR = root_path / 'data' / "epinoaa_nclimgrid"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    OUT_DIR = Path("binaries")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE = OUT_DIR / "county_h2a_prediction_climate_gdd_annual.parquet"
    return CACHE_DIR, OUTPUT_FILE


@app.cell
def _():
    START_YEAR = 2000
    END_YEAR = 2025
    MISSING_VAL = -999.00
    N_BINS = 10
    WET_DAY_MM = 1.0
    N_PRCP_BINS = 10
    return END_YEAR, N_BINS, N_PRCP_BINS, START_YEAR, WET_DAY_MM


@app.cell
def _():
    # Crop temperature preferences, in Celsius
    # --- 2. Crop GDD parameters (Celsius) ----------------------------------------
    CROPS = {
        # -- Field crops --
        "corn": {"base_C": 10.0, "cap_C": 30.0, "months":[4, 5, 6, 7, 8, 9], "cross_year": False},
        "soybean": {"base_C": 10.0, "cap_C": 30.0, "months":[4, 5, 6, 7, 8, 9, 10], "cross_year": False},
        "sorghum": {"base_C": 10.0, "cap_C": 30.0, "months": [4, 5, 6, 7, 8, 9], "cross_year": False},
        "rice": {"base_C": 10.0, "cap_C": None, "months": [5, 6, 7, 8, 9], "cross_year": False},
        "cotton": {"base_C": 15.6, "cap_C": 30.0, "months":[4, 5, 6, 7, 8, 9, 10], "cross_year": False},
        "winter_wheat": {"base_C": 0.0, "cap_C": 35.0, "months":[10, 11, 12, 1, 2, 3, 4, 5, 6], "cross_year": True},
        "spring_wheat": {"base_C": 0.0, "cap_C": 35.0, "months":[3, 4, 5, 6, 7], "cross_year": False},
        "barley": {"base_C": 0.0, "cap_C": 35.0, "months": [3, 4, 5, 6, 7], "cross_year": False},
        "canola": {"base_C": 5.0, "cap_C": None, "months": [3, 4, 5, 6], "cross_year": False},
        "sunflower": {"base_C": 6.7, "cap_C": None, "months":[5, 6, 7, 8, 9], "cross_year": False},

        # -- Specialty crops --
        "grape": {"base_C": 10.0, "cap_C": None, "months":[4, 5, 6, 7, 8, 9, 10], "cross_year": False},
        "citrus": {"base_C": 12.8, "cap_C": None, "months": list(range(1, 13)), "cross_year": False},
        "apple": {"base_C": 6.1, "cap_C": None, "months":[3, 4, 5, 6, 7, 8, 9, 10], "cross_year": False},
        "potato": {"base_C": 4.4, "cap_C": 30.0, "months":[4, 5, 6, 7, 8, 9], "cross_year": False},
        "sugar_beet": {"base_C": 1.1, "cap_C": 30.0, "months":[4, 5, 6, 7, 8, 9, 10], "cross_year": False},
        "tomato": {"base_C": 10.0, "cap_C": 30.0, "months":[5, 6, 7, 8, 9], "cross_year": False},
        "alfalfa": {"base_C": 5.0, "cap_C": None, "months":[3, 4, 5, 6, 7, 8, 9, 10], "cross_year": False},
        "peanut": {"base_C": 10.0, "cap_C": None, "months": [5, 6, 7, 8, 9, 10], "cross_year": False},
        "tobacco": {"base_C": 12.8, "cap_C": None, "months": [5, 6, 7, 8, 9], "cross_year": False},
    }
    return (CROPS,)


@app.cell
def _(CACHE_DIR, pl, us):
    ncei_fips_xwalk = pl.read_csv(CACHE_DIR / 'ncei_state_code.csv', infer_schema=False)
    ncei_fips_xwalk = ncei_fips_xwalk.with_columns(
        pl.col('state_name').map_elements(lambda x: us.states.lookup(x).fips, return_dtype=pl.String).alias('fips')
    ).with_columns(
        pl.col('ncei').str.pad_start(2, fill_char='0').alias('ncei')
    )
    return


@app.function
def get_parquet_url(year: int, month: int) -> str:
    ym = f"{year:04d}{month:02d}"
    yr_str = f"{year:04d}"
    file_name = f"{ym}.parquet"
    return f"https://noaa-nclimgrid-daily-pds.s3.amazonaws.com/EpiNOAA/v1-0-0/parquet/cty/YEAR={year}/STATUS=scaled/{file_name}"


@app.cell
def _(CACHE_DIR, Path, ssl, urllib):
    def download_monthly_parquet(year: int, month: int) -> Path:
        ym_str = f"{year}{month:02d}"
        cache_file = CACHE_DIR / f"{ym_str}.parquet"

        if cache_file.exists():
            print(f"{ym_str}.parquet already cached")
            return cache_file

        url = get_parquet_url(year, month)
        ctx = ssl._create_unverified_context()

        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30, context=ctx) as response:
                data = response.read()

                # Validate by checking for Parquet magic bytes ('PAR1')
                if not data.startswith(b'PAR1'):
                    print(f"ERROR: URL {url} did not return a Parquet file.")
                    # Print the first 250 characters to check if it's HTML or XML
                    preview = data[:250].decode('utf-8', errors='ignore').replace('\n', ' ')
                    print(f"Preview of downloaded content: {preview}...\n")
                    return None

                with open(cache_file, 'wb') as f_out:
                    f_out.write(data)

            return cache_file

        except urllib.error.URLError as e:
            print(f"Failed to download {year}-{month:02d}: {e}")
            return None

    return (download_monthly_parquet,)


@app.cell
def _(END_YEAR, START_YEAR, download_monthly_parquet):
    print(f"Downloading EpiNOAA monthly county parquets ({START_YEAR - 1}-{END_YEAR})...")
    for _yr in range(START_YEAR - 1, END_YEAR + 1):
        for _mo in range(1, 13):
            download_monthly_parquet(_yr, _mo)

    print("Downloads Complete.")
    return


@app.cell
def _(CACHE_DIR, START_YEAR, pl):
    all_daily = pl.read_parquet(CACHE_DIR / "*.parquet")
    all_daily = (
        all_daily
        .select([
            # Rename columns standard to EpiNOAA down to what your script expects
            pl.col("fips"),
            pl.col("date"),
            pl.col("tmin").str.strip_chars().cast(pl.Float64),
            pl.col("tmax").str.strip_chars().cast(pl.Float64),
            pl.col("tavg").str.strip_chars().cast(pl.Float64),
            pl.col("prcp").str.strip_chars().cast(pl.Float64)
        ])
        .with_columns([
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month")
        ])
        .filter(pl.col("year") >= START_YEAR - 1)
    )
    return (all_daily,)


@app.cell
def _(END_YEAR, N_BINS, N_PRCP_BINS, START_YEAR, WET_DAY_MM, all_daily, pl):
    # Compute decile breakpoints (pooled from all county-years)
    df_bounds = all_daily.filter(
        pl.col("year") >= START_YEAR,
        pl.col("year") <= END_YEAR
    )

    # TAVG breaks
    tavg_series = df_bounds.filter(pl.col("tavg").is_not_null())["tavg"]
    tavg_probs = [i / N_BINS for i in range(1, N_BINS)]
    tavg_bin_breaks = [tavg_series.quantile(p, interpolation="linear") for p in tavg_probs]

    # PRCP breaks (wet days only)
    prcp_series = df_bounds.filter((pl.col("prcp") >= WET_DAY_MM) & pl.col("prcp").is_not_null())["prcp"]
    prcp_probs =[i / N_PRCP_BINS for i in range(1, N_PRCP_BINS)]
    prcp_bin_breaks =[prcp_series.quantile(p, interpolation="linear") for p in prcp_probs]
    return df_bounds, prcp_bin_breaks, tavg_bin_breaks


@app.cell
def _(df_bounds, pl):
    # Compute annual metrics
    base_group_cols = ["fips", "year"]

    # Annual Temperature 
    ann_temp = df_bounds.group_by(base_group_cols).agg([
        pl.col("tmin").mean().alias("tmin_ann"),
        pl.col("tmax").mean().alias("tmax_ann"),
        pl.col("tavg").mean().alias("tavg_ann"),
        pl.col("tavg").is_not_null().sum().alias("n_days")
    ])
    return ann_temp, base_group_cols


@app.cell
def _(CROPS, START_YEAR, all_daily, base_group_cols, pl):
    # Crop GDDs
    crop_aggs =[]
    for crop, params in CROPS.items():
        b_C, c_C = params["base_C"], params["cap_C"]

        tmin_adj = pl.max_horizontal(pl.col("tmin"), b_C)
        tmax_adj = pl.min_horizontal(pl.col("tmax"), c_C) if c_C is not None else pl.col("tmax")
        daily_gdd = pl.max_horizontal((tmin_adj + tmax_adj) / 2.0 - b_C, 0.0)

        df_crop = all_daily.filter(pl.col("month").is_in(params["months"]))

        if params["cross_year"]:
            harvest_yr = pl.when(pl.col("month") >= 10).then(pl.col("year") + 1).otherwise(pl.col("year"))
        else:
            harvest_yr = pl.col("year")

        agg = df_crop.with_columns(
            daily_gdd.alias("gdd"),
            harvest_yr.alias("year")
        ).filter(
            pl.col("year") >= START_YEAR
        ).group_by(base_group_cols).agg([
            pl.col("gdd").sum().alias(f"GDD_{crop}"),
            pl.col("gdd").is_not_null().sum().alias(f"n_{crop}")
        ])
        crop_aggs.append(agg)
    return (crop_aggs,)


@app.cell
def _(N_BINS, base_group_cols, df_bounds, pl, tavg_bin_breaks):
    # Temperature Bins
    t_breaks =[-float('inf')] + tavg_bin_breaks + [float('inf')]
    bin_labels =[f"days_D{i:02d}" for i in range(1, N_BINS + 1)]
    t_bin_exprs =[]
    for _i in range(N_BINS):
        if _i == 0:
            cond = pl.col("tavg") < t_breaks[_i+1]
        elif _i == N_BINS - 1:
            cond = pl.col("tavg") >= t_breaks[_i]
        else:
            cond = (pl.col("tavg") >= t_breaks[_i]) & (pl.col("tavg") < t_breaks[_i+1])
        t_bin_exprs.append(cond.sum().alias(bin_labels[_i]))

    temp_bins = df_bounds.group_by(base_group_cols).agg(t_bin_exprs)
    return bin_labels, temp_bins


@app.cell
def _(WET_DAY_MM, base_group_cols, df_bounds, pl):
    # Precipitation Metrics
    prcp_metrics = df_bounds.group_by(base_group_cols).agg([
        pl.col("prcp").sum().alias("prcp_ann"),
        (pl.col("prcp") >= WET_DAY_MM).sum().alias("n_wet_days"),
        pl.when(pl.col("month").is_in([4,5,6,7,8,9])).then(pl.col("prcp")).otherwise(0.0).sum().alias("prcp_gs"),
        pl.when(pl.col("month").is_in([3,4,5])).then(pl.col("prcp")).otherwise(0.0).sum().alias("prcp_spring")
    ])
    return (prcp_metrics,)


@app.cell
def _(WET_DAY_MM, base_group_cols, df_bounds, pl):
    # Consecutive Dry Days (max_cdd_gs) in Apr-Sep
    cdd_df = df_bounds.filter(
        pl.col("month").is_in([4, 5, 6, 7, 8, 9])
    ).sort(["fips", "date"]).with_columns(
        pl.when(pl.col("prcp").is_null()).then(False).when(pl.col("prcp") < WET_DAY_MM).then(True).otherwise(False).alias("is_dry")
    ).with_columns(
        (~pl.col("is_dry")).cum_sum().over(["fips", "year"]).alias("run_id")
    ).filter(
        pl.col("is_dry")
    ).group_by(
        base_group_cols +["run_id"]
    ).agg(
        pl.len().alias("cdd_len")
    ).group_by(base_group_cols).agg(
        pl.col("cdd_len").max().alias("max_cdd_gs")
    )
    return (cdd_df,)


@app.cell
def _(
    N_PRCP_BINS,
    WET_DAY_MM,
    base_group_cols,
    df_bounds,
    pl,
    prcp_bin_breaks,
):
    # Precipitation Bins (Wet Days Only)
    p_breaks = [-float('inf')] + prcp_bin_breaks + [float('inf')]
    prcp_labels = [f"days_P{_i:02d}" for _i in range(1, N_PRCP_BINS + 1)]
    p_bin_exprs = []
    for _i in range(N_PRCP_BINS):
        if _i == 0:
            _cond = pl.col("prcp") < p_breaks[_i+1]
        elif _i == N_PRCP_BINS - 1:
            _cond = pl.col("prcp") >= p_breaks[_i]
        else:
            _cond = (pl.col("prcp") >= p_breaks[_i]) & (pl.col("prcp") < p_breaks[_i+1])
        p_bin_exprs.append(_cond.sum().alias(prcp_labels[_i]))

    prcp_bins = df_bounds.filter(
        (pl.col("prcp") >= WET_DAY_MM) & pl.col("prcp").is_not_null()
    ).group_by(base_group_cols).agg(p_bin_exprs)
    return prcp_bins, prcp_labels


@app.cell
def _(
    CROPS,
    END_YEAR,
    ann_temp,
    base_group_cols,
    bin_labels,
    cdd_df,
    crop_aggs,
    pl,
    prcp_bins,
    prcp_labels,
    prcp_metrics,
    temp_bins,
):
    # Combine, format output
    final_df = ann_temp
    for agg_df in crop_aggs + [temp_bins, prcp_metrics, cdd_df, prcp_bins]:
        final_df = final_df.join(agg_df, on=base_group_cols, how="left")

    final_df = final_df.with_columns(pl.col("max_cdd_gs").fill_null(0))
    for p_label in prcp_labels:
        final_df = final_df.with_columns(pl.col(p_label).fill_null(0))

    # Break FIPS back into components if desired
    final_df = final_df.with_columns([
        pl.col("fips").str.slice(0, 2).alias("state_fips"),
        pl.col("fips").str.slice(2, 3).alias("county_fips")
    ])

    # Reorder columns
    select_cols =["year", "fips", "state_fips", "county_fips", "tmin_ann", "tmax_ann", "tavg_ann", "n_days"]
    for _crop in CROPS.keys():
        select_cols +=[f"GDD_{_crop}", f"n_{_crop}"]
    select_cols += bin_labels +["prcp_ann", "prcp_gs", "prcp_spring", "n_wet_days", "max_cdd_gs"] + prcp_labels

    final_df = final_df.select(select_cols).sort(["fips", "year"]).filter(pl.col("year") <= END_YEAR)
    return (final_df,)


@app.cell
def _(END_YEAR, OUTPUT_FILE, START_YEAR, final_df):
    # Save
    final_df.write_parquet(OUTPUT_FILE)

    print(f"\nDone! {final_df.height} rows | {final_df.select('fips').n_unique()} unique counties | years {START_YEAR}-{END_YEAR}")
    print(f"Saved to: {OUTPUT_FILE}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
