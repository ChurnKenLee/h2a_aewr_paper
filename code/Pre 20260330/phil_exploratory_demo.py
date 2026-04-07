import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import altair as alt
    from scipy import stats
    from pyfixest.estimation import feols
    import pandas as pd # pyfixest currently requires pandas inputs
    import pyprojroot
    alt.data_transformers.enable("vegafusion")
    return alt, feols, mo, pd, pl, pyprojroot, stats


@app.cell
def _(pl, pyprojroot):
    root_path = pyprojroot.find_root(criterion='pyproject.toml')
    binary_path = root_path / 'Data Int'

    # Load Inputs
    h2a = pl.read_parquet(binary_path / "h2a_aggregated.parquet")
    aewr = pl.read_parquet(binary_path / "aewr.parquet")
    ppi = pl.read_parquet(binary_path / "ppi_2012.parquet")
    emp = pl.read_parquet(binary_path / "bea_caemp25n_data_year.parquet")
    return aewr, emp, h2a, ppi


@app.cell
def _(aewr, emp, h2a, pl, ppi):
    # Harmonize column data types
    h2a_cast = h2a.with_columns(
        pl.col('year').cast(pl.Int64).alias('year'),
        pl.col('nbr_workers_certified_start_year').alias('h2a_workers') # this is our primary definition of h2a usage
    )
    aewr_cast = aewr.with_columns(
        pl.col('year').cast(pl.Int64).alias('year'),
        pl.col('aewr').cast(pl.Float64).alias('aewr')
    )
    ppi_cast = ppi.with_columns(
        pl.col('year').cast(pl.Int64).alias('year')
    )
    # emp dataset is more complicated, need to convert 5 digit fips from float to str
    emp_cast = emp.with_columns(
        pl.col('year').cast(pl.Int64).alias('year'),
        pl.col('countyfips').cast(pl.Int64).cast(pl.String).alias('fips')
    ).with_columns(
        pl.col('fips').str.pad_start(length=5, fill_char='0')
    ).with_columns(
        state_fips_code = pl.col('fips').str.slice(length=2, offset=0),
        county_fips_code = pl.col('fips').str.slice(length=3, offset=2)
    )
    return aewr_cast, emp_cast, h2a_cast, ppi_cast


@app.cell
def _(aewr_cast, emp_cast, h2a_cast, pl, ppi_cast):
    # Build joined dataframe
    df = (emp_cast
        .join(h2a_cast, on=["year", "state_fips_code", "county_fips_code"], how="left")
        .join(ppi_cast, on="year", how="left")
        .join(aewr_cast, on=["year", "state_fips_code"], how="left")
        .with_columns([
            pl.col('h2a_workers').fill_null(0).alias("h2a_workers"),
            (pl.col("aewr") / pl.col("ppi_2012")).alias("real_aewr")
        ]))

    # Calculate H-2A employment share
    df = (df
        .with_columns(
            h2a_share = pl.col('h2a_workers') / pl.col('emp_farm')
        ))

    # Define exposure groups (baseline 2008)
    # Exposure is defined by share of H-2A workers in total agriculture employment in 2008
    usage_2008 = (df
        .filter(
            pl.col('year') == 2008
        ).with_columns(
            pl.col('h2a_share').quantile(interpolation='linear', quantile=0.5).alias('share_50th'),
            pl.col('h2a_share').quantile(interpolation='linear', quantile=0.66).alias('share_66th'),
            pl.col('h2a_share').quantile(interpolation='linear', quantile=0.75).alias('share_75th')
        ))

    usage_2008 = (usage_2008
        .with_columns(
            pl.when(pl.col("h2a_share") > pl.col('share_50th')).then(pl.lit("High Pre-Usage")).otherwise(pl.lit("Low Pre-Usage")).alias("exposure_group")
        )
        .select(["state_fips_code", "county_fips_code", "exposure_group"]).unique()
    )

    df = df.join(
        usage_2008, on=['state_fips_code', 'county_fips_code'], how='left'
    ).filter(pl.col("emp_farm") > 0)
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Slope/elasticity comparison between high vs low exposure share
    """)
    return


@app.cell
def _(alt, df, pd, pl, stats):
    # Calculate Log Growth Rates
    # We use log1p for workers because many counties have 0
    slope_df = (
        df
            .sort(["state_fips_code", "county_fips_code", "year"])
            .with_columns([
                pl.col("h2a_share").log1p().diff().over(["state_fips_code", "county_fips_code"]).alias("d_ln_h2a"),
                pl.col("real_aewr").log().diff().over(["state_fips_code", "county_fips_code"]).alias("d_ln_aewr")
            ])
            .drop_nulls(["d_ln_h2a", "d_ln_aewr"])
    )

    # Filter for outliers (common in H-2A growth rates)
    slope_df = slope_df.filter(
        (pl.col("d_ln_h2a").abs() < 2) & (pl.col("d_ln_aewr").abs() < 0.2)
    )
    clean_df = slope_df.drop_nulls(["exposure_group", "d_ln_aewr", "d_ln_h2a"])

    # 2. PYTHON-SIDE REGRESSION (Saves 90MB of overhead)
    # We calculate the line coordinates manually so Altair doesn't have to
    line_data_list = []
    for group in clean_df["exposure_group"].unique():
        subset = clean_df.filter(pl.col("exposure_group") == group)
    
        # Run simple linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            subset["d_ln_aewr"].to_list(), 
            subset["d_ln_h2a"].to_list()
        )
    
        # Create two points to define the line (Min and Max X)
        min_x, max_x = subset["d_ln_aewr"].min(), subset["d_ln_aewr"].max()
        line_data_list.append({"exposure_group": group, "x": min_x, "y": intercept + slope * min_x})
        line_data_list.append({"exposure_group": group, "x": max_x, "y": intercept + slope * max_x})

    # This dataframe is only 4 rows long!
    line_df = pd.DataFrame(line_data_list)

    # 3. SAMPLE DATA FOR DOTS
    # 3,000 dots is plenty for an EDA visual
    dots_df = clean_df.sample(n=3000, seed=42).to_pandas()

    # 4. PLOT (Total data sent to browser: ~200KB)
    _base = alt.Chart(dots_df).encode(
        x=alt.X('d_ln_aewr:Q', title='Δ Log Real AEWR'),
        y=alt.Y('d_ln_h2a:Q', title='Δ Log H-2A Workers'),
        color=alt.Color('exposure_group:N', title="Exposure Group")
    )

    points = _base.mark_point(opacity=0.3, size=15)

    # The regression line now uses the tiny 4-row dataframe
    lines = alt.Chart(line_df).mark_line(size=4).encode(
        x='x:Q',
        y='y:Q',
        color='exposure_group:N'
    )

    chart = (points + lines).properties(
        width=500, height=350,
        title="Differential Labor Demand Elasticity"
    )

    chart.display()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gap-on-gap plot to see whether H-2A gap in usage coincides with gap in AEWR
    """)
    return


@app.cell
def _(alt, df, pl):
    # 1. Aggregate usage by year and group
    # We calculate the mean number of H-2A workers per county in each group
    agg_usage = (
        df.group_by(["year", "exposure_group"])
        .agg(pl.col("h2a_share").mean().alias("mean_usage"))
    )

    # 2. Pivot to get groups into separate columns so we can calculate the Gap
    # Using standard .pivot()
    pivoted_usage = agg_usage.pivot(
        on="exposure_group",
        index="year",
        values="mean_usage"
    )

    # 3. Calculate the Gap
    # Note: Ensure these column names match exactly what you defined in the exposure_group logic
    # Based on the previous prompt: "High Exposure (Low Init. Usage)" vs "Low Exposure (High Init. Usage)"
    gap_df = pivoted_usage.with_columns(
        (pl.col("High Pre-Usage") - pl.col("Low Pre-Usage"))
        .alias("usage_gap")
    )

    # 4. Get the average annual Real AEWR to plot on the second axis
    avg_aewr = (
        df.group_by("year")
        .agg(pl.col("real_aewr").mean().alias("avg_real_aewr"))
    )

    # 5. Join them together
    final_gap_df = gap_df.join(avg_aewr, on="year").sort("year").to_pandas()

    # 6. Create the Dual-Axis Plot in Altair
    base = alt.Chart(final_gap_df).encode(
        x=alt.X('year:O', title='Year')
    )

    # Line 1: The Usage Gap (Left Axis)
    line_gap = base.mark_line(color='#1f77b4', strokeWidth=3).encode(
        y=alt.Y('usage_gap:Q', title='Usage Gap (High Usage - Low Usage Counties)', 
                axis=alt.Axis(titleColor='#1f77b4'))
    )

    # Line 2: Real AEWR (Right Axis)
    line_wage = base.mark_line(color='#d62728', strokeDash=[5,5], strokeWidth=3).encode(
        y=alt.Y('avg_real_aewr:Q', title='Average Real AEWR ($)', 
                axis=alt.Axis(titleColor='#d62728'))
    )

    # Combine with independent Y scales
    chart2 = alt.layer(line_gap, line_wage).resolve_scale(
        y='independent'
    ).properties(
        width=600,
        height=400,
        title="Strategy 2: The 'Gap-on-Gap' Identification Test"
    )

    chart2.display()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Mundlak mean-residual
    """)
    return


@app.cell
def _(alt, df, feols, pd, pl):
    clean_pdf = (
        df.with_columns([
            (pl.col("h2a_workers") / pl.col("emp_farm")).alias("intensity"),
            pl.col("real_aewr").log().alias("ln_aewr")
        ])
        .with_columns(
            pl.col("intensity").log1p().alias("ln_intensity")
        )
        # Drop rows where any key variable is null or infinite
        .filter(
            pl.col("ln_intensity").is_finite() & 
            pl.col("ln_aewr").is_finite() &
            pl.col("county_fips_code").is_not_null() &
            pl.col("year").is_not_null() &
            pl.col("exposure_group").is_not_null()
        )
        .to_pandas()
    )

    # 2. RUN REGRESSIONS
    # Now clean_pdf, residuals, and exposure_group are guaranteed to be the same length
    fit_y = feols("ln_intensity ~ 1 | county_fips_code + year", data=clean_pdf)
    fit_x = feols("ln_aewr ~ 1 | county_fips_code + year", data=clean_pdf)

    # 3. CONSTRUCT PLOT DATAFRAME (Safe now)
    # We use the index of clean_pdf implicitly by just creating the dict
    plot_df = pd.DataFrame({
        'resid_intensity': fit_y.resid(),
        'resid_wages': fit_x.resid(),
        'exposure_group': clean_pdf['exposure_group']
    })

    # 4. SAMPLE & STRIP (To stay under Marimo's 10MB limit)
    # We only need these 3 columns for the visualization
    final_plot_data = (
        plot_df[['resid_intensity', 'resid_wages', 'exposure_group']]
        .sample(n=min(2000, len(plot_df)), random_state=42)
        .reset_index(drop=True)
    )

    # 5. VISUALIZE
    scatter = alt.Chart(final_plot_data).mark_circle(size=20, opacity=0.4).encode(
        x=alt.X('resid_wages:Q', title='Real Wage Shock (Residualized ln_AEWR)'),
        y=alt.Y('resid_intensity:Q', title='Intensity Shock (Residualized ln_Intensity)'),
        color=alt.Color('exposure_group:N', title='Exposure Group')
    )

    # Add a reference line
    reg_line = scatter.transform_regression(
        'resid_wages', 'resid_intensity'
    ).mark_line(color='black', strokeDash=[4,2])

    chart4 = (scatter + reg_line).properties(
        width=500, height=350,
        title="Strategy 4: Identification from Within-County Variation"
    )

    chart4.display()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
