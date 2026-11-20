# Purpose: Interactive, read-only views of the panel-IV identifying variation.
# Inputs: data/processed/panel_iv_county_year.parquet,
#   data/processed/wage_reforms_cases.parquet, data/intermediate/aewr.parquet,
#   data/intermediate/fls_region.parquet.
# Outputs: none. Exploration only; nothing here selects a specification.
#
# Run with: uv run --no-sync marimo edit code/descriptives/panel_iv_intuition.py

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl

    from h2a.paths import INTERMEDIATE, PROCESSED

    alt.data_transformers.disable_max_rows()
    YEAR_MIN, YEAR_MAX = 2011, 2024
    return INTERMEDIATE, PROCESSED, YEAR_MAX, YEAR_MIN, alt, mo, np, pl


@app.cell
def _(mo):
    mo.md(r"""
    # Panel IV: what the identifying variation looks like

    Real AEWR varies only by AEWR region and year. With county and year fixed
    effects, the design identifies from how the 17 regional AEWRs move
    *relative to each other*. Each view below works at the level where that
    variation lives. "Residual" always means after removing the two fixed
    effects named in the view (no controls). These are descriptive views, not
    estimates of the preferred specification.
    """)
    return


@app.cell
def _(INTERMEDIATE, PROCESSED, YEAR_MAX, YEAR_MIN, pl):
    panel = pl.read_parquet(
        PROCESSED / "panel_iv_county_year.parquet",
        columns=[
            "county_fips", "countyname", "state_abbrev", "year",
            "aewr_region_id", "aewr_iv_cluster_id", "aewr_ppi",
            "z_wage_seasonal_composition_real",
            "emp_farm_2011", "nbr_workers_certified_start_year",
            "man_hours_certified_start_year", "nbr_applications_start_year",
            "cert_hours_at_aewr_start_year",
            "cert_hours_with_hourly_wage_and_aewr_start_year",
        ],
    ).filter(pl.col("year").is_between(YEAR_MIN, YEAR_MAX))

    region_names = (
        pl.read_parquet(INTERMEDIATE / "fls_region.parquet")
        .select("aewr_region_id", "region_name")
        .unique()
        .with_columns(pl.col("aewr_region_id").cast(pl.Int32))
    )
    panel = panel.with_columns(
        pl.col("aewr_region_id").cast(pl.Int32)
    ).join(region_names, on="aewr_region_id", how="left")
    return panel, region_names


@app.cell
def _(np, pl):
    def residualize(frame, columns, first, second, iterations=200):
        """Remove two sets of fixed effects by alternating projections."""
        out = frame.with_columns(
            [pl.col(c).alias(f"{c}_resid") for c in columns]
        )
        for _ in range(iterations):
            before = out.select([f"{c}_resid" for c in columns]).to_numpy()
            for group in (first, second):
                out = out.with_columns(
                    [
                        pl.col(f"{c}_resid") - pl.col(f"{c}_resid").mean().over(group)
                        for c in columns
                    ]
                )
            after = out.select([f"{c}_resid" for c in columns]).to_numpy()
            if np.nanmax(np.abs(after - before)) < 1e-10:
                break
        return out

    def slope(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        keep = np.isfinite(x) & np.isfinite(y)
        x, y = x[keep], y[keep]
        sxx = float((x * x).sum())
        b = float((x * y).sum() / sxx) if sxx > 0 else float("nan")
        resid = y - b * x
        r2 = 1 - float((resid**2).sum() / (y**2).sum()) if (y**2).sum() > 0 else float("nan")
        return b, r2, int(keep.sum())

    return residualize, slope


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Treatment variation: regional AEWR paths and their residuals
    """)
    return


@app.cell
def _(panel, pl, residualize):
    region_year = (
        panel.group_by("aewr_region_id", "region_name", "year")
        .agg(pl.col("aewr_ppi").mean())
        .sort("aewr_region_id", "year")
    )
    region_year = residualize(region_year, ["aewr_ppi"], "aewr_region_id", "year")
    return (region_year,)


@app.cell
def _(mo, region_names):
    region_choice = mo.ui.dropdown(
        options=sorted(region_names["region_name"].to_list()),
        value="Southeast",
        label="Highlight / case-study region",
    )
    region_choice
    return (region_choice,)


@app.cell
def _(alt, mo, pl, region_choice, region_year):
    _highlight = alt.condition(
        alt.datum.region_name == region_choice.value,
        alt.value(3), alt.value(0.8),
    )
    _levels = (
        alt.Chart(region_year.to_pandas())
        .mark_line()
        .encode(
            x=alt.X("year:O", title=None),
            y=alt.Y("aewr_ppi:Q", title="Real AEWR (2012 $)", scale=alt.Scale(zero=False)),
            color=alt.Color("region_name:N", title="Region"),
            strokeWidth=_highlight,
            tooltip=["region_name", "year", alt.Tooltip("aewr_ppi:Q", format=".2f")],
        )
        .properties(width=520, height=300, title="Real AEWR by region")
    )
    _heat = (
        alt.Chart(region_year.to_pandas())
        .mark_rect()
        .encode(
            x=alt.X("year:O", title=None),
            y=alt.Y("region_name:N", title=None),
            color=alt.Color(
                "aewr_ppi_resid:Q",
                title="Residual $",
                scale=alt.Scale(scheme="redblue", domainMid=0, reverse=True),
            ),
            tooltip=["region_name", "year", alt.Tooltip("aewr_ppi_resid:Q", format=".2f")],
        )
        .properties(width=520, height=300, title="AEWR after region and year means")
    )
    _resid = region_year["aewr_ppi_resid"]
    _ss = (_resid**2).sum()
    _top = region_year.with_columns(
        (pl.col("aewr_ppi_resid") ** 2 / _ss).alias("share_of_variation")
    ).sort("share_of_variation", descending=True)
    _top10 = _top.head(10)
    mo.vstack([
        mo.hstack([_levels, _heat]),
        mo.md(
            f"Residual SD: **${_resid.std():.2f}** over {region_year.height} region-years. "
            f"The 10 largest region-years carry **{_top10['share_of_variation'].sum():.0%}** "
            "of the identifying variation (sum of squared residuals)."
        ),
        _top10.select(
            "region_name", "year",
            pl.col("aewr_ppi").round(2), pl.col("aewr_ppi_resid").round(2),
            pl.col("share_of_variation").round(3),
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. First stage at the target-cluster-year level

    Each point is one of 85 target clusters in one year. Both axes remove
    cluster and year means. A steep, broad cloud is a healthy first stage; a
    slope carried by a few points is fragile even when the F is large.
    """)
    return


@app.cell
def _(mo):
    instrument_choice = mo.ui.dropdown(
        options={
            "Preferred (wage + seasonal/composition)": "z_wage_seasonal_composition_real",
        },
        value="Preferred (wage + seasonal/composition)",
        label="Instrument",
    )
    instrument_choice
    return (instrument_choice,)


@app.cell
def _(alt, instrument_choice, mo, panel, pl, residualize, slope):
    _z = instrument_choice.value
    _cluster_year = (
        panel.filter(pl.col(_z).is_not_null(), pl.col("aewr_iv_cluster_id").is_not_null())
        .group_by("aewr_iv_cluster_id", "region_name", "year")
        .agg(pl.col("aewr_ppi").mean(), pl.col(_z).mean().alias("z"))
    )
    _cluster_year = residualize(_cluster_year, ["aewr_ppi", "z"], "aewr_iv_cluster_id", "year")
    _b, _r2, _n = slope(_cluster_year["z_resid"], _cluster_year["aewr_ppi_resid"])
    _trim = _cluster_year.with_columns(pl.col("z_resid").abs().alias("_abs")).sort("_abs").head(
        int(_cluster_year.height * 0.95)
    )
    _bt, _, _ = slope(_trim["z_resid"], _trim["aewr_ppi_resid"])
    _points = (
        alt.Chart(_cluster_year.to_pandas())
        .mark_circle(size=30, opacity=0.6)
        .encode(
            x=alt.X("z_resid:Q", title="Instrument, residual"),
            y=alt.Y("aewr_ppi_resid:Q", title="Real AEWR, residual"),
            color=alt.Color("region_name:N", title="Region"),
            tooltip=["aewr_iv_cluster_id", "region_name", "year",
                     alt.Tooltip("z_resid:Q", format=".2f"),
                     alt.Tooltip("aewr_ppi_resid:Q", format=".2f")],
        )
    )
    _line = _points.transform_regression("z_resid", "aewr_ppi_resid").mark_line(color="black")
    mo.vstack([
        (_points + _line).properties(width=700, height=420),
        mo.md(
            f"Slope **{_b:.3f}** (R² {_r2:.2f}, {_n} cluster-years). "
            f"Dropping the 5% of points with the most extreme instrument residuals: "
            f"slope **{_bt:.3f}**."
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Reduced form at the region-year level

    Region-year totals of H-2A use divided by the region's total 2011 farm
    employment, then region and year means removed. This is 17 regions × 14
    years. If no slope is visible here, the county regressions are not finding
    a robust pattern either; they mostly reweight these same comparisons.
    """)
    return


@app.cell
def _(mo):
    outcome_choice = mo.ui.dropdown(
        options={
            "Certified workers / 2011 farm emp.": "workers",
            "Certified hours / 2011 farm emp.": "hours",
            "Applications / 2011 farm emp.": "applications",
            "log(certified workers)": "log_workers",
        },
        value="Certified workers / 2011 farm emp.",
        label="Outcome",
    )
    x_choice = mo.ui.dropdown(
        options={"Real AEWR": "aewr_ppi", "Preferred instrument": "z"},
        value="Real AEWR",
        label="Horizontal axis",
    )
    mo.hstack([outcome_choice, x_choice], justify="start")
    return outcome_choice, x_choice


@app.cell
def _(alt, mo, outcome_choice, panel, pl, residualize, slope, x_choice):
    _base = panel.filter(pl.col("emp_farm_2011") > 0)
    _ry = (
        _base.group_by("aewr_region_id", "region_name", "year")
        .agg(
            pl.col("aewr_ppi").mean(),
            (
                (pl.col("z_wage_seasonal_composition_real") * pl.col("emp_farm_2011")).sum()
                / pl.col("emp_farm_2011").filter(pl.col("z_wage_seasonal_composition_real").is_not_null()).sum()
            ).alias("z"),
            (pl.col("nbr_workers_certified_start_year").sum() / pl.col("emp_farm_2011").sum()).alias("workers"),
            (pl.col("man_hours_certified_start_year").sum() / pl.col("emp_farm_2011").sum()).alias("hours"),
            (pl.col("nbr_applications_start_year").sum() / pl.col("emp_farm_2011").sum()).alias("applications"),
            pl.col("nbr_workers_certified_start_year").sum().log().alias("log_workers"),
        )
    )
    _y, _x = outcome_choice.value, x_choice.value
    _ry = residualize(_ry, [_y, _x], "aewr_region_id", "year")
    _b, _r2, _n = slope(_ry[f"{_x}_resid"], _ry[f"{_y}_resid"])
    _points = (
        alt.Chart(_ry.to_pandas())
        .mark_circle(size=60, opacity=0.7)
        .encode(
            x=alt.X(f"{_x}_resid:Q", title=f"{x_choice.selected_key}, residual"),
            y=alt.Y(f"{_y}_resid:Q", title=f"{outcome_choice.selected_key}, residual"),
            color=alt.Color("region_name:N", title="Region"),
            tooltip=["region_name", "year",
                     alt.Tooltip(f"{_x}_resid:Q", format=".3f"),
                     alt.Tooltip(f"{_y}_resid:Q", format=".4f")],
        )
    )
    _line = _points.transform_regression(f"{_x}_resid", f"{_y}_resid").mark_line(color="black")
    mo.vstack([
        (_points + _line).properties(width=700, height=420),
        mo.md(f"Slope **{_b:.4g}** (R² {_r2:.3f}, {_n} region-years)."),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Where H-2A use actually is (2011–2024 certified workers)
    """)
    return


@app.cell
def _(alt, mo, panel, pl):
    _county = (
        panel.group_by("county_fips", "countyname", "region_name")
        .agg(pl.col("nbr_workers_certified_start_year").sum().alias("certified"))
        .sort("certified", descending=True)
        .with_row_index("rank", offset=1)
    )
    _total = _county["certified"].sum()
    _county = _county.with_columns((pl.col("certified").cum_sum() / _total).alias("cumulative_share"))
    _users = _county.filter(pl.col("certified") > 0).height
    _share = lambda k: _county.head(k)["certified"].sum() / _total
    _zero_cy = panel.select((pl.col("nbr_workers_certified_start_year") == 0).mean()).item()
    _curve = (
        alt.Chart(_county.filter(pl.col("rank") <= 1000).to_pandas())
        .mark_line()
        .encode(
            x=alt.X("rank:Q", title="County rank by total certified workers"),
            y=alt.Y("cumulative_share:Q", title="Cumulative share", axis=alt.Axis(format="%")),
            tooltip=["rank", "countyname", alt.Tooltip("cumulative_share:Q", format=".1%")],
        )
        .properties(width=520, height=300, title="Concentration of certified workers")
    )
    _by_year = (
        panel.group_by("year")
        .agg((pl.col("nbr_workers_certified_start_year") > 0).sum().alias("counties_using"))
        .sort("year")
    )
    _users_chart = (
        alt.Chart(_by_year.to_pandas())
        .mark_bar()
        .encode(x=alt.X("year:O", title=None), y=alt.Y("counties_using:Q", title="Counties with any certification"))
        .properties(width=420, height=300, title="Counties using H-2A each year")
    )
    mo.vstack([
        mo.md(
            f"{panel['county_fips'].n_unique():,} counties; **{_users:,}** ever certified in 2011–2024. "
            f"Top 10 counties: **{_share(10):.0%}** of certified workers; top 50: **{_share(50):.0%}**; "
            f"top 100: **{_share(100):.0%}**; top 500: **{_share(500):.0%}**. "
            f"County-years with zero certifications: **{_zero_cy:.0%}**. "
            "The county regressions weight every county-year equally."
        ),
        mo.hstack([_curve, _users_chart]),
        _county.head(25).select("rank", "countyname", "region_name", "certified", pl.col("cumulative_share").round(3)),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Does the AEWR bind?

    Left: share of certified hours (cases with an hourly wage and an AEWR)
    offered exactly at the AEWR, by region and year. Right: offered hourly wage
    divided by the worksite state's AEWR for hourly cases in the chosen start
    year (the case file covers start years through 2019). Mass at 1.0 means the floor binds; mass well above it means AEWR
    changes may not move labor costs much.
    """)
    return


@app.cell
def _(INTERMEDIATE, PROCESSED, YEAR_MAX, YEAR_MIN, panel, pl):
    _states = panel.select("state_abbrev").unique()
    _aewr = pl.read_parquet(INTERMEDIATE / "aewr.parquet").select("state_fips", "year", "aewr")
    _state_codes = pl.read_parquet(
        PROCESSED / "county_year_panel.parquet", columns=["state_fips", "state_abbrev"]
    ).unique()
    cases = (
        pl.read_parquet(PROCESSED / "wage_reforms_cases.parquet")
        .filter(pl.col("hourly"), pl.col("offered_wage") > 0, pl.col("start_date").is_not_null())
        .with_columns(pl.col("start_date").dt.year().cast(pl.Int32).alias("year"))
        .filter(pl.col("year").is_between(YEAR_MIN, YEAR_MAX))
        .join(_state_codes, left_on="worksite_state", right_on="state_abbrev", how="inner")
        .join(_aewr, on=["state_fips", "year"], how="inner")
        .filter(pl.col("aewr") > 0)
        .with_columns((pl.col("offered_wage") / pl.col("aewr")).alias("wage_to_aewr"))
    )
    return (cases,)


@app.cell
def _(cases, mo):
    # The wage-reform case file currently ends with FY2019 cases.
    bite_year = mo.ui.slider(
        int(cases["year"].min()), int(cases["year"].max()),
        value=int(cases["year"].max()), step=1, label="Start year", show_value=True,
    )
    bite_year
    return (bite_year,)


@app.cell
def _(alt, bite_year, cases, mo, panel, pl):
    _bite = (
        panel.group_by("region_name", "year")
        .agg(
            (
                pl.col("cert_hours_at_aewr_start_year").sum()
                / pl.col("cert_hours_with_hourly_wage_and_aewr_start_year").sum()
            ).alias("share_at_aewr")
        )
    )
    _heat = (
        alt.Chart(_bite.to_pandas())
        .mark_rect()
        .encode(
            x=alt.X("year:O", title=None),
            y=alt.Y("region_name:N", title=None),
            color=alt.Color("share_at_aewr:Q", title="Share at AEWR", scale=alt.Scale(domain=[0, 1])),
            tooltip=["region_name", "year", alt.Tooltip("share_at_aewr:Q", format=".0%")],
        )
        .properties(width=480, height=320, title="Share of certified hours paid at the AEWR")
    )
    _year_cases = cases.filter(pl.col("year") == bite_year.value).with_columns(
        pl.col("wage_to_aewr").clip(0.8, 2.0)
    )
    _hist = (
        alt.Chart(_year_cases.to_pandas())
        .mark_bar()
        .encode(
            x=alt.X("wage_to_aewr:Q", bin=alt.Bin(step=0.02), title="Offered wage / AEWR (clipped 0.8–2.0)"),
            y=alt.Y("count():Q", title="Cases"),
        )
        .properties(width=480, height=320, title=f"Hourly cases starting in {bite_year.value}")
    )
    _near = cases.group_by("year").agg(
        ((pl.col("wage_to_aewr") - 1).abs() <= 0.01).mean().alias("within_1pct")
    ).sort("year")
    mo.vstack([
        mo.hstack([_heat, _hist]),
        mo.md(
            "Share of hourly cases offering within 1% of the AEWR, by start year: "
            + ", ".join(f"{r['year']}: {r['within_1pct']:.0%}" for r in _near.iter_rows(named=True))
        ),
    ])
    return


@app.cell
def _(mo, region_choice):
    mo.md(f"""
    ## 6. Case study: {region_choice.value}

    Change the region in the dropdown under section 1. Top: the region's residual
    AEWR (section 1). Bottom: certified workers per 2011 farm worker in the
    region versus all other regions, indexed to 2011 = 100.
    """)
    return


@app.cell
def _(alt, mo, panel, pl, region_choice, region_year):
    _resid = (
        alt.Chart(region_year.filter(pl.col("region_name") == region_choice.value).to_pandas())
        .mark_bar()
        .encode(
            x=alt.X("year:O", title=None),
            y=alt.Y("aewr_ppi_resid:Q", title="Residual real AEWR ($)"),
            color=alt.condition(alt.datum.aewr_ppi_resid > 0, alt.value("#c0392b"), alt.value("#2e86c1")),
        )
        .properties(width=700, height=200)
    )
    _paths = (
        panel.filter(pl.col("emp_farm_2011") > 0)
        .with_columns(
            pl.when(pl.col("region_name") == region_choice.value)
            .then(pl.lit(region_choice.value))
            .otherwise(pl.lit("All other regions"))
            .alias("group")
        )
        .group_by("group", "year")
        .agg((pl.col("nbr_workers_certified_start_year").sum() / pl.col("emp_farm_2011").sum()).alias("rate"))
        .sort("group", "year")
    )
    _paths = _paths.with_columns(
        (100 * pl.col("rate") / pl.col("rate").filter(pl.col("year") == pl.col("year").min()).first().over("group")).alias("index")
    )
    _lines = (
        alt.Chart(_paths.to_pandas())
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O", title=None),
            y=alt.Y("index:Q", title="Certified per 2011 farm worker (2011 = 100)"),
            color=alt.Color("group:N", title=None),
            tooltip=["group", "year", alt.Tooltip("rate:Q", format=".4f"), alt.Tooltip("index:Q", format=".0f")],
        )
        .properties(width=700, height=280)
    )
    mo.vstack([_resid, _lines])
    return


if __name__ == "__main__":
    app.run()
