---
title: A09 05 Construct Chained Fisher County Index
marimo-version: 0.20.2
width: full
---

```python {.marimo}
import marimo as mo
from pathlib import Path
import pyprojroot
import polars as pl
import requests
import os
import io
import math
```

```python {.marimo}
root_path = pyprojroot.find_root(criterion='pyproject.toml')
binary_path = root_path / 'binaries'

# Core artifacts from A03/A09 pipeline
cdl_acres = (
    pl.scan_parquet(binary_path / 'croplandcros_county_crop_acres.parquet')
    .with_columns(pl.col("fips").str.slice(0, 2).alias("state_fips"))
)
qs_survey = pl.scan_parquet(binary_path / 'qs_survey_crops.parquet')
crosswalk = pl.scan_parquet(binary_path / 'nass_cdl_crosswalk.parquet')

# Panel parameters
BASE_YEAR = 2011
PANEL_START = 2008
PANEL_END = 2024
```

## Step 1: Prepare NASS prices (state + national fallback)

Price data for commodities is overwhelmingly reported at the state level in
QuickStats. National-level prices serve as a fallback for specialty crops
that lack state coverage.

```python {.marimo}
prices_state = (qs_survey
    .filter(
        (pl.col("observation_type") == "price")
        & (pl.col("agg_level_desc") == "STATE")
    )
    .join(crosswalk, on="nass_id", how="inner")
    .group_by(["year", "state_ansi", "cdl_code"])
    .agg(pl.mean("numeric_value").alias("price"))
    .rename({"state_ansi": "state_fips"})
)

# National fallback prices for crops missing at state level
prices_national = (qs_survey
    .filter(
        (pl.col("observation_type") == "price")
        & (pl.col("agg_level_desc") == "NATIONAL")
    )
    .join(crosswalk, on="nass_id", how="inner")
    .group_by(["year", "cdl_code"])
    .agg(pl.mean("numeric_value").alias("price_national"))
)
```

## Step 2: Three-tier yield resolution (county → AG DISTRICT → state)

Yield is hyper-local. In states like Kansas or Texas, irrigated western counties
can yield 2-3x dryland eastern counties. Instead of jumping from spotty county
data straight to a statewide average, we use NASS Agricultural Statistics
Districts (ASD) as a middle tier. NASS publishes yield at `agg_level_desc ==
'AG DISTRICT'` for most major crops; the `asd_code` column in QuickStats
identifies the district within each state.

The join cascade is: county yield → district yield → state yield, with
`pl.coalesce` across the three tiers.

```python {.marimo}
# Tier 1: County-level yields (sparse but most accurate)
yields_county = (qs_survey
    .filter(
        (pl.col("observation_type") == "yield")
        & (pl.col("agg_level_desc") == "COUNTY")
    )
    .join(crosswalk, on="nass_id", how="inner")
    .with_columns(
        pl.concat_str([pl.col("state_ansi"), pl.col("county_ansi")]).alias("fips")
    )
    .group_by(["year", "fips", "cdl_code"])
    .agg(pl.mean("numeric_value").alias("yield_county"))
)

# Tier 2: AG DISTRICT-level yields (intermediate geography, ~9 per state)
# asd_code is the 2-digit district identifier within a state.
# We construct a 4-character key: state_fips (2) + asd_code (2).
yields_district = (qs_survey
    .filter(
        (pl.col("observation_type") == "yield")
        & (pl.col("agg_level_desc") == "AG DISTRICT")
    )
    .join(crosswalk, on="nass_id", how="inner")
    .with_columns(
        pl.concat_str([pl.col("state_ansi"), pl.col("asd_code")]).alias("state_district")
    )
    .group_by(["year", "state_district", "cdl_code"])
    .agg(pl.mean("numeric_value").alias("yield_district"))
)

# Tier 3: State-level yields (broadest fallback)
yields_state = (qs_survey
    .filter(
        (pl.col("observation_type") == "yield")
        & (pl.col("agg_level_desc") == "STATE")
    )
    .join(crosswalk, on="nass_id", how="inner")
    .group_by(["year", "state_ansi", "cdl_code"])
    .agg(pl.mean("numeric_value").alias("yield_state"))
    .rename({"state_ansi": "state_fips"})
)
```

## Step 3: Build the county panel

We merge CDL acreage with prices and the three-tier yield cascade, enforce
strict dimensional integrity (no mixing bushels with acres), and apply a
5-year rolling state yield as a final imputation layer before dropping
observations that still lack price or yield.

```python {.marimo}
# Extract county → AG DISTRICT mapping from NASS county-level observations
county_to_district = (qs_survey
    .filter(pl.col("agg_level_desc") == "COUNTY")
    .with_columns(
        pl.concat_str([pl.col("state_ansi"), pl.col("county_ansi")]).alias("fips"),
        pl.concat_str([pl.col("state_ansi"), pl.col("asd_code")]).alias("state_district")
    )
    .select(["fips", "state_district"])
    .unique()
)

# 5-year rolling state average yield as final imputation layer
rolling_yield_state = (yields_state
    .sort("year")
    .with_columns(
        pl.col("yield_state")
            .rolling_mean(window_size=5, min_periods=2)
            .over(["state_fips", "cdl_code"])
            .alias("yield_rolling_5yr")
    )
)

# Build panel: CDL acreage → district key → prices → yields → imputation → quantity
panel = (
    cdl_acres
    # Attach AG DISTRICT key for the three-tier yield join
    .join(county_to_district, on="fips", how="left")
    # Join prices: state-level first, national fallback
    .join(prices_state, on=["year", "state_fips", "cdl_code"], how="left")
    .join(prices_national, on=["year", "cdl_code"], how="left")
    .with_columns(pl.coalesce(["price", "price_national"]).alias("price"))
    .drop("price_national")
    # Join yields: county → district → state cascade
    .join(yields_county, on=["year", "fips", "cdl_code"], how="left")
    .join(yields_district, on=["year", "state_district", "cdl_code"], how="left")
    .join(yields_state, on=["year", "state_fips", "cdl_code"], how="left")
    .with_columns(
        pl.coalesce(["yield_county", "yield_district", "yield_state"]).alias("yield_val")
    )
    .drop(["yield_county", "yield_district", "yield_state"])
    # 5-year rolling state yield as final imputation before dropping
    .join(
        rolling_yield_state.select(["year", "state_fips", "cdl_code", "yield_rolling_5yr"]),
        on=["year", "state_fips", "cdl_code"],
        how="left"
    )
    .with_columns(
        pl.coalesce(["yield_val", "yield_rolling_5yr"]).alias("final_yield")
    )
    .drop(["yield_val", "yield_rolling_5yr"])
    # Enforce strict dimensionality: drop if price or yield is missing.
    # quantity = total physical production (e.g., bushels) for county-crop-year.
    .drop_nulls(subset=["price", "final_yield"])
    .with_columns(
        (pl.col("acres") * pl.col("final_yield")).alias("quantity")
    )
    .select(["fips", "year", "state_fips", "cdl_code", "price", "quantity"])
)
```

## Step 4: Chained Fisher Index construction

A fixed-base Fisher (2011=100) suffers from two compounding problems over a
17-year panel with significant land use change:

1. **New goods**: Crops introduced after 2011 (e.g., hemp post-2018 Farm Bill)
   have no base-year quantity, so they only enter through Paasche.
2. **Disappearing goods**: Crops that exit (e.g., declining tobacco) leave ghost
   weight in Laspeyres (q0 > 0 but q_t = 0), widening the Laspeyres-Paasche
   spread artificially.

The methodologically correct solution is a **chained Fisher index**: compute
year-over-year bilateral Fisher links, then multiply them together. This lets
both the basket and reference prices update annually, handling entry and exit
symmetrically. The chain is anchored to BASE_YEAR = 100.

```python {.marimo}
# Materialize the panel — it's ~1-2M rows, well within memory.
panel_df = panel.collect()
```

```python {.marimo}
def compute_bilateral_fisher(df: pl.DataFrame, year_a: int, year_b: int) -> pl.DataFrame:
    """
    Compute county-level bilateral Fisher price index between year_a and year_b.
    
    Returns a DataFrame with columns [fips, laspeyres, paasche, fisher] where
    each value is the price ratio (year_b relative to year_a, as a multiplier).
    
    Uses an INNER join: only crops present in BOTH years contribute. This is the
    standard "matched set" approach used by BLS, NASS, and ABS for chained
    indices. New crops enter the chain in the first year-pair where they appear
    in consecutive years; exiting crops drop at their last consecutive pair.
    """
    df_a = (df
        .filter(pl.col("year") == year_a)
        .select([
            "fips", "cdl_code",
            pl.col("price").alias("p_a"),
            pl.col("quantity").alias("q_a")
        ])
    )
    df_b = (df
        .filter(pl.col("year") == year_b)
        .select([
            "fips", "cdl_code",
            pl.col("price").alias("p_b"),
            pl.col("quantity").alias("q_b")
        ])
    )

    bilateral = df_a.join(df_b, on=["fips", "cdl_code"], how="inner")

    bilateral = bilateral.with_columns([
        (pl.col("p_b") * pl.col("q_a")).alias("pb_qa"),
        (pl.col("p_a") * pl.col("q_a")).alias("pa_qa"),
        (pl.col("p_b") * pl.col("q_b")).alias("pb_qb"),
        (pl.col("p_a") * pl.col("q_b")).alias("pa_qb"),
    ])

    county_link = (bilateral
        .group_by("fips")
        .agg([
            pl.sum("pb_qa").alias("sum_pb_qa"),
            pl.sum("pa_qa").alias("sum_pa_qa"),
            pl.sum("pb_qb").alias("sum_pb_qb"),
            pl.sum("pa_qb").alias("sum_pa_qb"),
        ])
    )

    def safe_ratio(num: str, den: str) -> pl.Expr:
        return (
            pl.when((pl.col(den) == 0) | pl.col(den).is_null())
            .then(None)
            .otherwise(pl.col(num) / pl.col(den))
        )

    county_link = county_link.with_columns([
        safe_ratio("sum_pb_qa", "sum_pa_qa").alias("laspeyres"),
        safe_ratio("sum_pb_qb", "sum_pa_qb").alias("paasche"),
    ]).with_columns(
        (pl.col("laspeyres") * pl.col("paasche")).sqrt().alias("fisher")
    ).select(["fips", "laspeyres", "paasche", "fisher"])

    return county_link
```

```python {.marimo}
# Compute year-over-year bilateral links for every adjacent pair in the panel.
years = sorted(panel_df.get_column("year").unique().to_list())

bilateral_links = []
for i in range(len(years) - 1):
    _ya, _yb = years[i], years[i + 1]
    _link = compute_bilateral_fisher(panel_df, _ya, _yb).with_columns(
        pl.lit(_yb).alias("year")
    )
    bilateral_links.append(_link)

links_df = pl.concat(bilateral_links)
```

```python {.marimo}
# Chain the links via log-space cumulative sums, anchored at BASE_YEAR = 100.
#
# Each link in links_df at year=t represents the bilateral Fisher ratio for the
# transition from year (t-1) to year t. In log space, chaining is addition:
#
# Forward:  log_chain(t) = log(100) + sum_{s=BASE+1}^{t} log_link(s)
# Backward: log_chain(t) = log(100) - sum_{s=t+1}^{BASE} log_link(s)

_links_log = links_df.with_columns(
    pl.col("fisher").log().alias("log_fisher")
).select(["fips", "year", "log_fisher"])

_log_100 = math.log(100.0)

# Forward chain: cumsum of log-links from BASE_YEAR+1 onward
_forward = (_links_log
    .filter(pl.col("year") > BASE_YEAR)
    .sort(["fips", "year"])
    .with_columns(
        (pl.col("log_fisher").cum_sum().over("fips") + _log_100).alias("log_chain")
    )
    .select(["fips", "year", "log_chain"])
)

# Base year anchor
_base_anchor = (panel_df
    .select("fips").unique()
    .with_columns([
        pl.lit(BASE_YEAR).alias("year"),
        pl.lit(_log_100).alias("log_chain"),
    ])
)

# Backward chain: cumsum log-links in descending year order, negated.
# The output year for each accumulated row is (input_year - 1).
_backward = (_links_log
    .filter(pl.col("year") <= BASE_YEAR)
    .sort(["fips", "year"], descending=[False, True])
    .with_columns(
        (-(pl.col("log_fisher").cum_sum().over("fips")) + _log_100).alias("log_chain"),
        (pl.col("year") - 1).alias("chain_year"),
    )
    .select(["fips", pl.col("chain_year").alias("year"), "log_chain"])
    .filter(pl.col("year") >= PANEL_START)
)

# Combine all segments and exponentiate
chained_fisher = (
    pl.concat([_base_anchor, _forward, _backward])
    .with_columns(pl.col("log_chain").exp().alias("fisher_index"))
    .select(["fips", "year", "fisher_index"])
    .sort(["fips", "year"])
)
```

## Step 5: Fixed-base comparison indices

For diagnostic purposes, we also compute the traditional fixed-base Laspeyres,
Paasche, and Fisher indices against BASE_YEAR. This lets us measure the
Laspeyres-Paasche spread — a large spread indicates significant crop entry/exit
and validates the chained approach.

```python {.marimo}
_base_df = (panel_df
    .filter(pl.col("year") == BASE_YEAR)
    .select([
        "fips", "cdl_code",
        pl.col("price").alias("p0"),
        pl.col("quantity").alias("q0")
    ])
)

# National base-year price for imputing p0 on new crops (post-BASE_YEAR entries)
_national_base_prices = (panel_df
    .filter(pl.col("year") == BASE_YEAR)
    .group_by("cdl_code")
    .agg(pl.mean("price").alias("p0_national"))
)

# Left join preserves all post-BASE_YEAR crop introductions
_fixed_base_df = (panel_df
    .join(_base_df, on=["fips", "cdl_code"], how="left")
    .join(_national_base_prices, on="cdl_code", how="left")
    .with_columns([
        pl.col("q0").fill_null(0),
        pl.coalesce(["p0", "p0_national"]).alias("p0")
    ])
    .drop("p0_national")
    .drop_nulls(subset=["p0"])
    .with_columns([
        (pl.col("price") * pl.col("q0")).alias("p1_q0"),
        (pl.col("p0")    * pl.col("q0")).alias("p0_q0"),
        (pl.col("price") * pl.col("quantity")).alias("p1_q1"),
        (pl.col("p0")    * pl.col("quantity")).alias("p0_q1"),
    ])
)

def safe_index(num: str, den: str) -> pl.Expr:
    return (
        pl.when((pl.col(den) == 0) | pl.col(den).is_null())
        .then(None)
        .otherwise((pl.col(num) / pl.col(den)) * 100)
    )

fixed_base_index = (_fixed_base_df
    .group_by(["fips", "year"])
    .agg([
        pl.sum("p1_q0").alias("sum_p1_q0"),
        pl.sum("p0_q0").alias("sum_p0_q0"),
        pl.sum("p1_q1").alias("sum_p1_q1"),
        pl.sum("p0_q1").alias("sum_p0_q1"),
    ])
    .with_columns([
        safe_index("sum_p1_q0", "sum_p0_q0").alias("laspeyres_fixed"),
        safe_index("sum_p1_q1", "sum_p0_q1").alias("paasche_fixed"),
    ])
    .with_columns(
        (pl.col("laspeyres_fixed") * pl.col("paasche_fixed")).sqrt().alias("fisher_fixed")
    )
    .select(["fips", "year", "laspeyres_fixed", "paasche_fixed", "fisher_fixed"])
)
```

## Step 6: PPITW terms-of-trade deflation

The GDP Implicit Price Deflator measures broad macroeconomic inflation that does
not track agricultural input costs. In 2022, GDPDEF rose moderately while
fertilizer prices tripled — deflating by GDPDEF would make farmers appear
wealthier while their margins were collapsing.

Instead, we pull the USDA NASS Prices Paid Index (PPITW) from QuickStats. The
PPITW measures what producers actually pay for inputs, interest, taxes, and
wages, using a 2011=100 base that aligns with our index.

Our Fisher index is conceptually a Prices Received index. Dividing by PPITW
gives a terms-of-trade measure: the ratio of output prices to input costs. A
value above 100 means farmers are receiving relatively more for their output
than they are paying for inputs, compared to 2011.

```python {.marimo}
def _fetch_ppitw_from_parquet(bp: Path) -> pl.DataFrame | None:
    """Try to extract PPITW from the economics parquet created in A03."""
    try:
        qs_econ = pl.scan_parquet(bp / 'qs_survey_economics.parquet')
        result = (qs_econ
            .filter(
                pl.col("short_desc").str.to_uppercase().str.contains("PRICE PAID")
                & pl.col("short_desc").str.to_uppercase().str.contains("INDEX")
                & pl.col("short_desc").str.to_uppercase().str.contains("COMMODITIES AND SERVICES")
                & (pl.col("agg_level_desc") == "NATIONAL")
            )
            .select(["year", "numeric_value"])
            .drop_nulls()
            .group_by("year")
            .agg(pl.mean("numeric_value").alias("ppitw_raw"))
            .collect()
        )
        if result.height > 0:
            return result
    except Exception:
        pass
    return None


def _fetch_ppitw_from_api() -> pl.DataFrame | None:
    """Fetch PPITW directly from the NASS QuickStats API."""
    _api_key = os.getenv("NASS_QUICKSTATS_API_KEY")
    if not _api_key:
        return None

    # The composite PPITW (2011-base) in QuickStats:
    #   sector_desc=ECONOMICS, group_desc=PRICES PAID
    #   commodity_desc=COMMODITY TOTALS, statisticcat_desc=INDEX
    #   unit_desc contains "INDEX 2011"
    #   short_desc contains "COMMODITIES AND SERVICES"
    _params = {
        "key": _api_key,
        "source_desc": "SURVEY",
        "sector_desc": "ECONOMICS",
        "group_desc": "PRICES PAID",
        "commodity_desc": "COMMODITY TOTALS",
        "statisticcat_desc": "INDEX",
        "agg_level_desc": "NATIONAL",
        "short_desc__LIKE": "COMMODITIES AND SERVICES",
        "unit_desc__LIKE": "INDEX 2011",
        "format": "JSON",
    }
    try:
        _resp = requests.get(
            "https://quickstats.nass.usda.gov/api/api_GET/",
            params=_params,
            timeout=30,
        )
        _resp.raise_for_status()
        _payload = _resp.json()
        if "data" not in _payload or len(_payload["data"]) == 0:
            return None

        _rows = []
        for _rec in _payload["data"]:
            _val_str = _rec.get("Value", "").replace(",", "")
            try:
                _rows.append({
                    "year": int(_rec["year"]),
                    "ppitw_raw": float(_val_str),
                })
            except (ValueError, KeyError):
                continue

        if not _rows:
            return None

        return (
            pl.DataFrame(_rows)
            .group_by("year")
            .agg(pl.mean("ppitw_raw").alias("ppitw_raw"))
        )
    except Exception:
        return None


# Resolution order: local parquet → QuickStats API
ppitw_raw = _fetch_ppitw_from_parquet(binary_path)
_ppitw_source = "economics parquet"

if ppitw_raw is None:
    ppitw_raw = _fetch_ppitw_from_api()
    _ppitw_source = "QuickStats API"

if ppitw_raw is None:
    raise RuntimeError(
        "Could not load PPITW data. Ensure either:\n"
        "  (a) qs_survey_economics.parquet exists in binaries/ (from A03 01), or\n"
        "  (b) NASS_QUICKSTATS_API_KEY is set in your environment.\n"
        "Get a free key at https://quickstats.nass.usda.gov/api/"
    )

# Rebase to 2011=100
_base_ppitw = ppitw_raw.filter(pl.col("year") == BASE_YEAR).select("ppitw_raw").item()
ppitw = ppitw_raw.with_columns(
    ((pl.col("ppitw_raw") / _base_ppitw) * 100).alias("ppitw_index")
).select(["year", "ppitw_index"])

mo.md(f"PPITW loaded from **{_ppitw_source}** ({ppitw.height} year-observations)")
```

```python {.marimo}
# Compute terms-of-trade index: Fisher / PPITW × 100
# A value > 100 means output prices outpace input costs relative to 2011.
county_index = (chained_fisher
    .join(ppitw, on="year", how="left")
    .with_columns(
        ((pl.col("fisher_index") / pl.col("ppitw_index")) * 100)
        .alias("terms_of_trade_index")
    )
    # Merge the fixed-base diagnostics
    .join(fixed_base_index, on=["fips", "year"], how="left")
    # Laspeyres-Paasche spread as a diagnostic for entry/exit magnitude
    .with_columns(
        (pl.col("laspeyres_fixed") - pl.col("paasche_fixed")).alias("lp_spread_fixed")
    )
)
```

## Step 7: Export

```python {.marimo}
final_output = (county_index
    .select([
        "fips",
        "year",
        "fisher_index",          # Chained Fisher (2011=100)
        "terms_of_trade_index",  # Chained Fisher / PPITW × 100
        "laspeyres_fixed",       # Fixed-base diagnostic
        "paasche_fixed",         # Fixed-base diagnostic
        "fisher_fixed",          # Fixed-base Fisher diagnostic
        "lp_spread_fixed",       # Laspeyres minus Paasche (entry/exit diagnostic)
    ])
    .sort(["fips", "year"])
)

output_file = binary_path / 'county_chained_fisher_index.parquet'
final_output.write_parquet(output_file)
mo.md(f"Exported **{final_output.height:,}** rows to `{output_file}`")
```

```python {.marimo}
mo.md(f"""
### Diagnostics

**Panel**: {final_output.select('fips').n_unique()} counties × {final_output.select('year').n_unique()} years

**Chained Fisher range**: {final_output.select(pl.col('fisher_index').min()).item():.1f} – {final_output.select(pl.col('fisher_index').max()).item():.1f}

**Terms-of-trade range**: {final_output.select(pl.col('terms_of_trade_index').drop_nulls().min()).item():.1f} – {final_output.select(pl.col('terms_of_trade_index').drop_nulls().max()).item():.1f}

**Mean LP spread (fixed-base)**: {final_output.select(pl.col('lp_spread_fixed').drop_nulls().mean()).item():.2f} index points

A large positive LP spread indicates the Laspeyres is systematically overstating
(ghost weight from exiting crops) relative to Paasche — this is the primary
motivation for the chained approach.
""")
```

```python {.marimo}
final_output.head(20)
```
