# C02: Build the shared county-year panel

`01_build_county_panel.R` consumes
`data/intermediate/county_year_merged.parquet` and writes:

```text
data/processed/county_year_panel.parquet
```

It owns reusable outcomes, controls, consecutive-year control lags, 2011 farm
employment, AEWR-p25 gaps, cropland eligibility, and border-CZ status. It does
not create any design treatment, post period, fixed-effect factor, year dummy,
target cluster, or instrument.

Manual command:

```sh
Rscript code/c02_build/01_build_county_panel.R
```
