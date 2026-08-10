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

The panel's static predicted H-2A propensity comes from the one global cutoff
selected by `H2A_PREDICTION_CUTOFF_YEAR` in `code/paths.R`. The corresponding
one-row-per-county score is joined by county and repeated unchanged over panel
years. Predicted counts use fixed 2011 BEA farm employment, and predicted shares
divide by that same stored exposure.

The shared H-2A employer outcome uses the raw balanced-linkage employer count.
Conservative and high-recall counts remain in the panel for
linkage-sensitivity analyses.

Manual command:

```sh
Rscript code/c02_build/01_build_county_panel.R
```
