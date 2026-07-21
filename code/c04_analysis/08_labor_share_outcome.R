# Purpose: Estimate the DD response of farm labor's production-expense share.
# Input: data/processed/county_df_analysis_year.parquet.
# Output: outputs/tables/table_laborshare_dd.tex.
# Run after: code/c02_build/04_finalize_county_panel.R.

source(if (file.exists(file.path("code", "bootstrap_paths.R"))) {
  file.path("code", "bootstrap_paths.R")
} else {
  file.path("..", "bootstrap_paths.R")
})
source(path_code("c00_shared", "analysis_helpers.R"))
library(arrow)
library(tidyverse)
library(fixest)

county_df <- read_parquet(path_processed("county_df_analysis_year.parquet"))
samp_base <- analysis_sample(county_df)

#### Exhibit 20: Fisher Price Index Investigation — Labor Share DD --------------
# Fisher price index (fisher_index_ppi) construction:
#   Script: code/b01_derived/02_price_index_nass_synthetic_cdl.py
#   Inputs: CDL county-crop acreage (croplandcros_county_crop_acres.parquet)
#   with NASS synthetic prices/yields per crop (state, then national fallback)
#   Method: chained bilateral Fisher — geometric mean of Laspeyres and Paasche,
#           matched set = crops present in both consecutive years (inner join).
#           Base year 2011 = 100. Forward chain 2012-2022; backward 2010-2008.
#   Deflation: fisher_index_ppi = fisher_index / ppi_2012 [Build Dataset L1700]
#
# Quantity/composition proxy available in county_df:
#   share_farm_laborexp_prodexp = farm_laborexpense / farm_prodexp (BEA CAINC45)
#   This is a genuine share [0,1]. If AEWR increases shift production toward less
#   labor-intensive crops (substitution away from H-2A-intensive crops), this
#   share should fall. CDL crop-level acreage shares are NOT in county_df
#   (only any_cropland_2007); could be added from cdl_cropshares.parquet.

laborshare_dd_1 <- feols(
  share_farm_laborexp_prodexp ~
    aewr_cz_p25_l1 * postdummy | county_fe + year_fe,
  data = samp_base,
  vcov = ~cz_aewr_region_fe
)

laborshare_dd_2 <- feols(
  share_farm_laborexp_prodexp ~
    aewr_cz_p25_l1 *
    postdummy +
    ln_pop_census +
    emp_pop_ratio |
    county_fe + year_fe,
  data = samp_base,
  vcov = ~cz_aewr_region_fe
)

laborshare_dd_3 <- feols(
  share_farm_laborexp_prodexp ~
    aewr_cz_p25_l1 * postdummy | county_fe + year_fe,
  data = samp_no_border,
  vcov = ~cz_aewr_region_fe
)

laborshare_dd_4 <- feols(
  share_farm_laborexp_prodexp ~
    aewr_cz_p25_l1 *
    postdummy +
    ln_pop_census +
    emp_pop_ratio |
    county_fe + year_fe,
  data = samp_no_border,
  vcov = ~cz_aewr_region_fe
)

etable(
  laborshare_dd_1,
  laborshare_dd_2,
  laborshare_dd_3,
  laborshare_dd_4,
  tex = TRUE,
  title = "The Effect of the AEWR Wage Premium on Farm Labor Share of Production Expense",
  headers = c(
    "No Controls",
    "Controls",
    "No Border, No Controls",
    "No Border, Controls"
  ),
  dict = c(
    "share_farm_laborexp_prodexp" = "Labor share of farm production expense",
    "aewr_cz_p25_l1" = "Lagged AEWR vs 25th pct wage gap",
    "postdummy" = "Post",
    "ln_pop_census" = "Log population",
    "emp_pop_ratio" = "Employment-to-pop ratio"
  ),
  signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10),
  file = path_tables("table_laborshare_dd.tex"),
  replace = TRUE
)
