library(arrow); library(fixest)
folder_data <- paste0("C:/Users/", Sys.info()["user"], "/Dropbox/H-2A Paper/Data Int/")
county_df <- read_parquet(paste0(folder_data, "county_df_analysis_year.parquet"))
samp_base <- subset(county_df, any_cropland_2007 == 1 & county_simple_treatment_groups != "always takers")
dd_1 <- feols(
  h2a_cert_share_farm_workers_2011_start_year ~ aewr_cz_p25_l1 * postdummy | county_fe + year_fe,
  data = samp_base, vcov = ~cz_aewr_region_fe, demeaned = TRUE
)
cat("nrow samp_base:", nrow(samp_base), "\n")
cat("nobs dd_1:", nobs(dd_1), "\n")
cat("obsRemoved values:", dd_1$obs_selection$obsRemoved, "\n")
cat("str obs_selection:\n"); str(dd_1$obs_selection)
