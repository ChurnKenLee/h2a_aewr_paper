# FLS preliminary-vs-revised revision instrument: does any variation exist?
# Confirms (a) the preliminary value for year t and the revised value for
# year t come from different annual releases, and (b) the published AEWR for
# year t+1 equals the preliminary field-and-livestock wage for year t.
source("explorations/identification/id_helpers.R")
out_dir <- id_output_dir()

fls <- read_parquet(path_int("fls_region.parquet")) %>%
  select(estimate_year, aewr_region_id, region_name, revised_year, preliminary_year,
    field_livestock_revised, field_livestock_preliminary, source_zip, source_csv, table_title)
stopifnot(!anyDuplicated(fls[c("estimate_year", "aewr_region_id")]))

prelim <- fls %>% transmute(aewr_region_id, year = preliminary_year,
  prelim = field_livestock_preliminary, prelim_zip = source_zip, prelim_csv = source_csv)
revised <- fls %>% filter(!is.na(revised_year)) %>% transmute(aewr_region_id, year = revised_year,
  revised = field_livestock_revised, revised_zip = source_zip, revised_csv = source_csv)
rev <- inner_join(prelim, revised, by = c("aewr_region_id", "year")) %>%
  mutate(revision = revised - prelim, same_release = prelim_zip == revised_zip)

# Published AEWR: regional reference rate by state; region from the shared panel.
state_region <- read_parquet(path_processed("county_year_panel.parquet"),
  col_select = c("state_fips", "aewr_region_id")) %>% distinct()
stopifnot(!anyDuplicated(state_region$state_fips))
aewr <- read_parquet(path_int("aewr_rate_schedule.parquet")) %>%
  filter(occupation_scope %in% c("nonrange_reference", "fls_common_soc_reference"),
    worker_category == "covered_workers") %>%
  inner_join(state_region, by = "state_fips") %>%
  group_by(aewr_region_id, reference_year) %>%
  summarise(aewr = mean(aewr), aewr_states_disagree = n_distinct(aewr) > 1, .groups = "drop")
match_aewr <- prelim %>% mutate(reference_year = year + 1L) %>%
  inner_join(aewr, by = c("aewr_region_id", "reference_year")) %>%
  mutate(diff = aewr - prelim)

summ <- function(d, label) d %>% summarise(sample = label, region_years = n(),
  share_equal = mean(abs(revision) < 0.005, na.rm = TRUE),
  share_same_release = mean(same_release), mean_abs_revision = mean(abs(revision), na.rm = TRUE),
  max_abs_revision = max(abs(revision), na.rm = TRUE))
within_sd <- rev %>% filter(year >= 2010, year <= 2022) %>%
  mutate(r = revision - ave(revision, year)) %>% summarise(within_year_sd_2010_2022 = sd(r, na.rm = TRUE))
out <- bind_rows(summ(rev, "all"), summ(rev %>% filter(year >= 2001, year <= 2023), "2001-2023"),
  summ(rev %>% filter(year >= 2010, year <= 2022), "2010-2022")) %>%
  mutate(within_year_sd_2010_2022 = within_sd$within_year_sd_2010_2022)
aewr_summ <- match_aewr %>% summarise(region_years = n(), share_aewr_equals_prelim = mean(abs(diff) < 0.005),
  share_aewr_equals_prelim_2010_2024 = mean(abs(diff[reference_year >= 2010]) < 0.005),
  max_abs_diff = max(abs(diff)), states_disagree_cells = sum(aewr_states_disagree))
nonzero <- rev %>% filter(abs(revision) >= 0.005) %>% arrange(year)

id_write(rev, out_dir, "fls_revisions.csv")
id_write(out, out_dir, "fls_revision_summary.csv")
id_write(match_aewr, out_dir, "fls_aewr_match.csv")
id_write(aewr_summ, out_dir, "fls_aewr_match_summary.csv")
print(out, width = Inf); print(aewr_summ, width = Inf)
cat("Nonzero revisions:\n"); print(as.data.frame(nonzero %>% select(aewr_region_id, year, prelim, revised, revision, prelim_zip, revised_zip)))
cat("AEWR != preliminary:\n"); print(as.data.frame(match_aewr %>% filter(abs(diff) >= 0.005) %>% select(aewr_region_id, reference_year, prelim, aewr, diff)))
