# Purpose: Compile the version-3 Mundlak-Chamberlain specification registry
# and its outcome-free region-rank ledger.
# Outputs:
#   data/intermediate/mundlak_chamberlain_specification_registry.rds
#   outputs/tables/mc_specification_registry.csv
#   outputs/tables/mc_default_execution_registry.csv
#   outputs/tables/mc_rank_budget_audit.csv

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("designs", "mundlak_chamberlain", "design.R"))
source(path_code("designs", "mundlak_chamberlain", "helpers.R"))
source(
  path_code(
    "designs",
    "mundlak_chamberlain",
    "specification_program.R"
  )
)
library(dplyr)
library(readr)

dir.create(path_int(), recursive = TRUE, showWarnings = FALSE)
dir.create(path_tables(), recursive = TRUE, showWarnings = FALSE)

registry <- mc_specification_registry()
budget_audit <- mc_sp_region_budget_audit(registry)

saveRDS(
  list(
    design_version = MC_SPEC_PROGRAM_VERSION,
    generated_at = format(Sys.time(), tz = "UTC", usetz = TRUE),
    registry = registry,
    calendars = mc_sp_calendar_registry()
  ),
  path_int("mundlak_chamberlain_specification_registry.rds")
)
write_csv(
  registry,
  path_tables("mc_specification_registry.csv")
)
write_csv(
  mc_sp_execution_registry(registry, "compact"),
  path_tables("mc_default_execution_registry.csv")
)
write_csv(
  budget_audit,
  path_tables("mc_rank_budget_audit.csv")
)

message(
  "Compiled ",
  nrow(registry),
  " MC specifications across ",
  dplyr::n_distinct(registry$calendar_id),
  " calendar records; default execution registry contains ",
  nrow(mc_sp_execution_registry(registry, "compact")),
  " specifications."
)
