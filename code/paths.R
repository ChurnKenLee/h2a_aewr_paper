# Shared project paths. The `.here` marker at the repository root is the
# single source of truth for every supported R entry point.

project_path <- function(...) here::here(...)

path_root <- project_path
path_do <- function(...) here::here("Do", ...)
path_code <- function(...) here::here("code", ...)
path_json <- function(...) here::here("code", "json", ...)

path_data <- function(...) here::here("data", ...)
path_raw <- function(...) here::here("data", "raw", ...)
path_int <- function(...) here::here("data", "intermediate", ...)
path_processed <- function(...) here::here("data", "processed", ...)
path_cache <- function(...) here::here("data", "intermediate", "cache", ...)

path_outputs <- function(...) here::here("outputs", ...)
path_figures <- function(...) here::here("outputs", "figures", ...)
path_tables <- function(...) here::here("outputs", "tables", ...)
path_logs <- function(...) here::here("outputs", "logs", ...)

# Canonical H-2A prediction used by every shared and design-specific panel.
# Changing the training window is a source-controlled specification change and
# requires rebuilding the prediction artifact and all downstream panels.
# docs-ground:start prediction-selection
H2A_PREDICTION_CUTOFF_YEAR <- 2011L
H2A_PREDICTION_MODEL_SPEC <- "climate_norm_static_v1"
# docs-ground:end prediction-selection

# Calendar of the balanced shared county-year backbone. Sources that end
# earlier (BEA county employment and farm income stop at 2022) stay missing in
# later years; each design declares its own estimation window.
SHARED_PANEL_START_YEAR <- 2008L
SHARED_PANEL_END_YEAR <- 2025L

env_file <- here::here(".env")
if (file.exists(env_file)) {
  dotenv::load_dot_env(file = env_file)
}
rm(env_file)
