# Purpose: Request, cache, and convert the ACS extracts used by downstream scripts.
# Inputs: IPUMS extract specifications in code/json and IPUMS_API_KEY.
# Outputs: ACS one-year wage and five-year imputation Parquet files.

if (!exists("path_code", mode = "function")) {
  source(
    if (file.exists(file.path("code", "bootstrap_paths.R"))) {
      file.path("code", "bootstrap_paths.R")
    } else {
      file.path("..", "bootstrap_paths.R")
    }
  )
}
library(arrow)
library(tidyverse)
library(tidylog, warn.conflicts = FALSE)
library(ipumsr)
library(haven)

ipums_api_key <- Sys.getenv("IPUMS_API_KEY")
if (!nzchar(ipums_api_key)) {
  stop("IPUMS_API_KEY must be set in .env or the environment.")
}
set_ipums_api_key(ipums_api_key, save = FALSE)

dir.create(path_raw("acs"), recursive = TRUE, showWarnings = FALSE)

build_acs_parquet <- function(extract_id, spec_file, parquet_file) {
  ddi_path <- path_raw("acs", paste0(extract_id, ".xml"))
  data_path <- path_raw("acs", paste0(extract_id, ".dat.gz"))

  if (!file.exists(ddi_path) || !file.exists(data_path)) {
    message("Submitting missing ACS extract: ", extract_id)

    extract <- define_extract_from_json(path_json(spec_file)) %>%
      submit_extract() %>%
      wait_for_extract()

    temp_download_dir <- tempfile("ipums_acs_")
    dir.create(temp_download_dir)
    on.exit(unlink(temp_download_dir, recursive = TRUE), add = TRUE)

    downloaded_ddi <- download_extract(
      extract,
      download_dir = temp_download_dir
    )
    downloaded_data <- sub("\\.xml$", ".dat.gz", downloaded_ddi)

    file.copy(downloaded_ddi, ddi_path, overwrite = TRUE)
    file.copy(downloaded_data, data_path, overwrite = TRUE)
  }

  acs_data <- read_ipums_micro(
    ddi_path,
    data_file = data_path
  )

  acs_data <- acs_data %>%
    zap_labels() %>%
    zap_label()

  acs_data %>%
    write_parquet(
      path_int(parquet_file)
    )
}

build_acs_parquet(
  extract_id = "acs_1year_wages",
  spec_file = "acs_1year_wages_extract_spec.json",
  parquet_file = "acs_1year_for_wages.parquet"
)

build_acs_parquet(
  extract_id = "acs_5year_immigrant_status_imputation",
  spec_file = "acs_5year_immigrant_status_imputation_extract_spec.json",
  parquet_file = "acs_5year_for_immigrant_status_imputation.parquet"
)
