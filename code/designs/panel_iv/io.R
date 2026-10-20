# Local I/O and schema guards shared by the R stages and their tests.
IV_CALIBRATION_VERSION <- "panel_iv_deterministic_v1"

iv_require_columns <- function(data, fields, label) {
  missing <- setdiff(fields, names(data))
  if (length(missing)) stop(label, " missing columns: ", paste(missing, collapse = ", "))
  invisible(data)
}

iv_read <- function(path) {
  if (!file.exists(path)) stop("Missing required input: ", path)
  if (endsWith(path, ".csv")) return(readr::read_csv(path, show_col_types = FALSE))
  as.data.frame(arrow::read_parquet(path))
}

iv_require_calibration_version <- function(data) {
  if (!"calibration_version" %in% names(data) || !nrow(data) ||
      anyNA(data$calibration_version) || any(data$calibration_version != IV_CALIBRATION_VERSION))
    stop("Incompatible calibration cache; run a complete stage 04 rebuild first")
  data
}

iv_atomic_parquet <- function(data, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temporary <- tempfile(paste0(".", basename(path)), tmpdir = dirname(path))
  on.exit(unlink(temporary), add = TRUE)
  arrow::write_parquet(data, temporary, compression = "zstd")
  if (!file.rename(temporary, path)) stop("Atomic Parquet rename failed: ", path)
  invisible(path)
}

iv_merge_selected <- function(path, replacement, years, regions, keys) {
  iv_require_unique(replacement, keys, "Replacement cells")
  if (file.exists(path)) {
    old <- iv_read(path)
    selected <- old$source_year %in% years
    if (!is.null(regions)) selected <- selected & old$aewr_region_id %in% regions
    retained <- old[!selected, , drop = FALSE]
    if (nrow(retained)) {
      if (!identical(names(old), names(replacement)))
        stop("Incompatible calibration cache; run a complete stage 04 rebuild first")
      iv_require_calibration_version(old)
      replacement <- dplyr::bind_rows(retained, replacement)
    }
  }
  iv_require_unique(replacement, keys, "Merged cells")
  iv_sort(replacement, keys)
}
