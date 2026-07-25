#!/usr/bin/env Rscript

here::i_am("code/paths.R")

source(here::here("code", "paths.R"))

root <- normalizePath(here::here(), mustWork = TRUE)
stopifnot(
  identical(normalizePath(path_root(), mustWork = TRUE), root),
  identical(path_code("paths.R"), here::here("code", "paths.R")),
  identical(path_int("artifact.parquet"),
    here::here("data", "intermediate", "artifact.parquet")
  )
)

old_working_directory <- getwd()
on.exit(setwd(old_working_directory), add = TRUE)
setwd(path_code("c01_clean"))
stopifnot(
  identical(normalizePath(path_root(), mustWork = TRUE), root),
  identical(path_processed("panel.parquet"),
    file.path(root, "data", "processed", "panel.parquet")
  )
)

cat("path tests passed\n")
