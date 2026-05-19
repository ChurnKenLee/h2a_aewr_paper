## H2A Master Files ##
## Phil Hoxie ##
## January 11, 2024 ##

gc()
rm(list = ls())

## Master Paths ## -------------------------------------------------------------
bootstrap_source_file <- function() {
  frames <- sys.frames()

  for (i in rev(seq_along(frames))) {
    ofile <- frames[[i]]$ofile
    if (!is.null(ofile) && nzchar(ofile)) return(ofile)
  }

  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg) > 0) return(sub("^--file=", "", file_arg[[1]]))

  if (
    requireNamespace("rstudioapi", quietly = TRUE) &&
      rstudioapi::isAvailable()
  ) {
    active_file <- tryCatch(
      rstudioapi::getActiveDocumentContext()$path,
      error = function(e) NA_character_
    )
    if (length(active_file) > 0 && !is.na(active_file) && nzchar(active_file)) {
      return(active_file)
    }
  }

  NA_character_
}

bootstrap_is_root <- function(path) {
  all(file.exists(file.path(path, c(".env", ".here", "pyproject.toml")))) &&
    file.exists(file.path(path, "code", "paths.R"))
}

bootstrap_find_root <- function(start) {
  if (length(start) == 0 || is.na(start) || !nzchar(start)) return(NULL)

  start <- path.expand(start)
  if (file.exists(start) && !dir.exists(start)) start <- dirname(start)

  current <- normalizePath(start, winslash = "/", mustWork = FALSE)

  repeat {
    if (bootstrap_is_root(current)) return(current)

    parent <- dirname(current)
    if (identical(parent, current)) return(NULL)
    current <- parent
  }
}

project_root <- bootstrap_find_root(dirname(bootstrap_source_file()))
if (is.null(project_root)) project_root <- bootstrap_find_root(getwd())
if (is.null(project_root)) {
  stop("Could not find project root from H2A Master.R.")
}

source(file.path(project_root, "code", "paths.R"))
ensure_project_dirs()

folder_dir <- as_dir(path_root())
folder_do <- as_dir(path_do())
folder_data <- as_dir(path_int())
folder_raw <- as_dir(path_raw())
folder_output <- as_dir(path_outputs())

## Packages ## -----------------------------------------------------------------
library(tidyverse)
library(arrow)
library(tidylog, warn.conflicts = FALSE)

## Run Sub Scripts -------------------------------------------------------------

# state min wages

source(paste0(folder_do, "H2A Pull Min Wages.R"), echo = TRUE)

## Clean

source(paste0(folder_do, "H2A Clean and Load.R"), echo = TRUE)

## Build Dataset

source(paste0(folder_do, "H2A Build Dataset.R"), echo = TRUE)

## Analysis Figs and Tables

## DDD + Figs ##
source(paste0(folder_do, "H2A Analysis Figs and Tables.R"), echo = TRUE)
