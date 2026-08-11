# Purpose: Estimate the restartable version-3 specification grid.
#
# Optional environment filters used for smoke tests and distributed runs:
#   MC_SPEC_STAGE     primary, compact (default), or exhaustive
#   MC_SPEC_IDS       comma-separated exact IDs; overrides MC_SPEC_STAGE
#   MC_OUTCOME_IDS    comma-separated outcome IDs
#   MC_SPEC_MAX       maximum number of registry rows after filtering
#   MC_SPEC_WORKERS   forked outcome workers per specification (default 1)
#   MC_FIXEST_THREADS threads within each outcome worker (default 4)
#   MC_SPEC_MAX_DENSE_GIB maximum permitted single dense N x K matrix
#   MC_SPEC_MAX_PEAK_GIB estimated per-worker matrix working-set ceiling
#   MC_SPEC_FORCE     set to 1 to replace compatible checkpoints
#
# Full runs write one compact checkpoint per specification/outcome.  A
# checkpoint stores beta and the 17 x K state-error matrix, never a dense
# K x K covariance.

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
library(arrow)
library(dplyr)
library(fixest)
library(readr)
library(tidyr)

dir.create(path_int(), recursive = TRUE, showWarnings = FALSE)
dir.create(path_tables(), recursive = TRUE, showWarnings = FALSE)

registry_bundle <- readRDS(
  path_int("mundlak_chamberlain_specification_registry.rds")
)
if (
  !identical(
    registry_bundle$design_version,
    MC_SPEC_PROGRAM_VERSION
  )
) {
  stop("Unexpected specification-registry version.", call. = FALSE)
}
full_registry <- registry_bundle$registry
calendars <- registry_bundle$calendars

parse_filter <- function(variable) {
  value <- Sys.getenv(variable, unset = "")
  if (!nzchar(value)) {
    return(character())
  }
  trimws(strsplit(value, ",", fixed = TRUE)[[1L]])
}

specification_filter <- parse_filter("MC_SPEC_IDS")
if (length(specification_filter) > 0L) {
  absent <- setdiff(specification_filter, full_registry$spec_id)
  if (length(absent) > 0L) {
    stop(
      "Unknown MC specification IDs: ",
      paste(absent, collapse = ", "),
      call. = FALSE
    )
  }
  registry <- full_registry[
    full_registry$spec_id %in% specification_filter,
    ,
    drop = FALSE
  ]
  execution_stage <- "custom"
  registry$execution_stage <- execution_stage
  registry$execution_reason <- "explicit_spec_id"
  registry$execution_priority <- 1L
} else {
  execution_stage <- Sys.getenv(
    "MC_SPEC_STAGE",
    unset = MC_SPEC_DEFAULT_STAGE
  )
  registry <- mc_sp_execution_registry(
    full_registry,
    execution_stage
  )
}
maximum_specifications <- suppressWarnings(
  as.integer(Sys.getenv("MC_SPEC_MAX", unset = ""))
)
if (
  length(maximum_specifications) == 1L &&
    is.finite(maximum_specifications) &&
    maximum_specifications > 0L
) {
  registry <- head(registry, maximum_specifications)
}

outcome_filter <- parse_filter("MC_OUTCOME_IDS")
outcomes <- MC_OUTCOMES
if (length(outcome_filter) > 0L) {
  absent <- setdiff(outcome_filter, outcomes$outcome_id)
  if (length(absent) > 0L) {
    stop(
      "Unknown MC outcome IDs: ",
      paste(absent, collapse = ", "),
      call. = FALSE
    )
  }
  outcomes <- outcomes[
    outcomes$outcome_id %in% outcome_filter,
    ,
    drop = FALSE
  ]
}
workers <- suppressWarnings(
  as.integer(
    Sys.getenv(
      "MC_SPEC_WORKERS",
      unset = as.character(MC_SPEC_DEFAULT_WORKERS)
    )
  )
)
if (!is.finite(workers) || workers < 1L) {
  workers <- MC_SPEC_DEFAULT_WORKERS
}
fixest_threads <- mc_sp_configure_fixest_threads()
force_checkpoint <- identical(
  Sys.getenv("MC_SPEC_FORCE", unset = "0"),
  "1"
)

execution_registry <- registry
execution_registry$outcome_count <- nrow(outcomes)
execution_registry$workers <- workers
execution_registry$fixest_threads <- fixest_threads
execution_registry$dense_matrix_guard_gib <-
  mc_sp_numeric_environment(
    "MC_SPEC_MAX_DENSE_GIB",
    MC_SPEC_MAX_DENSE_MATRIX_GIB
  )
execution_registry$estimated_peak_guard_gib <-
  mc_sp_numeric_environment(
    "MC_SPEC_MAX_PEAK_GIB",
    MC_SPEC_MAX_ESTIMATED_PEAK_GIB
  )
write_csv(
  execution_registry,
  path_tables("mc_execution_registry.csv")
)
saveRDS(
  list(
    design_version = MC_SPEC_PROGRAM_VERSION,
    execution_stage = execution_stage,
    outcomes = outcomes$outcome_id,
    workers = workers,
    fixest_threads = fixest_threads,
    registry = execution_registry
  ),
  path_int("mundlak_chamberlain_execution_registry.rds")
)
message(
  "Execution stage '",
  execution_stage,
  "': ",
  nrow(registry),
  " specifications x ",
  nrow(outcomes),
  " outcomes; ",
  workers,
  " worker(s) x ",
  fixest_threads,
  " fixest thread(s)."
)

panel_cache_directory <- file.path(
  path_int(),
  "mundlak_chamberlain_specification_panels"
)
checkpoint_directory <- file.path(
  path_int(),
  "mundlak_chamberlain_specification_checkpoints"
)
dir.create(
  panel_cache_directory,
  recursive = TRUE,
  showWarnings = FALSE
)
dir.create(
  checkpoint_directory,
  recursive = TRUE,
  showWarnings = FALSE
)

atomic_save_rds <- function(object, target, compress = "gzip") {
  dir.create(dirname(target), recursive = TRUE, showWarnings = FALSE)
  temporary <- tempfile(
    pattern = paste0(".", basename(target), "."),
    tmpdir = dirname(target)
  )
  on.exit(unlink(temporary), add = TRUE)
  saveRDS(object, temporary, compress = compress)
  if (!file.rename(temporary, target)) {
    stop("Could not atomically install ", target, call. = FALSE)
  }
  invisible(target)
}

checkpoint_path <- function(spec_id, outcome_id) {
  file.path(
    checkpoint_directory,
    spec_id,
    paste0(outcome_id, ".rds")
  )
}

checkpoint_is_current <- function(target) {
  if (!file.exists(target) || force_checkpoint) {
    return(FALSE)
  }
  checkpoint <- tryCatch(
    readRDS(target),
    error = function(condition) NULL
  )
  if (
    is.null(checkpoint) ||
      !identical(
        checkpoint$design_version,
        MC_SPEC_PROGRAM_VERSION
      )
  ) {
    return(FALSE)
  }
  if (identical(checkpoint$status, "estimated")) {
    diagnostics <- checkpoint$diagnostics
    if (
      all(c("observations", "effective_parameters") %in%
        names(diagnostics))
    ) {
      resource <- mc_sp_resource_budget(
        diagnostics$observations[[1]],
        diagnostics$effective_parameters[[1]]
      )
      if (
        resource$dense_matrix_gib >
          resource$dense_matrix_guard_gib ||
          resource$estimated_peak_gib >
            resource$estimated_peak_guard_gib
      ) {
        return(FALSE)
      }
    }
  }
  TRUE
}

shared_panel <- read_parquet(
  path_processed("county_year_panel.parquet")
) |>
  as.data.frame()

load_calendar_panel <- function(calendar_id) {
  target <- file.path(
    panel_cache_directory,
    paste0(calendar_id, ".rds")
  )
  if (file.exists(target) && !force_checkpoint) {
    cached <- readRDS(target)
    if (
      identical(
        cached$metadata$design_version,
        MC_SPEC_PROGRAM_VERSION
      )
    ) {
      return(cached)
    }
  }
  calendar <- calendars[
    calendars$calendar_id == calendar_id,
    ,
    drop = FALSE
  ]
  message("Building calendar panel ", calendar_id)
  built <- mc_sp_build_calendar_panel(
    shared_panel,
    calendar
  )
  atomic_save_rds(built, target, compress = FALSE)
  built
}

estimate_one <- function(
  outcome_index,
  panel_bundle,
  specification
) {
  outcome_specification <- outcomes[
    outcome_index,
    ,
    drop = FALSE
  ]
  outcome_id <- outcome_specification$outcome_id[[1]]
  target <- checkpoint_path(
    specification$spec_id,
    outcome_id
  )
  if (checkpoint_is_current(target)) {
    return(data.frame(
      spec_id = specification$spec_id,
      outcome_id = outcome_id,
      action = "checkpoint_reused",
      stringsAsFactors = FALSE
    ))
  }
  message(
    "Estimating ",
    specification$spec_id,
    ": ",
    outcome_id
  )
  started <- proc.time()[["elapsed"]]
  checkpoint <- tryCatch(
    mc_sp_estimate_outcome(
      panel = panel_bundle$panel,
      metadata = panel_bundle$metadata,
      specification = specification,
      outcome_specification = outcome_specification
    ),
    error = function(condition) {
      list(
        status = "failed",
        design_version = MC_SPEC_PROGRAM_VERSION,
        spec_id = specification$spec_id,
        calendar_id = specification$calendar_id,
        outcome_id = outcome_id,
        diagnostics = data.frame(
          spec_id = specification$spec_id,
          calendar_id = specification$calendar_id,
          outcome_id = outcome_id,
          richness_tier = specification$richness_tier,
          horizon_count = specification$horizon_count,
          polynomial_degree =
            max(specification$polynomial_degrees),
          status = "failed",
          rejection_reason = conditionMessage(condition),
          stringsAsFactors = FALSE
        ),
        effects = data.frame(),
        influence = data.frame(),
        error = conditionMessage(condition)
      )
    }
  )
  checkpoint$design_version <- MC_SPEC_PROGRAM_VERSION
  checkpoint$elapsed_seconds <-
    proc.time()[["elapsed"]] - started
  atomic_save_rds(checkpoint, target)
  data.frame(
    spec_id = specification$spec_id,
    outcome_id = outcome_id,
    action = checkpoint$status,
    stringsAsFactors = FALSE
  )
}

progress_rows <- list()
for (calendar_id in unique(registry$calendar_id)) {
  panel_bundle <- load_calendar_panel(calendar_id)
  calendar_registry <- registry[
    registry$calendar_id == calendar_id,
    ,
    drop = FALSE
  ]
  for (specification_index in seq_len(nrow(calendar_registry))) {
    specification <- mc_sp_specification(
      calendar_registry[
        specification_index,
        ,
        drop = FALSE
      ]
    )
    result <- if (
      .Platform$OS.type == "unix" &&
        workers > 1L &&
        nrow(outcomes) > 1L
    ) {
      parallel::mclapply(
        seq_len(nrow(outcomes)),
        estimate_one,
        panel_bundle = panel_bundle,
        specification = specification,
        mc.cores = min(workers, nrow(outcomes)),
        mc.preschedule = FALSE
      )
    } else {
      lapply(
        seq_len(nrow(outcomes)),
        estimate_one,
        panel_bundle = panel_bundle,
        specification = specification
      )
    }
    progress_rows[[length(progress_rows) + 1L]] <-
      bind_rows(result)
    write_csv(
      bind_rows(progress_rows),
      path_tables("mc_specification_run_progress.csv")
    )
    message(
      "Execution progress: ",
      length(progress_rows),
      "/",
      nrow(registry),
      " specifications in stage '",
      execution_stage,
      "'."
    )
  }
  rm(panel_bundle)
  invisible(gc())
}

active_checkpoint_grid <- expand.grid(
  spec_id = registry$spec_id,
  outcome_id = outcomes$outcome_id,
  stringsAsFactors = FALSE
)
all_checkpoint_files <- file.path(
  checkpoint_directory,
  active_checkpoint_grid$spec_id,
  paste0(active_checkpoint_grid$outcome_id, ".rds")
)
all_checkpoint_files <- all_checkpoint_files[
  file.exists(all_checkpoint_files)
]
diagnostic_rows <- list()
effect_rows <- list()
influence_rows <- list()
basis_rows <- list()
history_rows <- list()
for (checkpoint_file in all_checkpoint_files) {
  checkpoint <- readRDS(checkpoint_file)
  if (
    !identical(
      checkpoint$design_version,
      MC_SPEC_PROGRAM_VERSION
    )
  ) {
    next
  }
  diagnostic_rows[[length(diagnostic_rows) + 1L]] <-
    checkpoint$diagnostics
  if (nrow(checkpoint$effects) > 0L) {
    effect_rows[[length(effect_rows) + 1L]] <-
      checkpoint$effects
  }
  if (nrow(checkpoint$influence) > 0L) {
    influence_rows[[length(influence_rows) + 1L]] <-
      checkpoint$influence
  }
  if (
    !is.null(checkpoint$common_basis_audit) &&
      nrow(checkpoint$common_basis_audit) > 0L
  ) {
    basis <- checkpoint$common_basis_audit
    basis$spec_id <- checkpoint$spec_id
    basis$outcome_id <- checkpoint$outcome_id
    basis_rows[[length(basis_rows) + 1L]] <- basis
  }
  if (
    !is.null(checkpoint$history_selection) &&
      nrow(checkpoint$history_selection) > 0L
  ) {
    history <- checkpoint$history_selection
    history$spec_id <- checkpoint$spec_id
    history$outcome_id <- checkpoint$outcome_id
    history_rows[[length(history_rows) + 1L]] <- history
  }
}

diagnostics <- bind_rows(diagnostic_rows)
effects <- bind_rows(effect_rows)
influence <- bind_rows(influence_rows)
common_basis <- bind_rows(basis_rows)
resolved_history <- bind_rows(history_rows)

write_csv(
  diagnostics,
  path_tables("mc_specification_model_diagnostics.csv")
)
write_csv(
  effects,
  path_tables("mc_specification_effects.csv")
)
write_csv(
  influence,
  path_tables("mc_reference_state_influence.csv")
)
write_csv(
  common_basis,
  path_tables("mc_common_basis_audit.csv")
)
write_csv(
  resolved_history,
  path_tables("mc_resolved_history_audit.csv")
)
write_csv(
  bind_rows(progress_rows),
  path_tables("mc_specification_run_progress.csv")
)

primary_target_ids <- registry_bundle$registry |>
  filter(
    .data$horizon_count == 3L,
    .data$preperiod_start == min(MC_BASELINE_YEARS),
    .data$preperiod_end == max(MC_BASELINE_YEARS),
    .data$analysis_start == min(MC_ANALYSIS_YEARS),
    .data$analysis_end == max(MC_ANALYSIS_YEARS),
    .data$polynomial_degree == 2L
  ) |>
  pull(.data$spec_id)
primary_complete <- all(
  expand.grid(
    spec_id = primary_target_ids,
    outcome_id = MC_OUTCOMES$outcome_id,
    stringsAsFactors = FALSE
  ) |>
    left_join(
      diagnostics |>
        select(
          .data$spec_id,
          .data$outcome_id,
          .data$status
        ),
      by = c("spec_id", "outcome_id")
    ) |>
    pull(.data$status) %in% c("estimated", "rejected")
)
if (primary_complete) {
  primary_selection <- mc_sp_primary_selection(
    registry_bundle$registry,
    diagnostics,
    MC_OUTCOMES
  )
  write_csv(
    primary_selection,
    path_tables("mc_primary_specification_selection.csv")
  )
  saveRDS(
    primary_selection,
    path_int("mundlak_chamberlain_primary_selection.rds")
  )
} else {
  message(
    "Primary selection is pending until all four current-calendar ",
    "H3-D2 richness tiers are resolved for every outcome."
  )
}

message(
  "Specification-program checkpoints processed: ",
  length(all_checkpoint_files),
  "."
)
