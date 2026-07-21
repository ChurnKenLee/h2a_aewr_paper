# Bootstrap the shared R environment from any directory inside the repository.

local({
  project_root <- normalizePath(
    getwd(),
    winslash = "/",
    mustWork = TRUE
  )

  repeat {
    is_root <- file.exists(file.path(project_root, ".here")) &&
      file.exists(file.path(project_root, "code", "paths.R"))
    if (is_root) {
      break
    }

    parent <- dirname(project_root)
    if (identical(parent, project_root)) {
      stop("Could not find the H-2A project root from ", getwd())
    }
    project_root <- parent
  }

  renv_activate <- file.path(project_root, "renv", "activate.R")
  active_project <- Sys.getenv("RENV_PROJECT", unset = "")
  if (nzchar(active_project)) {
    active_project <- normalizePath(
      active_project,
      winslash = "/",
      mustWork = FALSE
    )
  }

  if (file.exists(renv_activate) && !identical(active_project, project_root)) {
    Sys.setenv(RENV_PROJECT = project_root)
    sys.source(renv_activate, envir = globalenv())
  }

  sys.source(
    file.path(project_root, "code", "paths.R"),
    envir = globalenv()
  )
})
