# code/paths.R

is_project_root <- function(path) {
  dir.exists(file.path(path, "code")) &&
    file.exists(file.path(path, ".here")) &&
    file.exists(file.path(path, "pyproject.toml"))
}

env_root <- Sys.getenv("H2A_PROJECT_ROOT")

if (nzchar(env_root)) {
  ROOT <- env_root
} else {
  ROOT <- here::here()
}

ROOT <- normalizePath(ROOT, winslash = "/", mustWork = TRUE)

if (!is_project_root(ROOT)) {
  stop(
    "Could not find H-2A project root. ",
    "Set H2A_PROJECT_ROOT to the project folder, e.g. ",
    "H2A_PROJECT_ROOT=C:/Users/<name>/Dropbox/H-2A Paper"
  )
}

path_root <- function(...) file.path(ROOT, ...)
path_code <- function(...) file.path(ROOT, "code", ...)
path_json <- function(...) file.path(ROOT, "code", "json", ...)
path_raw <- function(...) file.path(ROOT, "data", "raw", ...)
path_int <- function(...) file.path(ROOT, "data", "intermediate", ...)
path_output <- function(...) file.path(ROOT, "output", ...)
