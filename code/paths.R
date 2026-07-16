# code/paths.R

# `.here` and this file identify the project independently of optional tooling
# configuration (for example, `pyproject.toml`).
PROJECT_MARKERS <- c(".here")

is_project_root <- function(path) {
  all(file.exists(file.path(path, PROJECT_MARKERS))) &&
    file.exists(file.path(path, "code", "paths.R"))
}

ancestor_paths <- function(path) {
  path <- normalizePath(path.expand(path), winslash = "/", mustWork = FALSE)
  paths <- path

  repeat {
    parent <- dirname(path)
    if (identical(parent, path)) {
      break
    }
    paths <- c(paths, parent)
    path <- parent
  }

  paths
}

candidate_roots <- unique(c(
  Sys.getenv("H2A_PROJECT_ROOT", unset = NA_character_),
  "/mnt/storage/Dropbox/projects/H-2A Paper", # Ken's root path
  paste0("C:/Users/", Sys.info()[["user"]], "/Dropbox/H-2A Paper"), # Phil's root path
  # Try other candidate paths if necessary
  file.path(Sys.getenv("HOME", unset = ""), "Dropbox/projects/H-2A Paper"),
  file.path(Sys.getenv("HOME", unset = ""), "Dropbox/H-2A Paper"),
  file.path(Sys.getenv("USERPROFILE", unset = ""), "Dropbox/H-2A Paper"),
  file.path(
    Sys.getenv("USERPROFILE", unset = ""),
    "OneDrive/Dropbox/H-2A Paper"
  ),
  file.path(Sys.getenv("OneDrive", unset = ""), "Dropbox/H-2A Paper"),
  file.path(Sys.getenv("OneDriveCommercial", unset = ""), "Dropbox/H-2A Paper"),
  ancestor_paths(getwd())
))

candidate_roots <- candidate_roots[
  !is.na(candidate_roots) & nzchar(candidate_roots)
]
candidate_roots <- normalizePath(
  path.expand(candidate_roots),
  winslash = "/",
  mustWork = FALSE
)

root_hits <- candidate_roots[vapply(
  candidate_roots,
  is_project_root,
  logical(1)
)]

if (length(root_hits) == 0) {
  stop(
    "Could not find H-2A project root. ",
    "Add this machine's root path to candidate_roots in code/paths.R."
  )
}

ROOT <- root_hits[[1]]

path_root <- function(...) file.path(ROOT, ...)
path_do <- function(...) file.path(ROOT, "Do", ...)
path_code <- function(...) file.path(ROOT, "code", ...)
path_json <- function(...) file.path(ROOT, "code", "json", ...)

path_data <- function(...) file.path(ROOT, "data", ...)
path_raw <- function(...) file.path(ROOT, "data", "raw", ...)
path_int <- function(...) file.path(ROOT, "data", "intermediate", ...)
path_processed <- function(...) file.path(ROOT, "data", "processed", ...)
path_cache <- function(...) {
  file.path(ROOT, "data", "intermediate", "cache", ...)
}

path_outputs <- function(...) file.path(ROOT, "outputs", ...)
path_figures <- function(...) file.path(ROOT, "outputs", "figures", ...)
path_tables <- function(...) file.path(ROOT, "outputs", "tables", ...)
path_logs <- function(...) file.path(ROOT, "outputs", "logs", ...)

library(dotenv)
dotenv::load_dot_env(file = path_root(".env"))

as_dir <- function(path) {
  paste0(normalizePath(path, winslash = "/", mustWork = FALSE), "/")
}
