from __future__ import annotations

import csv
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
import os

PROJECT = Path(".").resolve()
R_FILES = sorted((PROJECT / "Do").glob("*.R"))

READ_FUNS = {
    "read.csv",
    "read_csv",
    "read_parquet",
    "read_excel",
    "read_xlsx",
    "readRDS",
    "st_read",
}
WRITE_FUNS = {
    "write.csv",
    "write_csv",
    "write_parquet",
    "write_xlsx",
    "saveRDS",
    "ggsave",
    "writeLines",
}
SOURCE_FUNS = {"source"}


R_WALKER = r"""
args <- commandArgs(trailingOnly = TRUE)
files <- args

read_funs <- c(
  "read.csv", "read_csv", "read_parquet", "read_excel",
  "read_xlsx", "readRDS", "st_read"
)

write_funs <- c(
  "write.csv", "write_csv", "write_parquet", "write_xlsx",
  "saveRDS", "ggsave", "writeLines"
)

source_funs <- c("source")

env_paths <- list(
  folder_dir = "",
  folder_do = "Do/",
  folder_data = "Data Int/",
  folder_raw = "data/raw/",
  folder_output = "Output/"
)

is_missing_arg <- function(x) {
  isTRUE(tryCatch(identical(x, quote(expr = )), error = function(e) FALSE))
}

call_name <- function(x) {
  if (!is.call(x)) return(NA_character_)

  fn <- x[[1]]

  if (is.symbol(fn)) return(as.character(fn))

  if (is.call(fn)) {
    op <- as.character(fn[[1]])
    if (op %in% c("::", ":::")) {
      return(paste0(as.character(fn[[2]]), "::", as.character(fn[[3]])))
    }
  }

  NA_character_
}

strip_namespace <- function(fn) {
  if (is.na(fn)) return(NA_character_)
  sub("^.*::", "", fn)
}

expr_text <- function(x) {
  if (is.null(x)) return(NA_character_)

  tryCatch(
    paste(deparse(x, width.cutoff = 500L), collapse = " "),
    error = function(e) NA_character_
  )
}

resolve_path <- function(x) {
  if (is.null(x)) return(NA_character_)

  if (is.character(x)) return(paste0(x, collapse = ""))

  if (is.symbol(x)) {
    nm <- as.character(x)
    if (!is.null(env_paths[[nm]])) return(env_paths[[nm]])
    return(NA_character_)
  }

  if (!is.call(x)) return(NA_character_)

  fn <- strip_namespace(call_name(x))

  if (isTRUE(identical(fn, "paste0"))) {
    args <- as.list(x)[-1]
    parts <- lapply(seq_along(args), function(i) resolve_path(safe_get(args, i)))
    parts <- unlist(parts)
    if (length(parts) == 0 || any(is.na(parts))) return(NA_character_)
    return(paste0(parts, collapse = ""))
  }

  if (isTRUE(identical(fn, "file.path"))) {
    args <- as.list(x)[-1]
    parts <- lapply(seq_along(args), function(i) resolve_path(safe_get(args, i)))
    parts <- unlist(parts)
    if (length(parts) == 0 || any(is.na(parts))) return(NA_character_)
    return(do.call(file.path, as.list(parts)))
  }

  NA_character_
}

safe_get <- function(xs, i) {
  if (is.null(xs) || i < 1 || i > length(xs)) return(NULL)

  tryCatch({
    v <- xs[[i]]

    # Force evaluation enough to detect R's missing-argument sentinel.
    txt <- tryCatch(
      paste(deparse(v, width.cutoff = 500L), collapse = " "),
      error = function(e) NA_character_
    )

    if (is.na(txt) || identical(txt, "")) return(NULL)

    v
  }, error = function(e) NULL)
}

first_path_arg <- function(x, fn) {
  args <- as.list(x)[-1]
  if (length(args) == 0) return(NULL)

  nms <- names(args)
  if (is.null(nms)) nms <- rep("", length(args))

  get_named <- function(keys) {
    for (key in keys) {
      hit <- which(nms == key)
      if (length(hit) > 0) return(safe_get(args, hit[1]))
    }
    NULL
  }

  named <- get_named(c("file", "path", "filename", "con", "sink"))
  if (!is.null(named)) return(named)

  if (fn %in% c(
    "read.csv", "read_csv", "read_parquet", "read_excel",
    "read_xlsx", "readRDS", "st_read", "source"
  )) {
    return(safe_get(args, 1))
  }

  if (fn %in% c(
    "write.csv", "write_csv", "write_parquet",
    "write_xlsx", "saveRDS", "writeLines"
  )) {
    if (length(args) >= 2) return(safe_get(args, 2))
    return(safe_get(args, 1))
  }

  if (fn == "ggsave") {
    return(safe_get(args, 1))
  }

  NULL
}

walk <- function(x, script, parent_line = NA_integer_) {
  if (is.null(x)) return(invisible(NULL))
  if (!is.call(x)) return(invisible(NULL))

  fn_full <- call_name(x)
  fn <- strip_namespace(fn_full)

  direction <- NA_character_
  if (fn %in% read_funs) direction <- "read"
  if (fn %in% write_funs) direction <- "write"
  if (fn %in% source_funs) direction <- "source"

  if (!is.na(direction)) {
    path_arg <- first_path_arg(x, fn)

    rows[[length(rows) + 1]] <<- data.frame(
      script = script,
      line = parent_line,
      function_name = fn_full,
      direction = direction,
      path_expr = expr_text(path_arg),
      resolved_path = resolve_path(path_arg),
      call_expr = expr_text(x),
      stringsAsFactors = FALSE
    )
  }

  children <- as.list(x)[-1]

  if (length(children) > 0) {
    for (i in seq_along(children)) {
      node <- safe_get(children, i)

      if (!is.null(node)) {
        walk(
          x = node,
          script = script,
          parent_line = parent_line
        )
      }
    }
  }

  invisible(NULL)
}

line_for_expr <- function(expr) {
  sr <- attr(expr, "srcref")
  if (!is.null(sr)) return(as.integer(sr[[1]]))
  NA_integer_
}

rows <- list()

for (f in files) {
  exprs <- parse(f, keep.source = TRUE)

  for (i in seq_along(exprs)) {
    line <- line_for_expr(exprs[[i]])
    walk(exprs[[i]], f, line)
  }
}

out <- if (length(rows)) {
  do.call(rbind, rows)
} else {
  data.frame(
    script = character(),
    line = integer(),
    function_name = character(),
    direction = character(),
    path_expr = character(),
    resolved_path = character(),
    call_expr = character(),
    stringsAsFactors = FALSE
  )
}

write.csv(out, stdout(), row.names = FALSE, na = "")
"""

def run_r_ast_parser(r_files: list[Path]) -> list[dict[str, str]]:
    env = os.environ.copy()
    env.setdefault("TMPDIR", tempfile.gettempdir())

    proc = subprocess.run(
        ["Rscript", "--vanilla", "-e", R_WALKER, *map(str, r_files)],
        cwd=PROJECT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )

    if proc.returncode != 0:
        print("R stdout:")
        print(proc.stdout)
        print("R stderr:")
        print(proc.stderr)
        raise RuntimeError(f"R AST parser failed with exit code {proc.returncode}")

    return list(csv.DictReader(proc.stdout.splitlines()))


rows = run_r_ast_parser(R_FILES)

# Write manifest.
manifest = PROJECT / "phil_r_dependency_manifest.csv"
if rows:
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
else:
    print("No dependency calls found")

print(f"Wrote {manifest}")
print(f"Found {len(rows)} read/write/source calls")

# Classify resolved files.
by_path = defaultdict(set)
for row in rows:
    path = row["resolved_path"]
    if path:
        by_path[path].add(row["direction"])

print("\nClassification:")
for path, directions in sorted(by_path.items()):
    if directions == {"read"}:
        cls = "raw_candidate"
    elif "write" in directions and "read" in directions:
        cls = "intermediate"
    elif directions == {"write"}:
        cls = "output_or_terminal_write"
    elif directions == {"source"}:
        cls = "script_dependency"
    else:
        cls = "+".join(sorted(directions))

    print(f"{cls:24} {path}")
