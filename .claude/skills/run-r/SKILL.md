---
name: run-r
description: Run R code or an R script file for the H-2A project. Use this whenever you need to execute R — inspect data, test a snippet, run a sub-script, or check output. Pass either inline R code or a script filename as the argument.
---

Run R for the H-2A Paper project using the following setup:

**R executable:** `C:/Program Files/R/R-4.4.0/bin/Rscript.exe`

**Project root:** `C:/Users/pghox/Dropbox/H-2A Paper/`

**Standard path variables (always set these at the top of any R code you run):**
```r
folder_dir    <- paste0("C:/Users/", Sys.info()["user"], "/Dropbox/H-2A Paper/")
folder_do     <- paste0(folder_dir, "Do/")
folder_data   <- paste0(folder_dir, "Data Int/")
folder_output <- paste0(folder_dir, "Output/")
```

---

## How to use this skill

The argument (`$ARGUMENTS`) will be one of:

1. **A script filename** — e.g. `H2A Clean and Load.R`  
   Run it with:
   ```bash
   "C:/Program Files/R/R-4.4.0/bin/Rscript.exe" "C:/Users/pghox/Dropbox/H-2A Paper/Do/$ARGUMENTS"
   ```

2. **Inline R code** — e.g. `read_parquet("Data Int/aewr.parquet") |> head()`  
   Write the code to a temp file, prepend the path variables, then run it with Rscript.

3. **No argument** — prompt the user what they want to run.

---

## Guidelines

- Always prepend the four `folder_*` path variables when running inline snippets so scripts that reference them work correctly.
- Use `arrow::read_parquet()` for any `.parquet` files in `Data Int/`.
- Capture stdout and stderr together so error messages are visible.
- If a script takes a long time, warn the user before running.
- Do NOT source `H2A Master.R` unless the user explicitly asks — it runs the entire pipeline.
- Do NOT edit files in `Do/Pre 2026/` or `Do/Pre 20260330/` — those are archives.
