#!/usr/bin/env python3
"""Split the legacy Do R scripts into smaller generated R modules.

The parser is base R's own parser (`getParseData(parse(...))`). Python and
Polars handle indexing, section detection, provenance, and reconstruction.

This is intentionally conservative: generated modules preserve original source
text slices and avoid deparsing R. All source transformations are applied
through the JSONL patch table.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl

from h2a.paths import ROOT


GENERATOR_ID = "code/split_do_r_scripts.py"
GENERATED_PREFIX = "c"
SETUP_FILE = "c00_setup.R"
RUNNER_FILE = "c99_run_all.R"
SOURCE_ORDER = (
    "H2A Pull Min Wages.R",
    "H2A Clean and Load.R",
    "H2A Build Dataset.R",
    "H2A Analysis Figs and Tables.R",
)


PATCH_SCHEMA = {
    "module": pl.String,
    "op": pl.String,
    "selector": pl.String,
    "payload": pl.String,
    "reason": pl.String,
    "enabled": pl.Boolean,
    "allow_missing": pl.Boolean,
}


def read_patches(path: Path) -> pl.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pl.DataFrame(schema=PATCH_SCHEMA)

    patches = pl.read_ndjson(path)
    for name, dtype in PATCH_SCHEMA.items():
        if name not in patches.columns:
            default = True if dtype == pl.Boolean and name == "enabled" else None
            if dtype == pl.Boolean and name == "allow_missing":
                default = False
            patches = patches.with_columns(pl.lit(default, dtype=dtype).alias(name))

    return patches.with_columns(
        pl.col("enabled").fill_null(True),
        pl.col("allow_missing").fill_null(False),
    ).select([pl.col(name).cast(dtype, strict=False) for name, dtype in PATCH_SCHEMA.items()])


@dataclass(frozen=True)
class PatchApplyResult:
    text: str
    matched: int
    changed: bool


def apply_patch_text_result(text: str, patch: dict) -> PatchApplyResult:
    op = patch["op"]
    selector = patch["selector"]
    payload = patch.get("payload") or ""

    if op == "insert_before":
        idx = text.find(selector)
        if idx < 0:
            if patch.get("allow_missing"):
                return PatchApplyResult(text=text, matched=0, changed=False)
            raise ValueError(f"Selector not found for insert_before: {selector}")
        new_text = text[:idx] + payload + text[idx:]
        return PatchApplyResult(text=new_text, matched=1, changed=new_text != text)

    if op == "insert_after":
        idx = text.find(selector)
        if idx < 0:
            if patch.get("allow_missing"):
                return PatchApplyResult(text=text, matched=0, changed=False)
            raise ValueError(f"Selector not found for insert_after: {selector}")
        idx += len(selector)
        new_text = text[:idx] + payload + text[idx:]
        return PatchApplyResult(text=new_text, matched=1, changed=new_text != text)

    if op == "replace_regex":
        new_text, n = re.subn(selector, payload, text, flags=re.MULTILINE | re.DOTALL)
        if n == 0:
            if patch.get("allow_missing"):
                return PatchApplyResult(text=text, matched=0, changed=False)
            raise ValueError(f"Regex selector matched nothing: {selector}")
        return PatchApplyResult(text=new_text, matched=n, changed=new_text != text)

    if op == "replace_lines":
        start, end = [int(x) for x in selector.split(":")]
        lines = text.splitlines(keepends=True)
        new_text = "".join(lines[: start - 1] + [payload] + lines[end:])
        return PatchApplyResult(text=new_text, matched=1, changed=new_text != text)

    raise ValueError(f"Unknown patch op: {op}")


def apply_patch_text(text: str, patch: dict) -> str:
    return apply_patch_text_result(text, patch).text


def apply_module_patches(
    module_path: str,
    text: str,
    patches: pl.DataFrame,
) -> tuple[str, int, int, int]:
    patch_module_names = [module_path, unprefixed_module_path(module_path), "*"]
    module_patches = (
        patches
        .filter((pl.col("module").is_in(patch_module_names)) & pl.col("enabled"))
        .to_dicts()
    )

    matched_count = 0
    changed_count = 0
    for patch in module_patches:
        result = apply_patch_text_result(text, patch)
        text = result.text
        matched_count += result.matched
        changed_count += int(result.changed)

    return text, len(module_patches), matched_count, changed_count


def append_patch(path: Path, patch: dict[str, object]) -> None:
    """Append one patch record to the JSONL patch table."""
    row = {
        "module": patch.get("module", ""),
        "op": patch.get("op", ""),
        "selector": patch.get("selector", ""),
        "payload": patch.get("payload", ""),
        "reason": patch.get("reason", ""),
        "enabled": bool(patch.get("enabled", True)),
        "allow_missing": bool(patch.get("allow_missing", False)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


@dataclass(frozen=True)
class HeaderRef:
    pattern: str
    occurrence: int = 1


@dataclass(frozen=True)
class ModuleSpec:
    path: str
    source: str
    start: int | HeaderRef
    end_before: HeaderRef | None = None
    group: str = ""


SPLIT_SPECS: tuple[ModuleSpec, ...] = (
    ModuleSpec(
        path="01_pull_min_wages.R",
        source="H2A Pull Min Wages.R",
        start=1,
        group="pull_min_wages",
    ),
    ModuleSpec(
        path="02_clean_01_price_cz_ppi_minwage.R",
        source="H2A Clean and Load.R",
        start=1,
        end_before=HeaderRef(r"^##\s+H2A Data\b", 1),
        group="clean_load",
    ),
    ModuleSpec(
        path="02_clean_02_h2a_cdl_aux.R",
        source="H2A Clean and Load.R",
        start=HeaderRef(r"^##\s+H2A Data\b", 1),
        end_before=HeaderRef(r"^##\s+_+", 1),
        group="clean_load",
    ),
    ModuleSpec(
        path="02_clean_03_yearly_census_h2a_aewr.R",
        source="H2A Clean and Load.R",
        start=HeaderRef(r"^##\s+_+", 1),
        end_before=HeaderRef(r"^##\s+AEWR Region TS\b", 1),
        group="clean_load",
    ),
    ModuleSpec(
        path="02_clean_04_aewr_region_ts_diagnostics.R",
        source="H2A Clean and Load.R",
        start=HeaderRef(r"^##\s+AEWR Region TS\b", 1),
        end_before=HeaderRef(r"^##\s+BEA Data\b", 1),
        group="clean_load",
    ),
    ModuleSpec(
        path="02_clean_05_bea_population_county_panel.R",
        source="H2A Clean and Load.R",
        start=HeaderRef(r"^##\s+BEA Data\b", 1),
        group="clean_load",
    ),
    ModuleSpec(
        path="03_build_01_load_merge.R",
        source="H2A Build Dataset.R",
        start=1,
        end_before=HeaderRef(r"^##\s+Variable cleaning\b", 1),
        group="build_dataset",
    ),
    ModuleSpec(
        path="03_build_02_variable_cleaning.R",
        source="H2A Build Dataset.R",
        start=HeaderRef(r"^##\s+Variable cleaning\b", 1),
        end_before=HeaderRef(r"^##\s+lags of h2a variables\b", 1),
        group="build_dataset",
    ),
    ModuleSpec(
        path="03_build_03_lags_classification_write.R",
        source="H2A Build Dataset.R",
        start=HeaderRef(r"^##\s+lags of h2a variables\b", 1),
        end_before=HeaderRef(r"^#\s+---\s+Diagnostic:", 1),
        group="build_dataset",
    ),
    ModuleSpec(
        path="03_build_04_diagnostics_cleanup.R",
        source="H2A Build Dataset.R",
        start=HeaderRef(r"^#\s+---\s+Diagnostic:", 1),
        group="build_dataset",
    ),
    ModuleSpec(
        path="04_analysis_00_setup_data.R",
        source="H2A Analysis Figs and Tables.R",
        start=1,
        end_before=HeaderRef(r"^####\s+Exhibit 0:", 1),
        group="analysis",
    ),
    ModuleSpec(
        path="04_analysis_01_descriptive_figures.R",
        source="H2A Analysis Figs and Tables.R",
        start=HeaderRef(r"^####\s+Exhibit 0:", 1),
        end_before=HeaderRef(r"^##\s+Exhibit 13:", 1),
        group="analysis",
    ),
    ModuleSpec(
        path="04_analysis_02_main_dd_event_study.R",
        source="H2A Analysis Figs and Tables.R",
        start=HeaderRef(r"^##\s+Exhibit 13:", 1),
        end_before=HeaderRef(r"^##\s+Exhibit 16:", 1),
        group="analysis",
    ),
    ModuleSpec(
        path="04_analysis_03_summary_price_labor.R",
        source="H2A Analysis Figs and Tables.R",
        start=HeaderRef(r"^##\s+Exhibit 16:", 1),
        end_before=HeaderRef(r"^####\s+Exhibit 21:", 1),
        group="analysis",
    ),
    ModuleSpec(
        path="04_analysis_04_stacked_did_matching.R",
        source="H2A Analysis Figs and Tables.R",
        start=HeaderRef(r"^####\s+Exhibit 21:", 1),
        group="analysis",
    ),
)


HEADER_RE = re.compile(
    r"^\s*#{2,}\s*(.*?)\s*(?:[-_#]{3,}.*)?$"
    r"|^\s*#\s*---\s*(.*?)\s*---"
)
ARTIFACT_RE = re.compile(
    r"\b("
    r"read_parquet|read_csv|read\.csv|st_read|read_xlsx|read_dta|"
    r"write_parquet|write_csv|write\.csv|write_xlsx|ggsave|etable|cat\("
    r")\b"
)
ARTIFACT_CALLS = {
    "read_parquet",
    "read_csv",
    "read.csv",
    "st_read",
    "read_xlsx",
    "read_dta",
    "write_parquet",
    "write_csv",
    "write.csv",
    "write_xlsx",
    "ggsave",
    "etable",
    "cat",
}
LIBRARY_RE = re.compile(r"^\s*library\s*\(\s*([A-Za-z0-9_.]+)")


def project_root_from_script() -> Path:
    return ROOT


def default_patch_path(root: Path) -> Path:
    return Path(__file__).resolve().parent / "r_split_patches.jsonl"


def parse_args() -> argparse.Namespace:
    root = project_root_from_script()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--do-dir", type=Path, default=root / "Do")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--patches",
        type=Path,
        default=default_patch_path(root),
        help="JSONL patch table to apply after splitting. Defaults to r_split_patches.jsonl next to this script.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Accepted for compatibility; generated R files are replaced in place.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build metadata but do not write files.")
    parser.add_argument(
        "--skip-parse-check",
        action="store_true",
        help="Do not parse generated R modules after writing them.",
    )
    return parser.parse_args()


def read_source(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8-sig")
    if text and not text.endswith("\n"):
        text += "\n"
    return text.splitlines(keepends=True)


def r_parse_data(path: Path) -> pl.DataFrame:
    r_code = r'''
args <- commandArgs(trailingOnly = TRUE)
pd <- getParseData(parse(file = args[[1]], keep.source = TRUE))
if (is.null(pd)) {
  pd <- data.frame(
    line1 = integer(), col1 = integer(), line2 = integer(), col2 = integer(),
    id = integer(), parent = integer(), token = character(),
    terminal = logical(), text = character()
  )
}
write.csv(pd, stdout(), row.names = FALSE, na = "")
'''
    proc = subprocess.run(
        ["Rscript", "-e", r_code, str(path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"R parse failed for {path}:\n{proc.stderr}")
    df = pl.read_csv(io.StringIO(proc.stdout), infer_schema_length=0)
    expected = {"line1", "col1", "line2", "col2", "id", "parent", "token", "terminal", "text"}
    missing = expected.difference(df.columns)
    if missing:
        raise RuntimeError(f"R parse output for {path} is missing columns: {sorted(missing)}")
    return df.with_columns(
        pl.col("line1", "col1", "line2", "col2", "id", "parent").cast(pl.Int64, strict=False),
        pl.col("terminal")
        .str.to_lowercase()
        .is_in(["true", "t", "1"])
        .alias("terminal"),
        pl.col("text").fill_null(""),
    )


def r_parse_data_text(text: str) -> pl.DataFrame:
    r_code = r'''
txt <- readLines("stdin", warn = FALSE)
pd <- getParseData(parse(text = txt, keep.source = TRUE))
if (is.null(pd)) {
  pd <- data.frame(
    line1 = integer(), col1 = integer(), line2 = integer(), col2 = integer(),
    id = integer(), parent = integer(), token = character(),
    terminal = logical(), text = character()
  )
}
write.csv(pd, stdout(), row.names = FALSE, na = "")
'''
    proc = subprocess.run(
        ["Rscript", "-e", r_code],
        input=text,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"R parse failed for generated module text:\n{proc.stderr}")
    df = pl.read_csv(io.StringIO(proc.stdout), infer_schema_length=0)
    expected = {"line1", "col1", "line2", "col2", "id", "parent", "token", "terminal", "text"}
    missing = expected.difference(df.columns)
    if missing:
        raise RuntimeError(f"R parse output for generated module text is missing columns: {sorted(missing)}")
    return df.with_columns(
        pl.col("line1", "col1", "line2", "col2", "id", "parent").cast(pl.Int64, strict=False),
        pl.col("terminal")
        .str.to_lowercase()
        .is_in(["true", "t", "1"])
        .alias("terminal"),
        pl.col("text").fill_null(""),
    )


def line_text(lines: list[str], line_no: int) -> str:
    return lines[line_no - 1].rstrip("\r\n")


def find_header_line(lines: list[str], ref: HeaderRef, source: str) -> int:
    pattern = re.compile(ref.pattern)
    matches = [
        idx
        for idx in range(1, len(lines) + 1)
        if pattern.search(line_text(lines, idx).lstrip("\ufeff"))
    ]
    if len(matches) < ref.occurrence:
        headers = "\n".join(
            f"{idx}: {line_text(lines, idx)}"
            for idx in range(1, len(lines) + 1)
            if HEADER_RE.match(line_text(lines, idx).lstrip("\ufeff"))
        )
        raise ValueError(
            f"Could not find header {ref.pattern!r} occurrence {ref.occurrence} "
            f"in {source}. Available headers:\n{headers}"
        )
    return matches[ref.occurrence - 1]


def resolve_line(ref: int | HeaderRef, lines: list[str], source: str) -> int:
    if isinstance(ref, int):
        return ref
    return find_header_line(lines, ref, source)


def extract_sections(source_name: str, lines: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, raw in enumerate(lines, start=1):
        text = raw.rstrip("\r\n").lstrip("\ufeff")
        match = HEADER_RE.match(text)
        if not match:
            continue
        title = (match.group(1) or match.group(2) or "").strip(" #-_\t")
        if not title:
            continue
        rows.append({"source": source_name, "line": idx, "title": title, "raw": text})
    return rows


def text_slice(lines: list[str], line1: int, col1: int, line2: int, col2: int) -> str:
    selected = lines[line1 - 1 : line2]
    if not selected:
        return ""
    if line1 == line2:
        return selected[0][col1 - 1 : col2].rstrip("\r\n")
    selected[0] = selected[0][col1 - 1 :]
    selected[-1] = selected[-1][:col2]
    return "".join(selected).rstrip("\r\n")


def enclosing_expr(parse_df: pl.DataFrame, token_row: dict[str, object]) -> dict[str, object] | None:
    """Return the parse expression that contains a function call and its arguments."""
    by_id = {int(row["id"]): row for row in parse_df.to_dicts()}
    expr = by_id.get(int(token_row["parent"]))
    if expr is None:
        return None
    parent = by_id.get(int(expr["parent"]))
    if parent is not None and parent.get("token") == "expr":
        return parent
    return expr


def extract_path_parts(call_text: str) -> tuple[str, str, bool]:
    match = re.search(r"\b(path_[A-Za-z0-9_]+)\s*\((.*?)\)", call_text, flags=re.DOTALL)
    if not match:
        return "", "", False
    helper = match.group(1)
    args = match.group(2)
    strings = [
        part
        for groups in re.findall(r'"([^"]*)"|\'([^\']*)\'', args)
        for part in groups
        if part
    ]
    return helper, "/".join(strings), bool(re.search(r"\b(paste0|paste|sprintf|glue)\s*\(", args))


def assigned_object(call_text: str) -> str:
    match = re.match(r"\s*([A-Za-z.][A-Za-z0-9._]*)\s*(?:<-|=)\s*", call_text)
    return match.group(1) if match else ""


def artifact_role(kind: str) -> str:
    if kind.startswith("read") or kind == "st_read":
        return "read"
    if kind.startswith("write") or kind in {"ggsave", "etable", "cat("}:
        return "write"
    return ""


def extract_artifacts(
    source_name: str,
    lines: list[str],
    parse_df: pl.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    calls = parse_df.filter(
        (pl.col("terminal"))
        & (pl.col("token") == "SYMBOL_FUNCTION_CALL")
        & (pl.col("text").is_in(sorted(ARTIFACT_CALLS)))
    )
    for token_row in calls.to_dicts():
        expr = enclosing_expr(parse_df, token_row)
        if expr is None:
            continue
        call_text = text_slice(
            lines,
            int(expr["line1"]),
            int(expr["col1"]),
            int(expr["line2"]),
            int(expr["col2"]),
        ).strip()
        helper, relative_path, path_is_dynamic = extract_path_parts(call_text)
        kind = str(token_row["text"])
        if kind == "cat":
            kind = "cat("
        rows.append(
            {
                "source": source_name,
                "line": int(token_row["line1"]),
                "kind": kind,
                "role": artifact_role(kind),
                "assigned_object": assigned_object(call_text),
                "path_helper": helper,
                "relative_path": relative_path,
                "path_is_dynamic": path_is_dynamic,
                "text": call_text,
            }
        )

    if rows:
        return rows

    for idx, raw in enumerate(lines, start=1):
        text = raw.rstrip("\r\n")
        for match in ARTIFACT_RE.finditer(text):
            rows.append(
                {
                    "source": source_name,
                    "line": idx,
                    "kind": match.group(1),
                    "role": artifact_role(match.group(1)),
                    "assigned_object": assigned_object(text),
                    "path_helper": "",
                    "relative_path": "",
                    "path_is_dynamic": False,
                    "text": text.strip(),
                }
            )
    return rows


def collect_libraries(lines_by_source: dict[str, list[str]]) -> list[str]:
    libs: list[str] = []
    for source in SOURCE_ORDER:
        for line in lines_by_source[source]:
            match = LIBRARY_RE.match(line)
            if match and match.group(1) not in libs:
                libs.append(match.group(1))
    return libs


def source_sha(lines: Iterable[str]) -> str:
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def generated_module_path(path: str) -> str:
    if path.startswith(GENERATED_PREFIX):
        return path
    return f"{GENERATED_PREFIX}{path}"


def unprefixed_module_path(path: str) -> str:
    if path.startswith(GENERATED_PREFIX) and len(path) > 1 and path[1].isdigit():
        return path[1:]
    return path


def generated_header(module: dict[str, object]) -> str:
    return (
        "# Generated by code/split_do_r_scripts.py.\n"
        "# Edit the source Do script or split specification, then regenerate.\n"
        f"# Source: Do/{module['source']} lines {module['start_line']}-{module['end_line']}\n"
        f"# Source SHA256: {module['source_sha256']}\n\n"
        "if (!exists(\"path_processed\", mode = \"function\")) {\n"
        "  local({\n"
        "    split_current_file <- function() {\n"
        "      frames <- sys.frames()\n"
        "      for (idx in rev(seq_along(frames))) {\n"
        "        ofile <- frames[[idx]]$ofile\n"
        "        if (!is.null(ofile)) {\n"
        "          return(normalizePath(ofile, winslash = \"/\", mustWork = FALSE))\n"
        "        }\n"
        "      }\n"
        "\n"
        "      file_arg <- grep(\"^--file=\", commandArgs(FALSE), value = TRUE)\n"
        "      if (length(file_arg) > 0) {\n"
        "        return(normalizePath(sub(\"^--file=\", \"\", file_arg[[1]]), winslash = \"/\", mustWork = FALSE))\n"
        "      }\n"
        "\n"
        "      normalizePath(getwd(), winslash = \"/\", mustWork = FALSE)\n"
        "    }\n"
        "\n"
        f"    source(file.path(dirname(split_current_file()), \"{SETUP_FILE}\"))\n"
        "  })\n"
        "}\n\n"
    )


def setup_r(libraries: list[str]) -> str:
    library_lines = "\n".join(f"suppressPackageStartupMessages(library({lib}))" for lib in libraries)
    return f'''# Generated by code/split_do_r_scripts.py.
# Shared setup for generated split R modules.

split_current_file <- function() {{
  frames <- sys.frames()
  for (idx in rev(seq_along(frames))) {{
    ofile <- frames[[idx]]$ofile
    if (!is.null(ofile)) {{
      return(normalizePath(ofile, winslash = "/", mustWork = FALSE))
    }}
  }}

  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg) > 0) {{
    return(normalizePath(sub("^--file=", "", file_arg[[1]]), winslash = "/", mustWork = FALSE))
  }}

  normalizePath(getwd(), winslash = "/", mustWork = FALSE)
}}

split_dir <- dirname(split_current_file())
source(file.path(split_dir, "paths.R"))
ensure_project_dirs()
options(stringsAsFactors = FALSE)

{library_lines}

split_analysis_sample <- function(county_df) {{
  county_df %>%
    filter(
      any_cropland_2007 == 1,
      county_simple_treatment_groups != "always takers"
    )
}}

split_county_map <- function(cnt_shp = NULL) {{
  if (is.null(cnt_shp)) {{
    zip_path <- path_raw("county_shapefile", "tl_2020_us_county.zip")
    unzip(zip_path, exdir = tempdir())
    cnt_shp <- st_read(file.path(tempdir(), "tl_2020_us_county.shp"))
  }}

  county_map <- cnt_shp %>%
    mutate(statefip = as.numeric(STATEFP)) %>%
    filter(statefip <= 56 & statefip != 2 & statefip != 15)

  county_map <- st_simplify(
    county_map,
    preserveTopology = FALSE,
    dTolerance = 1000
  )
  county_map$countyfips <- as.numeric(str_c(
    county_map$STATEFP,
    county_map$COUNTYFP
  ))
  county_map
}}

split_load_analysis_inputs <- function(
  env = parent.frame(),
  include_date = FALSE,
  include_shape = FALSE,
  include_county_map = FALSE,
  include_h2a_data_ts = FALSE,
  include_h2a_data = FALSE,
  include_aewr_data_full = FALSE,
  include_county_df = FALSE,
  include_samples = FALSE
) {{
  if (include_date) {{
    assign(
      "date",
      paste0(
        substr(Sys.Date(), 1, 4),
        substr(Sys.Date(), 6, 7),
        substr(Sys.Date(), 9, 10)
      ),
      envir = env
    )
  }}

  if (include_shape || include_county_map) {{
    zip_path <- path_raw("county_shapefile", "tl_2020_us_county.zip")
    unzip(zip_path, exdir = tempdir())
    cnt_shp <- st_read(file.path(tempdir(), "tl_2020_us_county.shp"))
    assign("zip_path", zip_path, envir = env)
    assign("cnt_shp", cnt_shp, envir = env)
    if (include_county_map) {{
      assign("county_map", split_county_map(cnt_shp), envir = env)
    }}
  }}

  if (include_h2a_data_ts) {{
    assign("h2a_data_ts", read_parquet(path_processed("h2a_data_ts.parquet")), envir = env)
  }}
  if (include_h2a_data) {{
    assign("h2a_data", read_parquet(path_processed("h2a_data.parquet")), envir = env)
  }}
  if (include_aewr_data_full) {{
    assign("aewr_data_full", read_parquet(path_processed("aewr_data_full.parquet")), envir = env)
  }}
  if (include_county_df || include_samples) {{
    county_df <- read_parquet(path_processed("county_df_analysis_year.parquet"))
    assign("county_df", county_df, envir = env)
  }} else if (exists("county_df", envir = env, inherits = FALSE)) {{
    county_df <- get("county_df", envir = env)
  }}

  if (include_samples) {{
    samp_base <- split_analysis_sample(county_df)
    assign("samp_base", samp_base, envir = env)
    assign("samp_no_border", samp_base %>% filter(border_cz == 0), envir = env)
  }}

  invisible(env)
}}

split_prepare_bea_fips_xwalk <- function(full_county_set) {{
  bea_fips_xwalk <- read_csv(
    file = path_raw("geographic_crosswalks", "phil", "bea_fips_xwalk.csv")
  )

  county_list <- unique(select(full_county_set, fipscounty, countyname)) %>%
    mutate(indata = 1)

  bea_fips_xwalk <- merge(
    x = bea_fips_xwalk,
    y = county_list,
    by.x = "realfips",
    by.y = "fipscounty",
    all.x = TRUE,
    all.y = FALSE
  )

  bea_fips_xwalk %>%
    filter(county == 1) %>%
    select(realfips, beafips)
}}

split_apply_bea_fips_xwalk <- function(data, bea_fips_xwalk) {{
  data <- merge(
    x = data,
    y = bea_fips_xwalk,
    by.x = "countyfips",
    by.y = "beafips",
    all.x = TRUE,
    all.y = FALSE
  )

  data %>%
    rename(oldfips = countyfips) %>%
    mutate(countyfips = ifelse(!is.na(realfips), realfips, oldfips)) %>%
    select(-oldfips, -realfips)
}}

split_save_aewr_region_ts <- function(aewr_data, y_var, y_label, filename_prefix) {{
  for (i in sort(unique(aewr_data$aewr_region_num))) {{
    plot <- ggplot(
      data = subset(aewr_data, aewr_region_num == i),
      aes(x = year, y = .data[[y_var]])
    ) +
      geom_line() +
      labs(title = paste0("AEWR Region Number: ", i)) +
      xlab(y_label)

    ggsave(
      filename = path_figures("aewr_ts", paste0(filename_prefix, i, ".png")),
      plot,
      device = "png"
    )
  }}
  invisible(NULL)
}}
'''


def runner_r(modules: list[dict[str, object]]) -> str:
    groups: dict[str, list[str]] = {}
    for module in modules:
        groups.setdefault(str(module["group"]), []).append(str(module["path"]))

    group_lines = []
    for group, paths in groups.items():
        quoted = ", ".join(json.dumps(path) for path in paths)
        group_lines.append(f'  {json.dumps(group)} = c({quoted})')

    return f'''# Generated by code/split_do_r_scripts.py.
# Runs the split R pipeline in the same order as the legacy Do master.

split_current_file <- function() {{
  frames <- sys.frames()
  for (idx in rev(seq_along(frames))) {{
    ofile <- frames[[idx]]$ofile
    if (!is.null(ofile)) {{
      return(normalizePath(ofile, winslash = "/", mustWork = FALSE))
    }}
  }}

  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg) > 0) {{
    return(normalizePath(sub("^--file=", "", file_arg[[1]]), winslash = "/", mustWork = FALSE))
  }}

  normalizePath(getwd(), winslash = "/", mustWork = FALSE)
}}

split_dir <- dirname(split_current_file())
source(file.path(split_dir, "{SETUP_FILE}"))

source_in_group <- function(files, group_name) {{
  message("Running split group: ", group_name)
  group_env <- new.env(parent = globalenv())
  for (file in files) {{
    path <- file.path(split_dir, file)
    message("  source ", basename(path))
    sys.source(path, envir = group_env, keep.source = TRUE)
  }}
  invisible(group_env)
}}

pipeline_groups <- list(
{",\n".join(group_lines)}
)

for (group_name in names(pipeline_groups)) {{
  source_in_group(pipeline_groups[[group_name]], group_name)
}}
'''


def build_modules(lines_by_source: dict[str, list[str]]) -> list[dict[str, object]]:
    modules: list[dict[str, object]] = []
    for spec in SPLIT_SPECS:
        lines = lines_by_source[spec.source]
        start = resolve_line(spec.start, lines, spec.source)
        if spec.end_before is None:
            end = len(lines)
        else:
            end = find_header_line(lines, spec.end_before, spec.source) - 1
        if start > end:
            raise ValueError(f"Invalid split range for {spec.path}: {start}-{end}")
        original = lines[start - 1 : end]
        content = "".join(original)
        modules.append(
            {
                "path": generated_module_path(spec.path),
                "source": spec.source,
                "group": spec.group,
                "start_line": start,
                "end_line": end,
                "line_count": end - start + 1,
                "source_sha256": source_sha(original),
                "generated_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "content": content if content.endswith("\n") else f"{content}\n",
            }
        )
    return modules


def apply_patches_to_modules(modules: list[dict[str, object]], patches: pl.DataFrame) -> list[dict[str, object]]:
    patched_modules: list[dict[str, object]] = []
    for module in modules:
        content, patch_count, patch_match_count, patch_change_count = apply_module_patches(
            str(module["path"]),
            str(module["content"]),
            patches,
        )
        patched = dict(module)
        patched["baseline_sha256"] = module["generated_sha256"]
        patched["patch_count"] = patch_count
        patched["patch_match_count"] = patch_match_count
        patched["patch_change_count"] = patch_change_count
        patched["generated_sha256"] = hashlib.sha256(content.encode("utf-8")).hexdigest()
        patched["content"] = content
        patched_modules.append(patched)
    return patched_modules


def parse_check(paths: Iterable[Path]) -> None:
    for path in paths:
        proc = subprocess.run(
            ["Rscript", "-e", "args <- commandArgs(TRUE); parse(file = args[[1]])", str(path)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Generated R parse check failed for {path}:\n{proc.stderr}")


def is_generated_out_dir(out_dir: Path) -> bool:
    manifest_path = out_dir / "_metadata" / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return manifest.get("generated_by") == GENERATOR_ID


def generated_output_names(modules: list[dict[str, object]]) -> set[str]:
    names = {SETUP_FILE, RUNNER_FILE, "00_setup.R", "99_run_all.R"}
    for module in modules:
        path = str(module["path"])
        names.add(path)
        names.add(unprefixed_module_path(path))
    return names


def write_outputs(
    out_dir: Path,
    modules: list[dict[str, object]],
    sections_df: pl.DataFrame,
    artifacts_df: pl.DataFrame,
    module_artifacts_df: pl.DataFrame,
    tokens_df: pl.DataFrame,
    libraries: list[str],
    patches_df: pl.DataFrame,
    patch_path: Path,
    force: bool,
    dry_run: bool,
    skip_parse_check: bool,
) -> None:
    metadata_dir = out_dir / "_metadata"

    if dry_run:
        print(f"Would write {len(modules)} modules to {out_dir}")
        return

    if out_dir.exists():
        for name in generated_output_names(modules):
            target = out_dir / name
            if target.exists():
                target.unlink()
        if metadata_dir.exists():
            shutil.rmtree(metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / SETUP_FILE).write_text(setup_r(libraries), encoding="utf-8")
    for module in modules:
        path = out_dir / str(module["path"])
        body = generated_header(module) + str(module["content"])
        path.write_text(body, encoding="utf-8")
    (out_dir / RUNNER_FILE).write_text(runner_r(modules), encoding="utf-8")

    module_df = pl.DataFrame(
        [
            {key: value for key, value in module.items() if key != "content"}
            for module in modules
        ]
    )
    module_df.write_parquet(metadata_dir / "modules.parquet")
    sections_df.write_parquet(metadata_dir / "sections.parquet")
    artifacts_df.write_parquet(metadata_dir / "artifacts.parquet")
    module_artifacts_df.write_parquet(metadata_dir / "module_artifacts.parquet")
    tokens_df.write_parquet(metadata_dir / "tokens.parquet")
    patches_df.write_parquet(metadata_dir / "patches.parquet")

    manifest = {
        "generated_by": GENERATOR_ID,
        "out_dir": str(out_dir),
        "patch_path": str(patch_path),
        "libraries": libraries,
        "modules": module_df.to_dicts(),
    }
    (metadata_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not skip_parse_check:
        parse_check([out_dir / SETUP_FILE, *(out_dir / str(m["path"]) for m in modules), out_dir / RUNNER_FILE])


def strip_generated_header(text: str) -> str:
    marker = "\n\n"
    if text.startswith("# Generated by ") and marker in text:
        return text.split(marker, 1)[1]
    return text


def source_slice_text(do_dir: Path, module_row: dict[str, object]) -> str:
    source_path = do_dir / str(module_row["source"])
    lines = source_path.read_text(encoding="utf-8-sig").splitlines()
    start = int(module_row["start_line"])
    end = int(module_row["end_line"])
    return "\n".join(lines[start - 1 : end]) + "\n"


def module_diff(
    module_path: str,
    project_root: Path | None = None,
    out_dir: Path | None = None,
    include_generated_header: bool = False,
) -> str:
    root = (project_root or project_root_from_script()).resolve()
    split_dir = (out_dir or Path(__file__).resolve().parent).resolve()
    do_dir = root / "Do"
    modules = pl.read_parquet(split_dir / "_metadata" / "modules.parquet")
    rows = modules.filter(pl.col("path").is_in([module_path, generated_module_path(module_path)])).to_dicts()
    if not rows:
        raise ValueError(f"Unknown module: {module_path}")
    row = rows[0]
    original = source_slice_text(do_dir, row)
    final_module_path = str(row["path"])
    final = (split_dir / final_module_path).read_text(encoding="utf-8")
    if not include_generated_header:
        final = strip_generated_header(final)
    return "\n".join(
        difflib.unified_diff(
            original.splitlines(),
            final.splitlines(),
            fromfile=f"Do/{row['source']}:{row['start_line']}-{row['end_line']}",
            tofile=final_module_path,
            lineterm="",
        )
    )


def generate_split(
    project_root: Path | None = None,
    do_dir: Path | None = None,
    out_dir: Path | None = None,
    patch_path: Path | None = None,
    force: bool = False,
    dry_run: bool = False,
    skip_parse_check: bool = False,
) -> dict[str, object]:
    root = (project_root or project_root_from_script()).resolve()
    do_dir = (do_dir or root / "Do").resolve()
    out_dir = (out_dir or Path(__file__).resolve().parent).resolve()
    patch_path = (patch_path or default_patch_path(root)).resolve()

    lines_by_source = {source: read_source(do_dir / source) for source in SOURCE_ORDER}

    token_frames = []
    section_rows: list[dict[str, object]] = []
    artifact_rows: list[dict[str, object]] = []
    for source in SOURCE_ORDER:
        source_path = do_dir / source
        parsed = r_parse_data(source_path).with_columns(pl.lit(source).alias("source"))
        token_frames.append(parsed)
        section_rows.extend(extract_sections(source, lines_by_source[source]))
        artifact_rows.extend(extract_artifacts(source, lines_by_source[source], parsed))

    tokens_df = pl.concat(token_frames, how="diagonal_relaxed")
    sections_df = pl.DataFrame(section_rows) if section_rows else pl.DataFrame()
    artifacts_df = pl.DataFrame(artifact_rows) if artifact_rows else pl.DataFrame()

    modules = build_modules(lines_by_source)
    patches_df = read_patches(patch_path)
    modules = apply_patches_to_modules(modules, patches_df)
    libraries = collect_libraries(lines_by_source)

    module_artifact_rows: list[dict[str, object]] = []
    for module in modules:
        module_lines = str(module["content"]).splitlines(keepends=True)
        parsed_module = r_parse_data_text(str(module["content"]))
        for row in extract_artifacts(str(module["path"]), module_lines, parsed_module):
            row["module"] = str(module["path"])
            row["legacy_source"] = str(module["source"])
            module_artifact_rows.append(row)
    module_artifacts_df = (
        pl.DataFrame(module_artifact_rows)
        if module_artifact_rows
        else pl.DataFrame(schema=artifacts_df.schema)
    )

    summary = {
        "project_root": str(root),
        "do_dir": str(do_dir),
        "out_dir": str(out_dir),
        "patch_path": str(patch_path),
        "parsed_tokens": tokens_df.height,
        "detected_sections": sections_df.height if sections_df.width else 0,
        "detected_artifacts": artifacts_df.height if artifacts_df.width else 0,
        "detected_module_artifacts": module_artifacts_df.height if module_artifacts_df.width else 0,
        "patches": patches_df.height if patches_df.width else 0,
        "enabled_patches": patches_df.filter(pl.col("enabled")).height if patches_df.width else 0,
        "generated_modules": len(modules),
        "modules": [{key: value for key, value in module.items() if key != "content"} for module in modules],
    }

    print(f"Project root: {root}")
    print(f"Do dir:       {do_dir}")
    print(f"Out dir:      {out_dir}")
    print(f"Patch path:   {patch_path}")
    print(f"Parsed tokens: {tokens_df.height:,}")
    print(f"Detected sections: {sections_df.height if sections_df.width else 0:,}")
    print(f"Detected artifacts: {artifacts_df.height if artifacts_df.width else 0:,}")
    print(f"Detected module artifacts: {module_artifacts_df.height if module_artifacts_df.width else 0:,}")
    print(f"Enabled patches: {summary['enabled_patches']:,}")
    print(f"Generated modules: {len(modules):,}")
    for module in modules:
        print(
            f"  {module['path']}: {module['source']} "
            f"lines {module['start_line']}-{module['end_line']}"
        )

    write_outputs(
        out_dir=out_dir,
        modules=modules,
        sections_df=sections_df,
        artifacts_df=artifacts_df,
        module_artifacts_df=module_artifacts_df,
        tokens_df=tokens_df,
        libraries=libraries,
        patches_df=patches_df,
        patch_path=patch_path,
        force=force,
        dry_run=dry_run,
        skip_parse_check=skip_parse_check,
    )
    return summary


def main() -> int:
    args = parse_args()
    generate_split(
        project_root=args.project_root,
        do_dir=args.do_dir,
        out_dir=args.out_dir,
        patch_path=args.patches,
        force=args.force,
        dry_run=args.dry_run,
        skip_parse_check=args.skip_parse_check,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
