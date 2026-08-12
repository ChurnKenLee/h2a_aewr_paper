"""Artifact paths, atomic writes, and reproducibility hashes for MCW v4."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import polars as pl
import scipy

from h2a.paths import INTERMEDIATE, PROCESSED, ROOT, TABLES

BRANCH_DIR = ROOT / "code" / "designs" / "mundlak_chamberlain"
SOURCE_PANEL = PROCESSED / "county_year_panel.parquet"
ANALYSIS_PANEL = PROCESSED / "mundlak_chamberlain_county_year_v4.parquet"
REGISTRY_PATH = TABLES / "mc_v4_specification_registry.csv"
RESULTS_PATH = TABLES / "mc_v4_constructed_estimands.csv"
DIAGNOSTICS_PATH = TABLES / "mc_v4_diagnostics.csv"
COEFFICIENTS_PATH = TABLES / "mc_v4_coefficients.csv"
MANIFEST_PATH = INTERMEDIATE / "mundlak_chamberlain_v4_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def code_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted((BRANCH_DIR / "mcw").glob("*.py")):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment_record() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "duckdb": duckdb.__version__,
        "polars": pl.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    }


def atomic_write_frame(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.suffix == ".parquet":
        frame.write_parquet(temporary)
    elif path.suffix == ".csv":
        frame.write_csv(temporary)
    else:
        raise ValueError(f"Unsupported frame artifact extension: {path.suffix}")
    temporary.replace(path)


def atomic_write_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
    )
    temporary.replace(path)


def panel_key_hash(frame: pl.DataFrame) -> str:
    keys = frame.select("county_fips", "year").sort("county_fips", "year")
    return hashlib.sha256(keys.write_csv().encode()).hexdigest()
