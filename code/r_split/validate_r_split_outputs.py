#!/usr/bin/env python3
"""Snapshot and validate split-R parquet outputs against legacy-script outputs.

Typical workflow:

1. Run the legacy Do/*.R scripts and snapshot their data/processed parquet files.
2. Run c99_run_all.R from the split-script folder.
3. Validate the regenerated split outputs against the snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from h2a.paths import ROOT


DEFAULT_OUTPUTS = (
    "fred_state_minwages.parquet",
    "nass_fisher_price_index.parquet",
    "cz_file_2010.parquet",
    "cz_file_2010_small.parquet",
    "ppi_2012.parquet",
    "state_real_minwages.parquet",
    "h2a_predict.parquet",
    "h2a_data.parquet",
    "h2a_data_ts.parquet",
    "cdl_cropshares.parquet",
    "census_ag_cropland_year.parquet",
    "census_ag_cropland_2007_year.parquet",
    "h2a_data_year.parquet",
    "aewr_data_year.parquet",
    "aewr_data_full.parquet",
    "bea_caemp25n_data_year.parquet",
    "bea_cainc45_data_year.parquet",
    "census_pop_ests_year.parquet",
    "county_df_year.parquet",
    "county_df_analysis_year.parquet",
)


def project_root() -> Path:
    return ROOT


def parquet_signature(path: Path) -> dict[str, object]:
    file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        df = pl.read_parquet(path)
    except Exception as exc:
        metadata = pq.read_metadata(path)
        schema = metadata.schema.to_arrow_schema()
        return {
            "file": path.name,
            "rows": metadata.num_rows,
            "columns": metadata.num_columns,
            "schema": {field.name: str(field.type) for field in schema},
            "column_names": schema.names,
            "null_counts": {},
            "data_sha256": "",
            "file_sha256": file_sha256,
            "signature_method": f"parquet_metadata_file_hash_fallback:{type(exc).__name__}",
        }

    row_hashes = df.hash_rows(seed=0).to_numpy().tobytes()
    return {
        "file": path.name,
        "rows": df.height,
        "columns": df.width,
        "schema": {name: str(dtype) for name, dtype in df.schema.items()},
        "column_names": df.columns,
        "null_counts": df.null_count().row(0, named=True),
        "data_sha256": hashlib.sha256(row_hashes).hexdigest(),
        "file_sha256": file_sha256,
        "signature_method": "polars_row_hash",
    }


def collect_signatures(processed_dir: Path, outputs: tuple[str, ...]) -> dict[str, object]:
    signatures: dict[str, object] = {}
    missing: list[str] = []
    for name in outputs:
        path = processed_dir / name
        if path.exists():
            signatures[name] = parquet_signature(path)
        else:
            missing.append(name)
    return {"processed_dir": str(processed_dir), "signatures": signatures, "missing": missing}


def snapshot(args: argparse.Namespace) -> int:
    processed_dir = args.project_root / "data" / "processed"
    outputs = tuple(args.outputs or DEFAULT_OUTPUTS)
    baseline = collect_signatures(processed_dir, outputs)
    args.baseline.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    print(f"Wrote baseline for {len(baseline['signatures'])} files to {args.baseline}")
    if baseline["missing"]:
        print("Missing files:", ", ".join(baseline["missing"]))
    return 0


def validate(args: argparse.Namespace) -> int:
    processed_dir = args.project_root / "data" / "processed"
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    expected = baseline.get("signatures", {})
    current = collect_signatures(processed_dir, tuple(expected.keys()))
    allow_changed = set(args.allow_changed or [])

    failures: list[str] = []
    for name, expected_sig in expected.items():
        current_sig = current["signatures"].get(name)
        if current_sig is None:
            failures.append(f"{name}: missing current output")
            continue
        fields = ("rows", "columns", "schema", "column_names", "null_counts", "data_sha256", "file_sha256")
        for field in fields:
            if current_sig[field] != expected_sig[field] and name not in allow_changed:
                failures.append(f"{name}: {field} differs")

    for name in current["missing"]:
        failures.append(f"{name}: missing current output")

    if failures:
        print("Validation failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(f"Validation passed for {len(expected)} files.")
    if allow_changed:
        print("Allowed changed files:", ", ".join(sorted(allow_changed)))
    return 0


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("snapshot", "validate"),
        help="Create a legacy-output baseline or validate current outputs against one.",
    )
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(__file__).resolve().parent / "r_split_source_output_signatures.json",
    )
    parser.add_argument("--outputs", nargs="*", help="Optional processed parquet filenames.")
    parser.add_argument(
        "--allow-changed",
        nargs="*",
        default=[],
        help="Output filenames allowed to differ during validation.",
    )
    args = parser.parse_args()
    args.project_root = args.project_root.resolve()
    args.baseline = args.baseline.resolve()
    return args


def main() -> int:
    args = parse_args()
    if args.mode == "snapshot":
        return snapshot(args)
    return validate(args)


if __name__ == "__main__":
    raise SystemExit(main())
