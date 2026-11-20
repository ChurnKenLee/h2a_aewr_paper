"""User-authorized interim earnings from already cached public quarterly files."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import hashlib
    import json
    import os
    import runpy
    import zipfile
    from datetime import UTC, datetime
    from pathlib import Path

    import pandas as pd
    import polars as pl

    from h2a.paths import RAW, ROOT

    return Path, RAW, ROOT, UTC, datetime, hashlib, json, os, pd, pl, runpy, zipfile


@app.cell
def _(Path, ROOT, os, runpy):
    scratch = Path(os.environ["CHAPTER_SCRATCH"]).resolve()
    if scratch == ROOT or ROOT in scratch.parents:
        raise ValueError("Scratch must be outside the repository")
    os.environ["CD_EARNINGS_MODE"] = "test"
    # The existing exported notebook exposes its pure extractor without reading
    # earnings in test mode. Its frozen implementation remains unchanged.
    definitions = runpy.run_path(str(scratch / "05_extract_earnings.flat.py"))
    aggregate = definitions["aggregate_earnings"]
    verify = definitions["verify_hashes"]
    verify(ROOT / "explorations/chapter_directions/earnings_amendment.sha256")
    verify(ROOT / "explorations/chapter_directions/interim_amendment.sha256")
    return aggregate, scratch


@app.cell
def _(RAW, UTC, aggregate, datetime, hashlib, json, pd, pl, scratch, zipfile):
    source_dir = RAW / "qcew_episode_2025_2026"
    acquisition = pd.read_csv(source_dir / "MANIFEST.csv", dtype=str, keep_default_na=False)
    parts = [pl.read_parquet(scratch / "earnings_pre.parquet")]
    provenance = []
    columns = ["area_fips", "own_code", "industry_code", "agglvl_code", "year", "qtr", "disclosure_code",
               "month1_emplvl", "month2_emplvl", "month3_emplvl", "total_qtrly_wages", "avg_wkly_wage"]
    for year in [2025, 2026]:
        archive = source_dir / f"{year}_qtrly_singlefile.zip"
        entry = acquisition.loc[acquisition["file"] == archive.name]
        if len(entry) != 1:
            raise ValueError("Missing or duplicate acquisition metadata")
        with archive.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != entry.iloc[0]["sha256"]:
            raise ValueError(f"Cached archive hash mismatch: {archive.name}")
        selected = []
        with zipfile.ZipFile(archive) as bundle:
            members = [name for name in bundle.namelist() if name.endswith("singlefile.csv")]
            if len(members) != 1:
                raise ValueError("Expected one QCEW single-file CSV")
            with bundle.open(members[0]) as stream:
                for chunk in pd.read_csv(stream, usecols=columns, dtype=str, keep_default_na=False, chunksize=500_000):
                    keep = (chunk["own_code"] == "5") & (
                        (chunk["industry_code"].isin(["111", "112"]) & (chunk["agglvl_code"] == "75"))
                        | ((chunk["industry_code"] == "10") & (chunk["agglvl_code"] == "71")))
                    selected.append(pl.from_pandas(chunk.loc[keep]))
        table = aggregate(pl.concat(selected))
        if set(table["year"].unique()) != {year}:
            raise ValueError("Unexpected year in cached archive")
        parts.append(table)
        provenance.append({**entry.iloc[0].to_dict(), "path": str(archive), "opened_at": datetime.now(UTC).isoformat(),
                           "quarters": sorted(table["qtr"].unique().to_list())})
        print(year, "quarters:", sorted(table["qtr"].unique().to_list()), "records:", table.height, flush=True)
    interim = pl.concat(parts).sort("industry_code", "county_fips", "year", "qtr")
    required = {(2024, 4), (2025, 1), (2025, 4), (2026, 1)}
    for industry in ["111", "112", "10"]:
        have = set(interim.filter(pl.col("industry_code") == industry).select("year", "qtr").unique().iter_rows())
        if not required.issubset(have):
            raise ValueError(f"Declared interim window unavailable for {industry}")
    interim.write_parquet(scratch / "earnings_interim.parquet")
    (scratch / "earnings_interim_inputs.json").write_text(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    app.run()
