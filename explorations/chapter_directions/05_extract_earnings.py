"""Isolated three-series QCEW extractor; post-policy access is release-gated."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import hashlib
    import importlib.util
    import json
    import os
    import urllib.request
    import zipfile
    from datetime import UTC, datetime
    from pathlib import Path

    import pandas as pd
    import polars as pl

    from h2a.geography import assert_geo_columns
    from h2a.paths import RAW, ROOT

    return UTC, Path, RAW, ROOT, assert_geo_columns, datetime, hashlib, importlib, json, os, pd, pl, urllib, zipfile


@app.cell
def _(Path, ROOT, importlib, os):
    scratch = Path(os.environ["CHAPTER_SCRATCH"]).resolve()
    if scratch == ROOT or ROOT in scratch.parents:
        raise ValueError("CHAPTER_SCRATCH must be outside the repository")
    scratch.mkdir(parents=True, exist_ok=True)
    owner_spec = importlib.util.spec_from_file_location(
        "qcew_owner", ROOT / "code/a01_sources/04_02_qcew_quarterly_employment.py"
    )
    owner = importlib.util.module_from_spec(owner_spec)
    owner_spec.loader.exec_module(owner)
    return owner, scratch


@app.cell
def _(assert_geo_columns, owner, pl):
    def aggregate_earnings(frame):
        keys = ["county_fips", "year", "qtr", "industry_code"]
        selected = frame.filter(
            pl.col("own_code") == "5",
            ((pl.col("industry_code").is_in(["111", "112"])) & (pl.col("agglvl_code") == "75"))
            | ((pl.col("industry_code") == "10") & (pl.col("agglvl_code") == "71")),
            pl.col("area_fips").str.contains(r"^[0-9]{5}$"),
            ~pl.col("area_fips").str.ends_with("000"),
            ~pl.col("area_fips").str.ends_with("999"),
            ~pl.col("area_fips").is_in(owner.NONALLOCATABLE_AREA_FIPS),
        ).with_columns(
            pl.col("area_fips").map_elements(owner._canonical_county, return_dtype=pl.String).alias("county_fips"),
            pl.col("year", "qtr").cast(pl.Int32),
            pl.col("month1_emplvl", "month2_emplvl", "month3_emplvl", "total_qtrly_wages", "avg_wkly_wage").cast(pl.Float64, strict=False),
        ).filter(pl.col("county_fips").is_not_null())
        selected = selected.with_columns(
            (pl.col("area_fips") != pl.col("county_fips")).cast(pl.Int8).alias("priority"),
            ((pl.col("month1_emplvl") + pl.col("month2_emplvl") + pl.col("month3_emplvl")) / 3).alias("emp"),
        ).filter(pl.col("priority") == pl.col("priority").min().over(keys))
        if selected.select("area_fips", "year", "qtr", "industry_code").is_duplicated().any():
            raise ValueError("Duplicate source county-quarter-industry")
        selected = selected.with_columns(
            ((pl.col("disclosure_code") == "") & pl.col("emp").is_finite() & pl.col("avg_wkly_wage").is_finite()).fill_null(False).alias("disclosed"),
            (pl.col("avg_wkly_wage") * pl.col("emp")).alias("wage_numerator"),
        )
        result = selected.group_by(keys).agg(
            pl.col("disclosed").all(), pl.col("emp").sum(),
            pl.col("total_qtrly_wages").sum(), pl.col("wage_numerator").sum(),
            pl.len().alias("components"),
        ).with_columns(
            pl.when(pl.col("disclosed") & (pl.col("emp") > 0))
            .then(pl.col("wage_numerator") / pl.col("emp")).otherwise(None).alias("avg_wkly_wage"),
            pl.when(pl.col("disclosed") & (pl.col("emp") > 0))
            .then(pl.col("total_qtrly_wages") / pl.col("emp") / 13).otherwise(None).alias("weekly_wage_from_totals"),
        ).drop("wage_numerator").sort(keys)
        assert_geo_columns(result, ["county_fips"])
        if result.is_empty() or result.select(keys).is_duplicated().any():
            raise ValueError("Empty or duplicate QCEW output keys")
        return result

    return (aggregate_earnings,)


@app.cell
def _(Path, ROOT, UTC, datetime, hashlib, json, os, scratch):
    def verify_hashes(manifest):
        for line in manifest.read_text().splitlines():
            expected, name = line.split("  ", 1)
            path = ROOT / name
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(f"Hash mismatch: {name}")

    mode = os.environ.get("CD_EARNINGS_MODE", "pre")
    if mode not in {"pre", "lead", "post", "test"}:
        raise ValueError("Unknown earnings mode")
    verify_hashes(ROOT / "explorations/chapter_directions/registry.sha256")
    release = None
    if mode == "post":
        verify_hashes(ROOT / "explorations/chapter_directions/earnings_amendment.sha256")
        if datetime.now(UTC) < datetime(2026, 12, 2, 15, tzinfo=UTC):
            raise ValueError("Awaiting 2026 Q2 publication: do not open post-policy earnings")
        release = json.loads(Path(os.environ["CD_RELEASE_METADATA"]).read_text())
        if release.get("status") != "published" or "2026Q2" not in release.get("published_quarters", []):
            raise ValueError("Publication metadata must confirm the full 2026 Q2 release")
        if not release.get("source_url", "").startswith("https://www.bls.gov/") or not release.get("checked_at"):
            raise ValueError("Missing official publication metadata")
        checked_at = datetime.fromisoformat(release["checked_at"].replace("Z", "+00:00"))
        if checked_at.tzinfo is None or not datetime(2026, 12, 2, 15, tzinfo=UTC) <= checked_at <= datetime.now(UTC):
            raise ValueError("Publication verification timestamp must follow the release and not be in the future")
        (scratch / "release_metadata_used.json").write_text(json.dumps(release, indent=2))
    return mode, release, verify_hashes


@app.cell
def _(RAW, UTC, aggregate_earnings, datetime, hashlib, json, mode, os, pd, pl, scratch, urllib, zipfile):
    if mode == "lead":
        # Separate quarter slices preserve the prohibition on opening an annual
        # archive that also contains post-policy earnings before the release gate.
        lead_parts = []
        lead_manifest = []
        for lead_quarter in [1, 2]:
            lead_url = f"https://data.bls.gov/cew/data/api/2025/{lead_quarter}/industry/111.csv"
            lead_path = scratch / f"qcew_2025Q{lead_quarter}_111.csv"
            if not lead_path.exists():
                if os.environ.get("CD_ACQUIRE_LEAD") != "1":
                    raise ValueError("Supply public 2025Q1/Q2 crop slices or explicitly set CD_ACQUIRE_LEAD=1")
                with urllib.request.urlopen(lead_url, timeout=30) as lead_response:
                    lead_path.write_bytes(lead_response.read())
            lead_frame = pl.from_pandas(pd.read_csv(lead_path, dtype=str, keep_default_na=False))
            if set(lead_frame["year"].unique()) != {"2025"} or set(lead_frame["qtr"].unique()) != {str(lead_quarter)}:
                raise ValueError("Lead slice contains unexpected quarters")
            lead_parts.append(aggregate_earnings(lead_frame))
            lead_manifest.append({"path": str(lead_path), "source_url": lead_url,
                                  "sha256": hashlib.sha256(lead_path.read_bytes()).hexdigest(),
                                  "read_at": datetime.now(UTC).isoformat()})
        lead_history = pl.read_parquet(scratch / "earnings_pre.parquet").filter(pl.col("industry_code") == "111")
        pl.concat([lead_history, *lead_parts]).write_parquet(scratch / "earnings_lead.parquet")
        (scratch / "earnings_lead_inputs.json").write_text(json.dumps(lead_manifest, indent=2))
    if mode in {"pre", "post"}:
        years = range(2014, 2025) if mode == "pre" else range(2023, 2027)
        columns = ["area_fips", "own_code", "industry_code", "agglvl_code", "year", "qtr", "disclosure_code",
                   "month1_emplvl", "month2_emplvl", "month3_emplvl", "total_qtrly_wages", "avg_wkly_wage"]
        tables = []
        manifests = []
        for year in years:
            source_url = f"https://data.bls.gov/cew/data/files/{year}/csv/{year}_qtrly_singlefile.zip"
            last_modified = None
            archive = RAW / "qcew" / f"{year}_qtrly_singlefile.zip"
            if mode == "post":
                archive = scratch / "earnings_vintage" / archive.name
                archive.parent.mkdir(parents=True, exist_ok=True)
                if os.environ.get("CD_ACQUIRE") != "1":
                    raise ValueError("Full-window resume requires CD_ACQUIRE=1 for a consistent current vintage")
                # Only this deliberately invoked full-window command downloads files.
                with urllib.request.urlopen(source_url, timeout=120) as response, archive.open("wb") as target:
                    last_modified = response.headers.get("Last-Modified")
                    while block := response.read(2**20):
                        target.write(block)
            with archive.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            manifests.append({"file": str(archive), "sha256": digest, "source_url": source_url,
                              "last_modified": last_modified, "local_mtime": archive.stat().st_mtime,
                              "read_at": datetime.now(UTC).isoformat(), "mode": mode})
            parts = []
            with zipfile.ZipFile(archive) as bundle:
                members = [name for name in bundle.namelist() if name.endswith("singlefile.csv")]
                if len(members) != 1:
                    raise ValueError("Expected one QCEW single-file member")
                with bundle.open(members[0]) as stream:
                    for chunk in pd.read_csv(stream, usecols=columns, dtype=str, keep_default_na=False, chunksize=500_000):
                        keep = (chunk["own_code"] == "5") & (
                            (chunk["industry_code"].isin(["111", "112"]) & (chunk["agglvl_code"] == "75"))
                            | ((chunk["industry_code"] == "10") & (chunk["agglvl_code"] == "71")))
                        parts.append(pl.from_pandas(chunk.loc[keep]))
            table = aggregate_earnings(pl.concat(parts))
            if mode == "pre" and table["year"].max() > 2024:
                raise ValueError("Pre-policy reader encountered post-policy year")
            tables.append(table)
            print(f"{year}: {table.height} county-quarter-series records", flush=True)
        earnings = pl.concat(tables).sort("industry_code", "county_fips", "year", "qtr")
        if mode == "post":
            required = {(2024, 4), (2025, 1), (2025, 2), (2025, 4), (2026, 1), (2026, 2)}
            for industry in ["111", "112", "10"]:
                have = set(earnings.filter(pl.col("industry_code") == industry).select("year", "qtr").unique().iter_rows())
                if not required.issubset(have):
                    raise ValueError(f"Missing quarter for {industry}; nothing estimated")
        earnings.write_parquet(scratch / f"earnings_{mode}.parquet")
        (scratch / f"earnings_{mode}_inputs.json").write_text(json.dumps(manifests, indent=2))
    # No outputs from this terminal cell.


if __name__ == "__main__":
    app.run()
