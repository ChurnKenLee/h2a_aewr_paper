"""Verify the declared vintage and prepare this design's private work directory."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import hashlib
    import json
    import os
    import shutil
    import subprocess
    from datetime import UTC, datetime
    from pathlib import Path

    from h2a.paths import ROOT

    return ROOT, Path, UTC, datetime, hashlib, json, os, shutil, subprocess


@app.cell
def _(Path, ROOT, hashlib, json, os):
    branch = ROOT / "code/designs/policy_2025"
    work = Path(os.environ["POLICY2025_WORK"]).resolve()
    if work == ROOT or ROOT in work.parents:
        raise ValueError("Work directory must be outside the repository")
    for _folder in [work, work / "filer", work / "logs"]:
        _folder.mkdir(parents=True, exist_ok=True)

    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    vintage = json.loads((branch / "sources/input_vintage.json").read_text())
    for _line in (branch / "registry.sha256").read_text().splitlines():
        _expected, _name = _line.split("  ", 1)
        if digest(ROOT / _name) != _expected:
            raise ValueError(f"Design contract hash mismatch: {_name}")
    for _entry in vintage["inputs"]:
        if digest(ROOT / _entry["path"]) != _entry["sha256"]:
            raise ValueError(f"Input vintage changed; record a dated update before rebuilding: {_entry['path']}")
    return branch, digest, vintage, work


@app.cell
def _(Path, ROOT, branch, digest, os, shutil, subprocess, vintage, work):
    # An optional migration seed saves reconstruction; it never selects a vintage.
    seed = Path(os.environ["POLICY2025_LEGACY_SCRATCH"]).resolve() if os.environ.get("POLICY2025_LEGACY_SCRATCH") else None
    reused = []
    for entry in vintage["reusable_caches"]:
        cached = ROOT / entry["destination"]
        if not cached.exists() and seed is not None:
            cached = seed / entry["relative_path"]
        if cached.exists():
            if digest(cached) != entry["sha256"]:
                raise ValueError(f"Cached dependency does not match frozen vintage: {cached}")
            shutil.copy2(cached, work / entry["relative_path"])
            reused.append({"destination": entry["relative_path"], "source": str(cached), "sha256": entry["sha256"]})
    if not all((work / "filer" / name).exists() for name in ["filer_cases.parquet", "filer_ids.parquet"]):
        for notebook in ["01_filer_records", "02_rule_harness"]:
            flat = work / "filer" / f"{notebook}.flat.py"
            with (work / "logs" / f"{notebook}.log").open("w") as stream:
                subprocess.run(["uv", "run", "--no-sync", "marimo", "export", "script", str(branch / "inputs" / f"{notebook}.py"), "-o", str(flat), "--force"], cwd=ROOT, check=True, stdout=stream, stderr=subprocess.STDOUT)
                subprocess.run(["uv", "run", "--no-sync", "python", str(flat)], cwd=ROOT, env=dict(os.environ, EMPLOYER_PANEL_SCRATCH=str(work / "filer")), check=True, stdout=stream, stderr=subprocess.STDOUT)
    return (reused,)


@app.cell
def _(ROOT, UTC, datetime, json, reused, vintage, work):
    (work / "earnings_pre_inputs.json").write_text(json.dumps([
        {**entry, "file": str(ROOT / entry["path"]), "mode": "pre"}
        for entry in vintage["inputs"] if entry["path"].startswith("data/raw/qcew/")
    ], indent=2))
    (work / "inputs.json").write_text(json.dumps({
        "recorded_at": datetime.now(UTC).isoformat(),
        "design": "policy_2025_v1",
        "inputs": vintage["inputs"],
        "reused": reused,
        "outcomes_previously_inspected": True,
        "vintage": "FY2026 Q4 baseline (2026-11-21 update; 2026-11-25 unified minimum-wage vintage update): full-fiscal-year wage window, exposure unchanged from 2026-11-21, unified A01 FRED state minimum wages",
        "project_root": str(ROOT),
    }, indent=2))
    print(f"Verified {len(vintage['inputs'])} source inputs; reused {len(reused)} declared caches")


if __name__ == "__main__":
    app.run()
