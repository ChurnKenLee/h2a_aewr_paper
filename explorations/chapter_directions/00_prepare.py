"""Resolve scratch dependencies with their owners and record input provenance."""

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

    from h2a.paths import INTERMEDIATE, PROCESSED, ROOT

    return INTERMEDIATE, PROCESSED, ROOT, Path, UTC, datetime, hashlib, json, os, shutil, subprocess


@app.cell
def _(Path, ROOT, os, subprocess):
    scratch = Path(os.environ["CHAPTER_SCRATCH"]).resolve()
    if scratch == ROOT or ROOT in scratch.parents:
        raise ValueError("Scratch must be outside the repository")
    for _folder in [scratch, scratch / "filer", scratch / "benchmark", scratch / "logs"]:
        _folder.mkdir(parents=True, exist_ok=True)

    def run(command, log, extra_env=None):
        environment = dict(os.environ)
        environment.update(extra_env or {})
        with (scratch / "logs" / log).open("w") as stream:
            subprocess.run(command, cwd=ROOT, env=environment, check=True, stdout=stream, stderr=subprocess.STDOUT)

    return run, scratch


@app.cell
def _(Path, ROOT, hashlib, os, run, scratch, shutil):
    # Copy only known named inputs, never a whole prior scratch tree.
    copied = []
    for variable, folder, names in [
        ("CD_FILER_SOURCE", "filer", ["filer_cases.parquet", "filer_ids.parquet"]),
        ("CD_BENCHMARK_SOURCE", "benchmark", ["track1_pre_panel.csv", "track1_treatment_schedule.csv", "track1_post_estimates.csv"]),
        ("CD_EARNINGS_SOURCE", "benchmark", ["qcew_crop_county_quarter_2014_2024.parquet"]),
    ]:
        if os.environ.get(variable):
            for name in names:
                source = Path(os.environ[variable]) / name
                destination = scratch / folder / name
                if not destination.exists():
                    shutil.copy2(source, destination)
                    copied.append({"source": str(source), "destination": str(destination)})
    if not all((scratch / "filer" / name).exists() for name in ["filer_cases.parquet", "filer_ids.parquet"]):
        for notebook in ["01_filer_records", "02_rule_harness"]:
            flat = scratch / "filer" / f"{notebook}.flat.py"
            run(["uv", "run", "--no-sync", "marimo", "export", "script", f"explorations/employer_panel/{notebook}.py", "-o", str(flat), "--force"], f"{notebook}_export.log")
            run(["uv", "run", "--no-sync", "python", str(flat)], f"{notebook}.log", {"EMPLOYER_PANEL_SCRATCH": str(scratch / "filer")})
    if not all((scratch / "benchmark" / name).exists() for name in ["track1_pre_panel.csv", "track1_treatment_schedule.csv"]):
        run(["Rscript", "--vanilla", "explorations/identification/04_track1_power.R", str(scratch / "benchmark")], "rebuild_track1_dependencies.log")
    if not (scratch / "benchmark/track1_post_estimates.csv").exists():
        run(["Rscript", "--vanilla", "explorations/identification/08_track1_estimate.R", str(scratch / "benchmark")], "rebuild_track1_estimates.log")
    if not (scratch / "benchmark/qcew_crop_county_quarter_2014_2024.parquet").exists():
        run(["Rscript", "--vanilla", "explorations/episode_2025/06_qcew_quarterly_extract.R", str(scratch / "benchmark"), "2014", "2024"], "rebuild_original_earnings.log")
    run(["Rscript", "--vanilla", "explorations/episode_2025/09_domestic_wage_exposure.R", str(scratch / "benchmark")], "earnings_exposure.log")
    exposure = scratch / "benchmark/domestic_wage_exposure_frozen.csv"
    with exposure.open("rb") as _stream:
        if hashlib.file_digest(_stream, "sha256").hexdigest() != "6bd476d1bf0c4cb8d98e3072d1b7b1d6375176d18dffc62a71d6bb6e0e25eda3":
            raise ValueError("Rebuilt exposure differs from the frozen registry")
    frozen = ROOT / "explorations/episode_2025/domestic_wage_exposure_frozen.csv"
    if not frozen.exists():
        shutil.copy2(exposure, frozen)
    return (copied,)


@app.cell
def _(INTERMEDIATE, PROCESSED, ROOT, UTC, copied, datetime, hashlib, json, scratch, subprocess):
    inputs = [PROCESSED / "county_year_panel.parquet"]
    inputs += [INTERMEDIATE / name for name in [
        "h2a_allocated_worksites.parquet", "h2a_disclosure_records.parquet", "h2a_employer_record_membership.parquet",
        "aewr_rate_schedule.parquet", "aewr_occupation_skill_rates.parquet", "state_real_minwages.parquet",
        "qwi_county_ag_quarterly_earnings.parquet",
    ]]
    inputs += list((scratch / "filer").glob("*.parquet"))
    inputs += [scratch / "benchmark" / name for name in ["track1_pre_panel.csv", "track1_treatment_schedule.csv", "track1_post_estimates.csv", "domestic_wage_exposure_frozen.csv", "qcew_crop_county_quarter_2014_2024.parquet"]]
    entries = []
    for path in inputs:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        entries.append({"path": str(path), "sha256": digest, "bytes": path.stat().st_size, "local_mtime": path.stat().st_mtime})
    provenance = {"recorded_at": datetime.now(UTC).isoformat(), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "inputs": entries, "copied": copied, "outcomes_previously_inspected": ["Existing FY2026 wage and worksite decomposition results", "Wage extensions from this run as documented in implementation_audit"],
                  "earnings_access": "Pre-policy only; raw post-policy archives are not opened by preparation"}
    manifest = scratch / "inputs.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        previous_hashes = {entry["path"]: entry["sha256"] for entry in previous["inputs"]}
        for entry in entries:
            if entry["path"] in previous_hashes and previous_hashes[entry["path"]] != entry["sha256"]:
                raise ValueError(f"Input changed since preparation: {entry['path']}")
    else:
        manifest.write_text(json.dumps(provenance, indent=2))
    print(f"Verified {len(entries)} inputs")


if __name__ == "__main__":
    app.run()
