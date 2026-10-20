"""Audited annual AEWR references; these are not universal job-level wages."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path

import pdfplumber
import polars as pl
import us
from bs4 import BeautifulSoup

from h2a.geography import assert_geo_columns, normalize_geo_code

CRS_FILE = "20080326_RL32861_9a486634f79a5f9c680cc5ba021e579cc67210a5.pdf"
CRS_URL = "https://www.everycrsreport.com/files/" + CRS_FILE
COMMON_SOC = "45-2041;45-2091;45-2092;45-2093;45-2099;53-7064"


def state_identity(name: str) -> dict:
    name = name.strip().replace(".", "").replace("*", "")
    name = {"Idah": "Idaho", "US Virgin Islands": "Virgin Islands"}.get(name, name)
    state = next((s for s in [*us.STATES_AND_TERRITORIES, us.states.DC]
                  if name in (s.name, s.abbr, s.fips)), None)
    if state is None:
        raise ValueError(f"Unrecognized state in rate table: {name!r}")
    return {"state_name": state.name,
            "state_fips": normalize_geo_code(state.fips, "state_fips")}


def assert_unique(frame: pl.DataFrame, key: list[str]) -> None:
    if frame.is_empty() or frame.select(key).null_count().sum_horizontal().item():
        raise ValueError(f"Empty artifact or missing keys: {key}")
    if frame.select(key).is_duplicated().any():
        raise ValueError(f"Duplicate artifact keys: {key}")


def parse_fls_table(path: Path) -> list[dict]:
    """Require one complete two-column 49-state FLS table; reject wage means."""
    soup = BeautifulSoup(path.read_text(), "xml")
    candidates = []
    for table in soup.find_all("GPOTABLE"):
        title = table.get_text(" ", strip=True).upper()
        if "STATE" not in title or "ADVERSE EFFECT WAGE RATES" not in title:
            continue
        rows = []
        for row in table.find_all("ROW"):
            entries = [ent.get_text(strip=True) for ent in row.find_all("ENT")]
            if len(entries) != 2:
                continue
            try:
                value = float(entries[1].replace("$", "").replace(",", ""))
            except ValueError:
                continue
            identity = state_identity(entries[0])
            if not 0 < value < 100:
                raise ValueError(f"Invalid hourly rate in {path.name}")
            rows.append({**identity, "aewr": value})
        if rows:
            frame = pl.DataFrame(rows)
            assert_unique(frame, ["state_fips"])
            if set(frame["state_fips"]) != {s.fips for s in us.STATES if s.abbr != "AK"}:
                raise ValueError(f"Incomplete/non-FLS state table in {path.name}")
            candidates.append(rows)
    if len(candidates) != 1:
        raise ValueError(f"Expected one statutory FLS table in {path.name}")
    return candidates[0]


def parse_crs(path: Path) -> list[dict]:
    rows = []
    with pdfplumber.open(path) as pdf:
        for page_number in (10, 11):
            for source in pdf.pages[page_number - 1].extract_tables()[0][1:]:
                if len(source) != 20:
                    raise ValueError("CRS Table 1 schema changed")
                identity = state_identity(source[0])
                for year, value in zip(range(1990, 2009), source[1:], strict=True):
                    # Documented PDF extraction defect, not a general substitution.
                    if identity["state_fips"] == "38" and year == 2000 and value == "7/49":
                        value = "7.49"
                    rows.append({**identity, "year": year,
                                 "aewr": float(value.replace("$", "").replace(",", "")),
                                 "aewr_source_document": "20080326_RL32861",
                                 "aewr_source_url": CRS_URL,
                                 "aewr_reference_status": "published_fls_reference"})
    frame = pl.DataFrame(rows)
    assert_unique(frame, ["state_fips", "year"])
    if frame.height != 49 * 19:
        raise ValueError("Incomplete CRS annual history")
    return rows


def parse_nonstatutory_2009(path: Path) -> pl.DataFrame:
    soup = BeautifulSoup(path.read_text(), "xml")
    tables = [t for t in soup.find_all("GPOTABLE")
              if "State and National Average Certified" in t.get_text(" ", strip=True)]
    if len(tables) != 1:
        raise ValueError("2009 application-mean table signature changed")
    rows = []
    for row in tables[0].find_all("ROW"):
        values = [e.get_text(strip=True) for e in row.find_all("ENT")]
        if len(values) != 6 or us.states.lookup(values[0]) is None:
            continue
        try:
            wage = float(values[3].replace("$", "").replace(",", ""))
        except ValueError:
            wage = None
        rows.append({**state_identity(values[0]), "year": 2009,
                     "aewr_nonstatutory_application_wage_2009": wage,
                     "measure": "mean_certified_application_wage_after_2009_01_19",
                     "source_document": "2010-2731", "source_table": "Table 1, column 4",
                     "is_statutory_aewr": False})
    frame = pl.DataFrame(rows)
    assert_unique(frame, ["state_fips", "year"])
    return frame


def validate_schedule(frame: pl.DataFrame) -> None:
    assert_geo_columns(frame, ["state_fips"])
    assert_unique(frame, ["state_fips", "effective_start", "occupation_scope"])
    if frame.filter(pl.col("effective_end_exclusive") <= pl.col("effective_start")).height:
        raise ValueError("Nonpositive rate interval")
    ordered = frame.sort(["state_fips", "occupation_scope", "effective_start"])
    overlap = ordered.with_columns(
        pl.col("effective_end_exclusive").shift().over(
            ["state_fips", "occupation_scope"]).alias("previous_end")
    ).filter(pl.col("effective_start") < pl.col("previous_end"))
    if overlap.height:
        raise ValueError("Overlapping rate intervals")


def interval_rate(schedule: pl.DataFrame, state: str, start: date, end: date) -> dict:
    """Day-weight references on [start,end), retaining uncovered days.

    Does not establish job/cohort eligibility, competing floors or employment.
    """
    state = normalize_geo_code(state, "state_fips")
    if end <= start:
        raise ValueError("Exposure interval must contain at least one day")
    rate_days, covered = 0.0, 0
    for row in schedule.filter(pl.col("state_fips") == state).iter_rows(named=True):
        days = max(0, (min(end, row["effective_end_exclusive"])
                       - max(start, row["effective_start"])).days)
        covered += days
        rate_days += days * row["aewr"]
    if covered > (end - start).days:
        raise ValueError("Ambiguous/overlapping reference intervals")
    return {"reference_rate_day_sum": rate_days, "reference_covered_days": covered,
            "reference_uncovered_days": (end - start).days - covered,
            "reference_mean_rate": rate_days / covered if covered else None}


def parse_skill_workbooks(raw: Path) -> pl.DataFrame:
    """Retain published SOC, skill and worker-category distinctions verbatim."""
    rows = []
    releases = [
        ("DOL-OFLC_2025-2026_AEWRs_FINAL.xlsx", "2025-11", date(2025, 10, 2), date(2026, 8, 3)),
        ("DOL-OFLC_2026-2027_AEWRs_FINAL_valuesonly.xlsx", "2026-08", date(2026, 8, 3), None),
    ]
    for filename, directory, start, end in releases:
        sheets = pl.read_excel(raw / filename, sheet_id=0)
        for sheet, frame in sheets.items():
            combined = "(b)(1)(i)" in sheet
            offset = 2 if combined else 4
            for source in frame.iter_rows():
                if source[0] is None or not re.fullmatch(r"\d{2}", str(source[0])):
                    continue
                identity = state_identity(str(source[1]))
                if identity["state_fips"] != normalize_geo_code(source[0], "state_fips"):
                    raise ValueError("Workbook state name/code disagreement")
                soc = "FIELD_LIVESTOCK_COMBINED" if combined else str(source[2])
                if not combined and not re.fullmatch(r"\d{2}-\d{4}", soc):
                    raise ValueError("Invalid published occupational code")
                for category, category_offset in (("US", 0), ("H2A", 2)):
                    for skill in (1, 2):
                        raw_value = source[offset + category_offset + skill - 1]
                        try:
                            value = float(str(raw_value).replace("$", "").replace(",", ""))
                        except ValueError:
                            value = None
                        rows.append({**identity, "soc_code": soc, "skill_level": skill,
                            "worker_category": category, "hourly_aewr": value,
                            "source_rate_text": str(raw_value), "effective_start": start,
                            "effective_end_exclusive": end, "source_file": filename,
                            "source_sheet": sheet,
                            "source_url": f"https://flag.dol.gov/sites/default/files/{directory}/{filename}",
                            "applicability_status": "job_duties_cohort_skill_and_competing_floors_required"})
    result = pl.DataFrame(rows, infer_schema_length=None)
    assert_unique(result, ["state_fips", "soc_code", "skill_level", "worker_category", "effective_start"])
    assert_geo_columns(result, ["state_fips"])
    return result


def build_aewr_artifacts(raw: Path, registry_path: Path) -> dict:
    registry = pl.read_csv(registry_path, schema_overrides={"year": pl.Int64},
                           try_parse_dates=True)
    assert_unique(registry, ["document_number"])
    annual = parse_crs(raw / CRS_FILE)
    sources = [raw / CRS_FILE, registry_path, raw / "xml/2010-2731.xml",
               raw / "AEWR-2025.xlsx", raw / "xml/03-6559.xml"]
    correction = (raw / "xml/03-6559.xml").read_text()
    if not all(word in correction for word in ("Hawaii", "$9.29", "$9.42")):
        raise ValueError("Hawaii correction source changed")
    notice_rates = {}
    for record in registry.iter_rows(named=True):
        doc, year = record["document_number"], record["year"]
        path = raw / record["source_file"]
        sources.append(path)
        if year >= 2000:
            rows = parse_fls_table(path)
            if doc == "03-4500":
                for row in rows:
                    if row["state_fips"] == "15":
                        row["aewr"] = 9.42
            notice_rates[year] = rows
        else:
            notice_rates[year] = [
                {k: row[k] for k in ("state_fips", "state_name", "aewr")}
                for row in annual if row["year"] == year]
        if year >= 2010:
            annual.extend({**row, "year": year, "aewr_source_document": doc,
                           "aewr_source_url": record["source_url"],
                           "aewr_reference_status": "published_fls_reference"}
                          for row in notice_rates[year])
    annual.extend({**state_identity(s.name), "year": 2009, "aewr": None,
                   "aewr_source_document": "2010-2731",
                   "aewr_source_url": "https://www.govinfo.gov/content/pkg/FR-2010-02-12/pdf/2010-2731.pdf",
                   "aewr_reference_status": "oes_area_occupation_skill_determination_unavailable"}
                  for s in us.STATES if s.abbr != "AK")
    nonstatutory = parse_nonstatutory_2009(raw / "xml/2010-2731.xml")
    annual = pl.DataFrame(annual).join(
        nonstatutory.select("state_fips", "year", "aewr_nonstatutory_application_wage_2009"),
        on=["state_fips", "year"], how="left", validate="1:1").sort(["state_fips", "year"])
    assert_unique(annual, ["state_fips", "year"])
    assert_geo_columns(annual, ["state_fips"])
    if annual.height != 49 * 36 or annual.filter(pl.col("year") == 2009)["aewr"].null_count() != 49:
        raise ValueError("Unexpected annual coverage or nonstatutory contamination")
    # Validation never selects the published history's support.
    usda = pl.read_excel(raw / "AEWR-2025.xlsx").unpivot(
        index=["State", "Region"], variable_name="year", value_name="usda_aewr"
    ).rename({"State": "state_name", "Region": "usda_region"}).with_columns(
        pl.col("year").cast(pl.Int64), pl.col("usda_aewr").cast(pl.Float64, strict=False))
    audit = annual.join(usda, on=["state_name", "year"], how="left", validate="1:1").with_columns(
        (pl.col("aewr") - pl.col("usda_aewr")).alias("difference_from_usda_reference"),
        pl.col("usda_aewr").is_not_null().alias("usda_reference_available"))
    schedule_rows = []
    registry_rows = registry.sort("year").to_dicts()
    for i, record in enumerate(registry_rows):
        end = registry_rows[i + 1]["effective_start"] if i + 1 < len(registry_rows) else date(2025, 10, 2)
        if record["year"] == 2008:
            end = date(2009, 1, 17)
        for row in notice_rates[record["year"]]:
            schedule_rows.append({**row, "reference_year": record["year"],
                "effective_start": record["effective_start"], "effective_end_exclusive": end,
                "alternative_effective_start": record["alternative_effective_start"],
                "occupation_scope": "nonrange_reference" if record["year"] <= 2022 else "fls_common_soc_reference",
                "soc_codes": None if record["year"] <= 2022 else COMMON_SOC,
                "skill_level": "not_differentiated", "worker_category": "covered_workers",
                "source_document": record["document_number"], "source_url": record["source_url"],
                "rate_source": "crs_table_1" if record["year"] < 2000 else "fr_state_table",
                "applicability_status": record["applicability_status"],
                "requires_job_cohort_and_competing_floor_review": True})
    schedule = pl.DataFrame(schedule_rows, infer_schema_length=None).with_columns(
        pl.col("alternative_effective_start").cast(pl.Date))
    validate_schedule(schedule)
    skill_rates = parse_skill_workbooks(raw)
    sources.extend(raw / name for name in skill_rates["source_file"].unique().to_list())
    events_path = registry_path.with_name("aewr_policy_events.csv")
    events = pl.read_csv(events_path, try_parse_dates=True)
    assert_unique(events, ["event_id"])
    sources.append(events_path)
    manifest = {
        "schema_version": "aewr_source_v2", "annual_reference_years": [1990, 2025],
        "schedule_coverage_start": "1995-02-07", "schedule_coverage_end_exclusive": "2025-10-02",
        "unresolved": ["2009 OES area/SOC/skill determinations and transition cohorts",
                       "pre-1995 effective dates", "non-FLS occupations and range wage histories",
                       "job-specific prevailing/CBA/state-law coverage and injunction entity membership",
                       "2023--2025 other-occupation historical tables: uncached; network acquisition unavailable"],
        "inputs": [{"name": str(p.relative_to(raw)) if p.is_relative_to(raw) else p.name,
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(set(sources))],
        "annual_reference_semantics": "published FLS benchmark, not a dated universal wage obligation",
        "2009_semantics": "canonical null; means retained only in explicitly nonstatutory field"}
    return {"aewr.parquet": annual, "aewr_rate_schedule.parquet": schedule,
            "aewr_occupation_skill_rates.parquet": skill_rates,
            "aewr_policy_events.parquet": events,
            "aewr_reference_rate_source_audit.parquet": audit,
            "aewr_2009_nonstatutory_wages.parquet": nonstatutory,
            "aewr_source_manifest.json": manifest}


def write_artifacts(artifacts: dict, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name, value in artifacts.items():
        temporary = destination / f".{name}.tmp"
        if isinstance(value, pl.DataFrame):
            value.write_parquet(temporary)
        else:
            temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(destination / name)
