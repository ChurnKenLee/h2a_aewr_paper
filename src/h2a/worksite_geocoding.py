"""Resumable public worksite-address geocoding; never use employer headquarters."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import requests
import us

from h2a.aewr import assert_unique
from h2a.geography import normalize_geo_code

ENDPOINT = "https://geocoding.geo.census.gov/geocoder/geographies/addressbatch"
BENCHMARK = "Public_AR_Current"
VINTAGE = "Current_Current"
BATCH_SIZE = 10_000
MAX_BATCHES = 50


def address_key(street: str, city: str, state: str, zipcode: str) -> str:
    return hashlib.sha256(json.dumps([street, city, state, zipcode]).encode()).hexdigest()


def normalize_address(row: dict) -> dict:
    def clean(value):
        return re.sub(r"\s+", " ", str(value or "")).strip().upper()
    street = clean(row["source_street"])
    city, state, zipcode = (clean(row[k]) for k in ("source_city", "source_state", "source_zip"))
    state_object = next((s for s in [*us.STATES_AND_TERRITORIES, us.states.DC]
                         if state in (s.abbr, s.name.upper())), None)
    state = state_object.abbr if state_object else state
    zipcode = zipcode[:5] if re.fullmatch(r"\d{5}(-\d{4})?", zipcode) else zipcode
    ready = bool(street and len(street) <= 100 and state_object and (city or re.fullmatch(r"\d{5}", zipcode)))
    return {"address_id": address_key(street, city, state, zipcode), "street": street,
            "city": city, "state": state, "zip": zipcode, "requestable": ready}


def parse_census_batch(text: str, expected_ids: set[str]) -> list[dict]:
    rows = []
    for source in csv.reader(io.StringIO(text)):
        if len(source) < 3:
            raise ValueError("Malformed Census batch response")
        source += [""] * max(0, 12 - len(source))
        identifier, status = source[0], source[2]
        if identifier not in expected_ids or status not in {"Match", "No_Match", "Tie"}:
            raise ValueError("Unexpected ID/status in Census response")
        longitude = latitude = county = None
        if status == "Match":
            coordinates = source[5].split(",")
            if len(coordinates) != 2:
                raise ValueError("Matched address lacks a coordinate pair")
            longitude, latitude = map(float, coordinates)
            if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
                raise ValueError("Invalid geocoded coordinates")
            if source[8] and source[9]:
                county = normalize_geo_code(source[8], "state_fips") + normalize_geo_code(source[9], "county_code")
        rows.append({"address_id": identifier, "geocode_status": status,
                     "match_type": source[3] or None, "matched_address": source[4] or None,
                     "longitude": longitude, "latitude": latitude,
                     "census_county_fips_current": county,
                     "coordinate_quality": "Census_address_range_match_not_verified_farm_point" if status == "Match" else "no_unique_point"})
    if {r["address_id"] for r in rows} != expected_ids or len(rows) != len(expected_ids):
        raise ValueError("Census response does not cover each request exactly once")
    return rows


def source_location_records(raw: Path, intermediate: Path) -> tuple[pl.DataFrame, list[Path]]:
    addendum_path = intermediate / "h2a_addendum_b_with_fips.parquet"
    addendum = pl.read_parquet(addendum_path).with_row_index("source_row", offset=1).select(
        pl.concat_str(pl.lit("addendum_b:"), pl.col("source_row")).alias("location_record_id"),
        "case_number", "fiscal_year", "county_fips_list",
        pl.concat_str("worksite_address_1", "worksite_address_2", separator=" ", ignore_nulls=True).alias("source_street"),
        pl.col("worksite_city").alias("source_city"), pl.col("worksite_state").alias("source_state"),
        pl.col("worksite_zip").alias("source_zip"), pl.lit("addendum_b_employment").alias("source_dataset"),
        pl.lit("exact_allocation_id_when_retained_by_B01").alias("outcome_link_status"))
    records, inputs = [addendum], [addendum_path]
    # Main disclosure worksite streets exist from 2020 but were not retained in
    # the county-only location artifact. This explicit point-geocoding pass owns
    # their extraction; it does not substitute employer mailing addresses.
    for year, filename in [(2020, "H-2A_Disclosure_Data_FY2020.xlsx"),
                           (2021, "H-2A_Disclosure_Data_FY2021.xlsx"),
                           (2022, "H-2A_Disclosure_Data_FY2022_Q4.xlsx"),
                           (2023, "H-2A_Disclosure_Data_FY2023_Q4.xlsx"),
                           (2024, "H-2A_Disclosure_Data_FY2024_Q4.xlsx")]:
        matches = list((raw / "h2a").rglob(filename))
        if len(matches) != 1:
            raise ValueError(f"Expected one cached main disclosure: {filename}")
        path = matches[0]
        inputs.append(path)
        columns = ["CASE_NUMBER", "WORKSITE_ADDRESS", "WORKSITE_CITY", "WORKSITE_STATE", "WORKSITE_POSTAL_CODE"]
        frame = pl.read_excel(path, columns=columns, schema_overrides={c: pl.String for c in columns})
        frame = frame.with_row_index("source_row", offset=2).select(
            pl.concat_str(pl.lit(f"main_excel:{year}:"), pl.col("source_row")).alias("location_record_id"),
            pl.col("CASE_NUMBER").alias("case_number"), pl.lit(year, dtype=pl.Int32).alias("fiscal_year"),
            pl.lit(None, dtype=pl.String).alias("county_fips_list"),
            pl.col("WORKSITE_ADDRESS").alias("source_street"), pl.col("WORKSITE_CITY").alias("source_city"),
            pl.col("WORKSITE_STATE").alias("source_state"), pl.col("WORKSITE_POSTAL_CODE").alias("source_zip"),
            pl.lit("main_disclosure_worksite").alias("source_dataset"),
            pl.lit("case_worksite_signature_link_requires_ambiguity_audit").alias("outcome_link_status"))
        records.append(frame)
    return pl.concat(records), inputs


def build_worksite_geocodes(raw: Path, intermediate: Path, *, acquire: bool = False) -> dict:
    records, inputs = source_location_records(raw, intermediate)
    addresses = pl.DataFrame([normalize_address(row) for row in records.iter_rows(named=True)])
    records = records.with_columns(addresses["address_id"])
    addresses = addresses.unique("address_id").sort("address_id")
    assert_unique(records, ["location_record_id"])
    assert_unique(addresses, ["address_id"])
    records.write_parquet(intermediate / "h2a_worksite_geocode_records.parquet")
    addresses.write_parquet(intermediate / "h2a_worksite_geocode_addresses.parquet")
    queue = intermediate / "h2a_worksite_geocode_requests"
    cache = raw / "h2a" / "census_geocoder"
    queue.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    eligible = addresses.filter(pl.col("requestable"))
    requests_needed = (eligible.height + BATCH_SIZE - 1) // BATCH_SIZE
    if requests_needed > MAX_BATCHES:
        raise ValueError("Declared 50-batch resource guard exceeded; review source growth")
    parsed, batches = [], []
    network_failure = None
    for offset in range(0, eligible.height, BATCH_SIZE):
        batch = eligible.slice(offset, BATCH_SIZE).select("address_id", "street", "city", "state", "zip")
        request_text = batch.write_csv(include_header=False)
        request_hash = hashlib.sha256(request_text.encode()).hexdigest()
        (queue / f"{request_hash}.csv").write_text(request_text)
        response_path = cache / f"{request_hash}.csv"
        metadata_path = cache / f"{request_hash}.json"
        expected = set(batch["address_id"])
        status = "not_acquired"
        if not response_path.exists() and acquire and network_failure is None:
            try:
                response = requests.post(ENDPOINT, data={"benchmark": BENCHMARK, "vintage": VINTAGE},
                    files={"addressFile": ("addresses.csv", request_text, "text/csv")}, timeout=(10, 120))
                response.raise_for_status()
                parse_census_batch(response.text, expected)
                response_path.write_text(response.text)
                metadata_path.write_text(json.dumps({"endpoint": ENDPOINT, "benchmark": BENCHMARK,
                    "vintage": VINTAGE, "retrieved_utc": datetime.now(UTC).isoformat(),
                    "request_sha256": request_hash,
                    "response_sha256": hashlib.sha256(response.text.encode()).hexdigest()}, indent=2) + "\n")
            except (requests.RequestException, ValueError) as exc:
                network_failure = type(exc).__name__
        if response_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if metadata["request_sha256"] != request_hash or metadata["response_sha256"] != hashlib.sha256(response_path.read_bytes()).hexdigest():
                raise ValueError("Geocoder cache provenance mismatch")
            parsed.extend(parse_census_batch(response_path.read_text(), expected))
            status = "cached_response_validated"
        batches.append({"request_sha256": request_hash, "records": batch.height, "status": status})
    schema = {"address_id": pl.String, "geocode_status": pl.String, "match_type": pl.String,
              "matched_address": pl.String, "longitude": pl.Float64, "latitude": pl.Float64,
              "census_county_fips_current": pl.String, "coordinate_quality": pl.String}
    geocodes = addresses.join(pl.DataFrame(parsed, schema=schema), on="address_id", how="left", validate="1:1").with_columns(
        pl.col("geocode_status").fill_null(pl.when(pl.col("requestable")).then(pl.lit("not_acquired"))
                                            .otherwise(pl.lit("insufficient_address"))),
        pl.col("coordinate_quality").fill_null("no_coordinate"))
    geocodes.write_parquet(intermediate / "h2a_worksite_geocodes.parquet")
    manifest = {"schema_version": "public_worksite_geocoding_v1", "endpoint": ENDPOINT,
        "benchmark": BENCHMARK, "vintage": VINTAGE, "batch_size": BATCH_SIZE, "maximum_batches": MAX_BATCHES,
        "acquisition_requested": acquire, "network_failure": network_failure,
        "location_records": records.height, "unique_addresses": addresses.height,
        "requestable_addresses": eligible.height, "expected_requests": requests_needed,
        "matched_addresses": geocodes.filter(pl.col("geocode_status") == "Match").height,
        "batches": batches, "source_inputs": {str(p.relative_to(raw)) if p.is_relative_to(raw) else p.name:
            hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "spatial_rd_ready": False,
        "county_analysis_requires_point_geocodes": False,
        "remaining_requirements": ["unique actual worksite outcome linkage", "boundary-vintage and coordinate-quality audit",
                                   "declared geographic outcome and public exposure denominator", "job-specific wage assignment"],
        "farm_participation_design": "unsupported without a nonparticipant farm frame; not the county analysis"}
    (intermediate / "h2a_worksite_geocode_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {k: v for k, v in manifest.items() if k not in {"batches", "source_inputs"}}
