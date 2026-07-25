"""QWI response parsing shared by the extractor and its schema test."""

from __future__ import annotations

from typing import Any

import polars as pl

from h2a.geography import harmonize_county_fips_2010

QWI_FIELDS = ("Emp", "EmpS", "EmpTotal", "sEmp", "sEmpS", "sEmpTotal")
REQUIRED_FIELDS = {
    "time",
    "state",
    "county",
    "industry",
    "ownercode",
    "seasonadj",
    *QWI_FIELDS,
}


def parse_qwi_payload(payload: list[list[Any]]) -> pl.DataFrame:
    """Parse one nonempty Census QWI payload into the artifact schema."""
    if len(payload) < 2:
        raise ValueError("QWI payload must contain a header and at least one row")

    header = [str(value) for value in payload[0]]
    rows = [dict(zip(header, row, strict=True)) for row in payload[1:]]
    frame = pl.DataFrame(rows, infer_schema_length=None)

    missing_fields = REQUIRED_FIELDS.difference(frame.columns)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"QWI response is missing fields: {missing}")

    return (
        frame.with_columns(
            pl.col("state").cast(pl.String).str.pad_start(2, "0"),
            pl.col("county").cast(pl.String).str.pad_start(3, "0"),
            pl.col("industry").cast(pl.String),
            pl.col("time").cast(pl.String),
            *[
                pl.col(field).cast(pl.Float64, strict=False).alias(field)
                for field in QWI_FIELDS
            ],
        )
        .with_columns(
            pl.concat_str("state", "county")
            .map_elements(
                harmonize_county_fips_2010,
                return_dtype=pl.String,
            )
            .alias("county_fips"),
            pl.col("time")
            .str.extract(r"^(\d{4})-Q[1-4]$", 1)
            .cast(pl.Int16)
            .alias("year"),
            pl.col("time")
            .str.extract(r"^\d{4}-Q([1-4])$", 1)
            .cast(pl.Int8)
            .alias("qtr"),
        )
        .rename(
            {
                "industry": "industry_code",
                "Emp": "qwi_beginning_quarter_employment",
                "EmpS": "qwi_stable_employment",
                "EmpTotal": "qwi_any_quarter_employment",
                "sEmp": "qwi_beginning_quarter_employment_status",
                "sEmpS": "qwi_stable_employment_status",
                "sEmpTotal": "qwi_any_quarter_employment_status",
            }
        )
        .select(
            "county_fips",
            "year",
            "qtr",
            "industry_code",
            "ownercode",
            "seasonadj",
            "qwi_beginning_quarter_employment",
            "qwi_stable_employment",
            "qwi_any_quarter_employment",
            "qwi_beginning_quarter_employment_status",
            "qwi_stable_employment_status",
            "qwi_any_quarter_employment_status",
        )
    )
