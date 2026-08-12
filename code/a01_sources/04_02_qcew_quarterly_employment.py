"""Extract all-ownership QCEW 111/112 reference-month employment.

Inputs: quarterly QCEW single-file ZIP archives in ``data/raw/qcew``.
Output: ``data/intermediate/qcew_county_ag_quarterly_employment.parquet``.

The persistent row is a 2010-vintage county, year, quarter, and three-digit
industry.  QCEW does not publish a detailed-industry all-ownership row at this
aggregation level, so the producer sums the reported ownership components.
The total is usable only when every reported component is disclosed.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import polars as pl

from h2a.geography import assert_geo_columns, harmonize_county_fips_2010
from h2a.paths import INTERMEDIATE, RAW

FIRST_YEAR = 2000
LAST_YEAR = 2024
OUTPUT_PATH = INTERMEDIATE / "qcew_county_ag_quarterly_employment.parquet"
REFERENCE_MONTH = {
    1: "january",
    2: "april",
    3: "july",
    4: "october",
}
DETAILED_OWNERSHIP_CODES = ("1", "2", "3", "5")
AG_INDUSTRIES = ("111", "112")
NONALLOCATABLE_AREA_FIPS = {
    "02201",
    "02232",
    "02280",
    *(f"09{county:03d}" for county in range(110, 191, 10)),
    "51560",
}
QCEW_DTYPES = {
    "area_fips": pl.String,
    "own_code": pl.String,
    "industry_code": pl.String,
    "agglvl_code": pl.String,
    "year": pl.Int16,
    "qtr": pl.Int8,
    "disclosure_code": pl.String,
    "month1_emplvl": pl.Float64,
}
OUTPUT_KEYS = ["county_fips", "year", "qtr", "industry_code"]


def _canonical_county(value: str | None) -> str | None:
    if value == "02158":
        return "02270"
    if value in {"02063", "02066"}:
        return "02261"
    return harmonize_county_fips_2010(value)


def aggregate_ownership_components(source: pl.DataFrame) -> pl.DataFrame:
    """Aggregate reported ownership rows and enforce disclosure semantics."""
    required = set(QCEW_DTYPES)
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(
            "Quarterly QCEW input is missing columns: " + ", ".join(missing)
        )

    relevant = (
        source.filter(
            pl.col("own_code").is_in(DETAILED_OWNERSHIP_CODES),
            pl.col("agglvl_code") == "75",
            pl.col("industry_code").is_in(AG_INDUSTRIES),
            ~pl.col("area_fips").is_in(NONALLOCATABLE_AREA_FIPS),
            pl.col("area_fips").str.slice(2, 3) != "999",
        )
        .with_columns(
            pl.col("area_fips")
            .map_elements(_canonical_county, return_dtype=pl.String)
            .alias("county_fips")
        )
        .with_columns(
            (pl.col("area_fips") != pl.col("county_fips"))
            .cast(pl.Int8)
            .alias("qcew_geography_priority")
        )
    )

    # In transition years QCEW can carry old and successor county records.
    # Prefer the already-canonical source geography before summing ownerships.
    relevant = relevant.filter(
        pl.col("qcew_geography_priority")
        == pl.col("qcew_geography_priority").min().over(
            "county_fips", "year", "qtr", "industry_code"
        )
    )
    disclosed_row = pl.col("disclosure_code").fill_null("__missing__") == ""
    numeric_row = (
        pl.col("month1_emplvl").is_not_null()
        & pl.col("month1_emplvl").is_finite()
        & (pl.col("month1_emplvl") >= 0)
    )
    aggregated = (
        relevant.group_by(OUTPUT_KEYS)
        .agg(
            pl.col("own_code")
            .n_unique()
            .alias("qcew_reported_ownership_components"),
            disclosed_row.sum().alias("qcew_disclosed_ownership_components"),
            numeric_row.sum().alias("qcew_numeric_ownership_components"),
            disclosed_row.all().alias("qcew_employment_disclosed"),
            pl.when(disclosed_row)
            .then(pl.col("month1_emplvl"))
            .otherwise(None)
            .sum()
            .alias("_disclosed_employment_sum"),
        )
        .with_columns(
            pl.col("qtr")
            .replace_strict(REFERENCE_MONTH, return_dtype=pl.String)
            .alias("reference_month"),
            pl.when(pl.col("qcew_employment_disclosed"))
            .then(pl.col("_disclosed_employment_sum"))
            .otherwise(None)
            .alias("qcew_reference_month_emplvl"),
        )
        .drop("_disclosed_employment_sum")
        .select(
            "county_fips",
            "year",
            "qtr",
            "reference_month",
            "industry_code",
            "qcew_reported_ownership_components",
            "qcew_disclosed_ownership_components",
            "qcew_numeric_ownership_components",
            "qcew_employment_disclosed",
            "qcew_reference_month_emplvl",
        )
        .sort(*OUTPUT_KEYS)
    )
    validate_quarterly_employment(aggregated)
    return aggregated


def validate_quarterly_employment(frame: pl.DataFrame) -> None:
    """Validate keys, geography, and all-ownership disclosure semantics."""
    if frame.is_empty():
        raise ValueError("Quarterly QCEW output is empty")
    assert_geo_columns(frame, ["county_fips"])
    duplicate_cells = frame.group_by(OUTPUT_KEYS).len().filter(pl.col("len") > 1)
    if duplicate_cells.height:
        raise ValueError("Quarterly QCEW output has duplicate county cells")
    invalid = frame.filter(
        ~pl.col("qtr").is_in(REFERENCE_MONTH)
        | ~pl.col("industry_code").is_in(AG_INDUSTRIES)
    )
    if invalid.height:
        raise ValueError("Quarterly QCEW output has invalid quarter or industry")
    semantic_errors = frame.filter(
        (pl.col("qcew_reported_ownership_components") < 1)
        | (
            pl.col("qcew_reported_ownership_components")
            > len(DETAILED_OWNERSHIP_CODES)
        )
        | (
            pl.col("qcew_disclosed_ownership_components")
            > pl.col("qcew_reported_ownership_components")
        )
        | (
            pl.col("qcew_employment_disclosed")
            != (
                pl.col("qcew_disclosed_ownership_components")
                == pl.col("qcew_reported_ownership_components")
            )
        )
        | (
            pl.col("qcew_employment_disclosed")
            & (
                pl.col("qcew_numeric_ownership_components")
                != pl.col("qcew_reported_ownership_components")
            )
        )
        | (
            pl.col("qcew_employment_disclosed")
            != pl.col("qcew_reference_month_emplvl").is_not_null()
        )
        | (
            pl.col("qcew_reference_month_emplvl").is_not_null()
            & (
                ~pl.col("qcew_reference_month_emplvl").is_finite()
                | (pl.col("qcew_reference_month_emplvl") < 0)
            )
        )
    )
    if semantic_errors.height:
        raise ValueError(
            "Quarterly QCEW disclosure flags and employment totals disagree"
        )


def extract_year(zip_path: Path, year: int) -> pl.DataFrame:
    """Read, filter, and aggregate one quarterly single-file archive."""
    target_csv = f"{year}.q1-q4.singlefile.csv"
    with (
        zipfile.ZipFile(zip_path, mode="r") as archive,
        archive.open(target_csv) as extracted_file,
    ):
        year_df = pl.read_csv(
            extracted_file,
            columns=list(QCEW_DTYPES),
            schema_overrides=QCEW_DTYPES,
        )
    return aggregate_ownership_components(year_df)


def extract_quarterly_employment(
    output_path: Path = OUTPUT_PATH,
    *,
    first_year: int = FIRST_YEAR,
    last_year: int = LAST_YEAR,
) -> None:
    """Extract a year range and atomically write the calibration input."""
    if first_year > last_year:
        raise ValueError("first_year must not exceed last_year")
    qcew_path = RAW / "qcew"
    frames = []
    for year in range(first_year, last_year + 1):
        print(f"Extracting quarterly QCEW {year}", flush=True)
        frames.append(
            extract_year(qcew_path / f"{year}_qtrly_singlefile.zip", year)
        )

    quarterly_employment = pl.concat(frames, how="vertical_relaxed").sort(
        *OUTPUT_KEYS
    )
    validate_quarterly_employment(quarterly_employment)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        quarterly_employment.write_parquet(temporary, compression="zstd")
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(
        f"Wrote {quarterly_employment.height:,} county-industry-quarter rows "
        f"to {output_path}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing quarterly employment binary",
    )
    parser.add_argument("--first-year", type=int, default=FIRST_YEAR)
    parser.add_argument("--last-year", type=int, default=LAST_YEAR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if OUTPUT_PATH.exists() and not args.force:
        print(f"Output already exists at {OUTPUT_PATH}; use --force to rebuild")
        return
    extract_quarterly_employment(
        first_year=args.first_year,
        last_year=args.last_year,
    )


if __name__ == "__main__":
    main()
