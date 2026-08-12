# Purpose: Parse Farm Labor Survey wage and worker tables into regional panels.
# Inputs: FLS archives and the optional AEWR-region crosswalk in data/raw/fls.
# Outputs: fls_region, fls_state, paired quarterly worker/wage, and auxiliary
# moment Parquet files.

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    import csv
    import io
    import re
    from zipfile import ZipFile

    import polars as pl
    import us

    from h2a.paths import INTERMEDIATE, RAW

    return INTERMEDIATE, RAW, ZipFile, csv, io, pl, re, us


@app.cell
def _(RAW):
    fls_path = RAW / "fls"
    zip_paths = sorted(fls_path.glob("*.zip"))
    return (zip_paths,)


@app.cell
def _(RAW, pl, re, us):
    region_crosswalk_path = RAW / "fls" / "aewr_region_crosswalk.csv"
    if region_crosswalk_path.exists():
        region_crosswalk = pl.read_csv(region_crosswalk_path)
    else:
        # The published FLS geography is stable over the period used here.
        # Keep a self-contained fallback because the optional hand-built raw
        # crosswalk is not present on every machine running the pipeline.
        canonical_regions = (
            (1, "Pacific"),
            (2, "Mountain I"),
            (3, "Northern Plains"),
            (4, "Lake"),
            (5, "Northeast I"),
            (6, "Northeast II"),
            (7, "Cornbelt I"),
            (8, "Appalachian II"),
            (9, "Appalachian I"),
            (10, "Southeast"),
            (11, "Florida"),
            (12, "Delta"),
            (13, "Cornbelt II"),
            (14, "Southern Plains"),
            (15, "Mountain III"),
            (16, "Mountain II"),
            (17, "California"),
        )
        region_crosswalk = pl.DataFrame(
            {
                "aewr_region_id": [str(row[0]) for row in canonical_regions],
                "fls_table_name": [row[1] for row in canonical_regions],
                "fls_variant_long": [row[1] for row in canonical_regions],
                "fls_variant_abbrev": [
                    "FL" if row[0] == 11 else "CA" if row[0] == 17 else row[1]
                    for row in canonical_regions
                ],
                "ers_name": [row[1] for row in canonical_regions],
            }
        )
    region_lookup = {}
    state_lookup = {}

    def lookup_key(value):
        return re.sub(r"\s+", " ", value or "").strip().casefold()

    # Accept the region labels and abbreviations that appear across FLS vintages.
    for row in region_crosswalk.iter_rows(named=True):
        names = [
            row["fls_table_name"],
            row["fls_variant_long"],
            row["fls_variant_abbrev"],
            row["ers_name"],
        ]

        for _name in names:
            if _name:
                region_lookup[lookup_key(_name)] = (
                    row["aewr_region_id"],
                    row["fls_table_name"],
                )

        # Historical tables occasionally spell Cornbelt as two words or use
        # Arabic rather than Roman numerals.
        canonical_name = row["fls_table_name"]
        aliases = {
            canonical_name.replace("Cornbelt", "Corn Belt"),
            canonical_name.replace(" III", " 3")
            .replace(" II", " 2")
            .replace(" I", " 1"),
        }
        for alias in aliases:
            region_lookup[lookup_key(alias)] = (
                row["aewr_region_id"],
                canonical_name,
            )

    # State tables use a mix of postal abbreviations and full state names.
    for state in us.STATES:
        state_record = {
            "state_fips": state.fips,
            "state_abbreviation": state.abbr,
            "state_name": state.name,
        }
        state_lookup[lookup_key(state.name)] = state_record
        state_lookup[lookup_key(state.abbr)] = state_record
    return region_lookup, state_lookup


@app.cell
def _(csv, io, re, region_lookup, state_lookup):
    def clean(value):
        return re.sub(r"\s+", " ", value or "").strip()

    def is_annual_region_table(titles):
        title = clean(" ".join(titles))
        # Keep the annual AEWR-region wage table; state, SOC, and base-wage
        # tables are separate series with different coverage.
        return (
            re.search(r"annual average", title, re.IGNORECASE)
            and re.search(r"wage rates", title, re.IGNORECASE)
            and re.search(
                r"regions? and united states|by region and united states",
                title,
                re.IGNORECASE,
            )
            and not re.search(
                r"\bstate\b|standard occupational classification|\bSOC\b|base wage rates",
                title,
                re.IGNORECASE,
            )
        )

    def is_annual_state_table(titles):
        title = clean(" ".join(titles))
        return (
            re.search(r"annual average", title, re.IGNORECASE)
            and re.search(r"wage rates", title, re.IGNORECASE)
            and re.search(r"\bby state\b", title, re.IGNORECASE)
            and not re.search(
                r"standard occupational classification|\bSOC\b|base wage rates",
                title,
                re.IGNORECASE,
            )
        )

    def worker_type(cells):
        text = clean(
            " ".join(cell for cell in cells if not re.search(r"\b\d{4}\b", cell))
        ).casefold()
        text = text.replace("&", "and")

        if "all hired" in text or ("all" in text and "hired" in text):
            return "all_hired"
        if "field and livestock" in text or (
            "field" in text and "livestock" in text and "combined" in text
        ):
            return "field_livestock"
        if "livestock" in text and "field" not in text:
            return "livestock"
        if "field" in text and "livestock" not in text:
            return "field"
        return None

    def parse_number(value):
        value = clean(value).replace(",", "")
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def parse_annual_wage_table(
        text, source_zip, source_csv, table_filter, geography_lookup, geography_columns
    ):
        rows = list(csv.reader(io.StringIO(text)))
        titles = [
            clean(row[2]) for row in rows if len(row) > 2 and row[1].casefold() == "t"
        ]
        if not table_filter(titles):
            return []

        header_rows = [
            row[2:] for row in rows if len(row) > 1 and row[1].casefold() == "h"
        ]
        max_header_width = max((len(row) for row in header_rows), default=0)
        column_metadata = []

        # FLS headers are stacked across several "h" rows. Read down each
        # column to recover the worker type and year attached to that column.
        for idx in range(1, max_header_width):
            cells = [clean(row[idx]) if idx < len(row) else "" for row in header_rows]
            years = [
                int(year)
                for cell in cells
                for year in re.findall(r"\b(19\d{2}|20\d{2})\b", cell)
            ]
            kind = worker_type(cells)
            if years and kind:
                column_metadata.append((idx, kind, years[-1]))

        records = []
        table_title = " | ".join(titles)
        # Row type "d" contains table data. Blank, footnote, and aggregate rows
        # are ignored whenever their label does not match the target geography.
        for row in rows:
            if len(row) < 3 or row[1].casefold() != "d":
                continue

            geography = clean(re.sub(r"\s+\d+/\s*$", "", clean(row[2])))
            if not geography or geography.startswith("("):
                continue

            geography_match = geography_lookup.get(geography.casefold())
            if geography_match is None:
                continue

            for idx, kind, year in column_metadata:
                value = parse_number(row[idx + 2] if idx + 2 < len(row) else "")

                record = {
                    "source_zip": source_zip,
                    "source_csv": source_csv,
                    "table_title": table_title,
                    "year": year,
                    "worker_type": kind,
                    "wage": value,
                }
                record.update(geography_columns(geography_match))
                records.append(record)

        return records

    def parse_annual_region_table(text, source_zip, source_csv):
        return parse_annual_wage_table(
            text,
            source_zip,
            source_csv,
            is_annual_region_table,
            region_lookup,
            lambda match: {"aewr_region_id": match[0], "region_name": match[1]},
        )

    def parse_annual_state_table(text, source_zip, source_csv):
        return parse_annual_wage_table(
            text,
            source_zip,
            source_csv,
            is_annual_state_table,
            state_lookup,
            dict,
        )

    return parse_annual_region_table, parse_annual_state_table


@app.cell
def _(csv, io, re, region_lookup):
    reference_quarters = ("january", "april", "july", "october")
    release_months = {
        month: number
        for number, month in enumerate(
            (
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ),
            start=1,
        )
    }

    def clean_quarterly(value):
        return re.sub(r"\s+", " ", value or "").strip()

    def parse_quarterly_number(value):
        value = clean_quarterly(value).replace(",", "")
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def is_quarterly_worker_region_table(titles):
        title = clean_quarterly(" ".join(titles))
        return (
            re.search(r"\bregions?\b", title, re.IGNORECASE)
            and re.search(r"\bunited states\b", title, re.IGNORECASE)
            and re.search(r"\bnumber\b", title, re.IGNORECASE)
            and re.search(r"\bhours?\b.*\bworked\b", title, re.IGNORECASE)
            and re.search(
                r"\b(January|April|July|October)\b",
                title,
                re.IGNORECASE,
            )
            and not re.search(r"annual average", title, re.IGNORECASE)
            and not re.search(r"wage rates?", title, re.IGNORECASE)
        )

    def is_quarterly_wage_region_table(titles):
        title = clean_quarterly(" ".join(titles))
        return (
            re.search(r"\bregions?\b", title, re.IGNORECASE)
            and re.search(r"\bunited states\b", title, re.IGNORECASE)
            and re.search(r"\bwage rates?\b", title, re.IGNORECASE)
            and re.search(r"type of worker", title, re.IGNORECASE)
            and re.search(
                r"\b(January|April|July|October)\b",
                title,
                re.IGNORECASE,
            )
            and not re.search(r"annual average", title, re.IGNORECASE)
            and not re.search(
                r"type of farm|economic class of farm",
                title,
                re.IGNORECASE,
            )
        )

    def quarterly_wage_type(cells):
        text = clean_quarterly(" ".join(cells)).casefold().replace("&", "and")
        if "all hired" in text:
            return "all_hired"
        if "field and livestock" in text or (
            "field" in text and "livestock" in text and "combined" in text
        ):
            return "field_livestock"
        if "livestock" in text and "field" not in text:
            return "livestock"
        if "field" in text and "livestock" not in text:
            return "field"
        return None

    def parse_quarterly_worker_table(text, source_zip, source_csv):
        rows = list(csv.reader(io.StringIO(text)))
        titles = [
            clean_quarterly(row[2])
            for row in rows
            if len(row) > 2 and row[1].casefold() == "t"
        ]
        if not is_quarterly_worker_region_table(titles):
            return []

        table_title = " | ".join(titles)
        period_match = re.search(
            r"\b(January|April|July|October)\b[^|]{0,60}?"
            r"\b((?:19|20)\d{2})\b",
            table_title,
            re.IGNORECASE,
        )
        release_match = re.search(
            r"\bReleased\s+([A-Z][a-z]+)\s+(\d{1,2}),\s+"
            r"((?:19|20)\d{2})\b",
            table_title,
        )
        if period_match is None or release_match is None:
            return []

        quarter = period_match.group(1).casefold()
        year = int(period_match.group(2))
        release_month = release_months.get(release_match.group(1))
        if quarter not in reference_quarters or release_month is None:
            return []

        records = []
        for row in rows:
            if len(row) < 7 or row[1].casefold() != "d":
                continue

            geography = clean_quarterly(
                re.sub(r"\s+\d+/\s*$", "", clean_quarterly(row[2]))
            )
            if not geography or geography.startswith("("):
                continue

            geography_match = region_lookup.get(geography.casefold())
            if geography_match is None:
                continue

            records.append(
                {
                    "year": year,
                    "quarter": quarter,
                    "aewr_region_id": geography_match[0],
                    "region_name": geography_match[1],
                    "fls_hired_workers": parse_quarterly_number(row[3]),
                    "fls_hired_workers_150_days_or_more": (
                        parse_quarterly_number(row[4])
                    ),
                    "fls_hired_workers_149_days_or_less": (
                        parse_quarterly_number(row[5])
                    ),
                    "fls_gross_hours_worked": parse_quarterly_number(row[6]),
                    "release_year": int(release_match.group(3)),
                    "release_month": release_month,
                    "release_day": int(release_match.group(2)),
                    "source_zip": source_zip,
                    "source_csv": source_csv,
                    "table_title": table_title,
                }
            )

        return records

    def parse_quarterly_wage_table(text, source_zip, source_csv):
        rows = list(csv.reader(io.StringIO(text)))
        titles = [
            clean_quarterly(row[2])
            for row in rows
            if len(row) > 2 and row[1].casefold() == "t"
        ]
        if not is_quarterly_wage_region_table(titles):
            return []

        table_title = " | ".join(titles)
        period_match = re.search(
            r"\b(January|April|July|October)\b[^|]{0,60}?"
            r"\b((?:19|20)\d{2})\b",
            table_title,
            re.IGNORECASE,
        )
        release_match = re.search(
            r"\bReleased\s+([A-Z][a-z]+)\s+(\d{1,2}),\s+"
            r"((?:19|20)\d{2})\b",
            table_title,
        )
        if period_match is None or release_match is None:
            return []

        quarter = period_match.group(1).casefold()
        year = int(period_match.group(2))
        release_month = release_months.get(release_match.group(1))
        if quarter not in reference_quarters or release_month is None:
            return []

        header_rows = [
            row[2:] for row in rows if len(row) > 1 and row[1].casefold() == "h"
        ]
        max_header_width = max((len(row) for row in header_rows), default=0)
        column_metadata = []
        for idx in range(1, max_header_width):
            cells = [
                clean_quarterly(row[idx]) if idx < len(row) else ""
                for row in header_rows
            ]
            kind = quarterly_wage_type(cells)
            header_text = " ".join(cells).casefold()
            # Beginning in 2019 the regional tables place gross and base wage
            # columns side by side.  The FLS calibration uses the published
            # gross hourly series; the base-wage cells for the historical
            # transition quarters are explicitly unavailable.
            if kind and "base wage" not in header_text:
                column_metadata.append((idx, kind))

        if {kind for _, kind in column_metadata} != {
            "field",
            "livestock",
            "field_livestock",
            "all_hired",
        }:
            return []

        records = []
        for row in rows:
            if len(row) < 7 or row[1].casefold() != "d":
                continue
            geography = clean_quarterly(
                re.sub(r"\s+\d+/\s*$", "", clean_quarterly(row[2]))
            )
            if not geography or geography.startswith("("):
                continue
            geography_match = region_lookup.get(geography.casefold())
            if geography_match is None:
                continue

            record = {
                "year": year,
                "quarter": quarter,
                "aewr_region_id": geography_match[0],
                "region_name": geography_match[1],
                "release_year": int(release_match.group(3)),
                "release_month": release_month,
                "release_day": int(release_match.group(2)),
                "source_zip": source_zip,
                "source_csv": source_csv,
                "table_title": table_title,
            }
            for idx, kind in column_metadata:
                record[f"fls_{kind}_hourly_wage"] = parse_quarterly_number(
                    row[idx + 2] if idx + 2 < len(row) else ""
                )
            records.append(record)
        return records

    return (
        parse_quarterly_wage_table,
        parse_quarterly_worker_table,
        reference_quarters,
    )


@app.cell
def _(
    ZipFile,
    parse_annual_region_table,
    parse_annual_state_table,
    parse_quarterly_wage_table,
    parse_quarterly_worker_table,
    pl,
    zip_paths,
):
    region_records = []
    state_records = []
    quarterly_worker_records = []
    quarterly_wage_records = []
    for zip_path in zip_paths:
        with ZipFile(zip_path) as z:
            for name in z.namelist():
                name_lower = name.lower()
                if (
                    not name_lower.endswith(".csv")
                    or "_all" in name_lower
                    or "all_tables" in name_lower
                ):
                    continue

                text = z.read(name).decode("utf-8", errors="ignore")
                region_records.extend(
                    parse_annual_region_table(text, zip_path.name, name)
                )
                state_records.extend(
                    parse_annual_state_table(text, zip_path.name, name)
                )
                quarterly_worker_records.extend(
                    parse_quarterly_worker_table(text, zip_path.name, name)
                )
                quarterly_wage_records.extend(
                    parse_quarterly_wage_table(text, zip_path.name, name)
                )

    annual_wages_long = (
        pl.DataFrame(region_records) if region_records else pl.DataFrame()
    )
    state_wages_long = pl.DataFrame(state_records) if state_records else pl.DataFrame()
    quarterly_workers_long = (
        pl.DataFrame(quarterly_worker_records)
        if quarterly_worker_records
        else pl.DataFrame()
    )
    quarterly_wages_long = (
        pl.DataFrame(quarterly_wage_records)
        if quarterly_wage_records
        else pl.DataFrame()
    )

    def reshape_annual_wages(
        wages_long, geography_columns, preferred_columns, sort_columns
    ):
        output_rows = []
        if wages_long.is_empty():
            return pl.DataFrame()

        # Each annual table has two years. Store the older year as revised and
        # the newer year as preliminary for the estimate year. Some older cells
        # are blank or suppressed, but their header year still identifies them.
        group_columns = ["source_zip", "source_csv", "table_title"] + geography_columns
        for key, group in wages_long.group_by(
            group_columns,
            maintain_order=True,
        ):
            years = sorted(group["year"].unique().to_list())
            if len(years) not in {1, 2}:
                continue

            revised_year = years[0] if len(years) == 2 else None
            preliminary_year = years[-1]
            output_row = {
                "estimate_year": preliminary_year,
                "revised_year": revised_year,
                "preliminary_year": preliminary_year,
                "source_zip": key[0],
                "source_csv": key[1],
                "table_title": key[2],
            }
            output_row.update(dict(zip(geography_columns, key[3:])))

            for record in group.iter_rows(named=True):
                suffix = (
                    "revised"
                    if revised_year is not None and record["year"] == revised_year
                    else "preliminary"
                )
                output_row[f"{record['worker_type']}_{suffix}"] = record["wage"]

            output_rows.append(output_row)

        out = pl.DataFrame(output_rows) if output_rows else pl.DataFrame()
        if out.is_empty():
            return out

        return out.select(
            [column for column in preferred_columns if column in out.columns]
            + [column for column in out.columns if column not in preferred_columns]
        ).sort(sort_columns)

    out = reshape_annual_wages(
        annual_wages_long,
        ["aewr_region_id", "region_name"],
        [
            "estimate_year",
            "aewr_region_id",
            "region_name",
            "revised_year",
            "preliminary_year",
            "all_hired_revised",
            "all_hired_preliminary",
            "field_revised",
            "field_preliminary",
            "livestock_revised",
            "livestock_preliminary",
            "field_livestock_revised",
            "field_livestock_preliminary",
            "source_zip",
            "source_csv",
            "table_title",
        ],
        ["estimate_year", "aewr_region_id"],
    )
    state_out = reshape_annual_wages(
        state_wages_long,
        ["state_fips", "state_abbreviation", "state_name"],
        [
            "estimate_year",
            "state_fips",
            "state_abbreviation",
            "state_name",
            "revised_year",
            "preliminary_year",
            "all_hired_revised",
            "all_hired_preliminary",
            "field_revised",
            "field_preliminary",
            "field_livestock_revised",
            "field_livestock_preliminary",
            "source_zip",
            "source_csv",
            "table_title",
        ],
        ["estimate_year", "state_fips"],
    )
    return out, quarterly_wages_long, quarterly_workers_long, state_out


@app.cell
def _(
    out,
    pl,
    quarterly_wages_long,
    quarterly_workers_long,
    reference_quarters,
):
    # Worker counts and wage rates are two tables for the same survey week.
    # Pair them within a release before choosing a vintage. Prefer the release
    # that supplies the selected annual report; if that report omits an earlier
    # survey week, use the latest paired release no later than the annual one.
    annual_sources = (
        pl.concat(
            [
                out.select(
                    pl.col("revised_year").alias("year"),
                    pl.col("estimate_year").alias("annual_estimate_year"),
                    pl.col("source_zip").alias("annual_source_zip"),
                    pl.lit(1).alias("annual_vintage_priority"),
                ),
                out.select(
                    pl.col("preliminary_year").alias("year"),
                    pl.col("estimate_year").alias("annual_estimate_year"),
                    pl.col("source_zip").alias("annual_source_zip"),
                    pl.lit(0).alias("annual_vintage_priority"),
                ),
            ],
            how="vertical",
        )
        .filter(pl.col("year").is_not_null())
        .sort("year", "annual_vintage_priority", "annual_estimate_year")
        .unique(subset="year", keep="last", maintain_order=True)
        .drop("annual_vintage_priority")
    )

    pair_keys = [
        "year",
        "quarter",
        "aewr_region_id",
        "release_year",
        "release_month",
        "release_day",
        "source_zip",
    ]
    workers_for_pair = quarterly_workers_long.rename(
        {
            "source_csv": "worker_source_csv",
            "table_title": "worker_table_title",
        }
    )
    wages_for_pair = quarterly_wages_long.drop("region_name").rename(
        {
            "source_csv": "wage_source_csv",
            "table_title": "wage_table_title",
        }
    )
    paired_releases = workers_for_pair.join(
        wages_for_pair,
        on=pair_keys,
        how="inner",
        validate="1:1",
    )

    release_catalog = paired_releases.select(
        "source_zip",
        "release_year",
        "release_month",
        "release_day",
    ).unique()
    annual_sources = annual_sources.join(
        release_catalog.rename(
            {
                "source_zip": "annual_source_zip",
                "release_year": "annual_release_year",
                "release_month": "annual_release_month",
                "release_day": "annual_release_day",
            }
        ),
        on="annual_source_zip",
        how="left",
        validate="m:1",
    )
    selected_pairs = (
        paired_releases.join(annual_sources, on="year", how="inner", validate="m:1")
        .with_columns(
            pl.date("release_year", "release_month", "release_day").alias(
                "release_date"
            ),
            pl.date(
                "annual_release_year",
                "annual_release_month",
                "annual_release_day",
            ).alias("annual_release_date"),
            pl.when(pl.col("source_zip") == pl.col("annual_source_zip"))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("annual_release_match")
        )
        .filter(pl.col("release_date") <= pl.col("annual_release_date"))
        .sort(
            "year",
            "quarter",
            "aewr_region_id",
            "annual_release_match",
            "release_date",
        )
        .unique(
            subset=["year", "quarter", "aewr_region_id"],
            keep="last",
            maintain_order=True,
        )
        .with_columns(
            pl.when(pl.col("annual_release_match") == 1)
            .then(pl.lit("selected_annual_release"))
            .otherwise(pl.lit("latest_paired_before_annual_release"))
            .alias("release_selection_method")
        )
        .with_columns(
            pl.all_horizontal(
                pl.col("fls_hired_workers").is_not_null(),
                pl.col("fls_hired_workers_150_days_or_more").is_not_null(),
                pl.col("fls_hired_workers_149_days_or_less").is_not_null(),
                pl.col("fls_gross_hours_worked").is_not_null(),
            ).alias("fls_worker_values_available"),
            pl.all_horizontal(
                pl.col("fls_field_hourly_wage").is_not_null(),
                pl.col("fls_livestock_hourly_wage").is_not_null(),
                pl.col("fls_field_livestock_hourly_wage").is_not_null(),
                pl.col("fls_all_hired_hourly_wage").is_not_null(),
            ).alias("fls_wage_values_available"),
        )
        .with_columns(
            (
                pl.col("fls_worker_values_available")
                & pl.col("fls_wage_values_available")
            ).alias("fls_pair_values_available"),
            pl.when(
                pl.col("worker_table_title")
                .str.to_lowercase()
                .str.contains("survey was not conducted")
            )
            .then(pl.lit("survey_not_conducted"))
            .when(
                pl.col("fls_worker_values_available")
                & pl.col("fls_wage_values_available")
            )
            .then(pl.lit("published_values"))
            .otherwise(pl.lit("published_values_incomplete"))
            .alias("fls_pair_value_status"),
        )
        .sort("year", "quarter", "aewr_region_id")
    )

    shared_release_columns = [
        "year",
        "quarter",
        "aewr_region_id",
        "region_name",
        "release_year",
        "release_month",
        "release_day",
        "source_zip",
        "worker_source_csv",
        "worker_table_title",
        "wage_source_csv",
        "wage_table_title",
        "annual_source_zip",
        "release_selection_method",
        "fls_worker_values_available",
        "fls_wage_values_available",
        "fls_pair_values_available",
        "fls_pair_value_status",
    ]
    fls_region_quarterly_workers = selected_pairs.select(
        *shared_release_columns,
        "fls_hired_workers",
        "fls_hired_workers_150_days_or_more",
        "fls_hired_workers_149_days_or_less",
        "fls_gross_hours_worked",
    )
    fls_region_quarterly_wages = selected_pairs.select(
        *shared_release_columns,
        "fls_field_hourly_wage",
        "fls_livestock_hourly_wage",
        "fls_field_livestock_hourly_wage",
        "fls_all_hired_hourly_wage",
    )

    quarter_columns = []
    for quarter in reference_quarters:
        quarter_columns.extend(
            [
                pl.col("fls_hired_workers")
                .filter(pl.col("quarter") == quarter)
                .first()
                .alias(f"fls_hired_workers_{quarter}"),
                pl.col("fls_hired_workers_150_days_or_more")
                .filter(pl.col("quarter") == quarter)
                .first()
                .alias(f"fls_hired_workers_150_days_or_more_{quarter}"),
                pl.col("fls_hired_workers_149_days_or_less")
                .filter(pl.col("quarter") == quarter)
                .first()
                .alias(f"fls_hired_workers_149_days_or_less_{quarter}"),
            ]
        )

    fls_region_auxiliary_moments = (
        fls_region_quarterly_workers.group_by("year", "aewr_region_id", "region_name")
        .agg(quarter_columns)
        .with_columns(
            pl.sum_horizontal(
                *[
                    pl.col(f"fls_hired_workers_{quarter}").is_not_null().cast(pl.Int8)
                    for quarter in reference_quarters
                ]
            ).alias("fls_reference_weeks_observed"),
            pl.sum_horizontal(
                *[
                    pl.col(f"fls_hired_workers_150_days_or_more_{quarter}")
                    .is_not_null()
                    .cast(pl.Int8)
                    for quarter in reference_quarters
                ]
            ).alias("fls_duration_reference_weeks_observed"),
        )
        .with_columns(
            (
                (pl.col("fls_reference_weeks_observed") == 4)
                & (pl.col("fls_duration_reference_weeks_observed") == 4)
            ).alias("fls_auxiliary_moments_complete"),
            pl.sum_horizontal(
                *[
                    pl.col(f"fls_hired_workers_{quarter}")
                    for quarter in reference_quarters
                ]
            ).alias("fls_hired_workers_reference_week_total"),
            pl.sum_horizontal(
                *[
                    pl.col(f"fls_hired_workers_150_days_or_more_{quarter}")
                    for quarter in reference_quarters
                ]
            ).alias("fls_hired_workers_150_days_or_more_reference_week_total"),
        )
        .with_columns(
            pl.when(
                pl.col("fls_auxiliary_moments_complete")
                & (pl.col("fls_hired_workers_reference_week_total") > 0)
            )
            .then(
                pl.col("fls_hired_workers_150_days_or_more_reference_week_total")
                / pl.col("fls_hired_workers_reference_week_total")
            )
            .otherwise(None)
            .alias("fls_hired_worker_150_plus_share"),
            *[
                pl.when(
                    pl.col("fls_auxiliary_moments_complete")
                    & (pl.col("fls_hired_workers_reference_week_total") > 0)
                )
                .then(
                    pl.col(f"fls_hired_workers_{quarter}")
                    / pl.col("fls_hired_workers_reference_week_total")
                )
                .otherwise(None)
                .alias(f"fls_hired_worker_share_{quarter}")
                for quarter in reference_quarters
            ],
        )
        .sort("year", "aewr_region_id")
    )

    out_with_auxiliary = out.join(
        fls_region_auxiliary_moments.drop("region_name"),
        left_on=["estimate_year", "aewr_region_id"],
        right_on=["year", "aewr_region_id"],
        how="left",
    )
    return (
        fls_region_auxiliary_moments,
        fls_region_quarterly_wages,
        fls_region_quarterly_workers,
        out_with_auxiliary,
    )


@app.cell
def _(
    INTERMEDIATE,
    fls_region_auxiliary_moments,
    fls_region_quarterly_wages,
    fls_region_quarterly_workers,
    out_with_auxiliary,
    pl,
    state_out,
):
    from h2a.geography import assert_geo_columns

    # 17 regions * 24 year = 408 rows
    # 43 states * 9 years = 387 rows
    assert_geo_columns(out_with_auxiliary, ["aewr_region_id"])
    assert_geo_columns(state_out, ["state_fips"])
    assert_geo_columns(fls_region_quarterly_workers, ["aewr_region_id"])
    assert_geo_columns(fls_region_quarterly_wages, ["aewr_region_id"])
    assert_geo_columns(fls_region_auxiliary_moments, ["aewr_region_id"])
    supported_keys = {
        (str(region), year, quarter)
        for region in range(1, 18)
        for year in range(2010, 2022)
        for quarter in ("january", "april", "july", "october")
    }
    worker_keys = set(
        fls_region_quarterly_workers.select(
            "aewr_region_id", "year", "quarter"
        ).iter_rows()
    )
    wage_keys = set(
        fls_region_quarterly_wages.select(
            "aewr_region_id", "year", "quarter"
        ).iter_rows()
    )
    if not supported_keys.issubset(worker_keys & wage_keys):
        missing = sorted(supported_keys.difference(worker_keys & wage_keys))
        raise ValueError(
            "FLS worker/wage release pairs are incomplete for 2010-2021: "
            + ", ".join(f"{region}-{year}-{quarter}" for region, year, quarter in missing[:10])
        )
    supported_pairs = fls_region_quarterly_workers.filter(
        pl.col("year").is_between(2010, 2021)
    )
    known_nonconducted = supported_pairs.filter(
        ~pl.col("fls_pair_values_available")
    )
    valid_nonconducted = known_nonconducted.filter(
        (pl.col("year") == 2011)
        & (pl.col("quarter") == "april")
        & (pl.col("fls_pair_value_status") == "survey_not_conducted")
    )
    if known_nonconducted.height != 17 or valid_nonconducted.height != 17:
        raise ValueError(
            "Unexpected missing FLS worker/wage values in the supported period."
        )
    out_with_auxiliary.write_parquet(INTERMEDIATE / "fls_region.parquet")
    state_out.write_parquet(INTERMEDIATE / "fls_state.parquet")
    fls_region_quarterly_workers.write_parquet(
        INTERMEDIATE / "fls_region_quarterly_workers.parquet"
    )
    fls_region_quarterly_wages.write_parquet(
        INTERMEDIATE / "fls_region_quarterly_wages.parquet"
    )
    fls_region_auxiliary_moments.write_parquet(
        INTERMEDIATE / "fls_region_auxiliary_moments.parquet"
    )


if __name__ == "__main__":
    app.run()
