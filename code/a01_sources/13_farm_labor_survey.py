# Purpose: Parse Farm Labor Survey wage and worker tables into regional panels.
# Inputs: FLS archives and the optional AEWR-region crosswalk in data/raw/fls.
# Outputs: fls_region, fls_state, quarterly-worker, and auxiliary-moment Parquet files.

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    from zipfile import ZipFile
    import csv
    import io
    from h2a.paths import RAW, INTERMEDIATE
    import polars as pl
    import re
    import us

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
                "aewr_region_num": [row[0] for row in canonical_regions],
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
                    row["aewr_region_num"],
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
                row["aewr_region_num"],
                canonical_name,
            )

    # State tables use a mix of postal abbreviations and full state names.
    for state in us.STATES:
        state_record = {
            "state_fips_code": state.fips,
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
            lambda match: {"aewr_region_num": match[0], "region_name": match[1]},
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
                    "aewr_region_num": geography_match[0],
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

    return parse_quarterly_worker_table, reference_quarters


@app.cell
def _(
    ZipFile,
    parse_annual_region_table,
    parse_annual_state_table,
    parse_quarterly_worker_table,
    pl,
    zip_paths,
):
    region_records = []
    state_records = []
    quarterly_worker_records = []
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

    annual_wages_long = (
        pl.DataFrame(region_records) if region_records else pl.DataFrame()
    )
    state_wages_long = pl.DataFrame(state_records) if state_records else pl.DataFrame()
    quarterly_workers_long = (
        pl.DataFrame(quarterly_worker_records)
        if quarterly_worker_records
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
        ["aewr_region_num", "region_name"],
        [
            "estimate_year",
            "aewr_region_num",
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
        ["estimate_year", "aewr_region_num"],
    )
    state_out = reshape_annual_wages(
        state_wages_long,
        ["state_fips_code", "state_abbreviation", "state_name"],
        [
            "estimate_year",
            "state_fips_code",
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
        ["estimate_year", "state_fips_code"],
    )
    return out, quarterly_workers_long, state_out


@app.cell
def _(out, pl, quarterly_workers_long, reference_quarters):
    # Prefer the source used for that year's annual FLS estimates. Older
    # November releases do not repeat January and April, so fall back to the
    # latest same-year release for those observations. This keeps revisions
    # auditable while avoiding later publications silently replacing a target.
    annual_sources = out.select(
        pl.col("estimate_year").alias("year"),
        pl.col("source_zip").alias("annual_source_zip"),
    ).unique()

    fls_region_quarterly_workers = (
        quarterly_workers_long.join(annual_sources, on="year", how="inner")
        .with_columns(
            pl.when(pl.col("source_zip") == pl.col("annual_source_zip"))
            .then(pl.lit(2))
            .when(pl.col("release_year") == pl.col("year"))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("source_priority")
        )
        .sort(
            "year",
            "quarter",
            "aewr_region_num",
            "source_priority",
            "release_year",
            "release_month",
            "release_day",
        )
        .unique(
            subset=["year", "quarter", "aewr_region_num"],
            keep="last",
            maintain_order=True,
        )
        .drop("annual_source_zip", "source_priority")
        .sort("year", "quarter", "aewr_region_num")
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
        fls_region_quarterly_workers.group_by("year", "aewr_region_num", "region_name")
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
        .sort("year", "aewr_region_num")
    )

    out_with_auxiliary = out.join(
        fls_region_auxiliary_moments.drop("region_name"),
        left_on=["estimate_year", "aewr_region_num"],
        right_on=["year", "aewr_region_num"],
        how="left",
    )
    return (
        fls_region_auxiliary_moments,
        fls_region_quarterly_workers,
        out_with_auxiliary,
    )


@app.cell
def _(
    INTERMEDIATE,
    fls_region_auxiliary_moments,
    fls_region_quarterly_workers,
    out_with_auxiliary,
    state_out,
):
    # 17 regions * 24 year = 408 rows
    # 43 states * 9 years = 387 rows
    out_with_auxiliary.write_parquet(INTERMEDIATE / "fls_region.parquet")
    state_out.write_parquet(INTERMEDIATE / "fls_state.parquet")
    fls_region_quarterly_workers.write_parquet(
        INTERMEDIATE / "fls_region_quarterly_workers.parquet"
    )
    fls_region_auxiliary_moments.write_parquet(
        INTERMEDIATE / "fls_region_auxiliary_moments.parquet"
    )
    return


if __name__ == "__main__":
    app.run()
