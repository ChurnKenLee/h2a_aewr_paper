import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from zipfile import ZipFile
    import marimo as mo
    import csv
    import io
    import zipfile
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
    region_crosswalk = pl.read_csv(RAW / "fls" / "aewr_region_crosswalk.csv")
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

        header_rows = [row[2:] for row in rows if len(row) > 1 and row[1].casefold() == "h"]
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
def _(
    ZipFile,
    parse_annual_region_table,
    parse_annual_state_table,
    pl,
    zip_paths,
):
    region_records = []
    state_records = []
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
                region_records.extend(parse_annual_region_table(text, zip_path.name, name))
                state_records.extend(parse_annual_state_table(text, zip_path.name, name))

    annual_wages_long = pl.DataFrame(region_records) if region_records else pl.DataFrame()
    state_wages_long = pl.DataFrame(state_records) if state_records else pl.DataFrame()


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
    return out, state_out


@app.cell
def _(INTERMEDIATE, out, state_out):
    # 17 regions * 24 year = 408 rows
    # 43 states * 9 years = 387 rows
    out.write_parquet(INTERMEDIATE / "fls_region.parquet")
    state_out.write_parquet(INTERMEDIATE / "fls_state.parquet")
    return


if __name__ == "__main__":
    app.run()
