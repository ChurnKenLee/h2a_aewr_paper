import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import io
    import zipfile
    from h2a.paths import RAW, INTERMEDIATE
    import polars as pl
    import re

    return INTERMEDIATE, RAW, io, pl, re, zipfile


@app.cell
def _(INTERMEDIATE, RAW):
    oews_path = RAW / "oews"
    binary_path = INTERMEDIATE
    return binary_path, oews_path


@app.cell
def _(oews_path, re, zipfile):
    def oews_year_from_archive(archive_path):
        year_suffix = re.findall(r"\d+", archive_path.stem)[0][-2:]
        if year_suffix in {"97", "98", "99"}:
            return int("19" + year_suffix)
        return int("20" + year_suffix)


    def is_oews_estimate_file(member_name):
        file_name = member_name.rsplit("/", 1)[-1].lower()
        if not file_name.endswith((".xls", ".xlsx")):
            return False
        return file_name.startswith(("msa", "amsa", "oes", "bos"))


    # Create dict with list of OEWS estimate workbook members within each zip-year.
    oews_path_dict = {}
    for _archive_path in sorted(oews_path.glob("*.zip"), key=oews_year_from_archive):
        _year = oews_year_from_archive(_archive_path)
        with zipfile.ZipFile(_archive_path) as archive:
            _member_list = sorted(
                _member_name
                for _member_name in archive.namelist()
                if is_oews_estimate_file(_member_name)
            )
        oews_path_dict[_year] = [
            (_archive_path, _member_name) for _member_name in _member_list
        ]
    return (oews_path_dict,)


@app.cell
def _():
    oews_columns = [
        "area",
        "area_name",
        "occ_code",
        "occ_title",
        "tot_emp",
        "h_mean",
        "a_mean",
        "h_pct10",
        "h_pct25",
        "h_median",
        "h_pct75",
        "h_pct90",
        "a_pct10",
        "a_pct25",
        "a_median",
        "a_pct75",
        "a_pct90",
    ]
    oews_1997_columns = [
        "area",
        "area_name",
        "occ_code",
        "occ_title",
        "tot_emp",
        "h_mean",
        "a_mean",
        "h_median",
    ]
    oews_rename_pct_columns = {
        "h_wpct10": "h_pct10",
        "h_wpct25": "h_pct25",
        "h_wpct75": "h_pct75",
        "h_wpct90": "h_pct90",
        "a_wpct10": "a_pct10",
        "a_wpct25": "a_pct25",
        "a_wpct75": "a_pct75",
        "a_wpct90": "a_pct90",
    }
    return oews_1997_columns, oews_columns, oews_rename_pct_columns


@app.cell
def _(
    io,
    oews_1997_columns,
    oews_columns,
    oews_path_dict,
    oews_rename_pct_columns,
    pl,
    zipfile,
):
    def promote_oews_header(df):
        for row_index, row in enumerate(df.iter_rows()):
            values = [
                str(value).strip().lower() if value is not None else "" for value in row
            ]
            if (
                "area" in values
                and "occ_code" in values
                and ("occ_title" in values or "occ_titl" in values)
            ):
                header = [
                    value if value else f"column_{column_number + 1}"
                    for column_number, value in enumerate(values)
                ]
                data_df = df.slice(row_index + 1)
                data_df.columns = header
                return data_df
        raise ValueError("Could not find the OEWS header row.")


    def read_oews_member(zip_path, member_name, year):
        with zipfile.ZipFile(zip_path) as archive:
            workbook = io.BytesIO(archive.read(member_name))

        if year < 2001:
            new_df = pl.read_excel(
                workbook,
                has_header=False,
                infer_schema_length=0,
            )
            new_df = promote_oews_header(new_df)
        else:
            new_df = pl.read_excel(workbook, infer_schema_length=0)
            new_df = new_df.rename({column: column.lower() for column in new_df.columns})

        rename_map = {}
        if year == 2000:
            rename_map["occ_titl"] = "occ_title"
        if 1997 < year < 2003:
            rename_map.update(oews_rename_pct_columns)
        if year >= 2019:
            rename_map["area_title"] = "area_name"

        rename_map = {
            old_column: new_column
            for old_column, new_column in rename_map.items()
            if old_column in new_df.columns
        }
        if rename_map:
            new_df = new_df.rename(rename_map)

        columns_to_keep = oews_1997_columns if year == 1997 else oews_columns
        missing_columns = [
            column for column in columns_to_keep if column not in new_df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"{zip_path.name}:{member_name} is missing columns {missing_columns}."
            )

        return new_df.select(columns_to_keep)


    oews_year_dfs = []
    for _year, _zip_members in oews_path_dict.items():
        _member_df_list = []
        print(f"Loading files for year {_year}.")
        for _zip_path, _member_name in _zip_members:
            _member_df_list.append(read_oews_member(_zip_path, _member_name, _year))
        _combined_df = pl.concat(_member_df_list, how="diagonal_relaxed").with_columns(
            pl.col("tot_emp")
            .str.replace_all(",", "")
            .str.strip_chars()
            .cast(pl.Float64, strict=False),
            pl.col("a_mean")
            .str.replace_all(",", "")
            .str.strip_chars()
            .cast(pl.Float64, strict=False),
            pl.col("area").str.strip_chars(" "),
            pl.lit(_year).cast(pl.Int32).alias("year"),
        )
        oews_year_dfs.append(_combined_df)

    oews_df = pl.concat(oews_year_dfs, how="diagonal_relaxed")
    return (oews_df,)


@app.cell
def _(binary_path, oews_df, pl):
    # Drop Guam, Virgin Islands, Puerto Rico
    conus_oews_df = oews_df.filter(
        ~pl.col("area_name").str.contains(", PR|Puerto Rico|Guam|Virgin Islands")
    )

    # Save combined df as parquet
    conus_oews_df.write_parquet(binary_path / "oews.parquet")
    return


if __name__ == "__main__":
    app.run()
