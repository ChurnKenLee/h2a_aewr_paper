# Purpose: Parse H-2A disclosure and Addendum B files and match worksite
#   locations to 2010 counties with local, deterministic rules only.
# Inputs: DOL H-2A disclosure files, Census place/county/ZIP/county-subdivision
#   files and the pinned USGS GNIS national file.
# Outputs: h2a_disclosure_records.parquet, h2a_addendum_b_records.parquet and
#   h2a_location_pieces.parquet (one row per cleaned location piece with its
#   local match). 02_02 resolves unmatched pieces through Gemini/Places caches;
#   02_03 assigns counties and writes the with-FIPS files.

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import io
    import re
    import zipfile

    import addfips
    import marimo as mo
    import polars as pl

    from h2a.paths import INTERMEDIATE, RAW

    DC_STATEHOOD = 1  # Enables DC to be included in the state list

    import rapidfuzz
    import us

    return INTERMEDIATE, RAW, addfips, io, mo, pl, rapidfuzz, re, us, zipfile


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Extract all H-2A performance files and Census location crosswalks
    """)
    return


@app.cell
def _(INTERMEDIATE, RAW):
    binary_path = INTERMEDIATE
    h2a_path = RAW / "h2a"
    census_code_path = RAW / "geographic_crosswalks" / "census"
    return binary_path, census_code_path, h2a_path


@app.cell
def _(pl):
    # Helper function to read and uppercase all string columns
    def read_census_file(path):
        df = pl.read_csv(path, separator="|", infer_schema=False).with_columns(
            pl.col(pl.String).str.to_uppercase()
        )
        return df

    return (read_census_file,)


@app.cell
def _(addfips, pl):
    # Define a custom AddFIPS function that operates on a polars dataframe
    def add_fips_using_addfips(df, county_col, state_col, target_col):
        # Instantiate the library ONCE here
        af = addfips.AddFIPS(vintage=2010)

        # The 'row' object passed from pl.struct is a dictionary
        return df.with_columns(
            pl.struct([county_col, state_col])
            .map_elements(
                lambda row: (
                    af.get_county_fips(county=row[county_col], state=row[state_col])
                    or ""
                ),
                return_dtype=pl.String,
            )
            .alias(target_col)
        )

    return (add_fips_using_addfips,)


@app.cell
def _(re):
    # Load and process Census files
    # Strip suffixes from locality names for fuzzy string matching later
    place_suffixes = [
        " CITY",
        " TOWN",
        " VILLAGE",
        " CDP",
        " MUNICIPALITY",
        " BOROUGH",
        " TOWNSHIP",
        " CENSUS AREA",
        " CENSUS DESIGNATED PLACE",
        " COUNTY",
        " PARISH",
    ]

    # Construct the regex pattern: (?:suffix1|suffix2|...)$
    # re.escape is used to handle suffixes with special characters like '.'
    place_suffix_pattern = f"(?:{'|'.join(re.escape(s) for s in place_suffixes)})$"
    return (place_suffix_pattern,)


@app.cell
def _(census_code_path, pl, place_suffix_pattern, read_census_file):
    # Census county names
    census_county_2020 = read_census_file(census_code_path / "national_county2020.txt")
    census_county_2020 = (
        census_county_2020.with_columns(
            (pl.col("STATEFP") + pl.col("COUNTYFP")).alias("fips")
        )
        .select(
            [
                pl.col("STATE").alias("state"),
                pl.col("COUNTYNAME").alias("county"),
                pl.col("fips"),
            ]
        )
        .filter(pl.col("county") != "")
        .with_columns(pl.col("county").str.replace_all(place_suffix_pattern, ""))
    )
    return


@app.cell
def _(census_code_path, pl, place_suffix_pattern):
    census_county_2010 = (
        pl.read_csv(
            census_code_path / "national_county.txt",
            has_header=False,
            infer_schema=False,
        )
        .rename(
            {
                "column_1": "state",
                "column_2": "statefp",
                "column_3": "countyfp",
                "column_4": "county",
            }
        )
        .with_columns(
            (pl.col("statefp") + pl.col("countyfp")).alias("fips"),
            pl.col(pl.String).str.to_uppercase(),
        )
        .select(["state", "county", "fips"])
        .filter(pl.col("county") != "")
        .with_columns(pl.col("county").str.replace_all(place_suffix_pattern, ""))
    )
    return (census_county_2010,)


@app.cell
def _(census_code_path, census_county_2010, pl, place_suffix_pattern):
    census_placebycounty_2010 = (
        pl.read_csv(
            census_code_path / "national_places.txt",
            separator="|",
            infer_schema=False,
            encoding="latin-1",
        )
        .rename(
            {
                "STATE": "state",
                "STATEFP": "statefp",
                "PLACENAME": "place",
                "COUNTY\r": "county",
            }
        )
        .with_columns(pl.col(pl.String).str.strip_chars("\r"))
        .with_columns(pl.col(pl.String).str.to_uppercase())
    )
    # There are places that map to multiple counties
    # Explode those, match FIPS, then melt back
    census_place_agg_2010 = (
        census_placebycounty_2010.with_columns(
            pl.col("county").str.split(",").alias("county_list")
        )
        .select(["state", "statefp", "place", "county_list"])
        .explode("county_list", empty_as_null=True)
        .with_columns(
            # Counties after the first carry a leading space; trim before the
            # county-name join so multi-county places keep every county
            pl.col("county_list")
            .str.strip_chars()
            .str.replace_all(place_suffix_pattern, "")
            .alias("county"),
            pl.col("place").str.replace_all(place_suffix_pattern, ""),
        )
        .drop("county_list")
    )
    # census_placebycounty_2010 = add_fips_using_addfips(census_placebycounty_2010, 'county_list', 'state', 'fips')
    census_place_agg_2010 = census_place_agg_2010.join(
        census_county_2010, on=["state", "county"]
    )
    census_place_agg_2010 = census_place_agg_2010.group_by(["state", "place"]).agg(
        pl.col("fips").unique().sort().str.join(",")
    )
    return (census_place_agg_2010,)


@app.cell
def _(census_code_path, pl, place_suffix_pattern, read_census_file):
    # Census counties aggregated to Census places
    census_placebycounty_2020 = read_census_file(
        census_code_path / "national_place_by_county2020.txt"
    )
    census_place_agg_2020 = (
        census_placebycounty_2020.with_columns(
            (pl.col("STATEFP") + pl.col("COUNTYFP")).alias("fips")
        )
        .rename({"STATE": "state", "COUNTYNAME": "county", "PLACENAME": "place"})
        .group_by(["state", "place"])
        .agg(pl.col("fips").str.join(","))
        .filter(pl.col("place") != "")
        .with_columns(pl.col("place").str.replace_all(place_suffix_pattern, ""))
    )
    return


@app.cell
def _(census_code_path, pl):
    census_zip_2010 = pl.read_csv(
        census_code_path / "zcta_county_rel_10.txt", infer_schema=False
    )
    census_zip_agg_2010 = (
        census_zip_2010.rename({"ZCTA5": "zip", "GEOID": "fips"})
        .group_by("zip")
        .agg(pl.col("fips").unique().sort().str.join(","))
    )
    return (census_zip_agg_2010,)


@app.cell
def _(census_code_path, pl, read_census_file):
    # Census counties aggregated to ZIP
    census_zip_2020 = read_census_file(
        census_code_path / "tab20_zcta520_county20_natl.txt"
    )
    census_zip_agg_2020 = (
        census_zip_2020.group_by("GEOID_ZCTA5_20")
        .agg(pl.col("GEOID_COUNTY_20").str.join(","))
        .filter(pl.col("GEOID_ZCTA5_20") != "")
        .rename({"GEOID_ZCTA5_20": "zip", "GEOID_COUNTY_20": "fips"})
    )
    return


@app.cell
def _(rapidfuzz, re):
    # Normalize locality names identically for references and queries:
    # rapidfuzz default processing plus common abbreviation expansions
    name_abbreviations = {
        "ST": "SAINT",
        "STE": "SAINTE",
        "FT": "FORT",
        "MT": "MOUNT",
        "PT": "POINT",
    }
    trailing_name_suffix = re.compile(r"\s+(?:TWP|TOWNSHIP)$")

    def normalize_place_name(value):
        text = rapidfuzz.utils.default_process(str(value or "")).upper()
        tokens = [name_abbreviations.get(token, token) for token in text.split()]
        return trailing_name_suffix.sub("", " ".join(tokens)).strip()

    return (normalize_place_name,)


@app.cell
def _(normalize_place_name, pl):
    # Helper to create a state-based lookup with parallel lists. Rows are
    # sorted so that list order, and therefore every result, is reproducible.
    def create_ref_lookup(df, name_col):
        lookup = {}
        grouped = (
            df.select(["state", pl.col(name_col).alias("name"), "fips"])
            .unique()
            .sort(["state", "name", "fips"])
            .group_by("state", maintain_order=True)
            .agg([pl.col("name").alias("names"), pl.col("fips")])
            .to_dicts()
        )

        for row in grouped:
            normalized = [normalize_place_name(name) for name in row["names"]]
            # Exact index: normalized name -> set of county FIPS
            exact = {}
            for name, fips in zip(normalized, row["fips"], strict=True):
                exact.setdefault(name, set()).update(fips.split(","))
            lookup[row["state"]] = {
                "names": row["names"],
                "normalized": normalized,
                "fips": row["fips"],
                "exact": {
                    name: ",".join(sorted(counties)) for name, counties in exact.items()
                },
            }
        return lookup

    return (create_ref_lookup,)


@app.cell
def _(census_county_2010, census_place_agg_2010, create_ref_lookup):
    # Create state-based maps for rapidfuzz to match
    census_county_ref_map_2010 = create_ref_lookup(census_county_2010, "county")
    census_place_ref_map_2010 = create_ref_lookup(census_place_agg_2010, "place")

    census_county_ref_map_2020 = create_ref_lookup(census_county_2010, "county")
    census_place_ref_map_2020 = create_ref_lookup(census_place_agg_2010, "place")
    return census_county_ref_map_2010, census_place_ref_map_2010


@app.cell
def _(census_county_2010, normalize_place_name, pl):
    # Exact-name index for supplementary gazetteers: state -> normalized name
    # -> county FIPS. Names found in more than one county are ambiguous and
    # are left out so they cannot decide a match.
    valid_county_fips_2010 = set(census_county_2010["fips"].to_list())

    def build_exact_gazetteer(df, label):
        df = df.filter(pl.col("fips").is_in(list(valid_county_fips_2010)))
        counties = {}
        for state, name, fips in df.select(["state", "name", "fips"]).iter_rows():
            key = normalize_place_name(name)
            if state and key:
                counties.setdefault(state, {}).setdefault(key, set()).add(fips)
        index = {
            state: {key: next(iter(f)) for key, f in names.items() if len(f) == 1}
            for state, names in counties.items()
        }
        unique = sum(len(names) for names in index.values())
        total = sum(len(names) for names in counties.values())
        print(f"{label}: {unique} unique-county names of {total}")
        return index

    def gazetteer_lookup(state, query, index):
        return index.get(state, {}).get(normalize_place_name(query), "")

    return build_exact_gazetteer, gazetteer_lookup


@app.cell
def _(build_exact_gazetteer, census_code_path, pl, re):
    # Census 2010 county subdivisions: townships, New England towns, MCDs and
    # CCDs. Names may contain unquoted commas (Maryland election districts),
    # so the name is everything between the fifth and the last field.
    cousub_suffix_pattern = re.compile(
        r"\s+(?:charter township|township|town|city|village|borough|plantation|"
        r"gore|grant|purchase|location|UT|CCD|district|precinct|reservation|"
        r"unorganized territory)$",
        re.IGNORECASE,
    )
    _lines = (
        (census_code_path / "national_cousub.txt")
        .read_text(encoding="latin-1")
        .splitlines()
    )
    _records = []
    for _line in _lines[1:]:
        _fields = _line.split(",")
        if len(_fields) < 7:
            continue
        _name = ",".join(_fields[5:-1]).strip()
        if _name.lower().startswith("county subdivisions not defined"):
            continue
        _records.append(
            {
                "state": _fields[0],
                "name": cousub_suffix_pattern.sub("", _name),
                "fips": _fields[1] + _fields[2],
            }
        )
    census_cousub_2010 = pl.DataFrame(
        _records, schema={"state": pl.String, "name": pl.String, "fips": pl.String}
    )
    cousub_exact_index = build_exact_gazetteer(
        census_cousub_2010, "Census 2010 county subdivisions"
    )
    return (cousub_exact_index,)


@app.cell
def _(RAW, build_exact_gazetteer, io, pl, us, zipfile):
    from h2a.geography import harmonize_county_fips_2010

    # USGS GNIS Domestic Names, populated places only (pinned national file).
    # GNIS uses current county codes: drop Connecticut planning regions (its
    # towns are county subdivisions) and map post-2010 Alaska changes back.
    GNIS_TO_COUNTY_2010 = {"02158": "02270", "02063": "02261", "02066": "02261"}
    gnis_path = (
        RAW / "geographic_crosswalks" / "usgs_gnis" / "DomesticNames_National_Text.zip"
    )
    with zipfile.ZipFile(gnis_path) as _archive:
        _raw = _archive.read("Text/DomesticNames_National.txt")
    gnis = pl.read_csv(
        io.BytesIO(_raw),
        separator="|",
        quote_char=None,
        infer_schema=False,
        encoding="utf8-lossy",
    ).rename(lambda column: column.lstrip("\ufeff"))
    state_abbreviations = {s.name: s.abbr for s in us.states.STATES_AND_TERRITORIES}
    state_abbreviations["District of Columbia"] = "DC"
    gnis_populated_places = (
        gnis.filter(
            (pl.col("feature_class") == "Populated Place")
            & ~pl.col("feature_name").str.contains(r"\(historical\)")
            & (pl.col("county_numeric").fill_null("") != "")
        )
        .with_columns(
            state=pl.col("state_name").replace_strict(
                state_abbreviations, default=None
            ),
            fips=(pl.col("state_numeric") + pl.col("county_numeric")).replace(
                GNIS_TO_COUNTY_2010
            ),
            name=pl.col("feature_name"),
        )
        .filter(pl.col("state").is_not_null() & ~pl.col("fips").str.starts_with("091"))
        .with_columns(
            pl.col("fips").map_elements(
                harmonize_county_fips_2010, return_dtype=pl.String
            )
        )
        .select(["state", "name", "fips"])
    )
    gnis_exact_index = build_exact_gazetteer(
        gnis_populated_places, "USGS GNIS populated places"
    )
    return (gnis_exact_index,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Load H-2A performance files
    """)
    return


@app.cell
def _(load_workbook):
    def get_excel_column_names(excel_file_path, sheet_name=None):
        wb = load_workbook(excel_file_path, read_only=True)

        # Select the active sheet if none is specified
        if sheet_name is None:
            sheet = wb.active
        else:
            sheet = wb[sheet_name]

        # Iterate through the first row only to get the headers
        for row in sheet.iter_rows(min_row=1, max_row=1, values_only=True):
            return list(row)

        return []

    return


@app.cell
def _():
    # Get column names of all the H-2A Excel files
    # Names of H-2A program disclosure files from the DOL-OFLC
    h2a_filenames_dict = {
        2008: "H2A_FY2008.xlsx",
        2009: "H2A_FY2009.xlsx",
        2010: "H-2A_FY2010.xlsx",
        2011: "H-2A_FY2011.xlsx",
        2012: "H-2A_FY2012.xlsx",
        2013: "H2A_FY2013.xls",
        2014: "H-2A_FY14_Q4.xlsx",
        2015: "H-2A_Disclosure_Data_FY15_Q4.xlsx",
        2016: "H-2A_Disclosure_Data_FY16_updated.xlsx",
        2017: "H-2A_Disclosure_Data_FY17.xlsx",
        2018: "H-2A_Disclosure_Data_FY2018_EOY.xlsx",
        2019: "H-2A_Disclosure_Data_FY2019.xlsx",
        2020: "H-2A_Disclosure_Data_FY2020.xlsx",
        2021: "H-2A_Disclosure_Data_FY2021.xlsx",
        2022: "H-2A_Disclosure_Data_FY2022_Q4.xlsx",
        2023: "H-2A_Disclosure_Data_FY2023_Q4.xlsx",
        2024: "H-2A_Disclosure_Data_FY2024_Q4.xlsx",
        2025: "H-2A_Disclosure_Data_FY2025_Q4.xlsx",
    }

    # Names of H-2A program Addendum B files from the DOL-OFLC
    add_b_filenames_dict = {
        2020: "H-2A_FY2020_AddendumB_Employment.xlsx",
        2021: "H-2A_Addendum_B_Employment_FY2021.xlsx",
        2022: "H-2A_Addendum_B_Employment_Record_FY2022_Q4.xlsx",
        2023: "H-2A_Addendum_B_Employment_Record_FY2023_Q4.xlsx",
        2024: "H-2A_Addendum_B_Employment_Record_FY2024_Q4.xlsx",
        # DOL shortened the FY2025 Addendum B file name
        2025: "H-2A_Addendum_B_Employ_FY2025_Q4.xlsx",
    }
    return add_b_filenames_dict, h2a_filenames_dict


@app.cell
def _(pl):
    # Define the set of columns we want to add and remove over the years
    def mutate_list(base, add, remove):
        final = base + add
        if remove:
            final = [s for s in final if s not in remove]
        return final

    def s_map(cols):
        s_dict = {}
        for c in cols:
            if "DATE" in c:
                s_dict[c] = pl.Date
            else:
                s_dict[c] = pl.String
        return s_dict

    add_cols_dict = {}
    remove_cols_dict = {}

    add_cols_dict[2011] = [
        "BASIC_NUMBER_OF_HOURS",
        "NBR_WORKERS_REQUESTED",
        "REQUESTED_END_DATE_OF_NEED",
        "REQUESTED_START_DATE_OF_NEED",
    ]
    remove_cols_dict[2013] = ["NBR_WORKERS_REQUESTED"]
    # SOC_CODE_ID first appears in FY2014 and is renamed SOC_CODE in FY2015.
    add_cols_dict[2014] = [
        "SOC_CODE_ID",
        "WORKSITE_LOCATION_CITY",
        "WORKSITE_LOCATION_STATE",
    ]
    remove_cols_dict[2014] = ["ALIEN_WORK_CITY", "ALIEN_WORK_STATE"]
    add_cols_dict[2015] = [
        "CASE_NUMBER",
        "EMPLOYER_PHONE",
        "NBR_WORKERS_REQUESTED",
        "SOC_CODE",
        "WORKSITE_CITY",
        "WORKSITE_POSTAL_CODE",
        "WORKSITE_STATE",
    ]
    remove_cols_dict[2015] = [
        "CASE_NO",
        "REQUESTED_END_DATE_OF_NEED",
        "REQUESTED_START_DATE_OF_NEED",
        "SOC_CODE_ID",
        "WORKSITE_LOCATION_CITY",
        "WORKSITE_LOCATION_STATE",
    ]
    add_cols_dict[2016] = [
        "JOB_END_DATE",
        "JOB_START_DATE",
        "PRIMARY/SUB",
        "REQUESTED_END_DATE_OF_NEED",
        "REQUESTED_START_DATE_OF_NEED",
    ]
    remove_cols_dict[2016] = ["CERTIFICATION_BEGIN_DATE", "CERTIFICATION_END_DATE"]
    add_cols_dict[2017] = ["TRADE_NAME_DBA", "WORKSITE_COUNTY"]
    add_cols_dict[2018] = ["CASE_NO", "PRIMARY_SUB"]
    remove_cols_dict[2018] = ["CASE_NUMBER", "PRIMARY/SUB"]
    add_cols_dict[2019] = ["CASE_NUMBER", "PRMARY/SUB"]
    remove_cols_dict[2019] = ["CASE_NO", "PRIMARY_SUB"]
    # FLAG-era files rename the receipt date and add emergency-filing and
    # piece-rate fields.  The harmonized names below remain stable.
    add_cols_dict[2020] = [
        "ANTICIPATED_NUMBER_OF_HOURS",
        "EMERGENCY_FILING",
        "EMPLOYER_ADDRESS_1",
        "EMPLOYER_ADDRESS_2",
        "EMPLOYMENT_BEGIN_DATE",
        "EMPLOYMENT_END_DATE",
        "H2A_LABOR_CONTRACTOR",
        "PER",
        "PIECE_RATE_OFFER",
        "PIECE_RATE_UNIT",
        "RECEIVED_DATE",
        "REQUESTED_BEGIN_DATE",
        "REQUESTED_END_DATE",
        "TOTAL_WORKERS_H2A_CERTIFIED",
        "TOTAL_WORKERS_H2A_REQUESTED",
        "TOTAL_WORKERS_NEEDED",
        "TYPE_OF_EMPLOYER_APPLICATION",
        "WAGE_OFFER",
    ]
    remove_cols_dict[2020] = [
        "BASIC_NUMBER_OF_HOURS",
        "BASIC_RATE_OF_PAY",
        "BASIC_UNIT_OF_PAY",
        "EMPLOYER_ADDRESS1",
        "EMPLOYER_ADDRESS2",
        "JOB_END_DATE",
        "JOB_START_DATE",
        "NBR_WORKERS_CERTIFIED",
        "NBR_WORKERS_REQUESTED",
        "ORGANIZATION_FLAG",
        "PRMARY/SUB",
        "CASE_RECEIVED_DATE",
        "REQUESTED_END_DATE_OF_NEED",
        "REQUESTED_START_DATE_OF_NEED",
    ]
    add_cols_dict[2024] = ["AG_ASSN_OR_AGENCY_STATUS", "EMPLOYER_FEIN"]
    return add_cols_dict, mutate_list, remove_cols_dict, s_map


@app.cell
def _(add_cols_dict, h2a_filenames_dict, mutate_list, remove_cols_dict, s_map):
    # Generate set of schemas
    cols_dict = {}
    schema_dict = {}

    cols_dict[2008] = [
        "CASE_NO",
        "CASE_RECEIVED_DATE",
        "CASE_STATUS",
        "DECISION_DATE",
        "EMPLOYER_NAME",
        "EMPLOYER_ADDRESS1",
        "EMPLOYER_ADDRESS2",
        "EMPLOYER_CITY",
        "EMPLOYER_STATE",
        "EMPLOYER_POSTAL_CODE",
        "JOB_TITLE",
        "NBR_WORKERS_CERTIFIED",
        "CERTIFICATION_BEGIN_DATE",
        "CERTIFICATION_END_DATE",
        "BASIC_RATE_OF_PAY",
        "BASIC_UNIT_OF_PAY",
        "ALIEN_WORK_CITY",
        "ALIEN_WORK_STATE",
        "ORGANIZATION_FLAG",
    ]
    schema_dict[2008] = s_map(cols_dict[2008])

    for _y in range(2009, max(h2a_filenames_dict) + 1):
        add_list = add_cols_dict.get(_y, [])
        remove_list = remove_cols_dict.get(_y, [])

        cols_dict[_y] = mutate_list(cols_dict[_y - 1], add_list, remove_list)
        schema_dict[_y] = s_map(cols_dict[_y])
    return cols_dict, schema_dict


@app.cell
def _(s_map):
    # Columns and schema from the Addendum B files
    add_b_cols_dict = {}
    add_b_schema_dict = {}

    add_b_cols_dict[2020] = [
        "CASE_NUMBER",
        "JOB_ORDER_NUMBER",
        "NAME_OF_AGRICULTURAL_BUSINESS",
        "PLACE_OF_EMPLOYMENT_ADDRESS1",
        "PLACE_OF_EMPLOYMENT_ADDRESS2",
        "PLACE_OF_EMPLOYMENT_CITY",
        "PLACE_OF_EMPLOYMENT_STATE",
        "PLACE_OF_EMPLOYMENT_POSTAL_CODE",
        "REQUESTED_BEGIN_DATE",
        "REQUESTED_END_DATE",
        "TOTAL_WORKERS",
    ]
    add_b_cols_dict[2021] = add_b_cols_dict[2020]
    add_b_cols_dict[2022] = add_b_cols_dict[2020]
    add_b_cols_dict[2023] = add_b_cols_dict[2020]
    add_b_cols_dict[2024] = add_b_cols_dict[2020]
    # FY2025 adds the worksite county, which pre-2025 Addendum B files lack
    add_b_cols_dict[2025] = add_b_cols_dict[2020] + ["PLACE_OF_EMPLOYMENT_COUNTY"]

    for _y, _cols in add_b_cols_dict.items():
        add_b_schema_dict[_y] = s_map(_cols)
    return add_b_cols_dict, add_b_schema_dict


@app.cell
def _():
    # Define common column names
    h2a_rename_dict = {
        "CASE_NO": "case_number",
        "CASE_NUMBER": "case_number",
        "CASE_RECEIVED_DATE": "case_received_date",
        "CASE_STATUS": "case_status",
        "DECISION_DATE": "decision_date",
        "RECEIVED_DATE": "case_received_date",
        "EMPLOYER_NAME": "employer_name",
        "TRADE_NAME_DBA": "trade_name_dba",
        "EMPLOYER_ADDRESS1": "employer_address_1",
        "EMPLOYER_ADDRESS_1": "employer_address_1",
        "EMPLOYER_ADDRESS2": "employer_address_2",
        "EMPLOYER_ADDRESS_2": "employer_address_2",
        "EMPLOYER_CITY": "employer_city",
        "EMPLOYER_STATE": "employer_state",
        "EMPLOYER_POSTAL_CODE": "employer_postal_code",
        "EMPLOYER_PHONE": "employer_phone",
        "EMPLOYER_FEIN": "employer_fein",
        "NBR_WORKERS_REQUESTED": "nbr_workers_requested",
        "NBR_WORKERS_CERTIFIED": "nbr_workers_certified",
        "TOTAL_WORKERS_NEEDED": "nbr_workers_needed",
        "TOTAL_WORKERS_H2A_REQUESTED": "nbr_workers_requested",
        "TOTAL_WORKERS_H2A_CERTIFIED": "nbr_workers_certified",
        "REQUESTED_START_DATE_OF_NEED": "requested_begin_date",
        "REQUESTED_END_DATE_OF_NEED": "requested_end_date",
        "CERTIFICATION_BEGIN_DATE": "certification_begin_date",
        "CERTIFICATION_END_DATE": "certification_end_date",
        "REQUESTED_BEGIN_DATE": "requested_begin_date",
        "REQUESTED_END_DATE": "requested_end_date",
        "EMPLOYMENT_BEGIN_DATE": "job_begin_date",
        "EMPLOYMENT_END_DATE": "job_end_date",
        "JOB_START_DATE": "job_begin_date",
        "JOB_END_DATE": "job_end_date",
        "JOB_TITLE": "job_title",
        "BASIC_NUMBER_OF_HOURS": "number_of_hours",
        "ANTICIPATED_NUMBER_OF_HOURS": "number_of_hours",
        "BASIC_RATE_OF_PAY": "wage_rate",
        "WAGE_OFFER": "wage_rate",
        "BASIC_UNIT_OF_PAY": "wage_unit",
        "PER": "wage_unit",
        "PIECE_RATE_OFFER": "piece_rate_offer",
        "PIECE_RATE_UNIT": "piece_rate_unit",
        "SOC_CODE": "soc_code",
        "SOC_CODE_ID": "soc_code",
        "ALIEN_WORK_CITY": "worksite_city",
        "ALIEN_WORK_STATE": "worksite_state",
        "WORKSITE_LOCATION_CITY": "worksite_city",
        "WORKSITE_LOCATION_STATE": "worksite_state",
        "WORKSITE_CITY": "worksite_city",
        "WORKSITE_COUNTY": "worksite_county",
        "WORKSITE_STATE": "worksite_state",
        "WORKSITE_POSTAL_CODE": "worksite_zip",
        "PRIMARY/SUB": "primary_sub",
        "PRIMARY_SUB": "primary_sub",
        "PRMARY/SUB": "primary_sub",
        "ORGANIZATION_FLAG": "organization_flag",
        "TYPE_OF_EMPLOYER_APPLICATION": "type_of_employer_application",
        "H2A_LABOR_CONTRACTOR": "h2a_labor_contractor",
        "EMERGENCY_FILING": "emergency_filing",
        "AG_ASSN_OR_AGENCY_STATUS": "ag_association_or_agency",
    }

    # Common names in Addendum B files
    add_b_rename_dict = {
        "CASE_NUMBER": "case_number",
        "JOB_ORDER_NUMBER": "job_order_number",
        "NAME_OF_AGRICULTURAL_BUSINESS": "business_name",
        "PLACE_OF_EMPLOYMENT_ADDRESS1": "worksite_address_1",
        "PLACE_OF_EMPLOYMENT_ADDRESS2": "worksite_address_2",
        "PLACE_OF_EMPLOYMENT_CITY": "worksite_city",
        "PLACE_OF_EMPLOYMENT_COUNTY": "worksite_county",
        "PLACE_OF_EMPLOYMENT_STATE": "worksite_state",
        "PLACE_OF_EMPLOYMENT_POSTAL_CODE": "worksite_zip",
        "REQUESTED_BEGIN_DATE": "worksite_requested_begin_date",
        "REQUESTED_END_DATE": "worksite_requested_end_date",
        "TOTAL_WORKERS": "total_h2a_workers_requested",
    }
    return add_b_rename_dict, h2a_rename_dict


@app.cell
def _(mo, pl):
    # Cache reads while editing the notebook. Flat scripts run without a
    # Marimo runtime, so they use the same function without the decorator.
    def read_h2a_excel(path, cols, schema, rename_map):
        df = pl.read_excel(
            source=path, columns=cols, schema_overrides=schema, engine="calamine"
        )

        df = df.with_columns(
            pl.col(pl.Date).dt.to_string(),
            pl.col(pl.String).fill_null(""),
        ).rename(mapping=rename_map, strict=False)

        return df

    if mo.running_in_notebook():
        read_h2a_excel = mo.persistent_cache(read_h2a_excel)
    return (read_h2a_excel,)


@app.cell
def _(
    add_b_cols_dict,
    add_b_filenames_dict,
    add_b_rename_dict,
    add_b_schema_dict,
    cols_dict,
    h2a_filenames_dict,
    h2a_path,
    h2a_rename_dict,
    pl,
    read_h2a_excel,
    schema_dict,
):
    # Read the H-2A performance data files
    h2a_df = pl.DataFrame()
    for _y in sorted(h2a_filenames_dict):
        _path = h2a_path / h2a_filenames_dict[_y]
        print(f"Reading H-2A for year {_y}")
        _df = read_h2a_excel(_path, cols_dict[_y], schema_dict[_y], h2a_rename_dict)

        _df = _df.with_columns(fiscal_year=_y)
        h2a_df = pl.concat(items=[h2a_df, _df], how="diagonal")
        h2a_df = h2a_df.with_columns(pl.col(pl.String).fill_null(""))

    # Same with Addendum B files
    # Read the H-2A performance data files
    add_b_df = pl.DataFrame()
    for _y in sorted(add_b_filenames_dict):
        _path = h2a_path / add_b_filenames_dict[_y]
        print(f"Reading Addendum B for year {_y}")
        _df = read_h2a_excel(
            _path, add_b_cols_dict[_y], add_b_schema_dict[_y], add_b_rename_dict
        )

        _df = _df.with_columns(fiscal_year=_y)

        add_b_df = pl.concat(items=[add_b_df, _df], how="diagonal")
        add_b_df = add_b_df.with_columns(pl.col(pl.String).fill_null(""))
    # Files before FY2025 have no worksite county; keep the key column explicit
    if "worksite_county" not in add_b_df.columns:
        add_b_df = add_b_df.with_columns(pl.lit("").alias("worksite_county"))
    return add_b_df, h2a_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Add FIPS codes using worksite location information with local data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Given input location information (city, county, state, ZIP), we want to write functions that finds FIPS codes
    """)
    return


@app.cell
def _(add_b_df, h2a_df):
    # Set of unique worksite locations
    h2a_worksite_locations = h2a_df.select(
        ["worksite_city", "worksite_county", "worksite_state", "worksite_zip"]
    ).unique()
    # Addendum B county is reported from FY2025 and empty before
    add_b_worksite_locations = add_b_df.select(
        ["worksite_city", "worksite_county", "worksite_state", "worksite_zip"]
    ).unique()
    return add_b_worksite_locations, h2a_worksite_locations


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    These are the sets of functions that are used to perform local matching of FIPS
    """)
    return


@app.cell
def _(pl):
    # Expand multi-county entries
    def clean_county_explode(df, county_col, state_col):
        # Replace separators and explode
        df = df.with_columns(
            pl.col(county_col)
            .str.replace_all(" AND ", ",", literal=True)
            .str.replace_all(" & ", ",", literal=True)
            .str.replace_all("/", ",", literal=True)
            .str.split(",")
            .alias("county_list")
        ).explode("county_list", empty_as_null=True)

        # Clean whitespace and suffixes
        df = df.with_columns(pl.col("county_list").str.strip_chars()).with_columns(
            pl.col("county_list")
            .str.replace_all(" COUNTY", "", literal=True)
            .str.replace_all(" COUNTIES", "", literal=True)
            .str.replace_all(" PARISH", "", literal=True)
            .str.replace_all(" PARRISH", "", literal=True)
        )

        # Fix Louisiana typos using a series of replaces (native Polars is faster than a loop)
        # Only apply where state is LA
        la_typos = {
            "ST\\.\\w": "ST. ",
            "NORTH\\. ": "NORTH ",
            "SOUTH\\. ": "SOUTH ",
            "EAST\\. ": "EAST ",
            "WEST\\. ": "WEST ",
            "BATON ROGUE": "BATON ROUGE",
            "JEFF DAVIS": "JEFFERSON DAVIS",
            "IBERIAL": "IBERIA",
        }

        df = df.with_columns(
            pl.when(pl.col(state_col) == "LA")
            .then(pl.col("county_list").str.replace_many(la_typos))
            .otherwise(pl.col("county_list"))
            .alias("county_list")
        )

        # Fill nulls
        df = df.with_columns(pl.col(pl.String).fill_null(""))

        return df

    return (clean_county_explode,)


@app.cell
def _(pl):
    # Split city fields that list several places ("TONOPAH/WENDEN", "1) WENDEN
    # 2) TONOPAH", "YUMA, WELLTON, & YUMA VALLEY"), strip attachment notes, and
    # blank placeholders ("VARIOUS COUNTIES", "SEE ADDENDUM"). Single plain
    # names are left byte-identical so cached Places answers still join.
    needs_city_cleaning = r"(?i)[()/;&,]|\bAND\b|\d\s*\)|\bSEE\b|^\s*(?:VARIOUS|MULTIPLE)\b"
    city_placeholder = (
        r"(?i)^(?:VARIOUS|MULTIPLE|SEE\b|ATTACH|ADDENDUM|N/?A$|TBD$|NONE$|UNKNOWN$)"
    )

    def clean_city_explode(df, city_col):
        # Keep the unsplit value: cached Places answers are keyed on it
        df = df.with_columns(pl.col(city_col).alias(f"{city_col}_raw"))
        cleaned = (
            pl.col(city_col)
            .str.replace_all(r"\([^)]*\)", " ")
            .str.replace_all(r"(?i)\bSEE\s+(?:ATTACHMENTS?|ADDENDUM|ATTACHED)\b.*$", " ")
            .str.replace_all(r"\b\d+\s*\)\s*", ",")
            .str.replace_all(r"(?i)\s+(?:AND|&)\s+|[/;&]", ",")
            .str.split(",")
        )
        df = df.with_columns(
            pl.when(pl.col(city_col).str.contains(needs_city_cleaning))
            .then(cleaned)
            .otherwise(pl.concat_list(pl.col(city_col)))
            .alias(city_col)
        ).explode(city_col, empty_as_null=True)
        df = df.with_columns(
            pl.when(pl.col(city_col).fill_null("").str.contains(r"[(),/;&]|^\s|\s$"))
            .then(pl.col(city_col).str.strip_chars(" .-"))
            .otherwise(pl.col(city_col).fill_null(""))
            .alias(city_col)
        )
        return df.with_columns(
            pl.when(pl.col(city_col).str.contains(city_placeholder))
            .then(pl.lit(""))
            .otherwise(pl.col(city_col))
            .alias(city_col)
        )

    return (clean_city_explode,)


@app.cell
def _(pl):
    # Cleans ZIP codes, mostly by removing trailing 4 digits for ZIP+4 codes
    def clean_zip(df, zip_col):
        zip5 = df.with_columns(
            pl.col(zip_col)
            .str.replace_all(".", "", literal=True)
            .str.split("-")
            .list.get(0)
            .alias(zip_col)
        )
        return zip5

    return (clean_zip,)


@app.cell
def _(normalize_place_name, rapidfuzz):
    # Fuzzy matching only corrects typos; it never chooses among ambiguous names
    FUZZY_MIN_QUERY_LENGTH = 4
    FUZZY_MIN_MARGIN = 5.0

    # Find the locality whose name best matches the query within a state.
    # 1. An exact normalized name returns its full county set (score 100).
    # 2. Otherwise whole-string Levenshtein ratio; the best county set must be
    #    unique and beat every other county set by FUZZY_MIN_MARGIN points.
    #    Ambiguous or too-short queries return no match.
    def fuzz_search(state, query, ref_map):
        if not state or not query or state not in ref_map:
            return ("", 0.0, "")

        state_data = ref_map[state]
        normalized_query = normalize_place_name(query)
        if not normalized_query:
            return ("", 0.0, "")

        exact_fips = state_data["exact"].get(normalized_query)
        if exact_fips:
            return (exact_fips, 100.0, normalized_query)

        if len(normalized_query) < FUZZY_MIN_QUERY_LENGTH:
            return ("", 0.0, "")

        matches = rapidfuzz.process.extract(
            normalized_query,
            state_data["normalized"],
            processor=None,
            scorer=rapidfuzz.fuzz.ratio,
            limit=25,
        )
        if not matches:
            return ("", 0.0, "")

        # Best score per county set; ties among distinct sets are ambiguous
        best_by_fips = {}
        for _name, score, index in matches:
            fips = state_data["fips"][index]
            if score > best_by_fips.get(fips, (-1.0, ""))[0]:
                best_by_fips[fips] = (score, state_data["normalized"][index])
        ranked = sorted(best_by_fips.items(), key=lambda item: (-item[1][0], item[0]))
        best_fips, (best_score, best_name) = ranked[0]
        if len(ranked) > 1 and best_score - ranked[1][1][0] < FUZZY_MIN_MARGIN:
            return ("", 0.0, "")
        return (str(best_fips), float(best_score), str(best_name))

    return (fuzz_search,)


@app.cell
def _(us):
    # Function cleans state names using the US package and returns 2 character abbreviations
    # Returns empty string if no match; DC is included
    # Needed for using fuzzy match function because Census sources use abbreviations
    def clean_state_to_abbr(input):
        if input == "":
            return ""

        # Include DC
        if str(input).upper() == "DISTRICT OF COLUMBIA":
            return "DC"
        if str(input).upper() == "DC":
            return "DC"

        # US package lookup can handle names, abbreviations, FIPs
        state_obj = us.states.lookup(str(input).strip())

        if state_obj:
            return state_obj.abbr
        else:
            return ""

    return (clean_state_to_abbr,)


@app.cell
def _(
    add_fips_using_addfips,
    clean_city_explode,
    clean_county_explode,
    clean_state_to_abbr,
    clean_zip,
    fuzz_search,
    gazetteer_lookup,
    pl,
):
    # Apply functions in sequence to an input dataframe and produce dataframe with matched FIPS
    def add_fips_with_addfips_and_fuzzy_matching(
        df,
        county_col,
        city_col,
        state_col,
        zip_col,
        census_zip_df,
        census_county_map,
        census_place_map,
        cousub_index,
        gnis_index,
    ):
        # Initialize helper columns
        df = df.with_columns(
            [
                pl.col(city_col).alias("xcity"),
                pl.col(county_col).alias("xcounty"),
                pl.col(state_col)
                .map_elements(clean_state_to_abbr, return_dtype=pl.String)
                .alias("xstate"),
                pl.col(zip_col).alias("xzip"),
            ]
        )
        df = clean_city_explode(df, "xcity")
        df = clean_county_explode(df, "xcounty", "xstate")
        df = clean_zip(df, "xzip")

        # 1. Match via addfips
        df = add_fips_using_addfips(df, "county_list", "xstate", "fips_from_addfips")

        # 2. Match via Census ZIP (Join instead of Map)
        df = df.join(
            census_zip_df.select(
                [
                    pl.col("zip").alias("xzip"),
                    pl.col("fips").alias("county_fips_from_census_zip"),
                ]
            ),
            on="xzip",
            how="left",
        ).with_columns(pl.col("county_fips_from_census_zip").fill_null(""))

        # 3. Fuzzy Matching
        # Apply fuzzy logic
        df = df.with_columns(
            [
                pl.struct(["county_list", "xstate"])
                .map_elements(
                    lambda x: fuzz_search(
                        x["xstate"], x["county_list"], census_county_map
                    ),
                    return_dtype=pl.Object,
                )
                .alias("county_fuzzy_raw"),
                pl.struct(["xcity", "xstate"])
                .map_elements(
                    lambda x: fuzz_search(x["xstate"], x["xcity"], census_place_map),
                    return_dtype=pl.Object,
                )
                .alias("city_fuzzy_raw"),
                pl.struct(["county_list", "xstate"])
                .map_elements(
                    lambda x: fuzz_search(
                        x["xstate"], x["county_list"], census_place_map
                    ),
                    return_dtype=pl.Object,
                )
                .alias("ne_fuzzy_raw"),
            ]
        )
        # Expand the fuzzy results into individual columns
        df = df.with_columns(
            [
                pl.col("county_fuzzy_raw")
                .map_elements(lambda x: x[0], return_dtype=pl.String)
                .alias("fips_from_fuzzy_county"),
                pl.col("county_fuzzy_raw")
                .map_elements(lambda x: x[1], return_dtype=pl.Float64)
                .alias("fuzzy_county_score"),
                pl.col("city_fuzzy_raw")
                .map_elements(lambda x: x[0], return_dtype=pl.String)
                .alias("fips_from_fuzzy_city"),
                pl.col("city_fuzzy_raw")
                .map_elements(lambda x: x[1], return_dtype=pl.Float64)
                .alias("fuzzy_city_score"),
                pl.col("ne_fuzzy_raw")
                .map_elements(lambda x: x[0], return_dtype=pl.String)
                .alias("fips_from_fuzzy_ne"),
                pl.col("ne_fuzzy_raw")
                .map_elements(lambda x: x[1], return_dtype=pl.Float64)
                .alias("fuzzy_ne_score"),
            ]
        ).drop(["county_fuzzy_raw", "city_fuzzy_raw", "ne_fuzzy_raw"])

        # 5. Exact city names in supplementary gazetteers (unique county only)
        df = df.with_columns(
            [
                pl.struct(["xcity", "xstate"])
                .map_elements(
                    lambda x: gazetteer_lookup(x["xstate"], x["xcity"], cousub_index),
                    return_dtype=pl.String,
                )
                .alias("fips_from_cousub"),
                pl.struct(["xcity", "xstate"])
                .map_elements(
                    lambda x: gazetteer_lookup(x["xstate"], x["xcity"], gnis_index),
                    return_dtype=pl.String,
                )
                .alias("fips_from_gnis"),
            ]
        )

        return df

    return (add_fips_with_addfips_and_fuzzy_matching,)


@app.cell
def _(pl):
    def pick_best_fips(df, fuzzy_threshold):
        # 4. Define FIPS selection logic

        df = df.with_columns(
            pl.when(pl.col("county_fips_from_census_zip") != "")
            .then(pl.col("county_fips_from_census_zip"))
            .when(pl.col("fips_from_addfips") != "")
            .then(pl.col("fips_from_addfips"))
            # New England puts county name in city column so has extra match check
            .when(
                (pl.col("xstate").is_in(["CT", "ME", "MA", "NH", "RI", "VT"]))
                & (pl.col("fips_from_fuzzy_ne") != "")
                & (pl.col("fuzzy_ne_score") > (fuzzy_threshold + 10))
            )
            .then(pl.col("fips_from_fuzzy_ne"))
            # Use lower threshold fuzzy match result if matches agree
            .when(
                (pl.col("fips_from_fuzzy_county") != "")
                & (pl.col("fips_from_fuzzy_city") != "")
                & (pl.col("fips_from_fuzzy_county") == pl.col("fips_from_fuzzy_city"))
                & (pl.col("fuzzy_county_score") > fuzzy_threshold)
                & (pl.col("fuzzy_city_score") > fuzzy_threshold)
            )
            .then(pl.col("fips_from_fuzzy_county"))
            # Use higher threshold for fuzzy match result if matches disagree
            .when(
                (pl.col("fips_from_fuzzy_county") != "")
                & (pl.col("fips_from_fuzzy_city") != "")
                & (pl.col("fuzzy_county_score") > fuzzy_threshold + 5)
                & (pl.col("fuzzy_city_score") > fuzzy_threshold)
                & (pl.col("fuzzy_county_score") > pl.col("fuzzy_city_score"))
            )
            .then(pl.col("fips_from_fuzzy_county"))
            .when(
                (pl.col("fips_from_fuzzy_county") != "")
                & (pl.col("fips_from_fuzzy_city") != "")
                & (pl.col("fuzzy_county_score") > fuzzy_threshold)
                & (pl.col("fuzzy_city_score") > fuzzy_threshold + 5)
                & (pl.col("fuzzy_city_score") > pl.col("fuzzy_county_score"))
            )
            .then(pl.col("fips_from_fuzzy_city"))
            # Use highest threshold for singular fuzzy matches
            .when(
                (pl.col("fips_from_fuzzy_county") != "")
                & (pl.col("fuzzy_county_score") > fuzzy_threshold + 10)
            )
            .then(pl.col("fips_from_fuzzy_county"))
            .when(
                (pl.col("fips_from_fuzzy_city") != "")
                & (pl.col("fuzzy_city_score") > fuzzy_threshold + 10)
            )
            .then(pl.col("fips_from_fuzzy_city"))
            # Supplementary gazetteers only when Census rules give nothing
            .when(pl.col("fips_from_cousub") != "")
            .then(pl.col("fips_from_cousub"))
            .when(pl.col("fips_from_gnis") != "")
            .then(pl.col("fips_from_gnis"))
            .otherwise(pl.lit(""))
            .alias("matched_county_fips_list")
        )

        return df

    return (pick_best_fips,)


@app.cell
def _(
    add_b_worksite_locations,
    add_fips_with_addfips_and_fuzzy_matching,
    census_county_ref_map_2010,
    census_place_ref_map_2010,
    census_zip_agg_2010,
    cousub_exact_index,
    gnis_exact_index,
    h2a_worksite_locations,
    pick_best_fips,
):
    # Match FIPS for H-2A worksites
    h2a_worksite_locations_added_fips = add_fips_with_addfips_and_fuzzy_matching(
        h2a_worksite_locations,
        "worksite_county",
        "worksite_city",
        "worksite_state",
        "worksite_zip",
        census_zip_agg_2010,
        census_county_ref_map_2010,
        census_place_ref_map_2010,
        cousub_exact_index,
        gnis_exact_index,
    )
    h2a_worksite_locations_added_fips = pick_best_fips(
        h2a_worksite_locations_added_fips, 80
    )

    add_b_worksite_locations_added_fips = add_fips_with_addfips_and_fuzzy_matching(
        add_b_worksite_locations,
        "worksite_county",
        "worksite_city",
        "worksite_state",
        "worksite_zip",
        census_zip_agg_2010,
        census_county_ref_map_2010,
        census_place_ref_map_2010,
        cousub_exact_index,
        gnis_exact_index,
    )
    add_b_worksite_locations_added_fips = pick_best_fips(
        add_b_worksite_locations_added_fips, 80
    )
    return (
        add_b_worksite_locations_added_fips,
        h2a_worksite_locations_added_fips,
    )


@app.cell
def _(
    add_b_df,
    add_b_worksite_locations_added_fips,
    binary_path,
    h2a_df,
    h2a_worksite_locations_added_fips,
    pl,
):
    # Exchange artifacts for 02_02 (API queue) and 02_03 (county assignment)
    location_pieces = pl.concat(
        [
            h2a_worksite_locations_added_fips.with_columns(source=pl.lit("disclosure")),
            add_b_worksite_locations_added_fips.with_columns(source=pl.lit("addendum_b")),
        ],
        how="diagonal",
    )
    h2a_df.write_parquet(binary_path / "h2a_disclosure_records.parquet")
    add_b_df.write_parquet(binary_path / "h2a_addendum_b_records.parquet")
    location_pieces.write_parquet(binary_path / "h2a_location_pieces.parquet")
    print(
        location_pieces.group_by("source")
        .agg(
            pl.len().alias("pieces"),
            (pl.col("matched_county_fips_list") == "").sum().alias("unmatched_locally"),
        )
        .sort("source")
    )
    return


if __name__ == "__main__":
    app.run()
