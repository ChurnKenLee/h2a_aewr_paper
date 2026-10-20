# Purpose: Assign 2010 county FIPS lists to H-2A disclosure and Addendum B
#   records from local matches (02_01) and cached API answers (02_02), then
#   apply labeled record-level fallbacks.
# Inputs: h2a_disclosure_records.parquet, h2a_addendum_b_records.parquet,
#   h2a_location_pieces.parquet, h2a_location_api_counties.parquet and the
#   2010 county shapefile.
# Outputs: h2a_with_fips.parquet and h2a_addendum_b_with_fips.parquet.

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import re

    import addfips
    import polars as pl
    import us

    from h2a.geography import normalize_county_fips_list
    from h2a.paths import INTERMEDIATE, RAW

    return INTERMEDIATE, RAW, addfips, normalize_county_fips_list, pl, re, us


@app.cell
def _(INTERMEDIATE, normalize_county_fips_list, pl):
    # Each piece takes its local match, else its API answer. Pieces of one
    # location (split city or county fields) are unioned into one sorted,
    # duplicate-free county list per original location.
    LOCATION_KEYS = ["worksite_city", "worksite_county", "worksite_state", "worksite_zip"]
    PIECE_KEYS = ["source", "xcity", "county_list", "xstate", "xzip"]
    _pieces = pl.read_parquet(INTERMEDIATE / "h2a_location_pieces.parquet")
    _api = pl.read_parquet(INTERMEDIATE / "h2a_location_api_counties.parquet").select(
        [*PIECE_KEYS, "places_county_fips_list"]
    )
    _pieces = _pieces.join(_api, on=PIECE_KEYS, how="left").with_columns(
        pl.col("places_county_fips_list").fill_null("")
    ).with_columns(
        county_fips_list=pl.when(pl.col("matched_county_fips_list") != "")
        .then(pl.col("matched_county_fips_list"))
        .otherwise(pl.col("places_county_fips_list"))
    )

    def collapse(column):
        return (
            pl.col(column)
            .filter(pl.col(column) != "")
            .str.join(",")
            .str.split(",")
            .list.unique()
            .list.sort()
            .list.join(",")
            .map_elements(normalize_county_fips_list, return_dtype=pl.String)
        )

    location_counties = _pieces.group_by(["source", *LOCATION_KEYS]).agg(
        collapse("county_fips_list"),
        collapse("matched_county_fips_list"),
        collapse("places_county_fips_list"),
    )

    def attach(records_file, source):
        records = pl.read_parquet(INTERMEDIATE / records_file)
        counties = location_counties.filter(pl.col("source") == source).drop("source")
        located = records.join(counties, on=LOCATION_KEYS, how="left")
        if located.height != records.height:
            raise ValueError(f"{source}: location join changed the record count")
        return located

    h2a_located_df = attach("h2a_disclosure_records.parquet", "disclosure")
    add_b_located_df = attach("h2a_addendum_b_records.parquet", "addendum_b")
    return add_b_located_df, h2a_located_df


@app.cell
def _(RAW, addfips, re, us):
    import geopandas as gpd
    from shapely.geometry import Point

    # Record-level fallbacks for rows the location pass left without a county.
    # Each fill is labeled in county_fips_fallback_method; location-based
    # columns (matched/places) are left unchanged for audit.
    decimal_pair = re.compile(r"(-?\d{1,3}\.\d{3,})\s*[,;/\s]\s*(-?\d{1,3}\.\d{3,})")
    dms_value = re.compile(
        r"(-?\d{1,3})\s*[°º˚ ]\s*(\d{1,2})\s*['’′ ]\s*(\d{1,2}(?:\.\d+)?)?"
        r"\s*(?:\"|”|″|'')?\s*([NSEW])?"
    )
    county_in_text = re.compile(
        r"\b(?:WITHIN|IN|OF|THROUGHOUT|AROUND|ACROSS)\s+([A-Z][A-Z.' -]{1,30}?)\s+"
        r"(?:COUNTY|PARISH)\b"
    )
    county_af = addfips.AddFIPS(vintage=2010)
    county_polygons_2010 = gpd.read_file(
        f"/vsizip/{RAW / 'county_shapefile' / 'tl_2010_us_county10.zip'}"
        "/tl_2010_us_county10.shp",
        columns=["GEOID10", "STATEFP10"],
    ).to_crs(4326)

    def state_fips_of(value):
        state = us.states.lookup(str(value or "").strip()) if value else None
        if str(value or "").strip().upper() in ("DC", "DISTRICT OF COLUMBIA"):
            return "11"
        return state.fips if state else ""

    def parse_points(text):
        # Decimal pairs first; otherwise degree-minute-second pairs. Points are
        # kept only inside a US latitude/longitude box; positive longitudes are
        # read as west and swapped pairs are reordered.
        text = text.upper()
        pairs = [(float(a), float(b)) for a, b in decimal_pair.findall(text)]
        if not pairs:
            values = []
            for match in dms_value.finditer(text):
                deg, mins, sec, hemi = match.groups()
                value = abs(float(deg)) + float(mins) / 60 + float(sec or 0) / 3600
                values.append((-value if deg.startswith("-") or hemi in ("S", "W") else value, hemi))
            for i in range(0, len(values) - 1, 2):
                (lat, h1), (lon, h2) = values[i], values[i + 1]
                if h1 in ("E", "W") or h2 in ("N", "S"):
                    lat, lon = lon, lat
                pairs.append((lat, lon))
        points = []
        for lat, lon in pairs:
            if lat < 0 < lon:
                lat, lon = lon, lat
            lon = -abs(lon)
            if 17 <= lat <= 72 and -180 <= lon <= -60:
                points.append((lat, lon))
        return points

    def counties_from_points(rows):
        # rows: list of (row_id, state_fips, [(lat, lon), ...]); a point counts
        # only when its 2010 county lies in the row's reported state
        ids, geoms = [], []
        for row_id, _state, points in rows:
            for lat, lon in points:
                ids.append(row_id)
                geoms.append(Point(lon, lat))
        if not ids:
            return {}
        located = gpd.sjoin(
            gpd.GeoDataFrame({"row_id": ids}, geometry=geoms, crs=4326),
            county_polygons_2010,
            predicate="within",
            how="inner",
        )
        state_by_row = {row_id: state for row_id, state, _ in rows}
        found = {}
        for row_id, geoid, statefp in zip(
            located["row_id"], located["GEOID10"], located["STATEFP10"], strict=True
        ):
            if statefp == state_by_row[row_id]:
                found.setdefault(row_id, set()).add(geoid)
        return {row_id: ",".join(sorted(f)) for row_id, f in found.items()}

    def counties_from_text(text, state):
        names = {m.group(1).strip() for m in county_in_text.finditer(text.upper())}
        fips = {county_af.get_county_fips(name, state) for name in names} - {None}
        return ",".join(sorted(fips))

    def same_state_counties(county_lists, state_fips):
        # Polars evaluates UDFs on every row, so tolerate missing inputs
        if not county_lists or not state_fips:
            return ""
        counties = {
            c for value in county_lists for c in (value or "").split(",") if c[:2] == state_fips and c
        }
        return ",".join(sorted(counties))

    return (
        counties_from_points,
        counties_from_text,
        parse_points,
        same_state_counties,
        state_fips_of,
    )


@app.cell
def _(
    add_b_located_df,
    counties_from_points,
    counties_from_text,
    h2a_located_df,
    parse_points,
    pl,
    same_state_counties,
    state_fips_of,
):
    # 1. Addendum B: coordinates, then "within X County" text, in the address
    _add_b = add_b_located_df.with_row_index("_row").with_columns(
        _state_fips=pl.col("worksite_state").map_elements(state_fips_of, return_dtype=pl.String),
        _missing=pl.col("county_fips_list").fill_null("") == "",
    )
    _todo = _add_b.filter(pl.col("_missing")).select(
        "_row", "_state_fips", "worksite_state", "worksite_address_1", "worksite_address_2"
    )
    _point_rows, _text_fill = [], {}
    for _row, _sf, _state, _a1, _a2 in _todo.iter_rows():
        _text = f"{_a1} {_a2}"
        _points = parse_points(_text)
        if _points and _sf:
            _point_rows.append((_row, _sf, _points))
        if _sf:
            _fips = counties_from_text(_text, _state)
            if _fips:
                _text_fill[_row] = _fips
    _point_fill = counties_from_points(_point_rows)
    _direct = pl.DataFrame(
        [
            (row, _point_fill[row], "coordinates") if row in _point_fill
            else (row, _text_fill[row], "address_county")
            for row in sorted(set(_point_fill) | set(_text_fill))
        ],
        schema={"_row": pl.UInt32, "_fill": pl.String, "_method": pl.String},
        orient="row",
    )
    _add_b = _add_b.join(_direct, on="_row", how="left")

    # 2. Main file: union of the same case's directly located Addendum B
    #    counties within the main record's state
    _add_b_case_counties = (
        _add_b.with_columns(
            _county=pl.when(pl.col("_missing")).then(pl.col("_fill")).otherwise(pl.col("county_fips_list"))
        )
        .group_by("case_number", "fiscal_year")
        .agg(pl.col("_county").drop_nulls().alias("_add_b_counties"))
    )
    _main = h2a_located_df.with_columns(
        _state_fips=pl.col("worksite_state").map_elements(state_fips_of, return_dtype=pl.String),
        _missing=pl.col("county_fips_list").fill_null("") == "",
    ).join(_add_b_case_counties, on=["case_number", "fiscal_year"], how="left")
    _main = _main.with_columns(
        _fill=pl.when(pl.col("_missing") & pl.col("_add_b_counties").is_not_null())
        .then(
            pl.struct("_add_b_counties", "_state_fips").map_elements(
                lambda x: same_state_counties(x["_add_b_counties"], x["_state_fips"]),
                return_dtype=pl.String,
            )
        )
        .otherwise(pl.lit(""))
    )
    h2a_with_fips_df = _main.with_columns(
        county_fips_fallback_method=pl.when(pl.col("_fill") != "")
        .then(pl.lit("case_addendum_b_county_same_state"))
        .otherwise(pl.lit("")),
        county_fips_list=pl.when(pl.col("_fill") != "")
        .then(pl.col("_fill"))
        .otherwise(pl.col("county_fips_list")),
    ).drop(["_state_fips", "_missing", "_add_b_counties", "_fill"])

    # 3. Addendum B remainder: the case's main-file county within the same state
    _main_case_counties = (
        h2a_located_df.group_by("case_number", "fiscal_year")
        .agg(pl.col("county_fips_list").drop_nulls().alias("_main_counties"))
    )
    _add_b = _add_b.join(_main_case_counties, on=["case_number", "fiscal_year"], how="left")
    _add_b = _add_b.with_columns(
        _case_fill=pl.when(pl.col("_missing") & pl.col("_fill").is_null() & pl.col("_main_counties").is_not_null())
        .then(
            pl.struct("_main_counties", "_state_fips").map_elements(
                lambda x: same_state_counties(x["_main_counties"], x["_state_fips"]),
                return_dtype=pl.String,
            )
        )
        .otherwise(pl.lit(""))
    )
    add_b_with_fips_df = _add_b.with_columns(
        county_fips_fallback_method=pl.when(pl.col("_fill").is_not_null())
        .then(pl.col("_method"))
        .when(pl.col("_case_fill") != "")
        .then(pl.lit("case_main_county_same_state"))
        .otherwise(pl.lit("")),
        county_fips_list=pl.when(pl.col("_fill").is_not_null())
        .then(pl.col("_fill"))
        .when(pl.col("_case_fill") != "")
        .then(pl.col("_case_fill"))
        .otherwise(pl.col("county_fips_list")),
    ).sort("_row").drop(["_row", "_state_fips", "_missing", "_fill", "_method", "_main_counties", "_case_fill"])

    for _name, _df in [("H-2A", h2a_with_fips_df), ("Addendum B", add_b_with_fips_df)]:
        print(
            _name,
            dict(_df.group_by("county_fips_fallback_method").len().sort("county_fips_fallback_method").rows()),
            "still without county:",
            (_df["county_fips_list"].fill_null("") == "").sum(),
        )
    return add_b_with_fips_df, h2a_with_fips_df




@app.cell
def _(INTERMEDIATE, add_b_with_fips_df, h2a_with_fips_df):
    h2a_with_fips_df.write_parquet(INTERMEDIATE / "h2a_with_fips.parquet")
    add_b_with_fips_df.write_parquet(INTERMEDIATE / "h2a_addendum_b_with_fips.parquet")
    return


if __name__ == "__main__":
    app.run()
