import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import pyprojroot
    import polars as pl
    import re

    return pl, pyprojroot, re


@app.cell
def _(pyprojroot):
    root_path = pyprojroot.find_root(criterion='pyproject.toml')
    binary_path = root_path / 'binaries'
    json_path = root_path / 'code' / 'json'
    geo_path = root_path / 'data' / 'geographic_definitions'
    return (geo_path,)


@app.cell
def _(geo_path, pl):
    nber_counties = pl.read_csv(
        geo_path / 'county_adjacency2010.csv'
    ).select([
        'countyname', 'fipscounty'
    ]).unique().with_columns(
        pl.col('countyname').str.split_exact(', ',1).alias('fields')
    ).unnest(
        "fields"
    ).rename({
        'field_0':'nber_county_name',
        'field_1': 'nber_state_abbrev'
    })
    nber_counties
    return (nber_counties,)


@app.cell
def _(nber_counties, pl):
    test = nber_counties.filter(pl.col('nber_state_abbrev') == 'VA')
    test.sort(['fipscounty'])
    return


@app.cell
def _(geo_path, pl):
    bea_xwalk = pl.read_csv(
        geo_path / 'bea_fips_xwalk.csv'
    ).select([
        'realfips', 'beafips', 'bea_place'
    ]).rename({
        'realfips':'countyfips'
    })
    bea_xwalk
    return


@app.cell
def _(geo_path, pl, re):
    # Load and process Census files
    # Strip suffixes from locality names for fuzzy string matching later
    place_suffixes = [
        ' CITY',
        ' TOWN',
        ' VILLAGE',
        ' CDP',
        ' MUNICIPALITY',
        ' BOROUGH',
        ' TOWNSHIP',
        ' CENSUS AREA',
        ' CENSUS DESIGNATED PLACE',
        ' COUNTY',
        ' PARISH'
    ]

    # Construct the regex pattern: (?:suffix1|suffix2|...)$
    # re.escape is used to handle suffixes with special characters like '.'
    place_suffix_pattern = f'(?:{'|'.join(re.escape(s) for s in place_suffixes)})$'

    # Census county names
    census_county = pl.read_csv(geo_path / 'Gaz_counties_national.txt', separator='\t', encoding='latin1', infer_schema=False).with_columns(pl.col(pl.String).str.to_uppercase())
    census_places = pl.read_csv(geo_path / 'national_places.txt', separator='|', encoding='latin1', infer_schema=False).with_columns(pl.col(pl.String).str.to_uppercase())
    census_zcta = pl.read_csv(geo_path / 'zcta_county_rel_10.txt', separator=',', encoding='latin1', infer_schema=False).with_columns(pl.col(pl.String).str.to_uppercase())
    return census_county, census_places, place_suffix_pattern


@app.cell
def _(census_places):
    census_places
    return


@app.cell
def _(pl, place_suffix_pattern):
    census_county = (
        census_county
            .with_columns(
                (pl.col('STATEFP') + pl.col('COUNTYFP')).alias('fips')
            )
            .select(
                [
                    pl.col('STATE').alias('state'),
                    pl.col('COUNTYNAME').alias('county'),
                    pl.col('fips')
                ]
            )
            .filter(
                pl.col('county') != ''
            ).with_columns(
                pl.col('county').str.replace_all(place_suffix_pattern, '')
            )
    )
    return (census_county,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
