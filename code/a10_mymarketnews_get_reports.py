import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    from h2a.paths import CACHE, RAW, INTERMEDIATE
    import os
    import time
    import zipfile
    import polars as pl
    import requests
    from requests.auth import HTTPBasicAuth

    return CACHE, HTTPBasicAuth, RAW, os, pl, requests, time


@app.cell
def _(CACHE, HTTPBasicAuth, RAW, os):
    api_key = os.getenv("USDA_MYMARKETNEWS_API_KEY")
    api_auth = HTTPBasicAuth(api_key, "")
    base_url = "https://marsapi.ams.usda.gov/services/v1.2/reports/"

    reports_path = RAW / "mymarketnews_reports"
    work_path = CACHE / "mymarketnews_downloads"
    start_year = 2004
    end_year = 2025
    return api_auth, base_url, reports_path


@app.cell
def _(CACHE, api_auth, base_url, pl, requests):
    manifest_file = CACHE / "mymarketnews_reports_manifest.parquet"
    if manifest_file.exists():
        print("Manifest already downloaded")
        manifest = pl.read_parquet(manifest_file)
    else:
        _r = requests.get(url=base_url, auth=api_auth)
        manifest_json = _r.json()
        manifest = pl.from_dicts(manifest_json)
        manifest.write_parquet(manifest_file)
    return (manifest,)


@app.cell
def _(manifest, pl):
    # Get slug_id of terminal and shipping point reports
    terminal_market_slugs = (
        manifest.filter(
            pl.col("market_types") == ["Terminal"],
            pl.col("offices") != ["Washington, DC International - SC"],
        )
        .unique(pl.col("slug_id"))
        .get_column("slug_id")
    )

    shipping_point_slugs = (
        manifest.filter(
            pl.col("market_types") == ["Shipping Point"],
            pl.col("offices") != ["Washington, DC International - SC"],
        )
        .unique("slug_id")
        .get_column("slug_id")
    )
    return shipping_point_slugs, terminal_market_slugs


@app.cell
def _(
    api_auth,
    base_url,
    pl,
    reports_path,
    requests,
    terminal_market_slugs,
    time,
):
    report_header_file = reports_path / "mymarketnews_reports_terminal_headers.parquet"
    if report_header_file.exists():
        print("Load already downloaded report headers")
        report_headers = pl.read_parquet(report_header_file)
    else:
        report_headers = pl.DataFrame()
        for _slug_id in list(terminal_market_slugs):
            request_url = base_url + str(_slug_id)
            print(request_url)

            # Make request
            _r = requests.get(url=request_url, auth=api_auth)

            # Put response into dataframe
            report_header_json = _r.json()
            _df = pl.from_dicts(
                report_header_json["results"], infer_schema_length=None
            ).fill_null("")
            report_headers = pl.concat([report_headers, _df])

            time.sleep(1)

        report_headers.write_parquet(
            reports_path / "mymarketnews_reports_terminal_headers.parquet"
        )
    return


@app.cell
def _(
    api_auth,
    base_url,
    pl,
    reports_path,
    requests,
    shipping_point_slugs,
    sleep,
    terminal_market_slugs,
):
    # Grab reports
    try:
        reports_path.mkdir()
    except FileExistsError:
        print("Reports folder already exists")

    slug_dict = {
        "terminal": list(terminal_market_slugs),
        "shipping": list(shipping_point_slugs),
    }

    for report_type, slug_list in slug_dict.items():
        for year in range(2024, 2026):
            year_path = reports_path / str(year)
            try:
                year_path.mkdir()
            except FileExistsError:
                print("Year cache folder already exists")

            for slug in slug_list:
                year_slug_path = year_path / f"{report_type}_{year}_{slug}.parquet"
                if year_slug_path.exists():
                    print(f"Slug {slug} for {year} already exists")
                    continue

                slug_request_url = (
                    base_url
                    + str(slug)
                    + "?q="
                    + f"report_begin_date=01/01/{year}:12/31/{year}"
                    + "&allSections=true"
                )
                slug_response = requests.get(url=slug_request_url, auth=api_auth)
                try:
                    slug_json = slug_response.json()
                except:
                    sleep(1)
                    slug_response = requests.get(url=slug_request_url, auth=api_auth)

                slug_json = slug_response.json()

                if slug_json[1]["stats"]["totalRows"] == 0:
                    print(f"No results available for slug {slug}")

                elif slug_json[1]["stats"]["totalRows"] > 100000:
                    print("Too many rows, have to split request")

                    # January to June reports
                    slug1_request_url = (
                        base_url
                        + str(slug)
                        + "?q="
                        + f"report_begin_date=01/01/{year}:06/30/{year}"
                        + "&allSections=true"
                    )
                    slug1_response = requests.get(url=slug1_request_url, auth=api_auth)
                    slug1_json = slug1_response.json()
                    slug1_report_details = pl.from_dicts(
                        slug1_json[1]["results"], infer_schema_length=None
                    ).fill_null("")
                    print(
                        f"Slug {slug} report part 1 has {slug1_report_details.height} rows"
                    )

                    # July to December reports
                    slug2_request_url = (
                        base_url
                        + str(slug)
                        + "?q="
                        + f"report_begin_date=07/01/{year}:12/31/{year}"
                        + "&allSections=true"
                    )
                    slug2_response = requests.get(url=slug2_request_url, auth=api_auth)
                    slug2_json = slug2_response.json()
                    slug2_report_details = pl.from_dicts(
                        slug2_json[1]["results"], infer_schema_length=None
                    ).fill_null("")
                    print(
                        f"Slug {slug} report part 2 has {slug2_report_details.height} rows"
                    )

                    slug_report_combined = pl.concat(
                        [slug1_report_details, slug2_report_details]
                    )
                    slug_report_combined.write_parquet(year_slug_path)

                else:
                    slug_report_details = pl.from_dicts(
                        slug_json[1]["results"], infer_schema_length=None
                    ).fill_null("")
                    print(
                        f"Year {year} slug {slug} report has {slug_report_details.height} rows"
                    )
                    slug_report_details.write_parquet(year_slug_path)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
