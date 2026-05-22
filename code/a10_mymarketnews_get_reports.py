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
    import json

    return CACHE, HTTPBasicAuth, RAW, json, os, pl, requests, time


@app.cell
def _(CACHE, HTTPBasicAuth, RAW, os):
    api_key = os.getenv("USDA_MYMARKETNEWS_API_KEY")
    api_auth = HTTPBasicAuth(api_key, "")
    base_url = "https://marsapi.ams.usda.gov/services/v1.2/reports/"

    reports_path = RAW / "mymarketnews_reports"
    work_path = CACHE / "mymarketnews_downloads"
    no_reports_file = CACHE / "mymarketnews_no_reports.json"
    start_year = 2004
    end_year = 2025
    return api_auth, base_url, no_reports_file, reports_path


@app.cell
def _(requests, time):
    def get_json(url, auth, *, timeout=(20, 300), max_attempts=6):
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.get(url=url, auth=auth, timeout=timeout)
                response.raise_for_status()
                return response.json()
            except (requests.exceptions.RequestException, ValueError) as exc:
                last_error = exc
                if attempt == max_attempts:
                    break

                delay = min(300, 5 * 2 ** (attempt - 1))
                print(
                    f"Request failed on attempt {attempt}/{max_attempts}; "
                    f"retrying in {delay}s: {url}"
                )
                time.sleep(delay)

        raise RuntimeError(
            f"Request failed after {max_attempts} attempts: {url}"
        ) from last_error

    return (get_json,)


@app.cell
def _(CACHE, api_auth, base_url, get_json, pl):
    manifest_file = CACHE / "mymarketnews_reports_manifest.parquet"
    if manifest_file.exists():
        print("Manifest already downloaded")
        manifest = pl.read_parquet(manifest_file)
    else:
        manifest_json = get_json(base_url, api_auth)
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
    get_json,
    pl,
    reports_path,
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
            report_header_json = get_json(request_url, api_auth)

            # Put response into dataframe
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
def _(json, no_reports_file):
    # Keep a log of which year-slugs have no reports, so I do not re-query for reports that do not exist
    if no_reports_file.exists():
        no_reports = {
            (item["report_type"], int(item["year"]), str(item["slug"]))
            for item in json.loads(no_reports_file.read_text())
        }
    else:
        no_reports = set()


    def write_no_reports_cache():
        payload = [
            {"report_type": report_type, "year": year, "slug": slug}
            for report_type, year, slug in sorted(no_reports)
        ]
        tmp_file = no_reports_file.with_suffix(".tmp")
        tmp_file.write_text(json.dumps(payload, indent=2))
        tmp_file.replace(no_reports_file)

    return no_reports, write_no_reports_cache


@app.cell
def _(
    api_auth,
    base_url,
    get_json,
    no_reports,
    pl,
    reports_path,
    shipping_point_slugs,
    terminal_market_slugs,
    write_no_reports_cache,
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
        for year in range(2004, 2026):
            year_path = reports_path / str(year)
            try:
                year_path.mkdir()
            except FileExistsError:
                print("Year cache folder already exists")

            for slug in slug_list:
                year_slug_path = year_path / f"{report_type}_{year}_{slug}.parquet"
                if year_slug_path.exists():
                    print(f"Slug {slug} for {year} already exists; skipping")
                    continue

                cache_key = (report_type, year, str(slug))
                if cache_key in no_reports:
                    print(
                        f"No reports available for {report_type} slug {slug} in {year}; skipping"
                    )
                    continue

                slug_request_url = (
                    base_url
                    + str(slug)
                    + "?q="
                    + f"report_begin_date=01/01/{year}:12/31/{year}"
                    + "&allSections=true"
                )
                slug_json = get_json(slug_request_url, api_auth)

                if slug_json[1]["stats"]["totalRows"] == 0:
                    print(f"No results available for {report_type} slug {slug} in {year}")
                    no_reports.add(cache_key)
                    write_no_reports_cache()
                    continue

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
                    slug1_json = get_json(slug1_request_url, api_auth)
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
                    slug2_json = get_json(slug2_request_url, api_auth)
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


if __name__ == "__main__":
    app.run()
