import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    # Download county-quarter agricultural employment from the QWI API.

    # API responses are cached by state, industry, and year range so interrupted runs can resume without repeating completed requests.

    # Output: data/intermediate/qwi_county_ag_quarterly_employment.parquet
    return


@app.cell
def _():
    import json
    import os
    import dotenv
    import time

    import polars as pl
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    from h2a.paths import CACHE, INTERMEDIATE
    from h2a.geography import (
        assert_geo_columns,
    )
    from h2a.qwi import parse_qwi_payload

    return (
        CACHE,
        HTTPAdapter,
        INTERMEDIATE,
        Retry,
        assert_geo_columns,
        dotenv,
        json,
        os,
        parse_qwi_payload,
        pl,
        requests,
        time,
    )


@app.cell
def _(CACHE, INTERMEDIATE, dotenv, os):
    # Run settings
    FIRST_YEAR = 2000
    LAST_YEAR = 2024
    REFRESH_CACHE = False

    API_ENDPOINT = "https://api.census.gov/data/timeseries/qwi/sa"
    OUTPUT_PATH = INTERMEDIATE / "qwi_county_ag_quarterly_employment.parquet"
    CACHE_PATH = CACHE / "qwi"
    CACHE_PATH.mkdir(parents=True, exist_ok=True)
    dotenv.load_dotenv()
    CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")
    return (
        API_ENDPOINT,
        CACHE_PATH,
        CENSUS_API_KEY,
        FIRST_YEAR,
        LAST_YEAR,
        OUTPUT_PATH,
        REFRESH_CACHE,
    )


@app.cell
def _():
    # Request specs reference at https://www.census.gov/data/developers/data-sets/qwi.html
    INDUSTRIES = ("111", "112")
    STATE_FIPS = """
    01 02 04 05 06 08 09 10 11 12 13 15 16 17 18 19 20 21 22 23 24 25 26
    27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 44 45 46 47 48 49 50
    51 53 54 55 56
    """.split()
    QWI_FIELDS = ("Emp", "EmpS", "EmpTotal", "sEmp", "sEmpS", "sEmpTotal")
    REQUIRED_FIELDS = {
        "time",
        "state",
        "county",
        "industry",
        "ownercode",
        "seasonadj",
        *QWI_FIELDS,
    }
    return INDUSTRIES, QWI_FIELDS, REQUIRED_FIELDS, STATE_FIPS


@app.cell
def _(
    API_ENDPOINT,
    CACHE_PATH,
    CENSUS_API_KEY,
    FIRST_YEAR,
    HTTPAdapter,
    INDUSTRIES,
    LAST_YEAR,
    OUTPUT_PATH,
    QWI_FIELDS,
    REFRESH_CACHE,
    REQUIRED_FIELDS,
    Retry,
    STATE_FIPS,
    assert_geo_columns,
    json,
    parse_qwi_payload,
    pl,
    requests,
    time,
):
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))

    frames = []

    for state_fips in STATE_FIPS:
        for industry in INDUSTRIES:
            print(f"QWI state {state_fips}, NAICS {industry}", flush=True)

            cache_file = CACHE_PATH / (
                f"qwi_{state_fips}_{industry}_{FIRST_YEAR}_{LAST_YEAR}.json"
            )

            if cache_file.exists() and not REFRESH_CACHE:
                payload = json.loads(cache_file.read_text(encoding="utf-8"))
            else:
                params = {
                    "get": ",".join(QWI_FIELDS),
                    "for": "county:*",
                    "in": f"state:{state_fips}",
                    "time": f"from {FIRST_YEAR}-Q1 to {LAST_YEAR}-Q4",
                    "industry": industry,
                    "ownercode": "A05",
                    "seasonadj": "U",
                    "sex": "0",
                    "agegrp": "A00",
                    "key": CENSUS_API_KEY,
                }
                response = session.get(API_ENDPOINT, params=params, timeout=120)
                response.raise_for_status()

                if response.status_code == 204 or not response.content.strip():
                    payload = []
                else:
                    try:
                        payload = response.json()
                    except requests.exceptions.JSONDecodeError as exc:
                        content_type = response.headers.get("content-type", "unknown")
                        body_preview = " ".join(response.text.split())[:200]
                        raise RuntimeError(
                            f"QWI returned non-JSON content for state "
                            f"{state_fips}, NAICS {industry} "
                            f"(HTTP {response.status_code}, {content_type}): "
                            f"{body_preview or '<empty body>'}"
                        ) from exc

                if isinstance(payload, dict) and "error" in payload:
                    raise RuntimeError(str(payload["error"]))

                cache_file.write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )
                time.sleep(0.15)

            if not isinstance(payload, list):
                raise TypeError(
                    f"Unexpected QWI response for state {state_fips}, "
                    f"NAICS {industry}: {payload!r}"
                )
            if len(payload) < 2:
                print(
                    f"No QWI rows for state {state_fips}, NAICS {industry}; skipping",
                    flush=True,
                )
                continue

            try:
                frame = parse_qwi_payload(payload)
            except ValueError as exc:
                raise ValueError(
                    f"QWI response for state {state_fips}, "
                    f"NAICS {industry}: {exc}"
                ) from exc
            frames.append(frame)

    session.close()

    if not frames:
        raise RuntimeError("The QWI API returned no county-quarter rows.")

    qwi = pl.concat(frames, how="vertical_relaxed").sort(
        "county_fips",
        "year",
        "qtr",
        "industry_code",
    )
    assert_geo_columns(qwi, ["county_fips"])
    qwi.write_parquet(OUTPUT_PATH)
    print(f"Wrote {qwi.height:,} QWI rows to {OUTPUT_PATH}", flush=True)
    return


if __name__ == "__main__":
    app.run()
