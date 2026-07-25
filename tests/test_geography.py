"""Unit tests for the Python geographic-code contract."""

from __future__ import annotations

import polars as pl

from h2a.geography import (
    assert_geo_columns,
    county_code_from_county_fips,
    harmonize_county_fips_2010,
    normalize_county_fips_list,
    normalize_geo_code,
    state_from_county_fips,
)


def expect_error(function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except (TypeError, ValueError):
        return
    raise AssertionError("Expected geographic contract validation to fail")


def test_normalization() -> None:
    assert normalize_geo_code(1, "state_fips") == "01"
    assert normalize_geo_code(" 037 ", "county_code") == "037"
    assert normalize_geo_code(6037.0, "county_fips") == "06037"
    assert normalize_geo_code("00100", "cz_id") == "100"
    assert normalize_geo_code("01", "aewr_region_id") == "1"
    assert normalize_geo_code("0900001", "oews_area_code") == "0900001"
    assert harmonize_county_fips_2010("46102") == "46113"
    assert state_from_county_fips("06037") == "06"
    assert county_code_from_county_fips("06037") == "037"
    assert county_code_from_county_fips("46102") == "113"
    assert normalize_county_fips_list("06037, 46102") == "06037,46113"


def test_fail_fast_validation() -> None:
    expect_error(normalize_geo_code, "123", "state_fips")
    expect_error(normalize_geo_code, "06A37", "county_fips")
    expect_error(normalize_geo_code, "18", "aewr_region_id")

    valid = pl.DataFrame(
        {
            "state_fips": ["01", "06"],
            "county_fips": ["01001", "06037"],
            "cz_id": ["100", "200"],
        }
    )
    assert_geo_columns(valid, ["state_fips", "county_fips", "cz_id"])

    expect_error(
        assert_geo_columns,
        valid.with_columns(pl.col("county_fips").cast(pl.Int64)),
        ["county_fips"],
    )
    expect_error(
        assert_geo_columns,
        pl.DataFrame({"county_fips": [None, "06037"]}, schema={"county_fips": pl.String}),
        ["county_fips"],
    )
    expect_error(
        assert_geo_columns,
        pl.DataFrame({"county_fips": ["46102"]}),
        ["county_fips"],
    )
    expect_error(assert_geo_columns, valid, ["missing_geo_field"])


if __name__ == "__main__":
    test_normalization()
    test_fail_fast_validation()
    print("Python geography tests passed")
