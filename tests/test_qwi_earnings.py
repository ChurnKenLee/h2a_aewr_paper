"""Schema checks for the QWI earnings payload parser."""

import polars as pl

from h2a.qwi import QWI_EARNINGS_FIELDS, parse_qwi_earnings_payload

HEADER = [*QWI_EARNINGS_FIELDS, "time", "industry", "state", "county"]


def test_published_and_suppressed_cells_keep_values_and_status():
    payload = [
        HEADER,
        ["40", "30", "2500", "2400", "300000", "1", "1", "1", "1", "1",
         "2010-Q3", "111", "1", "1"],
        [None, None, None, None, None, "5", "5", "5", "5", "5",
         "2010-Q4", "112", "46", "102"],
    ]
    frame = parse_qwi_earnings_payload(payload)
    assert frame["county_fips"].to_list() == ["01001", "46113"]
    assert frame.schema["county_fips"] == pl.String
    assert frame["year"].to_list() == [2010, 2010]
    assert frame["qtr"].to_list() == [3, 4]
    assert frame["qwi_stable_monthly_earnings"].to_list() == [2500.0, None]
    assert frame["qwi_stable_monthly_earnings_status"].to_list() == [1, 5]
    assert frame["qwi_stable_employment"].to_list() == [30.0, None]


def rejection_message(payload):
    try:
        parse_qwi_earnings_payload(payload)
    except ValueError as error:
        return str(error)
    raise AssertionError("The payload was accepted")


def test_missing_field_and_empty_payload_are_rejected():
    without_earnings = [field for field in HEADER if field != "EarnS"]
    assert "missing fields: EarnS" in rejection_message(
        [without_earnings, ["1"] * len(without_earnings)]
    )
    assert "header and at least one row" in rejection_message([HEADER])
