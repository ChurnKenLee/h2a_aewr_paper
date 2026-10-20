"""Public geocoder response/identity safeguards without network requests."""

from h2a.worksite_geocoding import normalize_address, parse_census_batch


def test_missing_worksite_street_never_uses_a_mailing_address():
    row = {"source_street": "", "source_city": "Athens", "source_state": "Georgia", "source_zip": "30601",
           "employer_address_1": "100 EXAMPLE STREET"}
    value = normalize_address(row)
    assert not value["requestable"]
    assert value["street"] == ""
    assert value["state"] == "GA"


def test_response_retains_unmatched_and_string_geography():
    response = 'a,"input",Match,Exact,"matched","-84.1,33.2",12,L,13,001,010100,1000\nb,"input",No_Match\n'
    result = parse_census_batch(response, {"a", "b"})
    assert result[0]["census_county_fips_current"] == "13001"
    assert result[0]["longitude"] == -84.1
    assert result[1]["longitude"] is None
    assert result[1]["coordinate_quality"] == "no_unique_point"


def test_partial_or_duplicate_responses_cannot_be_promoted():
    for response in ('a,"input",No_Match\n', 'a,"input",No_Match\na,"input",No_Match\n'):
        try:
            parse_census_batch(response, {"a", "b"})
        except ValueError:
            pass
        else:
            raise AssertionError("Incomplete geocoder response was accepted")
