"""Small schema test for the QWI API parser."""

from h2a.qwi import parse_qwi_payload


def test_parse_qwi_payload() -> None:
    payload = [
        [
            "Emp",
            "EmpS",
            "EmpTotal",
            "sEmp",
            "sEmpS",
            "sEmpTotal",
            "time",
            "state",
            "county",
            "industry",
            "ownercode",
            "seasonadj",
        ],
        [
            "100",
            "75",
            "130",
            "1",
            "1",
            "1",
            "2020-Q2",
            "6",
            "1",
            "111",
            "A05",
            "U",
        ],
    ]
    parsed = parse_qwi_payload(payload)
    row = parsed.row(0, named=True)
    assert row["county_fips"] == "06001"
    assert row["year"] == 2020
    assert row["qtr"] == 2
    assert row["qwi_beginning_quarter_employment"] == 100
    assert row["qwi_stable_employment"] == 75


if __name__ == "__main__":
    test_parse_qwi_payload()
    print("QWI extractor tests passed")
