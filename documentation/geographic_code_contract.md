# Geographic code contract

Pipeline artifacts use one name and one character representation for each
geographic identifier:

| Field | Representation |
| --- | --- |
| `state_fips` | Two digits |
| `county_code` | Three-digit county component |
| `county_fips` | Five-digit county FIPS, harmonized to the 2010 county vintage |
| `neighbor_county_fips` | Five-digit neighboring county FIPS |
| `cz_id` | Unpadded digits |
| `aewr_region_id` | Unpadded digits from `1` through `17` |
| `oews_area_code` | Digits of source-defined width; leading zeroes preserved |

All fields are strings. Producers normalize source-specific fields when they
first create a persistent artifact, and downstream consumers join directly on
the canonical fields. Final supported artifacts are checked for nonempty,
unique keys.

The project uses 2010 county geography. The shared normalizers map the current
Oglala Lakota County, South Dakota code `46102` to its 2010-vintage Shannon
County code `46113`.
Source-specific names such as `state_ansi`, `county_ansi`, `GeoFIPS`, and
`GEOID10` may appear only while reading raw data; they are not persistent
artifact fields.

R scripts that normalize identifiers source `code/c00_shared/geography.R`;
Python producers import `h2a.geography`. `scripts/run_tests.sh` checks the
final supported panel keys.
