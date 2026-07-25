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
first create a persistent artifact. Consumers join directly on the canonical
fields and fail when a required field is missing, malformed, incorrectly
typed, or unexpectedly null.

The project uses 2010 county geography. The shared normalizers map the current
Oglala Lakota County, South Dakota code `46102` to its 2010-vintage Shannon
County code `46113`.
Source-specific names such as `state_ansi`, `county_ansi`, `GeoFIPS`, and
`GEOID10` may appear only while reading raw data; they are not persistent
artifact fields.

R scripts source `code/c00_shared/geography.R`. Python scripts import
`h2a.geography`. The B-stage
`08_validate_geography_contract.R` script provides a read-only check after
artifacts have been rebuilt.
