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
BEA combined Virginia reporting areas (`519xx`) require a separate
source-specific concordance: county-level BEA producers map each combined area
to the principal 2010 county identified by
`data/raw/geographic_crosswalks/phil/bea_fips_xwalk.csv`. They do not treat a
BEA combined-area code as a county FIPS or allocate its value to independent
cities.
Source-specific names such as `state_ansi`, `county_ansi`, `GeoFIPS`, and
`GEOID10` may appear only while reading raw data; they are not persistent
artifact fields.

R scripts that normalize identifiers source `code/c00_shared/geography.R`;
Python producers import `h2a.geography`. The C01 merge, shared-panel build,
and design-specific panel builders check that their supported artifacts are
nonempty and unique on their declared keys.
