+++
title = "Geographic code contract"
description = "Canonical geographic identifiers shared by R and Python producers."

[extra]
scopes = ["code/c00_shared", "code/c01_clean", "code/c02_build", "src/h2a"]
+++


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

The annual QCEW producer publishes `county_fips`, not the source field
`area_fips`. It prefers an already-canonical 2010 code when old and new QCEW
codes coexist in a transition year, maps later Oglala Lakota and Kusilvak
records to their 2010 predecessors, and combines the Chugach and Copper River
records into 2010-vintage Valdez-Cordova. Undefined `xx999` areas and source
geographies that cannot be allocated to 2010 counties are excluded rather
than treated as counties or repaired by consumers.

R scripts that normalize identifiers source `code/c00_shared/geography.R`;
Python producers import `h2a.geography`. The C01 merge, shared-panel build,
and design-specific panel builders check that their supported artifacts are
nonempty and unique on their declared keys.

{{ grounding(path="code/c00_shared/geography.R", anchor="geographic-code-contract-r", sha256="79622391dad204af12c046ee90147524c89a51b5e00ce3610b7e09cdae6a7050") }}

{{ grounding(path="src/h2a/geography.py", anchor="geographic-code-contract-python", sha256="764ab9a606d6b512dcff7a57ec64cb3ce7f651b443eef3619f5532c0924c7ee4") }}
