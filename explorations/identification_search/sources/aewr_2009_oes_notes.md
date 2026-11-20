# A2 acquisition notes: 2009 OES-based AEWRs and FY2006–07 H-2A disclosures

Phase 2 acquisition for strategy A2 in `markdowns/identification_search_agent_plan.md`.
Retrieved 2026-11-05. Scripts are in `explorations/identification_search/A2/`
(`00`–`02`). No producer, retained output or manuscript file was changed.

## 1. The OES-based AEWR schedules (2008 rule) — acquired

### Legal basis (primary text, local)

- 20 CFR 655.108(e) as published in the 2008 final rule (73 FR 77110,
  `data/raw/aewr_2009_oes/E8-29309.htm`, govinfo):
  - The AEWR "will be based on published wage data for the occupation, skill
    level, and geographical area from the [BLS] OES survey."
  - The NPC obtains it "using the On-line Wage Library (OWL) found on the
    Foreign Labor Certification Data Center Web site".
  - The AEWR "shall not be less than the July 24, 2009 Federal minimum wage of
    $7.25." This floor applies from the rule's start (2009-01-17), before the
    federal minimum itself rose. `aewr_hourly` = max(OWL wage, 7.25).
- 655.108(a): the employer offers the highest of the AEWR "in effect at the
  time recruitment for a position is begun", the prevailing wage, and the
  federal or state minimum wage. The wage year attaches to the NPC wage
  determination and recruitment start, not strictly to the filing date.
- 655.108(g): definitions of skill Levels I–IV.
- 655.93: sheepherders and other special-procedure occupations keep their
  special procedures (monthly rates under TEGLs). The OWL 45-2093 rows are
  **not** the herder AEWR. The 2010 rule's Table 1 also excludes monthly-paid
  range and combine workers.

### Source files

FLC Data Center downloads, from Wayback Machine captures. The live site is
gone, and DataLumos project 100447 returns a Cloudflare challenge (HTTP 403)
to scripted access.

| Raw file (`data/raw/aewr_2009_oes/`) | Content | Wayback capture |
| --- | --- | --- |
| `OWL_2009_TEXT.zip`, `OWL_2009_Download.zip` (mdb) | FLC wage data 7/2008–6/2009 | 20100103170946, 20100103170640 |
| `OWL_2010_TEXT.zip`, `OWL_2010_Download.zip` (mdb) | FLC wage data 7/2009–6/2010 | 20100103171029, 20100103170850 |
| `OWL_2008_TEXT.zip`, `OWL_2008_Download.zip` | 7/2007–6/2008 (not used under the 2008 rule; kept for reference) | 20100103171222, 20100206144208 |
| `Download_aspx_20090226212449.html`, `Download_aspx_20100103170404.html` | FLC download page: maps file names to wage years | as named |
| `OWL_UserGuide.doc`, `geography_zip_wayback20070728.zip` | OWL user guide; 2006 area-definition file | 20090220061150, 20070728233616 |
| `E8-29309.htm` | 2008 final rule, govinfo | not Wayback |

URL pattern: `https://web.archive.org/web/<capture>/http://flcdatacenter.com/download/<file>`.

**Mapping file names to wage years.** The archived download page says so
directly:

- 2009-02-26 capture: "Download the 7/2008 - 6/2009 FLC Wage Data" →
  `OWL_2009_*`. The same page says "FLC Wage Data updated July 1, 2008".
- 2010-01-03 capture: "Download the 7/2009 - 6/2010 FLC Wage Data …
  Updated July 1, 2009" → `OWL_2010_*`.

**OES vintage (verified).** Each OWL area × SOC row gives Levels I–IV.
OFLC's convention is that (L1 + 2·L4)/3 equals the OES mean. Comparing that
implied mean with the local BLS files (`data/raw/oews/oesm0*ma.zip`) for
five agricultural SOCs at GeoLvl 1 shows:

- `OWL_2009` = **BLS OES May 2007**: 1,301 of 1,301 rows within $0.02;
  median gap $0.003.
- `OWL_2010` = **BLS OES May 2008**: 1,266 of 1,266 rows within $0.02.
- Every other pairing has a median gap of $0.45–1.18.

The structure RTF inside every zip still says "7/2007–6/2008". That is a stale
template and should be ignored.

### Which wage year applied to which dates

| Wage year (OWL file) | OES data | Applies under the 2008 rule |
| --- | --- | --- |
| 7/2008–6/2009 (`OWL_2009`) | May 2007 | Wage determinations and recruitment from 2009-01-17 to 2009-06-30 |
| 7/2009–6/2010 (`OWL_2010`) | May 2008 | From 2009-07-01 until the 2010 rule (filings before 2010-03-15), plus grandfathered 2008-rule work |

The case-level check (section "Verification") confirms the 2009-07-01 switch
by receipt month. Matches at an OWL 45-2092 Level I rate above $7.25 were:

- `OWL_2009`: 14.8% of hourly cases received 2009-01-17 to 2009-06-30, and
  1.3% from 2009-07-01 to 2010-03-14.
- `OWL_2010`: 0.3% before 2009-07-01, then 13.7% in July 2009 and 16.3% over
  2009-07-01 to 2010-03-14.

The suspension rule (effective 2009-06-29) and its injunction do not appear
in the OWL files. Their dating remains open (see the exploration note).

### Tables built

**`aewr_2009_oes_schedule.csv`: 131,768 rows.**

- One row per wage year × state × OES area × SOC × skill level. A multi-state
  MSA appears once per state.
- Coverage:
  - 2 wage years.
  - 573 areas (2008–09) and 574 areas (2009–10).
  - 54 state codes, including DC, PR, GU and VI.
  - 26 SOCs: all `45-*`, plus 11-9011, 11-9012, 37-3011, 47-2061, 49-3041,
    51-3022, 53-3032, 53-3033, 53-7062 and 53-7064.
  - Levels I–IV.
- Columns:
  - `owl_hourly_wage`: OWL ALC "All Industries" value as published.
  - `aewr_hourly`: max(OWL, 7.25).
  - `floor_725_binds`.
  - `owl_geo_level`: 1 = own area; 2 = area plus contiguous areas; 3–4 =
    wider fallbacks, per the 2008 rule preamble.
  - `oes_area_code`: string; 7-digit nonmetro codes are left-padded, since
    the OWL drops the leading zero (`100001` → `0100001`).
  - `oes_area_code_owl`: the raw code.
  - Source URL, retrieval date, confidence.
- 45-2092 Level I across areas:
  - 2008–09: mean $7.68, SD $0.98, range $6.22–14.58. 36% of areas are below
    $7.25. The within-state SD of the AEWR is $0.64.
  - 2009–10: mean $8.02, SD $1.01. 24% of areas are below $7.25. Within-state
    SD $0.71.
- 45-2092 GeoLvl (2008–09): 348 areas at level 1, 209 at level 2, and 16 at
  levels 3–4.

**`oes_area_county_crosswalk_2009.csv`: 9,958 rows.**

- One row per wage year × OWL Geography row (county, or New England town
  shown as "COUNTY (TOWN)"), with `county_fips`.
- County FIPS come from OWL names matched to
  `data/intermediate/oews_area_definitions.parquet` (2008/2009 vintages, read
  only), in this order:
  1. county name;
  2. New England town name;
  3. county of a matched town;
  4. truncated-name prefix;
  5. 24 hand assignments (2010-vintage FIPS: Shannon SD → 46113; Clifton
     Forge VA → 51005).
- 3,141 counties are mapped in each wage year. 41 counties (all New England)
  span more than one OES area; use the town rows or weight them.
- Unmatched: PR (81), GU (46), VI (3), Yellowstone National Park (MT) and the
  Oak Ridge Reservation (TN).
- Area definitions differ between the two wage years. In 2009–10 the AZ
  nonmetro areas were renumbered, and 29420 (Lake Havasu), 37380 (Palm Coast)
  and 14600 (Bradenton, replacing 42260) were added. The 2010 download page
  notes the Palm Coast fix.

### Verification

Outputs: `aewr_2009_oes_verification.csv`, plus scratch files
`$S/A2/verify_state_table1.csv` and `$S/A2/verify_case_match_by_month.csv`.

1. **Against Table 1 of the 2010 rule** (column 4 = local
   `aewr_2009_nonstatutory_wages`):
   - Local FY2009 hourly cases received after 2009-01-19 reproduce the
     Table 1 state means: corr 0.978, position-weighted mean |gap| $0.055,
     49 states.
   - The schedule-implied floor, max(45-2092 Level I, 7.25) for the worksite
     area, averages $0.47 below Table 1 (corr 0.52). It is at or below
     Table 1 (+$0.05) in 90% of states.
   - This fits Table 1 being a certified wage that also reflects prevailing
     wages, higher levels and other SOCs. The 2010 rule says 60% of 2008-rule
     applications were certified above the AEWR.
   - Exceptions where the floor exceeds Table 1:
     - MA (+$1.15): Level I in the MA areas is above most offers, which
       suggests a different SOC or area was used.
     - VA (+$0.22), CT (+$0.15), WV (+$0.19).
2. **Case level.** Share of hourly cases whose offered wage equals an OWL rate
   for the worksite county's area, by receipt regime:

   | Received | Cases | At $7.25 | OWL 08–09 45-2092 L-I | OWL 09–10 45-2092 L-I | Either, any SOC/level |
   | --- | --- | --- | --- | --- | --- |
   | before 2009-01-17 | 1,853 | 0.4% | 1.9% | 0.1% | 23.8% (coincidence baseline) |
   | 2009-01-17 – 06-30 | 2,489 | 28.6% | 41.6% | 20.7% | 60.3% |
   | 2009-07-01 – 2010-03-14 | 3,275 | 23.5% | 21.1% | 37.5% | 65.8% |
   | 2010-03-15 on | 1,077 | 0.5% | 0.5% | 1.9% | 39.8% |

   - Level I matches include the $7.25 floor.
   - The above-$7.25 Level I matches (section "Which wage year applied")
     isolate the wage year.
   - The 28.6% at exactly $7.25 in the first half of 2009 confirms the $7.25
     floor applied before the federal minimum reached $7.25 on 2009-07-24.

### Gaps (item 1)

- **Suppressed 45-2092 cells.** OWL leaves 45-2092 blank in some areas:
  - 2008–09: Bakersfield, El Centro, Fresno, Madera and Modesto CA; Bowling
    Green KY; MA BOS3; Hagerstown MD-WV; Fargo ND-MN; Jefferson City MO;
    NM BOS3; Corvallis and Salem OR; OR BOS4; PR BOS2; Brownsville and
    McAllen TX; Roanoke VA.
  - 2009–10: Anchorage; GA 1300003; NC 3700001; PR 7200002; Brownsville;
    El Paso.
  - How the NPC set the AEWR there is not documented in these files. It may
    have used another SOC (for example 45-2091 or 45-2099) or a statewide
    figure. Confidence for those cells is "no_wage_published".
- **SOC per case is missing.** `soc_code` is empty in the local FY2009–11
  disclosure records, so the occupation and level each case used must be
  inferred (job title, or a match against the schedule).
- **Herders and range workers** are outside the OWL schedule (special
  procedures). Their 2009 monthly rates were not acquired.
- **The date convention is approximate.** The wage year attaches to the
  wage-determination or recruitment date, which the disclosure data lack;
  receipt date is a proxy. Filings in July–August 2009 can carry 2008–09
  determinations.

## 2. FY2006 and FY2007 H-2A disclosures — acquired

| Raw file (`data/raw/h2a_disclosures_fy2006_2007/`) | Wayback capture |
| --- | --- |
| `H2a_FY2006.zip` (mdb), `H2a_FY2006_text.zip` (source name `H2a_%20FY2006_text.zip`) | 20100920024720, 20100920023721 |
| `H2a_FY2007.zip` (mdb + format doc), `H2a_FY2007_text.zip` | 20100920025454, 20100920025123 |
| `CaseH2a_aspx_20080117.html` | FLC Data Center H-2A page ("by fiscal year (October to September)") |

Tidy table: **`h2a_disclosures_fy2006_2007.csv`, 15,209 rows**.

| FY | Rows | Cases | Certified workers | Requested workers |
| --- | --- | --- | --- | --- |
| 2006 | 7,469 | 4,264 | 72,510 | 88,391 |
| 2007 | 7,740 | 4,704 | 89,575 | 93,339 |

For comparison, the local FY2008 file has 102,662 certified workers.

Fields kept:

- Both years: case number, status, certification begin and end dates,
  employer name, address, city, state and ZIP, job title, workers requested
  and certified, basic hours per week, rate and unit of pay.
- FY2007 only: application type, decision date, agent, and alien work state.
- Derived: `received_date_from_case_number`, decoded from the case number
  (`A|C-YYDDD-NNNNN`, Julian day). In FY2007 every decoded receipt date
  precedes the decision date.

Gaps (item 2):

- **No worksite county or city in either year.** FY2006 has employer
  location only. FY2007 adds the work state, which equals the employer state
  in 96.7% of rows. County assignment needs employer ZIP or city → county,
  which is a design-stage choice.
- **FY2006 is an early extract (January 2007).** It includes
  Received/Incomplete/Modification statuses and has no decision date.
- **185 case numbers appear in both FY2006 and FY2007**, so deduplicate
  before counting.
- **Multi-row cases.** About 7.5% of cases have several rows (association
  members).
- **No SOC field.** Job titles only.
- **Hours.** Weekly hours are present for 99.99% of rows, unlike FY2008–10.
- **Overlap with the local FY2008+ records:** 3 case numbers.

## Checksums (sha256)

```
486f66445d10a8764f76da7ea92145790ede2f13afd7209e8dfbc7ff7ed83df3  aewr_2009_oes/OWL_2009_TEXT.zip
e8bdcad18f3c429cca0d7783f192b2826e1628ffccf9afdf63432b870feeb832  aewr_2009_oes/OWL_2010_TEXT.zip
c990afd559a02e0b58fe1c34125db46b8633eccca61fd52bec29f5b6bc9da40a  aewr_2009_oes/OWL_2009_Download.zip
efe05d559b2262b9c074a5f184cb83de268155ffa358552048921b8b1aa420fb  aewr_2009_oes/OWL_2010_Download.zip
c932e472476abb2d300433c3b61d74868ec54c110b86028b62ae5884e6a7c9cb  aewr_2009_oes/OWL_2008_TEXT.zip
ed6b12ed48dffa4120dfd9ea3535342d6719c69934c431f4298d2d18137fe916  aewr_2009_oes/OWL_2008_Download.zip
5a9160a3ce45072cf824066386d28b2fef892dcdbf9229edc5e3d629ba253e41  aewr_2009_oes/E8-29309.htm
e43350d355e3089c96ea9500e78cc66dfb1dea1ac32ffb12ce0f336449e3339f  h2a_disclosures_fy2006_2007/H2a_FY2006_text.zip
482778e41afa9b5ddc8b7cb8597a6253cc8ebc8fb2050a5ba41b580ed1320c3f  h2a_disclosures_fy2006_2007/H2a_FY2006.zip
2fbc85b1b36c8a617008f1cc279b87081e9a8b498e3ef73c0442bf66d6f7f923  h2a_disclosures_fy2006_2007/H2a_FY2007_text.zip
9507e5db263eb03aed4febfe0c44d52bc8dced061ee3b469a1743221f4666f2d  h2a_disclosures_fy2006_2007/H2a_FY2007.zip
```

`data/` is git-ignored, so the raw downloads are local only. Re-download them
from the Wayback URLs above if needed.
