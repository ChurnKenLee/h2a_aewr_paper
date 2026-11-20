# A3 acquisition notes: 2023-rule occupation (OEWS) AEWRs and the court orders that bound them

This is the Phase 2 acquisition for strategy A3 in `markdowns/identification_search_agent_plan.md`.
Files were retrieved 2026-11-05. The scripts are `explorations/identification_search/A3/00_aewr_2023_oews_schedule.R`
(the tables) and `01_verify_schedule.R` (the checks). No producer, retained output or manuscript file
was changed.

## 1. Occupation AEWRs for non-field SOCs: acquired

### Rule

The rule comes from 88 FR 12760 (FR Doc 2023-03756), effective 2023-03-30, together with OFLC's
implementation FAQ of 2023-03-09 and the FLAG AEWR page.

- **Field-and-livestock SOCs** keep the state or regional FLS AEWR. These are the six SOCs
  45-2041, 45-2091, 45-2092, 45-2093, 45-2099 and 53-7064.
  - FLS reports no wage for AK, DC, GU, PR and VI. There the six SOCs get one OEWS-based rate: the
    employment-weighted mean of the statewide OEWS hourly means (655.120(b)(1)(i)(B)).
- **Every other non-range SOC** takes the *statewide* OEWS annual average hourly wage (H_MEAN) for the
  SOC. If OEWS publishes no statewide wage, the *national* H_MEAN applies (655.120(b)(1)(ii)).
- **Jobs that combine SOCs** take the highest applicable AEWR across all of their SOCs
  (655.120(b)(5); FAQ Q2; FR 2025-12910, footnote 1).
  - That covers an FLS rate paired with an OEWS rate, and two OEWS rates.
  - FAQ example 2: a livestock job that also hauls cattle in semi-trucks of 26,001 lb GVW or more on
    public roads is 45-2093 plus 53-3032, so it pays the 53-3032 rate.
- **Which job orders are covered.** The rule follows the ETA-790/790A submission date (FAQ Q3):
  - job orders submitted on or after 2023-03-30 use the 2023 method;
  - job orders submitted earlier, and cases already certified, keep the 2010 method until the next
    AEWR adjustment that applies to the SOC.
- **Mid-contract updates.** A higher updated AEWR binds during the contract (655.120(b)(3)). A lower one
  does not reduce the job-order wage (655.120(b)(4)).

### Vintages (`aewr_2023_2025_oews_vintages.csv`)

| OEWS data | Effective (default group) | Kansas-order group | Authority |
| --- | --- | --- | --- |
| May 2021 | 2023-03-30 to 2023-06-30 | same | Final rule; FLAG workbook named "Effective 3.30.23 to 6.30.23" |
| May 2022 | 2023-07-01 to 2024-07-07 | same | FR 2023-12896 ("effective July 1, 2023"); FLAG archive |
| May 2023 | 2024-07-08 to 2025-07-10 | to 2025-07-24 | FR 2024-13809 ("applicable July 8, 2024"); FLAG archive "to July 10, 2025" |
| May 2024 | 2025-07-11 to 2025-08-28 | 2025-07-25 to 2025-08-28 | FR 2025-12910 ("effective July 11, 2025 … Kansas … July 25, 2025"); FLAG archive "July 11, 2025 through August 28, 2025" |

The end date of the May 2024 vintage comes from the vacatur (section 3). It is not a scheduled update.

The May 2024 workbook link is no longer on the live FLAG page. It was recovered from the archive PDF
`AEWR Page Archive 8.28.25.pdf` at `https://flag.dol.gov/sites/default/files/2025-07/BLS-OEWS_National_State_Data_May%202024_Release.xlsx`.

### Tables produced

**`aewr_2023_2025_occupation_rates.csv`** has 178,146 rows. They cover:

- 54 areas: 50 states, DC, GU, PR and VI;
- 824–825 SOCs per vintage, with the six FLS SOCs excluded;
- 4 vintages.

The key is `state_fips` × `soc_code` × `oews_reference`. The row universe is every code on OFLC's state
sheet (the detailed SOCs plus seven OEWS aggregate codes such as 47-4090), joined with the national
detailed codes. `rate_basis` takes four values:

| `rate_basis` | Rows | Meaning |
| --- | --- | --- |
| `statewide` | 131,179 | Statewide H_MEAN |
| `national_fallback` | 33,752 | Statewide value suppressed (`*`) or row absent. Mostly GU and VI (about 75–80% of SOCs), then CO in May 2024 and thinly covered SOCs |
| `statewide_topcoded_unavailable` | 422 | `#`, a mean of at least $115/h. Not determinable from the workbook; irrelevant for H-2A |
| `unavailable` | 12,793 | No hourly mean is published (annual-only occupations such as teachers) |

The table carries both the default effective dates and the Kansas-group dates, plus `kansas_order_state`.

- **`aewr_2023_2025_oews_field_group_rates.csv`:** the OEWS-based field-and-livestock AEWRs for AK,
  DC, GU, PR and VI in each vintage, transcribed from OFLC's archived AEWR tables. The other 49 states'
  FLS rates are already in `data/intermediate/aewr_rate_schedule.parquet`.
- **`aewr_2023_2025_regime_periods.csv`:** the regime calendar for non-field SOCs from before the rule
  to the 2025 IFR, with the cohort rule and the entity exceptions.

### Confidence

| Rows | Confidence | Why |
| --- | --- | --- |
| Statewide rows, and national rows where the state row shows `*` | High | These are OFLC's own lookup values |
| National rows where the state row is absent | Medium-high | The rule's fallback, applied by construction |
| Colorado, May 2024 vintage (825 rows) | Medium | See below |

**Colorado, May 2024.** Neither OFLC's workbook nor BLS's May 2024 state file has Colorado statewide
rows, so the rule gives the national rate. However, OFLC's FY2026 IFR file shows Colorado-specific
skill-II values, about 5.9% above the national mean. What OFLC actually applied in Colorado from
2025-07-11 to 2025-08-28 is therefore uncertain. It is a 49-day window, and the disclosures have no
non-field Colorado cases in it.

## 2. The Kansas v. DOL injunction: acquired (scope and dates)

- **The order.** *Kansas et al. v. U.S. DOL*, No. 2:24-cv-00076-LGW-BWC (S.D. Ga.), preliminary
  injunction of 2024-08-26 against the 2024 Farmworker Protection Rule (89 FR 33898). That rule had
  amended 655.120(b)(2) to make AEWR updates effective on the date of FR publication.
- **Scope.** The scope is quoted in FR 2024-29549, footnote and text, and on the archived DOL FPR page
  (2025-07-23 capture).
  - **States (17):** AR, FL, GA, ID, IN, IA, KS, LA, MO, MT, NE, ND, OK, SC, TN, TX and VA. The test
    is "work performed at places of employment located in" these states.
  - **Entities:** "Miles Berry Farm and members of the Georgia Fruit and Vegetable Growers Association
    as of August 26, 2024", wherever they work.
- **Effect on the AEWR series.** The Kansas group gets a 14-day lag on updates published after the
  injunction:
  - the 2024-12-16 FLS update starts 2024-12-30 for the Kansas group (already in the producer's
    `aewr_policy_events`);
  - the 2025-07-11 OEWS update starts 2025-07-25 for the Kansas group (FR 2025-12910, DATES).
  - The 2023-07-01 and 2024-07-08 OEWS updates predate the injunction and applied uniformly.
- **Other injunctions do not move AEWR dates.**
  - *Barton v. DOL* (E.D. Ky., 2024-11-25) is a preliminary injunction.
  - *IFPA v. DOL* (S.D. Miss., 2024-11-25) is a §705 stay.
  - FR 2024-29549, footnote 2, says neither affects 655.120(b)(2). DOL's 2024-12-20 guidance is
    in the raw directory.
- **Coding limit.** Entity membership (GFVGA members, Miles Berry Farm) cannot be observed in the
  disclosures. Coding the Kansas group by worksite state is the feasible approximation. The
  verification script does this.

## 3. A third legal event for A3: the Teche Vermilion injunction and vacatur

This was found during acquisition and was not in the brief. It matters directly for A3.

**Preliminary injunction.** *Teche Vermilion Sugar Cane Growers Ass'n v. Su*, No. 6:23-cv-831 (W.D. La.)
enjoins enforcement of the 2023 AEWR rule itself.

- **Dates.** The order is dated 2024-09-18 and was amended 2024-12-05 (ECF 61).
- **Workers covered:** H-2A workers in Louisiana sugar cane farming and processing.
- **Employers covered:** the named plaintiffs, and members as of 2024-11-07 of Teche Vermilion SCGA,
  Cora Texas Growers & Harvesters and the American Sugar Cane League.
  - Exhibit 1 of the amended order lists the members over about 150 pages. It is in the raw directory
    and has not been extracted.
- **Job orders covered:** those filed after 2024-09-19, plus those pending on that date.

The IFR preamble also cites an amendment of 2025-07-16 (2025 WL 1969937). It was not obtained.

**Vacatur.** The final judgment of 2025-08-25 (filed 2025-08-26) made the injunction permanent and
**vacated** the 2023 rule.

- OFLC announced on 2025-08-28 that AEWRs follow the 2010 method.
- The FLAG page then listed one FLS AEWR per state for **all** non-range occupations, effective from
  2025-08-29 by the FLAG archive's dating. AK, DC, GU, PR and VI were listed as N/A.
- This was confirmed by Wayback captures of 2025-08-30, 2025-09-12 and 2025-09-28, and by the IFR
  preamble (FR 2025-19365). No FR notice was found for the switch itself.

**Consequences for the A3 design:**

- The "2025 cut" for non-field SOCs happened on **2025-08-29**, through the vacatur, not on 2025-10-02.
  - From 2025-08-29 to 2025-10-01 non-field jobs paid the FLS rate.
  - From 2025-10-02 they paid the IFR skill rates (skill II equals the May 2024 statewide mean).
- Louisiana sugarcane truck-driver jobs of covered employers stopped receiving the OEWS rate from
  2024-09-19. This is an entity-level exception inside the A3 treated group.

The producer's `aewr_policy_events.parquet` has neither the 2025-07-11 OEWS update nor these two
court events (section 6).

Two other suits challenged the same rule. Per the IFR preamble, neither changed the rates:

- *USA Farm Labor v. Su* (W.D.N.C., No. 1:23-cv-00096): the preliminary injunction was denied, and an
  appeal is pending in the Fourth Circuit.
- *Florida Growers Ass'n v. Su* (M.D. Fla., No. 8:23-cv-00889): the preliminary injunction was denied,
  and the case was later stayed.

## 4. Verification

The results are in `aewr_2023_2025_verification.csv`. Offered wages only were used; no quantity
outcome was examined.

1. **OFLC workbook against BLS (May 2024).** The OFLC workbook's state H_MEAN is identical to BLS
   `oesm24st` (Wayback 2025-06-01) on all 36,844 rows. BLS also has no Colorado statewide rows.
2. **Field-group recomputation (May 2024).** Recomputing the AK, DC, GU, PR and VI field-group AEWRs
   from the workbook (TOT_EMP × H_MEAN) reproduces OFLC's published 20.04, 20.79, 10.67, 10.36 and
   13.69 exactly.
3. **FY2026 (b)(1)(ii) rates in `aewr_occupation_skill_rates.parquet` (2025-10-02 vintage).**
   - There are 32,982 overlapping state-SOCs.
   - On every one of the 32,312 statewide-basis cells, the US-worker skill-II rate equals the
     acquired May 2024 statewide mean.
   - Skill I is below the mean in every cell (median ratio 0.72).
   - The only mismatches are the 670 Colorado cells (section 1). There the IFR uses Colorado-specific
     figures rather than the national mean.
   - So the 2025-10-02 skill-II rate is, by construction, the last 2023-rule rate for non-field SOCs.
     For non-field SOCs, the shock from the 2023 rule to the IFR runs mostly through the skill-I
     option and the H-2A housing adjustment.
4. **Offered hourly wages, FY2022–FY2026 disclosures, non-field SOCs.** The rate in force is assigned
   by case-receipt date. The rule keys on the job-order date, a few weeks earlier, so cases near a
   boundary often match the neighbouring vintage. "Kansas states" means the 17-state proxy.

| Receipt period | n | At current OEWS rate | At previous rate | At next rate | At FLS rate |
| --- | --- | --- | --- | --- | --- |
| Before 2023-03-30 (placebo, May 2021 rate) | 1,270 | 0.000 | – | 0.000 | 0.733 |
| May 2021 (2023-03-30 to 06-30) | 279 | 0.323 | – | 0.315 | 0.204 |
| May 2022 | 1,575 | 0.717 | 0.012 | 0.081 | 0.112 |
| May 2023 | 1,767 | 0.757 | 0.012 | 0.074 | 0.092 |
| May 2024 (2025-07-11 to 08-28) | 109 | 0.752 | 0.009 | – | 0.156 |
| Vacatur (2025-08-29 to 10-01) | 125 | 0.208 (May 2024) | | | **0.648** |
| IFR (from 2025-10-02; skill II = May 2024 mean) | 1,130 | 0.319 | | | 0.000 |

What the table shows:

- **2023-rule window.** Pooled over the four vintages, 70.8% of 3,730 non-field hourly offers equal the
  acquired rate to the cent. Another 11% equal the FLS rate, which reflects SOC coding and
  pre-rule job orders. About 92–93% sit exactly at one of the floors.
- **May 2021 window.** It splits between the May 2021 and May 2022 rates. This fits disclosures
  reporting the wage after the 2023-07-01 mid-contract update.
- **Placebo.** No offer before the rule equals the May 2021 rate.
- **Vacatur period.** Offers jump back to FLS, confirming the 2025-08-29 switch.
- **By SOC.** Exact-match shares in the 2023-rule window are listed in `aewr_2023_2025_offer_match_by_soc.csv`:

  | SOC | Occupation | n | Exact-match share |
  | --- | --- | --- | --- |
  | 53-3032 | Heavy truck drivers | 1,017 | 0.854 |
  | 49-3041 | Farm equipment mechanics | 314 | 0.825 |
  | 45-1011 | First-line supervisors | 294 | 0.731 |
  | 53-3053 | Shuttle drivers | 755 | 0.657 |
  | 47-2061 | Construction laborers | 499 | 0.639 |

- **Teche check, Louisiana 53-3032.**
  - From 2023-03-30 to 2024-09-19: 83% at the OEWS rate and 0% at FLS (n = 35).
  - After 2024-09-19: 21% at OEWS and 58% at FLS (n = 19). This is consistent with the injunction.
- **National-fallback cells** match in only 19% of 47 offers. Most of those cases carry odd SOCs, such
  as 13-1074 farm labor contractors and 51-3022, with offers at the FLS rate. The matches that do occur
  (MN 47-3012, CT 49-3041, LA and MS 11-9013, NM 19-4012, MI 45-2021) support the fallback convention.

## 5. Raw files (`data/raw/aewr_2023_oews_occupation/`)

| Path | Source |
| --- | --- |
| `flag/BLS-OEWS_National_State_Data_May_202{1,2,3,4}*.xlsx` | flag.dol.gov AEWR page and archive. URLs are in `aewr_2023_2025_oews_vintages.csv` |
| `flag/Historical_AEWR_*.pdf`, `flag/Range_and_Non-Range_*.pdf`, `flag/AEWR_Page_Archive_8.28.25.pdf`, `flag/AEWR_Archive_Range_Occupations_Update_1.20.26.pdf` | OFLC archived AEWR tables, flag.dol.gov/sites/default/files/… |
| `flag/flag_aewr_page_20261105.html` | Live FLAG AEWR page |
| `wayback/flag_aewr_page_wayback_<ts>.html` | `web.archive.org/web/<ts>id_/https://flag.dol.gov/wage-data/adverse-effect-wage-rates`, ts = 20230322091505, 20230711142016, 20240709143434, 20241231223949, 20250711203143, 20250830184252, 20250928163831 |
| `wayback/dol_farmworker_protection_rule_page_wayback_20250723.html` | Wayback, dol.gov/agencies/eta/foreign-labor/farmworker-protection-final-rule (Kansas scope) |
| `court/Teche-H2A-Order-9.18.24.pdf`, `court/Teche-H2A-Amended-order.pdf` (with Exhibit 1 member lists), `court/1-Judgment-Teche-Vermillion-v-Su-8.26.25.pdf` | aglaw.psu.edu copies of W.D. La. ECF 46, 61 and 87 |
| `dol_notices/OFLC_announcement_2025-08-28_vacatur_mn_deed_copy.pdf` | mn.gov/deed copy of the OFLC announcement. dol.gov returns "Access Denied" to scripted requests |
| `dol_notices/2023-AEWR-Final-Rule-FAQs_implementation_2023-03-09.pdf` | Wayback 20230313121713 of dol.gov |
| `dol_notices/DOL_guidance_2024-12-20_Barton_IFPA.pdf`, `dol_notices/TEN_27-23_Change_2.pdf` | Wayback copies of dol.gov (FPR injunction guidance) |
| `dol_notices/CRS_R48833_H2A_wage_regulation_history.pdf` | congress.gov CRS R48833 (background) |
| `federal_register/2023-03756.htm`, `federal_register/2025-19365.htm` | govinfo: the 2023 final rule and the 2025 IFR |
| `bls/oesm24st_wayback_20250601024221.zip` | Wayback copy of bls.gov/oes/special-requests/oesm24st.zip (bls.gov blocks scripted requests) |

The OEWS-update notices FR 2023-12896, 2024-13809 and 2025-12910, and the FLS notice 2024-29549, were
already cached in `data/raw/aewr/`.

## 6. Gaps and cautions

- **Colorado, May 2024 vintage.** The rate actually applied is uncertain (section 1).
- **Teche** (a combined coding sketch follows this list):
  - Entity-level coverage needs the Exhibit 1 member lists, which are not extracted, matched to
    employer names.
  - The 2025-07-16 amendment was not obtained.
  - A sugarcane-and-Louisiana proxy is the fallback.
- **Kansas entities outside the 17 states.** GFVGA members and Miles Berry Farm cannot be observed,
  so code by worksite state.
- **No FR notice for the 2025-08-29 switch.** The date rests on the FLAG archive label and the OFLC
  announcement of 2025-08-28.
  - The cohort rule for job orders pending on 2025-08-25 to 08-28 is not documented in what was found.
  - Treat late-August 2025 receipts as a transition window.
- **Primary SOC only.** Disclosures report one SOC, so the highest-applicable rule for multi-SOC jobs
  cannot be applied from the disclosures. The design must use the pre-rule cell SOC, as planned.
- **Not determinable.** Top-coded (`#`) and annual-only cells have no rate. None is an H-2A-relevant SOC.
- **Producer drift, for the lead to decide; not changed here.** `aewr_policy_events.parquet`:
  - still says "historical lookup workbook not cached";
  - lacks the 2025-07-11 OEWS update (Kansas 07-25);
  - lacks the Teche PI (2024-09-19) and vacatur (2025-08-25 / 08-29).
  - Its 2025 FLS row runs to 2025-10-02, which is consistent with the vacatur period.
  - The A1 review's point stands: the shared at-AEWR flag lacks these occupation floors.

As a sketch only, the combined coding these limits imply:

- **2023-rule cohort:** job order submitted between 2023-03-30 and 2025-08-28.
- **Non-field SOC:** the pre-rule cell SOC is outside the six FLS SOCs.
- **Teche exception:** not a Louisiana sugarcane member with an order filed after 2024-09-19.
- **Rate:** the vintage in force at work performance, using the Kansas dates for the 17 states.
