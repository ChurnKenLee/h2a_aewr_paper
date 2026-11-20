# State agricultural overtime schedule, 2008-2027: method and source notes

Retrieved 2026-11-02. Companion to `state_ag_overtime_schedule.csv` (same directory).

## Scope and coding conventions

- Federal baseline: FLSA s. 13(b)(12), 29 U.S.C. 213(b)(12), exempts employees employed in agriculture
  from federal overtime. All variation in the CSV is state law.
- One row per state x employer-size class x sector scope x regime spell. `effective_start` is
  2008-01-01 when the regime predates 2008; `effective_end` is the last day of the spell (inclusive) and is
  blank when the regime continues through 2027-12-31.
- `weekly_threshold_hours` / `daily_threshold_hours`: hours after which 1.5x the regular rate is owed.
  Blank means no weekly (daily) ag overtime trigger.
- `applies_to_h2a`: TRUE when the state rule covers the workers and contains no H-2A carve-out (state wage
  laws generally cover H-2A workers, and 20 CFR 655.135(e) requires H-2A employers to comply with all
  applicable state law). FALSE on rows with no ag overtime simply means no state ag overtime applies to
  anyone, H-2A included. Only New York's agency guidance says so expressly (NYSDOL FLFLPA FAQ Q14).
- Colorado 2024-2026 uses employer classes (`highly_seasonal_employer`, `small_employer`,
  `other_employer`) rather than headcount classes; the 56-hour peak-week threshold for highly seasonal
  employers is described in `exceptions_note`, not in the weekly column.
- Confidence: high = official statute/regulation/agency text read directly; medium = statute text via a
  secondary host (FindLaw, mirrors) or secondary source for a point; low = not verified.

## States with an agricultural overtime threshold at any point 2008-2027

Weekly (and daily, CA only) thresholds for hired farm workers. "none" = no ag OT.

| State | 2008 baseline | Changes 2008-2027 | Notes |
|---|---|---|---|
| CA | Wage Order 14: OT after 10 h/day + 7th day | 26+ employees: 55/9.5 (2019), 50/9 (2020), 45/8.5 (2021), 40/8 (2022+). 25 or fewer: same steps 2022-2025 | 7th-day premium throughout; 2x after 12 h at full phase-in |
| NY | none | 60 (2020-01-01), 56 (2024), 52 (2026); 48 in 2028 ... 40 in 2032 | 24-h day of rest; OT for rest-day work |
| WA | none | Dairy: 40 from 2020-11-05. Other ag: 55 (2022), 48 (2023), 40 (2024+) | |
| OR | none | 55 (2023-24), 48 (2025-26), 40 (2027+) | Excludes ORS 653.020(1) workers |
| CO | none | 60 (2022-11-01 to 2023); 2024: 48 highly seasonal (56 in up to 22 peak weeks) / 56 small / 54 other; 2025-26: 48 (highly seasonal 56 peak); 56 uniform from 2027-01-01 (SB26-121) | Range workers exempt |
| MN | 48 | no change | Salaried ag workers and piece-rate sugar beet laborers excluded |
| MD | 60 | no change | Only employers above the 500 ag-worker-day test |
| HI | 40 (48 in up to 20 designated weeks for qualifying seasonal/processing employers) | no change | Employers with 20+ employees; coffee-harvest weeks excluded; medium confidence |
| VA | none | 40 from 2021-07-01 to 2022-06-30 (Virginia Overtime Wage Act), then none | Medium confidence; FLSA 213(a)(6) workers excluded |
| DC | 40 | no change | No ag exemption; negligible farm employment |
| MA | none for farm labor | none | Post-harvest cleaning/sorting/packing work is not exempt (Arias-Villano, 2019): 40 |
| ME | none for farm labor | none | Egg operations with >300,000 layers covered at 40 (start date unverified) |
| KY | none weekly | none | KRS 337.050 7th-day rule may reach farm workers (low confidence) |

All other states (AL, AK, AZ, AR, CT, DE, FL, GA, ID, IL, IN, IA, KS, LA, MI, MS, MO, MT, NE, NV, NH,
NJ, NM, NC, ND, OH, OK, PA, RI, SC, SD, TN, TX, UT, VT, WV, WI, WY) had no farm-worker overtime in
2008-2027. Some have no state OT law at all; the rest exempt agriculture (citations in the CSV).

Spells within a state x employer class x sector are contiguous. Where a regime change also changes the
sector or class label (CO 2022/2024/2027, OR 2023, VA 2021-22), the label changes along with it. When
building a panel, map each state-year to the row whose label fits the population of interest. For a single
state-year series, use: CA `26_or_more_employees`; CO `other_employer` for 2024-26 (the highly seasonal
48/56 regime is the relevant one for most H-2A crop employers); WA `all_agriculture_excluding_dairy`.

## Sources by state (core states researched directly)

- **California**: AB 1066, Stats. 2016, ch. 313 (chaptered 2016-09-12), Lab. Code ss. 857-864 (s. 860
  phase-in; s. 863 governor suspension power, never used per DLSE pages). DLSE pages:
  https://dir.ca.gov/dlse/Overtime-for-Agricultural-Workers.html and
  https://dir.ca.gov/dlse/Overtime-for-Agricultural-Workers-FAQ.html. Enrolled bill:
  https://www.leginfo.ca.gov/pub/15-16/bill/asm/ab_1051-1100/ab_1066_bill_20160830_enrolled.htm.
  Pre-2019: IWC Wage Order 14 (8 CCR 11140), OT after 10 h/day and on the 7th day, with no weekly-hours
  trigger. Seventh-day premium (1.5x first 8 h, 2x after 8 h) continues for all sizes. Double time after
  12 h/day from full phase-in (2022 large / 2025 small). Size is counted as under SB 3 (2016); DLSE does
  not say whether H-2A or seasonal workers count. Sheepherders: DLSE Opinion Letter 2019-07-26 holds
  AB 1066 overrides the former Wage Order 14 sheepherder exclusion (litigated; see the DIR SB 143 study,
  https://www.dir.ca.gov/Reports/California-Sheepherders-and-Goat-Herders-SB-143-Study.pdf).
- **New York**: Farm Laborers Fair Labor Practices Act, L. 2019, ch. 105 (S6578/A8419, signed
  2019-07-17), effective 2020-01-01: OT after 60 h/week plus 24 consecutive hours of rest per week (work on
  the rest day paid at 1.5x). Farm Laborers Wage Board recommended the phase-down; the Commissioner accepted
  it in 2022 and NYSDOL finalized 12 NYCRR 190-2.4 in Feb. 2023 (secondary reports). Schedule: 56 (2024),
  52 (2026), 48 (2028), 44 (2030), 40 (2032). Sources: NYSDOL FAQ PDF
  https://dol.ny.gov/farm-laborers-fair-labor-practices-act-faq (all farm sizes covered; H-2A/J-1/TN
  covered, Q14); NYSDOL release 2025-12-19
  https://dol.ny.gov/news/new-york-state-department-labor-reminds-new-yorkers-decrease-farm-worker-overtime-threshold-0.
  Employers get the refundable Farm Employer Overtime Credit.
- **Washington**: Martinez-Cuevas v. DeRuyter Bros. Dairy, 196 Wn.2d 506, 475 P.3d 164 (2020-11-05),
  struck the ag exemption as applied to dairy workers. ESSB 5172, Laws of 2021 ch. 249 (effective
  2021-07-25): RCW 49.46.130(6) gives 55 (2022), 48 (2023), 40 (2024); dairy employees excluded from the
  phase-in and owed OT after 40 from 2020-11-05 (L&I FAQ). Sources: https://app.leg.wa.gov/RCW/default.aspx?cite=49.46.130,
  https://www.lni.wa.gov/workers-rights/agriculture-policies/agricultural-overtime-faq. 2025-26 bills for
  50-hour seasonal weeks (HB 1597, SB 5487) were retained in committee and not enacted (secondary trackers).
- **Oregon**: HB 4002, Or. Laws 2022 ch. 115; ORS 653.271-653.273: 55 (2023-24), 48 (2025-26), 40 (2027+),
  weekly only. Exclusions via ORS 653.020(1) (small piece-rate hand-harvest employers, family, commuting
  short-season hand harvesters, young piece-rate harvesters, salaried range livestock workers) and white-collar
  ag employees. Refundable employer tax credit. Source: https://www.oregonlegislature.gov/bills_laws/ors/ors653.html;
  BOLI 2025 employer guide. Pre-2023 exemption citation (OAR) not verified (medium).
- **Colorado**: SB21-087 directed CDLE rulemaking; overtime rules now in COMPS Order, 7 CCR 1103-1 Rule
  2.3.2 (first adopted 2022 as Agricultural Labor Conditions Rules). Schedule per CDLE INFO #12A (rev.
  2025-12-18): none before 2022-11-01; 60 (2022-11-01 to 2023-12-31); 2024: 48 for highly seasonal employers
  (56 in up to 22 peak weeks), 56 small employers, 54 others; 2025-26: 48 (highly seasonal 56 in peak
  weeks). SB26-121 (Colo. Sess. Laws 2026 ch. 84, signed 2026-05-04) sets a uniform 56-hour threshold from
  2027-01-01 and removes CDLE ag-OT rulemaking authority. Exempt: open-range livestock workers paid the
  range salary, decision-making livestock managers, family members. The CDLE PDF itself returned HTTP 403;
  its content was read via a mirror (https://ezel.ai/labor-guidance/co/info-12a) and cross-checked against
  the Legislative Council final fiscal note for SB26-121 (https://leg.colorado.gov/bill_files/117211/download).
- **Minnesota**: Minn. Stat. 177.25 subd. 1 (48-hour OT) applies to farm workers (DLI: since 1974);
  exclusions in 177.23 subd. 7(1)-(2) for salaried ag workers (two or fewer salaried workers, or salary of at
  least 48 h at minimum wage plus 17 h at 1.5x); piece-rate sugar beet hand laborers exempt if pay is at least
  minimum + $0.40/h. Sources: https://www.revisor.mn.gov/statutes/cite/177.25, /177.23,
  https://www.dli.mn.gov/agriculture.
- **Maryland**: Md. Code, Lab. & Empl. s. 3-420(c): OT may be computed after 60 h/week for employees engaged
  in agriculture who are exempt from FLSA OT. s. 3-403(12) excludes employers using no more than 500
  agricultural-worker days in each quarter of the prior year; plus family and short-season commuting
  piece-rate hand harvesters. Sources: mgaleg.maryland.gov statute text pages for 3-420 and 3-403.
- **Hawaii** (medium confidence): HRS 387-1 excludes ag workers in any workweek in which the employer has
  fewer than 20 employees and in any workweek of coffee harvesting. HRS 387-3(a) OT after 40; 387-3(e)
  lets agriculture-plus-first-processing and seasonal-processing-linked employers designate up to 20 weeks
  per July-June year with OT only after 48. Coverage of ag workers at FLSA-covered employers relies on the
  HRS 387-1 exclusion (12) proviso ("if the maximum workweek established ... under the FLSA is greater ...
  section 387-3 shall apply"). The official capitol.hawaii.gov HRS pages returned HTTP 403, so the text was
  read via FindLaw. The National Agricultural Law Center compilation (current through 2026-05-06)
  summarizes Hawaii only as "over 48 hours", which is incomplete. DLIR guidance confirming the 40-hour
  baseline for ag was not found. Treat as medium.

Cross-check: National Agricultural Law Center, "Overtime for Agricultural Workers" state compilation
(https://nationalaglawcenter.org/state-compilations/agpay/overtime/), current through 2026-05-06, lists
the same eight states: CA, CO, HI, MD, MN, NY, OR, WA. Its CO and NY entries are out of date or simplified.

## Non-core states

Researched by two parallel passes (official statute text where reachable; FindLaw or the US DOL state
minimum wage/overtime table where official sites blocked access, labeled secondary). Legislative activity
that did NOT produce ag overtime:
- Maine: LD 398 (2023) had its OT stripped, then was vetoed 2023-07-19 (override failed 61-61). LD 2273 (2024)
  was minimum wage only and vetoed. LD 589 (2025) was introduced with an OT phase-in but enacted as P.L. 2025
  c. 232 with a farm minimum wage only (from 2026-01-01).
- Massachusetts: 55-hour bills (2019, 2021, 2023-24) died. H.3107 (2025) went to a study order on 2026-04-02.
  S.3132 was reported favorably 2026-07-09 and sent to Senate Ways and Means, not enacted as of retrieval.
- Vermont: H.403 (60 h from 2026) and H.347 (phase to 40 by 2036) were not enacted. A 2024 study committee
  recommended 60 h.
- New Jersey: farm-labor exemption N.J.S.A. 34:11-56a4(b)(1). Repeal bills (A4451, A4549, S2354, S2764/A4620,
  S1075 in 2026) never passed a chamber.
- Nevada: SB 172 (2025) as introduced removed the ag exemption (8 h/day, 40 h/week). The OT language was
  amended out in April 2025, and the bill died in the Assembly.
- Illinois: HB 3395 and SB 2003 (2023-24) died.
- Virginia: 2025 HB 1625 (farm minimum wage) was vetoed. 2026 HB 20/SB 121 bring farm laborers under the
  state minimum wage from 2027-01-01 (not overtime).
- Michigan, Connecticut, Alaska, Pennsylvania, Rhode Island, New Mexico: no farmworker OT enactment or veto found.

## Rows lacking primary sources or with reduced confidence

- **Low**: ME large-egg-operation row (start date of the carve-out unverified). The KY seventh-day-rule
  applicability (in the KY row's `seventh_day_rule` text; row confidence reflects the weekly coding).
- **Medium**:
  - HI (statute read via FindLaw; ag coverage depends on HRS 387-1(12) proviso; no DLIR confirmation).
  - VA 2021-22 window (reading of enacted text plus a settled class action; no ruling).
  - MA post-harvest row.
  - DC.
  - OR pre-2023 exemption citation.
  - Several no-OT rows based on secondary hosts or the US DOL state table: AZ, IN, NH, NJ, NM, OK, RI, SC,
    TX, WI. The `notes` field of each row carries `[source_type: ...]`.
- **Not fetched directly from the official host** (403 or blocked): CA leginfo statute page (used the DLSE
  pages and the enrolled bill instead); CDLE INFO #12A PDF (used a mirror plus the Legislative Council fiscal
  note); capitol.hawaii.gov; Justia. NY's 12 NYCRR 190-2.4 text was not read directly; the schedule rests on
  NYSDOL releases and FAQ plus secondary reports of the Feb. 2023 adoption.

## H-2A job orders: must state overtime pay be disclosed?

- **Form ETA-790A, Item A.8g** (OMB 1205-0466), general instructions:
  https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/Form_ETA-790-790A_Instructions.pdf. The employer
  answers Yes/No on "whether overtime pay will be available, whether voluntarily offered by the employer or
  required by an applicable Federal, State, or local law. If 'Yes' is selected, use only one (1) Addendum C
  to fully describe: the wage rate(s) to be paid for any such overtime hours; the circumstances under which
  the wage rate(s) for overtime hours will be paid; whether overtime wage rates will vary between places of
  employment; and where the overtime pay is required by law, the applicable federal, State, or local law
  requiring the overtime pay." So state-law overtime must be disclosed in the clearance order (Addendum C).
- **20 CFR 655.122(l)(4)** (added by the 2024 "Improving Protections for Workers in Temporary
  Agricultural Employment" final rule, 89 FR 33898, Apr. 29, 2024; amendatory text at 89 FR 34060) requires
  the job order to state, if applicable: (i) that overtime hours may be available; (ii) the overtime wage
  rate(s); (iii) the circumstances under which they are paid; (iv) "Where the overtime pay is required by
  law, the applicable Federal, State, or local law requiring the overtime pay." Source: eCFR current text of
  20 CFR 655.122.
- **Status of the 2024 rule**: DOL suspended enforcement of the 2024 rule on 2025-06-20 and proposed
  rescinding most of it (NPRM, 90 FR 28919, July 2, 2025). The NPRM proposes to **retain** the overtime
  disclosure edits to 655.122(l)(4) (90 FR 28931; the proposed regulatory text at 90 FR 28943 renumbers it
  as (l)(3)). I found no final rescission rule through 2026-11-02. Also, Kansas v. DOL (S.D. Ga. 2024)
  enjoined the 2024 rule in several states before the suspension (from memory; not re-verified here).
- **Other provisions**: 20 CFR 655.135(e) requires H-2A employers to comply with all applicable federal,
  state and local employment-related laws. 655.122(q) makes the job order the work contract when there is
  no separate contract. A sampled job order (H-300-26005-534767, seasonaljobs.dol.gov; state not checked)
  states overtime "over 48 hours per state laws". Sampled Hawaii coffee-harvest job orders list no OT, consistent with the HRS
  coffee-harvest exclusion.
- **Implication for the paper**: from about 2019-2020 onward (form revision), job orders in SeasonalJobs.dol.gov
  should carry Addendum C overtime text that could validate the state schedule empirically. Before the
  form revision this was not a discrete item (not verified).

## Method

Primary statute, regulation and agency pages were fetched with WebFetch where possible. Where official
hosts blocked automated fetching (CA leginfo, CO CDLE PDF, HI capitol site, Justia), I used official mirrors
or code hosts and lowered confidence where only a secondary host was available. Non-core states (all states
except CA, CO, HI, MD, MN, NY, OR, WA) were researched by two parallel research passes that recorded
the state's overtime statute (if any) and its agricultural exemption, and checked 2008-2026 legislative
activity. Their results were merged as one row per state.
