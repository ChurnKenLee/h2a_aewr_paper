# State minimum-wage coverage of hired farmworkers, 2008–2026: research notes

> Repository note (2026-11-02): compiled by web research and adopted as the
> hand-maintained source table `code/a01_sources/state_ag_minwage_coverage.csv`.
> That table adds `applicable_floor` (`state_minimum`, `federal_minimum` or
> `ag_specific_rate`), numeric `ag_min_wage` and `floor_rule_note`, and splits
> New Jersey's 2019+ farm schedule into one row per rate (66 rows). Partial
> coverage without a separate rate (HI, MI 2014–16, NM, PA) is coded
> `state_minimum` because the exclusions do not reach hourly, H-2A-size,
> housed crews. `code/c01_clean/04_state_minimum_wages.R` reads the table.

Companion to `state_ag_minwage_coverage.csv` (58 rows: one per state per coverage spell, covering 50 states and DC).
Research was done on 2026-11-02 with web search and fetch only. No files in the H-2A_Paper repository were modified.

## Method

1. **Secondary screen.** I started from the National Agricultural Law Center (NALC) table "Minimum Wage for Agricultural Workers" (current through 2026-04-14), which flags each state as covering agricultural workers (Yes / Yes* / No). The DOL WHD "State Minimum Wage Laws" page (updated 2026-07-01) was checked as well. It flags agriculture only for Puerto Rico, so it was not useful here.
2. **Primary check.** For every state whose own minimum exceeded the federal minimum at any point in 2008–2026, plus every state on the suspect list, I read the definitions and exemption sections of the statute. Sources were the legislature site where it could be fetched, and otherwise a verbatim statute mirror (FindLaw, oregon.public.law). Where the question turned on a date, I also read session laws, administrative rules, or official agency posters.
3. **Coding rule** for `state_min_covers_hired_farmworkers`:
   - **TRUE.** The state minimum applies to ordinary hired farmworkers of an H-2A-size (FLSA-covered) employer. FLSA-mirroring exemptions do not change this coding. Those exemptions are: the 500 man-day small-farm test; local hand-harvest piece-rate workers who commute daily and worked fewer than 13 weeks in agriculture the year before; immediate family; piece-rate minors on a parent's farm; and range livestock workers. They do not reach housed seasonal crews.
   - **FALSE.** Agriculture is excluded, or there is no state minimum, so only the federal floor applies. FALSE is also used when the state sets a separate agricultural rate that is at or below the federal rate, because the federal floor then binds. This applies to MA 2008–2014 ($1.60) and WI 2008 to 2009-07-23 ($5.15). The nonbinding rate is still recorded in `ag_specific_rate`.
   - **partial.** Coverage depends on something that matters for crop workers. Examples: a separate lower agricultural rate above the federal rate (MA 2015+, NJ 2019+); an employer-size test stricter than the FLSA, or a crop exclusion (HI); a general exclusion of piece-rate or dairy workers (NM); a piece-rate harvest exclusion (MI 2014–2016); or coverage limited to seasonal workers (PA).
4. **Dates.** `effective_start = 2008-01-01` means the regime predates 2008, or that I could not confirm a later origin. A blank `effective_end` means the regime runs through 2026. The research window ends 2026-12-31, but changes already enacted for 2027 are described in `notes` (VA, NJ).
5. **Confidence.** "high" means primary or verbatim statute text was read and nothing suggests a change within the window. "medium" means the source was an official agency summary, a secondary source, or a statute whose history back to 2008 was not verified. "low" means the claim is not verified (TX only, which is moot).
6. **Multi-year rates.** `ag_specific_rate` for NJ lists each year's rate inside one spell, because the coverage regime stays the same while the rate changes every January 1.

## Disagreements with the current hard-coded list

The current list treats these as "agriculture exempt, federal floor": AK, DE, GA, IN, KS, KY, MA, ME, NE, NH, NJ, OK, RI, VA, VT, WA, WV, WY. AL, LA, MS, SC and TN are treated as federal floor because they have no state minimum.

| State | Current list | Finding | Effective dates | Basis |
|---|---|---|---|---|
| **WA** | exempt | **Covered (TRUE)** | entire window | RCW 49.46.010(4)(a) excludes only local, commuting hand-harvest piece-rate workers employed fewer than 13 weeks in agriculture the prior year. The old agricultural exemption was from overtime. |
| **NJ** | exempt | **Covered at the general rate (TRUE)**, then **separate lower farm rate (partial)** | TRUE 2008-01-01 to 2019-06-30; partial from 2019-07-01 | P.L.2019 c.32 and NJDOL MW-220 posters. Farm rate by year: 8.85 (from 2019-07-01), 10.30 (2020), 10.44 (2021), 11.05 (2022), 12.01 (2023), 12.81 (2024), 13.40 (2025), 14.20 (2026), 15.00 (2027). |
| **MA** | exempt (federal) | Separate agricultural minimum of **$8.00** (partial). Before 2015 the statutory figure was $1.60 and nonbinding (FALSE). | partial from 2015-01-01 | M.G.L. c.151 §2A, amended by St. 2014 c.144 §32 (effective 2015-01-01). The general minimum still excludes "agricultural and farm work" (§2). |
| **RI** | exempt | **Covered (TRUE)** | entire window | R.I. Gen. Laws 28-12-3 and 28-12-2(5) have no agricultural exclusion. The agricultural exemption in 28-12-4.3(a)(9) applies only to the overtime sections 28-12-4.1 and 4.2. NALC's "No" cites that overtime section, which I think is a mistake. |
| **ME** | exempt | Exempt through 2025; **covered from 2026-01-01** | FALSE 2008-01-01 to 2025-12-31; TRUE from 2026-01-01 | 26 M.R.S. 663(3)(A) excludes agriculture, except farms with more than 300,000 laying birds. P.L. 2025 c.232 (LD 589, signed 2025-06-10) created 26 M.R.S. 651–655: a separate agricultural minimum with a $14.65 base plus CPI, which Maine DOL reports as $15.10 for 2026, equal to the general minimum. |
| **PA** | not listed (covered) | **partial** | entire window, but it matters only from 2008 to 2009-07-23, while PA's $7.15 exceeded the federal rate | The Minimum Wage Act exempts "labor on a farm" (43 P.S. 333.105(a)(1)). The Seasonal Farm Labor Act (43 P.S. 1301.201) requires seasonal farm workers, including anyone housed in employer quarters, to be paid the state minimum anyway. |
| **WI** | not listed (covered) | **FALSE** until 2009-07-23 | 2008-01-01 to 2009-07-23 | DWD 272.05 set a separate agricultural rate of $5.15, below federal, while the general WI rate was $6.50. From 2009-07-24 all WI rates equal $7.25 (TRUE, moot). |
| **NC** | not listed | FALSE (moot) | entire window | G.S. 95-25.14(a)(2). The state rate equals the federal rate, so there is no practical effect. |
| **HI** | not listed (covered) | **partial** | entire window | HRS 387-1 excludes agriculture in any workweek in which the employer has fewer than 20 employees, and in any workweek in which the worker is harvesting coffee. |
| **NM** | not listed (covered) | **partial** | from 2008-01-01 (Laws 2007 SB 324) | NMSA 50-4-21(C) has FLSA-style agricultural exclusions plus exclusions for employees in "milk production" or range livestock, for handling and packing of agricultural commodities, and for all "employees compensated upon piecework". |
| **MI** | not listed (covered) | **partial** for 2014-05-27 to 2016-09-12, otherwise TRUE | see dates | 2014 PA 138 §10(4) and MCL 408.940(4) exclude piece-rate harvest workers until a piece-rate scale exists. The agency took that position in 2016. R 408.711 was amended effective 2016-09-13 to require a scale equivalent to the minimum hourly wage. |
| **CO** | not listed (covered) | TRUE throughout, with medium confidence before 2022 | — | Before 2022, only farm jobs exempt from the FLSA minimum were excluded (NELP 2019; CDLE INFO for COMPS #37). From 2022-01-01 (COMPS #38, implementing SB21-087), all agricultural employers are covered, and range workers get a weekly-salary alternative. Press accounts conflict; see the low-confidence list. |

Entries on the current list that are **confirmed** (FALSE for the whole window): AK, DE, GA, IN, KS, KY, NE, NH, OK, VA, VT, WV, WY, plus the no-law states AL, LA, MS, SC, TN. Notes on the specific questions in the brief:

- **VA.** The 2020 Virginia Minimum Wage Act amendments (2020 Acts cc. 1204 and 1242) raised the rates from 2021-05-01 but **kept** the exclusion of "any person employed as a farm laborer or farm employee." Va. Code 40.1-28.9 also excludes "any individual employed as a temporary foreign worker as governed by 20 C.F.R. Part 655", which covers H-2A and H-2B workers. 2026 Acts cc. 357 and 358 delete the farm-laborer exclusion **effective 2027-01-01**. The temporary-foreign-worker exclusion stays, so **H-2A workers remain excluded even after 2027**, while domestic farmworkers become covered. NALC's "Yes" for VA is wrong for 2026.
- **WV.** The exclusion is a true agricultural exclusion: 21-5C-1(f)(11) excludes "any individual employed in agriculture" as the FLSA defines it. It is not just a small-employer rule, although the 6-employee threshold and the pre-2015 rule excluding employers with 80% FLSA-covered staff also apply.
- **NH.** RSA 279:21 I says the minimum "shall not apply to ... farm labor." Since 2011 (2011, 204:1) the state minimum is tied to the federal minimum, so the exclusion mattered only until 2009-07-23.
- **NE.** Neb. Rev. Stat. 48-1202(3)(a) excludes "any individual employed in agriculture." This matters because NE's rate exceeded the federal rate from 2015 and reached $15.00 in 2026.
- **DE, VT.** Both statutes exclude "any individual employed in agriculture": 19 Del. C. 901(3)a and 21 V.S.A. 383(2)(A).

States checked whose state minimum exceeded the federal minimum and which **cover** H-2A-size farm employers (TRUE): AZ, AR, CA, CO, CT, DC, FL, IA (2008–09), IL, MD, MI (except 2014–16), MN, MO, MT, NV, NY, OH, OR, RI, SD, WA. The moot states with state rate equal to federal, ID, ND, TX and UT, are recorded TRUE with exemptions.

## Rows where primary statute text was not obtained, or confidence is limited

- **AK (medium).** akleg.gov returned 403. The exemption ("individual employed in agriculture," AS 23.10.055) is confirmed from the Alaska DOLWD *Summary of Alaska Wage and Hour Act* (2025) and older bill texts in search snippets.
- **CO 2008–2021 (medium).** The text of Minimum Wage Orders #22–#35 was not retrieved, and CDLE pages returned 403. The coding relies on NELP's 2019 comment ("workers on small farms who are not subject to the FLSA are not covered by Colorado's MWO") and a search snippet from CDLE INFO #1 for COMPS #37. It **conflicts** with 2021 press statements that farms only had to pay the federal minimum, and with the SB21-087 summary language "removes the exemption of agricultural labor from state and local minimum wage laws." This is the row most worth re-verifying against archived MWO text.
- **IL (medium).** ilga.gov failed TLS certificate verification. I did not bypass it. The wording of 820 ILCS 105/3(d) comes from search snippets of ILGA bill texts (P.A. 100-192, SB0454) and NALC.
- **HI (medium).** The official HRS site returned 403. The text comes from SLH 2024 Act 73, which reprints 387-1. When the 20-employee/coffee paragraph was first enacted was not confirmed for 2008.
- **ME 2026 rate (medium).** The statute (26 M.R.S. 652) gives a $14.65 base plus CPI. The $15.10 figure comes from Maine DOL as quoted in search results; the maine.gov page returned 403.
- **MI (medium, all spells).** The pre-2014 statute (MCL 408.381 et seq.) was not fetched. The exact effective day of 2014 PA 138 was not verified; I used 2014-05-27. Reading the 2016 R 408.711 amendment as ending the piece-rate exclusion is my inference.
- **NJ pre-2019 (medium).** Inferred from statute structure and the 2021 NJDOL poster showing a common $8.85 rate on 2019-01-01. The 2021 farm rate conflicts between NJDOL posters: $10.44 on MW-220 (1/21), $10.30 on MW-220 (1/26).
- **NM (medium).** How the general "piecework" exclusion is applied to farmworkers in practice was not verified.
- **NV (medium).** The pre-2014 NRS 608.250 text was not fetched. The description comes from a 2014 Nevada B&I press release and secondary summaries.
- **NY (medium).** Labor Law Art. 19-A text was not fetched directly. The source is the NYSDOL Part 190 order (snippet) and an archived NYSDOL page.
- **PA (medium).** The Seasonal Farm Labor Act text comes from FindLaw; the PA Agriculture copy did not resolve.
- **SD (medium).** No definitions section excluding agriculture was found. NALC agrees with coverage.
- **ND (medium), UT (medium), TX (low).** All three are moot because the state rate equals federal. ND relies on NDSU Extension and NALC; TX agricultural provisions were not verified.
- **AR, DC, ID (medium).** Statute text came from a mirror (FindLaw) or the DC Council code. History back to 2008 was not separately verified.

## Sources consulted (all retrieved 2026-11-02)

Secondary / screening:
- National Agricultural Law Center, "Minimum Wage for Agricultural Workers" (current through 2026-04-14): https://nationalaglawcenter.org/state-compilations/agpay/minimumwage/
- U.S. DOL WHD, State Minimum Wage Laws (updated 2026-07-01): https://www.dol.gov/agencies/whd/minimum-wage/state

Primary / official by state:
- AK: https://labor.alaska.gov/lss/forms/Summary_of_Alaska_Wage_and_Hour_Act-01-2025.pdf
- AZ: https://azleg.gov/ars/23/00362.01.htm
- AR: https://codes.findlaw.com/ar/title-11-labor-and-industrial-relations/ar-code-sect-11-4-203/
- CA: https://www.dir.ca.gov/IWC/IWCArticle14.pdf
- CO: https://leg.colorado.gov/bills/sb21-087 ; COMPS #38 https://www.sos.state.co.us/CCR/GenerateRulePdf.do?ruleVersionId=10120&fileName=7+CCR+1103-1 ; NELP comment https://www.nelp.org/app/uploads/2019/08/NELP-Comments-Revising-Colorado-Minimum-Wage-Order-August-2019.pdf
- CT: https://codes.findlaw.com/ct/title-31-labor/ct-gen-st-sect-31-58/
- DE: https://delcode.delaware.gov/title19/c009/index.html
- DC: https://code.dccouncil.gov/us/dc/council/code/sections/32-1004
- FL: https://www.flsenate.gov/Laws/Statutes/2025/448.110
- GA: https://codes.findlaw.com/ga/title-34-labor-and-industrial-relations/ga-code-sect-34-4-3/
- HI: https://data.capitol.hawaii.gov/sessions/sessionlaws/Years/SLH2024/SLH2024_Act73.pdf
- ID: https://legislature.idaho.gov/statutesrules/idstat/title44/t44ch15/sect44-1504/
- IN: https://codes.findlaw.com/in/title-22-labor-and-safety/in-code-sect-22-2-2-3/
- IA: https://www.legis.iowa.gov/docs/code/91D.1.pdf
- KS: https://codes.findlaw.com/ks/chapter-44-labor-and-industries/ks-st-sect-44-1202/
- KY: https://codes.findlaw.com/ky/title-xxvii-labor-and-human-rights/ky-rev-st-sect-337-010/
- ME: https://legislature.maine.gov/statutes/26/title26sec663.html ; https://legislature.maine.gov/statutes/26/title26sec651.html ; https://legislature.maine.gov/statutes/26/title26sec652.html ; https://legislature.maine.gov/legis/bills/display_ps.asp?LD=589
- MD: https://mgaleg.maryland.gov/mgawebsite/Laws/StatuteText?article=gle&section=3-403&enactments=false
- MA: https://malegislature.gov/Laws/GeneralLaws/PartI/TitleXXI/Chapter151/Section2A ; https://malegislature.gov/Laws/GeneralLaws/PartI/TitleXXI/Chapter151/Section2 ; https://malegislature.gov/Laws/SessionLaws/Acts/2014/Chapter144
- MI: https://www.legislature.mi.gov/Laws/MCL?objectName=mcl-408-940 ; https://www.law.cornell.edu/regulations/michigan/Mich-Admin-Code-R-408-711 ; https://www.michiganpublic.org/politics-government/2016-04-15/advocates-say-state-failing-to-protect-farmworkers-under-minimum-wage-law
- MN: https://www.revisor.mn.gov/statutes/cite/177.23
- MO: https://codes.findlaw.com/mo/title-xviii-labor-and-industrial-relations/mo-rev-st-290-507/ ; https://codes.findlaw.com/mo/title-xviii-labor-and-industrial-relations/mo-rev-st-290-500/
- MT: https://mca.legmt.gov/bills/mca/title_0390/chapter_0030/part_0040/section_0060/0390-0030-0040-0060.html
- NE: https://nebraskalegislature.gov/laws/statutes.php?statute=48-1202
- NV: https://www.business.nv.gov/news-media/press-releases/2014/labor/nevada-supreme-court-issues-ruling-on-minimum-wage-exemptions/
- NH: https://gc.nh.gov/rsa/html/XXIII/279/279-21.htm ; https://gc.nh.gov/rsa/html/XXIII/279/279-1.htm
- NJ: https://www.nj.gov/labor/wageandhour/assets/PDFs/MW-220(1-26)WageAndHourAbstractPoster.pdf ; https://www.nj.gov/labor/wageandhour/assets/PDFs/Employer%20Poster%20Packet/MW-220_1-21.pdf ; https://www.nj.gov/labor/lwdhome/press/2026/20260929_min_wage.shtml
- NM: https://codes.findlaw.com/nm/chapter-50-employment-law/nm-st-sect-50-4-21/ ; https://www.nmlegis.gov/Sessions/07%20Regular/final/SB0324.html
- NY: https://forms.labor.ny.gov/WP/CR190.pdf
- NC: https://www.ncleg.gov/EnactedLegislation/Statutes/HTML/BySection/Chapter_95/GS_95-25.14.html
- ND: https://ndsu.edu/pubweb/~saxowsky/frm&agbusmgt/ref_topics/miniwageovertime.htm
- OH: https://codes.ohio.gov/ohio-revised-code/section-4111.14
- OK: https://oksenate.gov/sites/default/files/2019-12/os40.pdf (40 O.S. 197.4)
- OR: https://oregon.public.law/statutes/ors_653.020
- PA: https://codes.findlaw.com/pa/title-43-ps-labor/pa-st-sect-43-1301-201/ ; https://codes.findlaw.com/pa/title-43-ps-labor/pa-st-sect-43-1301-103/
- RI: https://webserver.rilegislature.gov/Statutes/TITLE28/28-12/28-12-2.htm ; .../28-12-3.htm ; .../28-12-4.1.htm ; .../28-12-4.3.htm ; https://webserver.rilegislature.gov/BillText24/SenateText24/S2021.htm
- SD: https://sdlegislature.gov/api/Statutes/60-11-3.html ; https://sdlegislature.gov/api/Statutes/60-11-5.html ; https://sdlegislature.gov/api/Statutes/60-11-3.2.html
- UT: https://codes.findlaw.com/ut/title-34-labor-in-general/ut-code-sect-34-40-104/
- VT: https://legislature.vermont.gov/statutes/section/21/005/00383
- VA: https://law.lis.virginia.gov/vacode/title40.1/chapter3/section40.1-28.9/
- WA: https://app.leg.wa.gov/RCW/default.aspx?cite=49.46.010
- WV: https://code.wvlegislature.gov/21-5C-1/
- WI: https://docs.legis.wisconsin.gov/ruletext/CR%2008-069 ; https://www.dwd.wisconsin.gov/eworkboard/min-wage
- WY: https://codes.findlaw.com/wy/title-27-labor-and-employment/wy-st-sect-27-4-201/

## Caveats for use in the paper

- This file records **coverage**, not the binding wage level. To build a binding farm minimum, combine it with a dated schedule of general state minimum rates, which is not part of this deliverable. Then take `max(federal, state rate)` where coverage is TRUE, use `ag_specific_rate` where coverage is partial because of a separate rate, and use the federal rate where coverage is FALSE.
- For H-2A workers, the AEWR usually exceeds every state minimum, so state coverage mostly matters for domestic crew members and for piece-rate guarantees.
- Several state ballot or statutory schedules (NE, MO, FL, etc.) raised rates during the window. The coverage coding above does not depend on those rates.
