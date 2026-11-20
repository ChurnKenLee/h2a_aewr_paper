# Employer design: exploration plan

**Status:**
- Written 2026-11-13 at the user's request.
- Updated the same day with the user's decisions (below).
- Step 1 has been run: `markdowns/employer_panel_step1.md`. Steps 2–5 have
  not.
- No producer, retained artifact, registry or specification has changed.

## Decisions recorded (2026-11-13)

| # | Decision |
| --- | --- |
| 1 | Build a filer unit beside the worksite business (user: yes). |
| 2 | Contact fields may be used as they are. The data are public DOL records (user). |
| 3 | No geocoding for now. The user works with worksite county, city, place and ZIP, not street addresses. |
| 4 | The manual review is set aside for now. A5 below is on hold. |
| 5 | Linkage rules must work for the whole period from FY2008. They may read only fields present in every year: name, mailing address and agent. FEIN (FY2024 on) and phone (FY2015 on) serve as answer keys for scoring, and at most as a labeled sensitivity tier. The 0.98 precision target is measured against the FEIN. |

Consequences for the text below, which is otherwise as first written:

- A2: the filer identifier is not anchored on the FEIN. It comes from the
  rule set scored in step 1.
- A3: contact email, housing address and the other FY2020-on fields cannot
  enter the rules. They remain possible answer keys for FY2020–FY2023.
- A3: the geocoding row is dropped.

**Goal:** find out what an employer-level panel can add to the AEWR
question, and build a panel good enough to trust the answer.

**What to expect.** The employer unit adds no treatment variation: the AEWR
is regional through FY2025 and statewide after. A better panel can deliver
three things:

1. a cleaner and possibly more precise version of the 2025-rule design;
2. one contrast with a different identifying assumption (the same filer
   across states);
3. trustworthy entry and exit margins.

It cannot rescue 2008–2024, where the problem is shared state-by-year shocks
and not the unit.

## What was checked (2026-11-13)

All from retained intermediates; FEIN comparisons use FY2024–FY2026, the only
years with FEINs.

| Fact | Number |
| --- | --- |
| Filers (by FEIN) in FY2024 that file again in FY2025 | 89.7% of 13,789; 93.2% weighted by certified workers |
| Worksite businesses (balanced id) in FY2024 found again in FY2025 | 60.4% of 51,649 |
| Same, FY2015–FY2018 (phone available) | 74–81% |
| Same, FY2008–FY2014 (no phone) | 52–70%; conservative tier 31–51%, high-recall 64–79% |
| Same, FY2019 to FY2020 (the Addendum B break) | 64% |
| Review labels so far (131 of 200 pairs) | accepted links: 85 true, 3 uncertain, 0 false. Near-threshold rejections: 26 true, 1 false |
| FEINs with more than one case that use more than one normalized name | 7.8% of 14,552; more than one ZIP 5.6%; more than one phone 14.8% |
| Normalized name × state pairs that map to more than one FEIN | 0.9% of 20,350 |
| Labor-contractor filings, FY2025 | 3,950 of 24,526 cases; 43% of certified workers |
| Units with worksites in two or more states of one former AEWR region, FY2025 | 438 filers holding 11.3% of certified positions; 237 worksite businesses holding 2.7% |
| Worksite addresses geocoded | 0 of 252,821 requestable addresses |

Three readings:

- **The linkage is precise and misses many true links.** No reviewed accepted
  link is false. Nearly every reviewed near-threshold rejection is a true
  match.
- **Most apparent exit is not exit.** Nine in ten filers return the next
  year. The 36% exit rate in the employer decomposition
  (`markdowns/track1_employer_decomposition.md`) is mostly failed links and
  turnover in the farms a filer names, not filers leaving.
- **The two units carry very different evidence.**

| | Filer (the case's employer) | Worksite business (Addendum B row after FY2019) |
| --- | --- | --- |
| Who it is | the entity that files the job order and sets the offer; includes labor contractors | the farm where the work is done |
| Identity fields | name, mailing address, phone from FY2015, FEIN from FY2024 | name and place-of-employment address only; ZIP on 80% |
| Current linkage | not a unit; used only as alias evidence | the v2 unit (`content/contracts/employer-linkage.md`) |

Fields in the raw files that the linkage does not use:

| Years | Fields |
| --- | --- |
| FY2008 on | agent or attorney name, city and state |
| FY2014 on | law-firm name |
| FY2019 | NAICS code, primary crop, worksite ZIP and county |
| FY2025 file (earlier post-2019 years to be checked) | two previous trade names, contact person's name, phone and email, phone and email to apply, housing address, worksite street address, preparer and law-firm FEIN, NAICS code |

## Part A. Building a better panel

Everything in Part A runs in scratch under a new
`explorations/employer_panel/`. It reads retained artifacts and raw files and
writes nothing to `data/intermediate`. Promotion to the producer is step 5
and needs separate approval.

### A0. Where do the apparent exits come from?

- **What:** for FY2024 to FY2025, classify every worksite business that
  disappears by whether its filer (same FEIN) filed again. For those whose
  filer continued, draw a sample and read the names.
- **Why:** it splits apparent exit into failed links, farms that a continuing
  contractor stopped naming, and real exit. The fixes differ.
- **Output:** a three-way split by count and by certified workers, and 100
  read pairs.

### A1. A scoring harness with FEIN as truth

- **What:** rerun the linkage on FY2024–FY2026 filer records with the FEIN
  blanked, then score every rule's links against the FEIN. Report pairwise
  precision and recall by rule and tier.
- **Second truth set:** FY2015–FY2019 with the phone blanked, scored against
  phone-linked pairs. This is the test for rules meant for FY2008–FY2014,
  which have neither field.
- **Why:** every change below gets a measured precision and recall before it
  is considered. The manual review measures precision on 200 pairs; it says
  little about recall.
- **Limit:** the truth covers filer rows only. Addendum B rows have no FEIN,
  so their recall is judged by A0 samples and by manual labels.

### A2. A filer unit beside the worksite business

- **What:** a second identifier, `filer`, for the case's employer.
  - FY2024 on: FEIN.
  - Earlier years: carry the FEIN back along exact name and phone or address
    chains; then phone (FY2015 on); then name and address.
- **Why:** it is the wage-setting unit, it has the strong identifiers, and it
  runs through the FY2020 break unchanged. It is the natural unit for offered
  wages, filing, and entry and exit.
- **Not a replacement.** The worksite business stays the unit for which
  farms use H-2A workers and for the county business counts.
- **Check:** linked without FEINs, the FY2024-to-FY2025 return rate should be
  close to the FEIN figure of 89.7%.

### A3. Recall fixes for the worksite-business tiers

Each is scored in the harness and kept only if precision holds.

| Fix | What changes | Expected size |
| --- | --- | --- |
| Canonical name key | Drop legal-form tokens (LLC, INC, CORP), site and field numbers, trailing "#2", parentheticals and state suffixes before the exact-name rules | Small: such suffixes are on 1–3% of FY2025 Addendum names. Legal-form variants are not yet counted |
| Filer as a block | Link Addendum rows with the same canonical name under the same filer across years, whatever the ZIP | Likely the largest: a contractor's farms recur under the same filer |
| Thresholds | Lower the 97 and 98 similarity cut-offs where measured precision stays above the target | Medium: about 94,000 balanced-tier candidates are rejected on score |
| Exact name within state, across ZIPs | Move from high-recall to balanced if precision holds | Medium: name × state is 99.1% unique against FEINs for filers; untested for farms |
| Address standardization | Standard street abbreviations, unit numbers and PO boxes before the exact-address rules | Unknown |
| Unused raw fields | Previous trade names, contact email and phone, housing address, agent or attorney, as corroborating contacts under the existing more-than-ten-names guard | Unknown until the per-year inventory is done |
| Agent or attorney for FY2008–FY2014 | Same name, same state, same agent as a corroborated link in the years with no phone | Targets the 52–70% return rates before FY2015 |
| Geocoded addresses (optional) | Match on Census-geocoded coordinates, not address strings | Needs acquisition of 252,821 addresses; ask first |

A probabilistic matcher (Fellegi–Sunter) could be scored in the same harness
as a challenger. It would need a new dependency, so it is not proposed unless
the rule fixes fall short.

### A4. The FY2020 break

- **What:** measure links across FY2019–FY2020 separately for stand-alone
  filers, association members and contractor client farms.
- **Why:** before FY2020 an association's members appear as sub-employer
  rows; after, as Addendum B rows. The filer unit covers filers. Members and
  client farms need the A3 fixes.
- **Output:** cross-break return rates by group, beside the within-era
  rates.

### A5. Review labels

- Finish the 69 unlabeled pairs; the v2 review status is still `pending`.
- Add a recall sample: candidate pairs the rules did not link, stratified by
  era and by rule that came closest.
- The contract requires manual labels. I can prepare each pair with its
  evidence and a suggested label for you to confirm or overrule.

### A6. The panel itself

- **Employer-year table** for each unit, with first and last year, gaps, and
  entry and exit flags built on symmetric windows.
- **Risk set:** an employer is at risk from its first year. Never-filers are
  not observed and are not invented.
- **Linkage uncertainty:** every estimate is run under at least two tiers,
  and worker- or hours-weighted versions are reported, since failed links
  concentrate in small units.
- **Contractor–farm links:** a filer-by-worksite-business table from the
  Addendum rows, for the share of positions that go through contractors.

### Success and failure for Part A

- **Success:** FEIN-blanked recall rises materially at precision of at least
  0.98; the non-FEIN filer return rate is within 2 points of 89.7%; return
  rates before FY2015 and across FY2020 move toward the FY2015–FY2018 level.
- **Partial:** the filer unit works from FY2015 only, and the worksite
  business stays near v2. Designs are then limited to FY2015 on.
- **Failure:** recall rises only by giving up precision. Keep v2 and use the
  filer unit for FY2024 on, where the FEIN does the work.

## Part B. What to try with it

| # | Design | Estimand | What the panel changes | Earlier verdict | Check before registering |
| --- | --- | --- | --- | --- | --- |
| B1 | 2025-rule design on filers, with filer effects | Same as the 2025-rule design: response to the statutory rate within former region | Within-filer changes in positions and hours; exit and entry with a true base rate near 10% | County version: elasticities below about −1 excluded | Placebo-year error on FY2023–24 and FY2024–25 against the county version |
| B2 | Same filer across states under the 2025 rule, filer × year effects | Reallocation of a filer's positions and offers toward states with deeper cuts | Support rises from 2.7% to 11.3% of positions | Killed at triage on the worksite-business unit (MDE above 2) | Dose spread after filer × year effects; placebo-year MDE |
| B3 | Contractor share | Share of a county's or farm's positions that come through contractors, on the 2025-rule dose | Needs A6's filer-by-farm table | Not tried | Descriptive first; an outcome layer for B1, not its own design |
| B4 | Bindingness with corrected floors | Differential response of employers paying above the floor | Occupation AEWRs from the 2023 rule in the at-floor flag; FEIN-linked persistence | Failed: above-floor status is transient, and in 2025 above-floor employers cut offers as much | Share still above the floor two years later; stop if below 70% |

B2 is the only row with a different identifying assumption. It removes
every filer-level shock and leaves state-specific demand shocks that line up
with the cut. It measures movement across states within a filer, which is
not the total demand elasticity, and it rests mostly on labor contractors.

B1 and B2 read FY2026 outcomes that have already been seen at the county
level. Each needs a frozen registry and is labeled post-outcome relative to
the 2025-rule design.

Not pursued, with reasons:

- **A selection model.** A cleaner panel supplies no variable that moves
  filing but not the offer, and filing did not respond to the cut.
- **Employer effects on 2008–2024 regional variation.** This is the plain
  panel with finer units.
- **Employer-level border pairs.** The number of border segments does not
  change.
- **Later windows and the August 2026 schedule.** Already examined and set
  aside (`markdowns/episode_2025_prechecks.md`).

## Order of work and stop points

| Step | Work | Reads | Stop for |
| --- | --- | --- | --- |
| 1 | A0, A1, the per-year inventory of unused raw fields | Retained artifacts and raw headers; FY2026 identity fields only | Report on where exits come from and each rule's precision and recall |
| 2 | A2, A3, A4 in scratch; A5 pairs prepared | Same | Comparison of the scratch identifiers with v2; your labels |
| 3 | B1, B2 and B4 checks on the scratch panel | Outcomes through FY2025 only | Which designs to register, if any |
| 4 | Registries, then estimation for the survivors | FY2026 under frozen registries | Results |
| 5 | Promotion to the producer as v3, only if a design is adopted | — | Separate approval |

Step 5 is large. It changes `02_04`, `src/h2a/employer_linkage.py`, the
contract page and its grounding digest, `EMPLOYER-001`, the review sample,
the county business counts from b01, and every design pinned to the v2
review status. Raw-field extraction belongs in `02_01`, which owns the
disclosure parse.
