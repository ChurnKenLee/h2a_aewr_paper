# Employer panel, step 1: where exits come from and which rules work from FY2008

- **Date:** 2026-11-13.
- **Plan:** `markdowns/employer_design_agent_plan.md`, step 1.
- **Notebooks:** `explorations/employer_panel/` (see its README).
- **Outputs:** scratch directory
  `/tmp/claude-1000/-home-ken-Dropbox-projects-H-2A-Paper/0dae8120-5cae-40b7-a1f7-ef23357b9b56/scratchpad/employer_panel`
  (called `$P` below). Every number here comes from a CSV or log there.
- **Data read:** identity fields for FY2008–FY2026 Q3 and certified worker
  counts. No design outcome was estimated. No retained artifact, producer or
  linkage contract changed.

"Filer" means the employer named on the case. "Worksite business" is the
current v2 linkage unit.

## Verdict

- **A filer identifier that works from FY2008 is feasible.** A rule set that
  reads only the name, mailing address and agent reaches 0.99 precision and
  0.97 recall against the FEIN, which it never sees.
- **The current exact rules lose one returning filer in ten.** The proposed
  set loses 2.6%.
- **Its quality looks stable back to FY2015.** Scored against the phone, its
  precision and recall are the same in FY2015–19, FY2020–23 and FY2024–26.
- **FY2008–FY2010 remain a problem.** Return rates there are 59–70% under
  any name rule, against 76–88% from FY2011. I could not show whether that
  is real.
- **Of worksite businesses that vanish between FY2024 and FY2025, one in
  seven lost its filer.** The rest were named by a filer that came back.

## 1. Fields that exist in every year

`$P/raw_field_inventory.csv`.

| Field | First year | Share of cases |
| --- | --- | --- |
| Employer name, mailing address | FY2008 | 99–100% |
| Agent or attorney name | FY2008 | 84–92% |
| Law firm | FY2014 | 85–92% |
| Phone | FY2015 | 100% |
| NAICS code | FY2017 | 78% rising to 100% |
| Contact email, housing address, phone and email to apply | FY2020 | 96–100% |
| FEIN | FY2024 | 100% |
| Previous trade names | FY2024 | column present, always empty |

Only the first two rows can enter a rule meant for the whole period. Phone
and FEIN are used below as answer keys only.

## 2. Rule sets scored against the FEIN

`02_rule_harness.py`; `$P/harness_rule_sets.csv`, `harness_return_errors.csv`.

- **Sample:** 67,783 filer cases in FY2024–FY2026, all with a FEIN.
- **Precision:** for a case, the share of cases grouped with it that have
  its FEIN, averaged over cases.
- **Recall:** for a case, the share of cases with its FEIN that are grouped
  with it, averaged over cases.
- **Returning filer missed:** of FY2024 cases whose FEIN files again in
  FY2025, the share whose rule identifier does not.
- Each set contains the one above it and is closed over all years.

| Set | Adds | Precision | Recall | Returning filer missed |
| --- | --- | --- | --- | --- |
| R0 | Exact name at the same address or state and ZIP, with the v2 test for a distinctive name | 0.996 | 0.891 | 10.0% |
| R1 | Short names allowed ("WSJ Farms", "E & E Farm") | 0.996 | 0.903 | 8.3% |
| R2 | Core name at the same address or state and ZIP | 0.995 | 0.935 | 5.5% |
| R3 | Core name in the same state and city | 0.995 | 0.945 | 4.6% |
| R4 | Same distinctive words at the same address, ZIP or city | 0.993 | 0.946 | 4.5% |
| R5 | Similar name at the same address (ratio 85) or state and ZIP (94) | 0.991 | 0.960 | 3.5% |
| R6 | Core name with the same state and agent | 0.991 | 0.967 | 2.9% |
| **R7** | **Core name in the same state** | **0.990** | **0.969** | **2.6%** |
| R8 | Core name in any state | 0.975 | 0.983 | 1.4% |
| R9 | Same address under any name | 0.928 | 0.989 | 0.6% |

- **R7 is the loosest set that meets the 0.98 target.** R8 does not.
- **R0 is the current exact rules without phone or FEIN.** It is the
  baseline for what name and address alone do today.
- **The core name does most of the work.** It drops legal-form words,
  parentheticals such as "(Petition 2)", numbers and plural endings.
- **The v2 test for a distinctive name costs recall.** It blocks names with
  no word of four letters or more, even at an identical address.

### The FEIN is not a perfect answer key

- Under R0, 5.0% of FY2024 cases whose FEIN does not return are still found
  in FY2025 by exact name and place.
- I read ten groups that these rules formed across two FEINs:
  - three are FEIN typos (one digit apart);
  - four are the same name in the same city under two FEINs;
  - three are false merges, all from the state-wide name rule ("Johnson
    Farms" in two North Dakota towns).
- Measured precision is therefore a lower bound.

### Is the quality the same in earlier years?

The phone is the only answer key before FY2024. It is noisier: 15% of FEINs
use more than one phone.

| R7 scored against the phone | Precision | Recall |
| --- | --- | --- |
| FY2024–FY2026 | 0.922 | 0.941 |
| FY2020–FY2023 | 0.913 | 0.940 |
| FY2015–FY2019 | 0.919 | 0.932 |

The same rule set scores 0.990 and 0.969 against the FEIN in FY2024–26. If
the gap between the two answer keys is constant, quality is the same back to
FY2015. Nothing can score FY2008–FY2014 directly.

The two similarity thresholds were chosen on the FEIN window they are scored
on. The phone windows did not inform them.

## 3. Return rates by year

Share of filers in a year that file again the next year
(`$P/harness_return_rates.csv`).

| Fiscal year | R0 | R7 | Phone | FEIN |
| --- | --- | --- | --- | --- |
| 2008 | 0.45 | 0.61 | | |
| 2009 | 0.43 | 0.59 | | |
| 2010 | 0.51 | 0.70 | | |
| 2011 | 0.59 | 0.76 | | |
| 2012 | 0.59 | 0.79 | | |
| 2013 | 0.72 | 0.84 | | |
| 2014 | 0.73 | 0.82 | | |
| 2015 | 0.76 | 0.84 | 0.81 | |
| 2016 | 0.77 | 0.85 | 0.82 | |
| 2017 | 0.81 | 0.87 | 0.85 | |
| 2018 | 0.79 | 0.87 | 0.84 | |
| 2019 | 0.72 | 0.84 | 0.78 | |
| 2020 | 0.80 | 0.88 | 0.83 | |
| 2021 | 0.81 | 0.88 | 0.84 | |
| 2022 | 0.79 | 0.88 | 0.84 | |
| 2023 | 0.80 | 0.88 | 0.83 | |
| 2024 | 0.79 | 0.88 | 0.84 | 0.90 |

- **R7 runs through the FY2020 form change.** The FY2019 rate is 0.84. R0
  dips to 0.72 there.
- **From FY2013 the R7 series is flat at 0.82–0.88.**
- **FY2008–FY2012 are lower, and FY2008–FY2010 much lower.** What I checked:
  - It is not regional: Kentucky returns 59% in FY2008 and 86% in FY2012.
  - It is not mostly missed links. Of the 1,819 FY2008 filers that R7 does
    not find in FY2009, 14% have another name filing from their address,
    and in a sample of 400, 3% have a name in the state with a ratio of 85
    or more.
  - The FY2008 file has two case-number series (2,038 "A" and 3,604 "C"
    cases); from FY2009 nearly all are "C".
- I do not know whether the early turnover is real. It needs a check
  against published program counts before FY2008–FY2010 enter any exit
  outcome.

## 4. Where apparent exits of worksite businesses come from

`03_exit_diagnosis.py`; `$P/exit_diagnosis.csv`. Of 51,649 worksite
businesses (v2 balanced) in FY2024, 20,476 (39.6%) are absent in FY2025.

| What happened | Businesses | Share of the absent | Share of FY2024 certified workers |
| --- | --- | --- | --- |
| Filer did not file again | 3,012 | 14.7% | 4.3% |
| Filer filed and named the same core name again | 7,483 | 36.5% | 6.5% |
| Filer filed and named a similar name (ratio 85 or more) | 1,710 | 8.4% | 1.4% |
| Filer filed and named nothing like it | 8,271 | 40.4% | 8.2% |

- **The second row is missed links.** A rule "same filer, same core name"
  would recover them. Many of these rows have no ZIP, which the current
  exact-name rule requires.
- **The third row is mixed.** "Taylor Farms - Viera" and "Vierra" are one
  site. "Dole Fresh Vegetables - Loveland" and "- Vel" are two ranches.
- **Many Addendum names are a grower plus a ranch**, such as "DOLE FRESH
  VEGETABLES (grower): West Pendergast". The current unit counts each ranch
  as a business. A grower-level key would take the part before the ranch.
- **The fourth row is mostly contractor filings** (80%). It is turnover in
  the farms or ranches a contractor names, or renaming too large to match.
- Absent businesses hold 20% of FY2024 certified workers. Those whose filer
  left hold 4%.

## What I recommend for step 2

1. **Adopt R7 as the candidate filer identifier**, with R5 and R8 as the
   tight and loose sensitivities. Build the filer-year panel on it in
   scratch.
2. **Tighten the state-wide name rule**, the source of the false merges
   read above, for example by requiring that the name is rare in the state.
3. **For worksite businesses:** add "same filer, same core name" and a
   grower-level key, then rerun the exit split.
4. **Resolve FY2008–FY2010** against published counts, or start
   employer-level outcomes in FY2011.

## Limits

- The scores are for filers. Addendum B rows have no answer key; section 4
  is the only evidence on them.
- Rule sets were scored on the years they will be least needed for. The
  phone comparison is the only check on earlier years, and it stops at
  FY2015.
- Agent names are people and firms mixed, and a firm's agents change. The
  agent rule adds 0.7 points of recall.
- The filer unit includes labor contractors and, before FY2020, association
  masters.
