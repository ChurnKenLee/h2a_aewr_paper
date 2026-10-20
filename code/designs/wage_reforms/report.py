"""Generate and validate the historical wage-reform evidence report."""

from __future__ import annotations

import json
from datetime import date
from itertools import pairwise

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from analysis import (
    BRANCH,
    REPORTS,
    TABLES,
    configuration,
    digest,
    relative,
    unique_key,
    write_json,
)

from h2a.geography import assert_geo_columns
from h2a.paths import OUTPUTS, PROCESSED, ROOT

FIGURES = OUTPUTS / "figures" / "wage_reforms"


def table(name: str) -> pl.DataFrame:
    return pl.read_csv(TABLES / f"{name}.csv", infer_schema_length=None)


def markdown(headers: list[str], rows: list[list]) -> str:
    text = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        text.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(text)


def one(frame: pl.DataFrame, **conditions) -> dict:
    for field, value in conditions.items():
        frame = frame.filter(pl.col(field) == value)
    if frame.height != 1:
        raise ValueError(f"Expected one report cell: {conditions}, found {frame.height}")
    return frame.row(0, named=True)


def figures() -> list:
    FIGURES.mkdir(parents=True, exist_ok=True)
    paths = []
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    counts = table("weekly_cutoff_counts")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for column, event in enumerate(configuration()["reversal_cutoffs"]):
        year = date.fromisoformat(event).year
        for calendar_year, color in [(year, "#155e75"), (2011, "#a3a3a3"), (2012, "#c5a25d")]:
            data = counts.filter((pl.col("event") == event) & (pl.col("calendar_year") == calendar_year)).sort("week")
            x = data["week"].to_numpy() + 0.5
            axes[0, column].plot(x, data["all_application_case_years"], label=str(calendar_year), color=color)
            axes[1, column].plot(x, data["hourly_median_wage"], label=str(calendar_year), color=color)
        axes[0, column].set_title(f"Calendar cutoff: {event[5:]}")
        axes[0, column].set_ylabel("All disclosure case-years received")
        axes[1, column].set_ylabel("Median hourly offer ($), certified cases")
        axes[1, column].set_xlabel("Weeks from the filing cutoff")
        for row in range(2):
            axes[row, column].axvline(0, color="#b91c1c", linestyle="--", linewidth=1)
            axes[row, column].grid(alpha=0.15)
        axes[0, column].legend(frameon=False)
    fig.suptitle("Wage-method reversal: wages and filing composition", fontsize=14)
    for suffix in ["png", "pdf"]:
        path = FIGURES / f"reversal_cutoffs.{suffix}"
        fig.savefig(path, dpi=180)
        paths.append(path)
    plt.close(fig)

    coverage = table("coverage").filter(pl.col("fiscal_year").is_between(2013, 2018))
    responses = table("cohort_responses").filter(
        (pl.col("cohort") == "herder") & (pl.col("outcome") == "certified_participation")
    ).sort("year")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].plot(coverage["fiscal_year"], coverage["herder_median_monthly_wage"], "o-", color="#155e75")
    axes[0].set_ylabel("Median offered monthly wage ($)")
    axes[0].set_xlabel("Disclosure fiscal year")
    axes[0].set_title("Large observed wage increase")
    for row in responses.iter_rows(named=True):
        if row["status"] != "estimated_diagnostic":
            continue
        axes[1].errorbar(row["year"], 100 * row["coefficient"],
                         yerr=[[100 * (row["coefficient"] - row["ci_low"])],
                               [100 * (row["ci_high"] - row["coefficient"])]],
                         fmt="o", color="#155e75", capsize=3)
    axes[1].plot(2014, 0, "o", color="#a3a3a3")
    axes[1].axhline(0, color="#737373", linewidth=1)
    axes[1].set_ylabel("Herder-minus-hourly participation change (pp)")
    axes[1].set_xlabel("Disclosure fiscal year; 2014 reference")
    axes[1].set_title("Decline precedes the wage increase")
    for axis in axes:
        axis.axvline(2015.5, color="#b91c1c", linestyle="--", linewidth=1)
        axis.grid(alpha=0.15)
    fig.suptitle("Herder phase-in: descriptive evidence, not a causal elasticity", fontsize=13)
    for suffix in ["png", "pdf"]:
        path = FIGURES / f"herder_phase_in.{suffix}"
        fig.savefig(path, dpi=180)
        paths.append(path)
    plt.close(fig)
    return paths


def render() -> None:
    config = configuration()
    sources = json.loads((BRANCH / "policy_sources.json").read_text())
    source_map = {row["id"]: row for row in sources["sources"]}
    coverage = table("coverage")
    local = table("local_wage_responses")
    responses = table("cohort_responses")
    geography = table("herder_geography_support")
    ddd = table("candidate_ddd_support")
    retention = table("linkage_retention_audit")
    weekly = table("weekly_cutoff_counts")
    build_manifest = json.loads((REPORTS / "wage_reforms_build_manifest.json").read_text())
    figure_paths = figures()

    def citation(source_id: str, label: str) -> str:
        return f"[{label}]({source_map[source_id]['url']})"

    local_rows = []
    for row in local.filter(pl.col("role") == "actual_cutoff").iter_rows(named=True):
        if row["status"] == "estimated_diagnostic":
            local_rows.append([
                row["event"], f"±{row['window_days']} days", row["n"], row["clusters"],
                f"{100 * np.expm1(row['coefficient']):+.1f}%",
                f"[{100 * np.expm1(row['ci_low']):+.1f}%, {100 * np.expm1(row['ci_high']):+.1f}%]",
            ])
        else:
            local_rows.append([row["event"], row["window_days"], row["n"], row.get("clusters"), row["status"], "unavailable"])
    local_table = markdown(["Cutoff", "Window", "Cases", "AEWR regions", "Wage discontinuity", "Conditional 95% interval"], local_rows)
    coverage_rows = []
    for row in coverage.filter(pl.col("fiscal_year").is_between(2013, 2018)).iter_rows(named=True):
        coverage_rows.append([row["fiscal_year"], f"{row['herder_positions']:,.0f}",
                              f"{row['positions_share_percent']:.2f}%", row["herder_businesses"],
                              f"{row['businesses_share_percent']:.2f}%", row["herder_cases"],
                              f"${row['herder_median_monthly_wage']:,.2f}"])
    coverage_table = markdown(["FY", "Herder positions", "% of program positions", "Herder businesses",
                               "% of program businesses", "Herder cases", "Median monthly offer"], coverage_rows)
    cohort_rows = []
    selected = responses.filter((pl.col("cohort") == "herder") & (pl.col("outcome") == "certified_participation"))
    for row in selected.iter_rows(named=True):
        cohort_rows.append([row["year"], f"{100 * row['coefficient']:+.1f} pp",
                            f"[{100 * row['ci_low']:+.1f}, {100 * row['ci_high']:+.1f}] pp",
                            row["herder_businesses"], row["hourly_businesses"], row["clusters"],
                            f"{100 * row['conditional_mde80']:.1f} pp"])
    response_table = markdown(["FY", "Participation contrast vs FY2014", "Conditional 95% interval",
                               "Herder firms", "Hourly firms", "States", "Conditional MDE"], cohort_rows)
    geography_rows = [[row["geography_screen"], row["businesses"], row["explicit_sheep_businesses"]]
                      for row in geography.filter(pl.col("fiscal_year") == 2014).iter_rows(named=True)]
    geography_table = markdown(["FY2014 observed geography screen", "Herder businesses", "Explicit sheep titles"], geography_rows)
    ddd_protected = ddd.filter(pl.col("candidate_protected") & (pl.col("support_group") == "explicit_monthly_sheep"))
    protected_firms = int(ddd_protected["baseline_businesses"].sum())
    protected_states = ddd_protected["baseline_state"].n_unique()
    ddd_state_description = ", ".join(f"{row['baseline_state']}: {row['baseline_businesses']}" for row in ddd_protected.iter_rows(named=True))
    return_2018 = one(retention, cohort="herder", baseline_type="herder", geography_supported=True, fiscal_year=2018)
    fit_2015 = one(responses, cohort="herder", year=2015, outcome="certified_participation")
    fit_2018 = one(responses, cohort="herder", year=2018, outcome="certified_participation")
    before = one(weekly, event="2010-03-15", calendar_year=2010, week=-1)
    after = one(weekly, event="2010-03-15", calendar_year=2010, week=0)
    delete = table("cohort_state_deletions").filter(
        (pl.col("cohort") == "herder") & (pl.col("year") == 2018)
        & (pl.col("outcome") == "certified_participation") & (pl.col("status") == "estimated_diagnostic")
    )
    wage_2018 = one(responses, cohort="herder", year=2018, outcome="log_wage")
    text = f"""# Historical wage reforms: investigation and decision

The **2009–2010 reversal remains the stronger identification candidate**, provided
its local occupation/skill wage schedules can be recovered. The 2010 reversal
has a visible offered-wage response. A filing-date RD alone is not credible
because filing composition changes around the deadlines and the rules also
changed recruitment and enforcement.

The **herder phase-in has a large wage first stage but currently lacks a convincing
comparison group**. Its apparent employer-participation response mostly precedes
the wage increase, and its employer-retention measurement is sensitive to the
linkage tier. Neither investigation presently supports a causal wage elasticity.
These are findings of the bounded investigation, not a preference selected by
statistical significance.

## 1. The reversal: price response and sorting

The 2008 rule became effective January 17, 2009; the 2010 rule applied from
March 15, 2010. The latter reinstated FLS wages for new-rule applications while
old-rule contracts retained their wage principles. Cases filed before March 15
with starts on/after June 1 are kept out of the 2010 local fit because the
transition guidance makes their classification ambiguous. The attempted June
2009 suspension was enjoined and is not an implemented third switch.
{citation('transition_2010', 'DOL transition guidance')}; {citation('injunction_2009', 'court order')}.

The following local linear fits use **offered hourly wages of positive-certified
cases**, with state and normalized job-title fixed effects, separate slopes on
each side, equal case weight and AEWR-region CRV1 inference. Intervals use
t(G−1). Reported percentage changes and interval endpoints exponentiate the log
coefficients. All three windows were declared before fitting; none is promoted
as a causal specification.

{local_table}

The weekly all-case disclosure count around March 15, 2010 falls from
**{before['all_application_case_years']:,} in the preceding week to {after['all_application_case_years']:,} in the following week**.
Among positive-certified cases the corresponding counts are
{before['all_positive_cases']:,} and {after['all_positive_cases']:,}. The all-case
audit includes denied/withdrawn/other zero-certified records; the wage sample
conditions on positive certification. These count changes are a sorting and
composition diagnostic, not a formal density test or a causal estimate of
reduced application demand. Same-calendar-date 2011/2012 wage and count
comparisons are retained in `local_wage_responses.csv` and the figure. Annual
AEWR notices can affect these comparison windows too; they are not guaranteed
untreated placebos.

![Wages and filings around the reversal](../figures/wage_reforms/reversal_cutoffs.png)

Early SOC and requested-position fields are absent. The source has job titles
and offered wages, but a generic title does not establish a skill level. The
local regression is consequently a **wage-response diagnostic, not an IV first
stage for a verified statutory exposure**. The 2009 state application-average
wages are not legal rates and are not used anywhere in this branch.
{citation('rule_2010', '2010 final rule')}.

The next defensible design is a **continuous-exposure reversal**: freeze each
baseline job's geography and occupation/skill, construct the gap between its
counterfactual FLS and FLC schedules, and interact that independently assigned
gap with the two implemented regimes. Use fixed employer cohorts with zeros,
plus market-by-job cells for entrants, and distinguish incumbent return from
entry among all potential farms. Common national reform effects can be absorbed;
identification still requires bundled nonwage changes not to have differential
effects correlated with the wage gap. A return to the old wage method need not
undo entry sunk costs immediately.

The historical FLC archive is identified, but **its files have not been recovered
locally**. The official current archive only reaches 2021–2022. The DataLumos
collection identifies original DOL wage files back to FY2007, including the
geographic and occupational tables. Folder/download access failed here, and
shell network access could not resolve the host. The local BLS OES archives
contain wage statistics, not verified FLC Levels I–IV. No substituted schedule
or assigned skill has been fabricated.
{citation('historical_flc', 'historical DOL data archive')}; {citation('current_flc', 'official downloads')}.

A second, more speculative route is **predicted renewal timing**: use an
employer's preannouncement filing or crop-calendar pattern to predict whether
its next application would fall under the old or new wage rule. This avoids
using its chosen postannouncement filing date as assignment. It would still
need a strong timing first stage and calendar-seasonality comparisons; this
investigation does not assume that predictable seasonality is random assignment.

## 2. Herders: large wage changes, thin unaffected support

The national monthly floor is $1,206.31 from November 16, 2015 through 2016,
$1,389.67 in 2017, and $1,584.22 in 2018. The last figure incorporates DOL's
correction to an inconsistent printed notice. Higher applicable state or CBA
floors continue to bind, and wage increases also apply during existing
contracts. **Earlier-filed herder contracts are not an untreated grandfathered
group.** The 2015 rule retained the earlier 364-day sheep/goat and ten-month
other-livestock need limits; those are distinct from visa duration.
{citation('herder_initial', 'initial floor')}; {citation('herder_2017', '2017 floor')};
{citation('herder_2018_correction', '2018 correction')}; {citation('herder_faq', 'ongoing-contract guidance')};
{citation('herder_rule', '2015 rule')}.

Monthly pay plus an explicit herder/range-livestock title is the strict screening
definition. It is not verified legal coverage. All monthly-pay cases and their
titles are retained separately. In particular, FY2011–2012 often use generic
farm-and-ranch-animal titles, so apparent changes in title-screened coverage do
not establish changes in the regulatory population.

{coverage_table}

Positions are **certified positions, not unique workers or visas**. Cases are
counted once using the upstream national-audit singleton/master rule. Employer
counts use distinct reviewed v2 balanced worksite businesses on positive source
rows. Case counts increase substantially during the phase-in while positions
are much steadier: a case-count outcome can reflect filing organization as well
as labor demand. No master total is assigned in full to each member business.

The FY2014 geography audit partitions the screened businesses as follows:

{geography_table}

Associated-case states include every observed row on the business's baseline
cases. This is deliberately conservative: a joint case may list other members'
states without proving that the same worker travels to each state. Conversely,
a single observed state does not prove a complete itinerary. Reported low
monthly wages in higher-floor states cannot be converted directly into wage
exposure.

For an explicit-sheep triple difference, the fixed FY2011 cohort with the strict
geography screen has **{protected_firms} candidate protected employers across
{protected_states} state(s)** ({ddd_state_description}). The proposed California/
Oregon/Hawaii comparison therefore does not provide many independent untreated
policy assignments. Hawaii's earlier prevailing wage is not assumed to remain
a statutory floor. California's floor changes independently and depends on
total employer size, which certified H-2A positions do not measure. The response
window ends in 2018 before California's additional 2019 overtime phase-in.
{citation('california_floor', 'California wage order')}; {citation('california_overtime', 'overtime history')}.

## 3. Employer response: the pre-period is a problem

The herder diagnostic fixes baseline business type and geography in FY2011.
It follows all those businesses through FY2018, including zeros when no
positive certification is observed. The outcome counts any subsequent program
use, even if the business changes job type. Mixed baseline types and ambiguous
geographies remain in the audit. The response contrasts changes for herder and
hourly nonherder businesses within the same baseline state, relative to FY2014.
It is an incumbent-cohort comparison; it is not entry among all farms, and
absence from a disclosure year is not proof that a business closed.

{response_table}

The contrast is already **{100 * fit_2015['coefficient']:+.1f} percentage points in FY2015**,
which ends September 30, before the November 16 wage increase. By FY2018 it is
{100 * fit_2018['coefficient']:+.1f} points. Thus the apparent post-2015 difference
largely inherits an earlier decline. Anticipation of the rule or the June 2014
litigation is possible, but cannot be distinguished here from sector trends,
other rule provisions, or measurement changes. FY2014 is not asserted to be an
uncontaminated pre-period. {citation('mendoza', 'Mendoza litigation')}.

![Herder wage and participation evidence](../figures/wage_reforms/herder_phase_in.png)

The original fixed cohort contains {return_2018['baseline_businesses']} geographically
supported strict herder businesses. By FY2018, {return_2018['balanced_returns']}
have a positive-certified return under the balanced IDs. Allowing a return
through any baseline-linked high-recall ID reaches
**{return_2018['high_recall_reachable_returns']}**; the conservative tier reaches
{return_2018['conservative_reachable_returns']}. These are alternative identity
reachability diagnostics for the **same baseline entities**, not corrected
business counts or interchangeable causal estimates. Exact nonempty raw
name/address/state/postal matching flags
{return_2018['unlinked_exact_identity_return_candidates']} unlinked return candidates
for this cell. That narrow zero does not validate all apparent exits. The full
linkage-retention table remains alongside the results.

Wage-change regressions condition on observing wages in both years and the
same baseline pay/title type. For FY2018, the geographically comparable wage
sample contains only {wage_2018['herder_businesses']} herder businesses and
{wage_2018['hourly_businesses']} hourly businesses in {wage_2018['clusters']} states.
These survivor comparisons are not population first stages and do not justify
forming a Wald elasticity from the participation and wage coefficients.

## 4. Precision and what would change the decision

The conditional MDE column is
[t(.975, G−1) + normal(.8)] × CRV1 standard error for 80% power at two-sided
5%. It is a local approximation under the stated error reference, **not a
size-validated power simulation or an observed-power calculation**. State
clustering also does not absorb a common national sheep-industry shock. The
FY2018 participation coefficient ranges from
{100 * delete['coefficient'].min():+.1f} to {100 * delete['coefficient'].max():+.1f}
percentage points when each state is deleted in turn. These deletions are
influence checks, not a choice of preferred sample.

For the reversal, the priority data are historical FLC wage vintages covering
2008–2010, their area/county definitions and occupation/skill schedules, and
baseline job descriptions or wage determinations that resolve occupation and
skill without using postreform wages. Exposure must also respect recruitment
vintage, competing floors and transition cohorts. The present data establish
why that recovery is worth pursuing; they do not supply its missing exclusion
restriction.

For herders, the priority is a reviewed employer/contract panel covering the
2013–2018 disclosure-format transitions, verified range duties and itineraries,
and actual state/CBA obligations. A larger verified high-floor comparison
group could revive the triple difference. If such a group cannot be recovered,
the phase-in mostly supplies national herder time variation: the 2017 and 2018
steps were announced in 2015, and adding herder-by-year controls absorbs that
variation. Bigger wage changes alone do not solve that identification problem.

**Current decision:** retain the reversal as the lead candidate for source
recovery; retain the herder wage/coverage evidence but do not promote the simple
herder DiD or thin protected-state triple difference to a causal specification.

## Reproduction and audit trail

Run `./scripts/run_wage_reforms.sh` from the repository root. The runner exports
the four Marimo applications before execution and uses local inputs only.
The exact specification is `code/designs/wage_reforms/design.json`; sourced
policy facts and archive-access limits are in `policy_sources.json`. Source
files, other designs, the shared panel, the manuscript and welfare are not
rewritten by this runner.

The build contains {build_manifest['audit']['positive_case_years']:,}
positive-certified case-years and
{build_manifest['audit']['all_source_case_years']:,} total source case-years.
There are {build_manifest['audit']['application_case_years_without_unique_valid_receipt']}
all-case rows without a unique valid receipt date and
{build_manifest['audit']['case_numbers_in_multiple_fiscal_vintages']} case numbers
appearing in multiple positive fiscal vintages. Fiscal-year/case is the declared
source key; no claim of globally unique worker or application events is made.
Raw dates are parsed without hand repairs; the certification/job/requested
date-field priority accommodates the FY2016 disclosure field change. Missing
hours and requested positions remain missing.

All windows, placebo fits, failed-fit statuses, cohort paths, title/wage quality,
geography support, linkage reachability, and state deletions are retained in
`outputs/tables/wage_reforms/`. PNG and PDF figures are in
`outputs/figures/wage_reforms/`. The build and report provenance JSONs record
SHA-256 hashes, and the final validation checks source freshness, unique keys,
fixed assignments, complete cohort years, zero/wage semantics, arithmetic and
the complete declared fit menu. Successful validation establishes reproducibility,
not causal identification.
"""
    report_path = REPORTS / "wage_reforms.md"
    report_path.write_text(text)
    tracked = [BRANCH / "design.json", BRANCH / "policy_sources.json", BRANCH / "analysis.py",
               BRANCH / "report.py", REPORTS / "wage_reforms_build_manifest.json"]
    tracked.extend(sorted(TABLES.glob("*.csv")))
    tracked.extend(sorted(BRANCH.glob("0*.py")))
    manifest = {"version": config["version"], "causal_elasticity_estimated": False,
                "inputs": {relative(p): digest(p) for p in tracked},
                "outputs": {relative(p): digest(p) for p in [report_path, *figure_paths]}}
    write_json(REPORTS / "wage_reforms_provenance.json", manifest)
    print(relative(report_path))


def validate() -> None:
    config = configuration()
    checks = []
    for name in ["wage_reforms_build_manifest.json", "wage_reforms_provenance.json"]:
        manifest = json.loads((REPORTS / name).read_text())
        if manifest["version"] != config["version"]:
            raise ValueError("Incompatible report/build registry")
        for category in ["inputs", "outputs"]:
            for file, expected in manifest[category].items():
                if digest(ROOT / file) != expected:
                    raise ValueError(f"Stale {category} artifact: {file}")
        checks.append(f"{name}: all input/output hashes match")
    cases = pl.read_parquet(PROCESSED / "wage_reforms_cases.parquet")
    applications = pl.read_parquet(PROCESSED / "wage_reforms_application_cases.parquet")
    panel = pl.read_parquet(PROCESSED / "wage_reforms_business_year.parquet")
    unique_key(cases, ["fiscal_year", "case_number"])
    unique_key(applications, ["fiscal_year", "case_number"])
    unique_key(panel, ["cohort", "employer_id_balanced", "fiscal_year"])
    assert_geo_columns(cases, ["aewr_region_id"], allow_null=["aewr_region_id"])
    if cases.filter(pl.col("certified") <= 0).height:
        raise ValueError("Nonpositive cases in certified-case sample")
    if applications.filter(pl.col("any_certified")).height != cases.height:
        raise ValueError("All-case and positive-case universes disagree")
    if panel.filter((pl.col("certified_participation") == 0) & pl.col("log_wage").is_not_null()).height:
        raise ValueError("A nonparticipating business has an observed wage")
    if not set(panel["certified_participation"].unique()).issubset({0, 1}):
        raise ValueError("Invalid participation coding")
    if panel.filter(pl.col("fiscal_year") == pl.col("baseline_year")).filter(pl.col("certified_participation") != 1).height:
        raise ValueError("Baseline cohort not defined on baseline certification")
    invariance = panel.group_by("cohort", "employer_id_balanced").agg([
        pl.col(c).n_unique().alias(c) for c in ["baseline_type", "baseline_state", "geography_supported"]
    ])
    if invariance.filter(pl.any_horizontal(pl.col("baseline_type", "baseline_state", "geography_supported") != 1)).height:
        raise ValueError("A baseline assignment changes over time")
    for cohort in config["cohorts"]:
        subset = panel.filter(pl.col("cohort") == cohort["name"])
        expected = cohort["years"][1] - cohort["years"][0] + 1
        if subset.group_by("employer_id_balanced").len().filter(pl.col("len") != expected).height:
            raise ValueError("Incomplete baseline cohort panel")
        if sorted(subset["fiscal_year"].unique()) != list(range(cohort["years"][0], cohort["years"][1] + 1)):
            raise ValueError("Wrong cohort years")
    checks.append("Unique keys, canonical geographic strings, fixed assignments and complete zero-inclusive cohorts")
    coverage = table("coverage")
    aggregate = cases.group_by("fiscal_year").agg(pl.col("certified").sum().alias("recomputed"))
    comparison = coverage.join(aggregate, on="fiscal_year", validate="1:1")
    np.testing.assert_allclose(comparison["all_certified_positions"], comparison["recomputed"], rtol=0, atol=1e-8)
    np.testing.assert_allclose(coverage["positions_share_percent"],
                               100 * coverage["herder_positions"] / coverage["all_certified_positions"], atol=1e-10)
    local = table("local_wage_responses")
    unique_key(local, ["event", "calendar_year", "window_days"])
    expected_local = {(event, year, window) for event in config["reversal_cutoffs"]
                      for year in [date.fromisoformat(event).year, 2011, 2012]
                      for window in config["local_windows_days"]}
    if set(local.select("event", "calendar_year", "window_days").iter_rows()) != expected_local:
        raise ValueError("Incomplete local response menu")
    responses = table("cohort_responses")
    unique_key(responses, ["cohort", "outcome", "year"])
    expected_responses = {(c["name"], outcome, year) for c in config["cohorts"]
                          for outcome in ["certified_participation", "log_wage"]
                          for year in range(c["years"][0] + 1, c["years"][1] + 1)
                          if year != c["reference_year"]}
    if set(responses.select("cohort", "outcome", "year").iter_rows()) != expected_responses:
        raise ValueError("Incomplete cohort response menu")
    for frame in [local, responses]:
        fitted = frame.filter(pl.col("status") == "estimated_diagnostic")
        if fitted.filter((pl.col("clusters") < 2) | (pl.col("ci_low") > pl.col("coefficient"))
                         | (pl.col("coefficient") > pl.col("ci_high"))).height:
            raise ValueError("Invalid fitted diagnostic")
    checks.append("Case totals/share arithmetic and complete declared response menus, including failures")
    sources = json.loads((BRANCH / "policy_sources.json").read_text())
    rates = sources["herder_reference_schedule"]
    if [row["monthly_aewr"] for row in rates] != [1206.31, 1389.67, 1584.22]:
        raise ValueError("Unreviewed monthly floor or uncorrected 2018 typo")
    for left, right in pairwise(rates):
        if left["effective_to_exclusive"] != right["effective_from"]:
            raise ValueError("Herder reference schedule has a gap or overlap")
    provenance = json.loads((REPORTS / "wage_reforms_provenance.json").read_text())
    if provenance["causal_elasticity_estimated"]:
        raise ValueError("Diagnostic branch has promoted a causal elasticity")
    checks.append("Source-traced monthly schedule, corrected 2018 value and no causal promotion")
    result = {"status": "passed", "version": config["version"], "checks": checks,
              "positive_cases": cases.height, "all_application_cases": applications.height,
              "business_year_rows": panel.height, "local_fits": local.height, "cohort_fits": responses.height}
    write_json(REPORTS / "wage_reforms_validation.json", result)
    print(json.dumps(result))
