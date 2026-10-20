> Historical dispatch, no longer an active panel-IV task. The user retired
> Goal 1, its extensions and generated results. Follow the current panel-IV
> README and canonical design; the border material below is retained unchanged.

# Dispatch the panel-IV and AEWR-border goals

Run two independent Codex sessions in this repository's **existing checkout**.
Session A owns the panel-IV revision; session B owns the border-design revision.
Each session may delegate independent work to subagents. Use no worktrees,
clones, or branch switches.

This file supplies the dispatch instructions. Reading or creating it does not
activate either goal. The commands below activate them when you submit them.

## 1. Open and name two sessions

In two terminal tabs, enter this repository using the existing direnv/devenv
environment and start Codex:

```bash
cd /home/ken/Dropbox/projects/H-2A_Paper
codex
```

In the first session, submit:

```text
/rename H2A IV weight shocks
```

In the second session, submit:

```text
/rename H2A border redesign
```

Keep both sessions running. A goal belongs to its chat, so each session has its
own goal state. See the [official Goals guide](https://developers.openai.com/cookbook/examples/codex/using_goals_in_codex).

## 2. Establish the shared baseline before either session edits

Paste this ordinary message into **session A** and wait for it to finish:

```text
Read TWO_GOAL_DISPATCH.md in full. Prepare the shared-checkout run described
there without starting either research goal or changing repository files.

Create /tmp/h2a-iv-border-dispatch for this run. If that directory contains
an earlier run, preserve it under a distinct archival name before starting.
Never move or overwrite a coordination directory used by an active run.

Inspect git status and record the current commit, branch, relevant tracked
diffs, untracked design files, and an input/output inventory in baseline.md.
Preserve scoped copies or hashes sufficient to identify subsequent changes
and reproduce the current designs. Include existing uncommitted code and
untracked files; HEAD alone is not the starting design. Exclude secrets and
environment files. Inventory large artifacts rather than copying all data.

Run the required grounding verification and scope snapshots for both design
directories. Read every instruction, README, canonical page, and assumption
named by those snapshots. Inspect the actual runners and artifact paths.

Create ownership.md with the exact file/artifact allocations from this guide,
resolved against the current checkout, and record any previously unknown
shared dependencies. Initialize A.md and B.md with their assigned scopes.
After initialization, A.md has only session A as writer and B.md only B.
Do not infer file ownership from git status alone.

Record the common inputs that both sessions will treat as immutable during
the run, the verification result, and any pre-existing failures. Report the
baseline path and readiness to launch. Do not start analyses or delegate
implementation during this preparation step.
```

Do not run this initialization again while either goal is active. After a
restart, reuse and read the existing coordination records.

## 3. Activate both goals

Paste this into **session A**:

```text
/goal Complete Goal 1 in TWO_GOAL_DISPATCH.md: propose and implement a panel-IV construction that isolates entropy-calibrated survey-weight movements from donor-wage movements, with a frozen-wage implementation, a properly assessed Hamilton alternative, and the specified identification and validation evidence. Follow the entire dispatch guide and /tmp/h2a-iv-border-dispatch/ownership.md. Use parallel subagents for independent methodological research, code tracing, and review, with exclusive file ownership for edits. Preserve the shared checkout and existing work. Act as session A and perform the shared integration responsibilities after both branches are ready. Continue through implementation and required checks; report empirical limitations honestly.
```

Paste this into **session B**:

```text
/goal Complete Goal 2 in TWO_GOAL_DISPATCH.md: diagnose the existing AEWR-border design's lack of power, develop and implement the strongest defensible border-based redesign supported by local data, and deliver the specified power, identification, inference, and robustness evidence. Success requires credible identification and informative precision, not a significant coefficient. If the available design cannot meet that standard, finish the bounded investigation and implementation with an explicit infeasibility finding and a concrete data/design requirement. Follow the entire dispatch guide and /tmp/h2a-iv-border-dispatch/ownership.md. Use parallel subagents for independent design research, power diagnosis, and review, with exclusive file ownership for edits. Act as session B, preserve existing work, and coordinate shared integration with session A.
```

These prompts explicitly request subagents. Each lead should delegate bounded
tasks that can proceed alongside useful local work, then integrate their
findings. Do not launch several agents that all edit the same estimator. The
[official subagent guide](https://developers.openai.com/codex/multi-agent)
describes delegation and inspection through `/agent`.

## 4. Shared execution and ownership contract

### File ownership

| Owner | Writable scope |
| --- | --- |
| A | `code/designs/panel_iv/`, its tests, README and local `AGENTS.md`; `scripts/run_panel_iv.sh`; `content/designs/panel-iv.md`; artifacts explicitly produced by this branch |
| B | `code/designs/border_discontinuity/`, its tests, README and local `AGENTS.md`; `scripts/run_border_discontinuity.sh`; `content/designs/border-discontinuity.md`; artifacts explicitly produced by this branch |
| A, during coordinated integration | Shared documentation such as `content/contracts/causal-outcomes-and-model-inputs.md`, `agent/assumptions.toml`, root and runner READMEs, and reviewed grounding generation |
| Neither during concurrent implementation | Source/derived/shared-panel producers, `code/paths.R`, `src/h2a/paths.py`, shared helpers, hand-maintained crosswalks, raw data, shared input panels, dependency manifests/locks, unrelated designs, and manuscript sources |

Resolve actual artifact filenames before writing. Panel-IV outputs include
`panel_iv_*`, `iv_*`, `table_iv_*`, and `fig_iv_*` products; border outputs use
`border_discontinuity_*`. These are discovery hints, not permission to overwrite
every matching file. Record the owning producer for each exact output.

Create new retained methodological reports at
`outputs/reports/panel_iv_weight_shocks.md` and
`outputs/reports/border_redesign.md`, owned by A and B respectively. Keep their
generating code and machine-readable supporting results with the relevant
branch. Store exchange artifacts in `data/intermediate`, analysis panels in
`data/processed`, and retained results in `outputs`.

### Coordination and preservation

- Both sessions read `baseline.md` and `ownership.md` before editing. Their
  subagents inherit the same scope limits and repository instructions.
- Each lead alone maintains its status file, using states such as `working`,
  `waiting-on-input`, `ready-for-integration`, and `validated`. Record active
  commands, changed paths, specification decisions, validation, requests, and
  which artifacts are stable. Update at meaningful milestones and before
  releasing inputs to another reader.
- Read the other status file before dependent work and during integration.
  These files are an explicit coordination convention, not automatic messaging
  between otherwise independent chats.
- Request changes outside your scope in your own status file. The owner makes
  them. Ownership transfers require acknowledgements in both status files
  before edits begin; do not resolve an overlap by editing simultaneously.
- Preserve pre-existing changes and retained baseline evidence. Do not stage,
  commit, stash, reset, clean, switch branches, or revert unfamiliar changes.
  Compare against the recorded working baseline, not just the last commit.
- Do not point analysis outputs or editable Python imports at another checkout.
  Both sessions use the repository path helpers and existing environment.
- Serialize heavy estimation, recovery, and bootstrap commands with a common
  OS lock. For example, from the repository root:

  ```bash
  flock /tmp/h2a-iv-border-dispatch/heavy.lock ./scripts/run_panel_iv.sh
  ```

  ```bash
  flock /tmp/h2a-iv-border-dispatch/heavy.lock ./scripts/run_border_discontinuity.sh
  ```

  Acquire the same lock for narrower heavy entry points, too. Do not nest a
  second acquisition inside a runner already holding it. Use a background
  command session and short status polls while waiting so the lead remains
  responsive. Lightweight reading, coding, and review can proceed concurrently.
  Preserve the runners' thread, memory, seed, and other resource guards.
- If a shared producer or dependency change proves necessary, record the
  concrete reason and affected consumers. Coordinate one owner and quiesce
  affected consumers before changing or rebuilding shared inputs. Do not add
  downstream schema repairs. The design requests do not authorize a full source
  refresh, credentialed acquisition, or an unrelated dependency upgrade.

### Authorization for empirical revisions

The user's requests authorize the documented design changes needed for these
two goals. Existing instructions that freeze the old donor-wage construction
or old primary border specification describe the starting design. They must
not prevent the requested revisions merely because the old choices were
previously preferred.

Before changing such a choice, enumerate the old and new estimand, treatment,
timing, sample, normalization, weights, controls, fixed effects, inference,
support, and outputs. Preserve the old specification as a labeled comparator.
Update the local design instructions, registry, README, canonical design page,
and affected assumptions to describe the resulting design. Do not weaken an
integrity check merely to pass validation. Routine implementation choices
within this scope do not require renewed permission.

All research-integrity, source-data, geography, reproducibility, and execution
requirements in `AGENTS.md` continue to apply. Leave unrelated specifications
unchanged. An inconclusive result is a finding, not permission to keep searching
until the coefficient or p-value looks favorable.

## 5. Goal 1: isolate survey-weight movements in panel IV

### User objective

The current instrument combines entropy-balanced survey weights and wages in
dissimilar donor clusters. Propose ways to separate these moving parts, then
implement the revision in panel IV. Evaluate preperiod average wages and a
Hamilton-filtered wage trend, together with reasonable alternatives.

### Establish the current implementation

Start from `code/designs/panel_iv/README.md`, its canonical design page, local
instructions, `design.R`, and stages 04--09. Recheck the scoped snapshot and
read all named context before planning edits.

At dispatch preparation, the branch uses county entropy weights, the mapped
OEWS-area Big-Six hourly-wage proxy, source year `t-1`, soft calibration with
`rho = 0.10`, fixed target/donor clusters, and target-area overlap exclusion.
The wage-only and wage-plus-seasonal/composition calibrations are distinct;
controlled column 4 is the existing preferred specification. Verify these
facts against executable code when starting.

The OEWS measure is not a directly observed county farm wage. Identify the
exact wage source, units, geographic support, observation years, and publication
timing. A farm-wage series constructed from QCEW or BEA would be an explicit
measurement change with its own denominator and coverage requirements, never
a silent fallback. Baseline the actual current estimates if valid local inputs
permit; label retained estimates as historical until their provenance is checked.

### Required methodological work

1. **Write the exact instrument equation.** Trace county calibration through
   donor selection, overlap exclusion, missingness, renormalization, aggregation,
   deflation, and policy-year assignment. Identify every source of time variation.
   Explain whether “survey-weight shock” means a change from a fixed reference,
   the posterior-minus-prior calibration adjustment, or an innovation relative
   to a predetermined prediction. These objects are not interchangeable.

2. **Separate weight, wage, and interaction components.** On a declared fixed
   support, let `a[g,c,s]` be the final normalized donor weight and `w[c,s]` the
   real donor wage, where `s = t-1`. For fixed reference weights `a0[g,c]` and
   fixed wage loadings `b[c]`, verify the accounting identity:

   ```text
   Z[g,t]  = sum_c a[g,c,t-1] * w[c,t-1]
   Z0[g]   = sum_c a0[g,c] * b[c]

   Z[g,t] - Z0[g]
     = sum_c (a[g,c,t-1] - a0[g,c]) * b[c]
     + sum_c a0[g,c] * (w[c,t-1] - b[c])
     + sum_c (a[g,c,t-1] - a0[g,c]) * (w[c,t-1] - b[c])
   ```

   These are an algebraic diagnostic, not a proof of exclusion. Handle changes
   in eligible support explicitly rather than attributing them to survey shocks.
   Report component variances and covariances, including after the estimation
   controls and fixed effects are removed.

3. **Compare a small, declared menu before selecting an implementation.**

   | Candidate | Required assessment |
   | --- | --- |
   | Fixed preperiod average real wage by donor county, or a transparently aggregated fixed donor-cluster wage | Choose the window using genuine prepolicy availability; deflate each baseline observation into the declared common units before averaging; keep the resulting loading fixed |
   | A single preperiod reference-year wage | Useful benchmark for the amount of baseline measurement noise introduced or reduced by averaging |
   | Pooled or shrunk preperiod wage loadings | Estimate only from admissible baseline data; freeze fitted loadings and tuning choices; retain heterogeneity that identifies the instrument |
   | A preperiod-fitted trend extrapolation | Distinguish fixed parameters from a time-varying loading; explain what stochastic wage movements remain and whether the advisor's requested restriction is met |
   | Hamilton forecast/trend construction | Specify information sets, annual-data horizon and lags, coefficient-estimation window, and support; distinguish the predicted component from the residual |

   The menu is a set of candidates to investigate, not an instruction to run
   an exhaustive grid. Explain advantages, identifying assumptions, feasibility,
   sample consequences, and selection criteria in a short sourced memo.

4. **Make the Hamilton assessment substantive.** Read Hamilton's original
   [filter paper](https://www.nber.org/papers/w23429) and verify the formula
   against the chosen data frequency. Do not import quarterly lag choices into
   a short annual panel without justification. Audit the usable history after
   horizons and lags. A lagged regressor does not prevent look-ahead if its
   coefficients are fitted on the full future sample. Document any pooled fit,
   recursive fit, or preperiod-only fit and its additional assumptions.

   A Hamilton fitted trend generally varies over time. Multiplying changing
   weights by that trend is not literally a fixed-wage instrument. Treat this
   as a distinct sensitivity unless the construction demonstrably isolates
   the requested component. If the history cannot support a defensible fit,
   report the shortfall and implement the feasible frozen-loading alternative;
   do not silently use future observations or fabricated history.

5. **Audit the calibration itself.** Freezing the wage multiplier does not
   freeze wage information used inside the entropy optimization. Trace the
   FLS wage targets, OEWS wage features, QCEW composition/seasonal features,
   annual county prior, and state raking. Distinguish recovered pseudo-survey
   weights from observed sampling weights and from exogenous sampling errors.
   Explain whether the residual variation can plausibly be called a survey
   shock. If isolating the requested shock requires freezing calibration
   features or priors, propose and document that additional revision, rebuild
   at the owning recovery stage, and compare it separately.

6. **State the exclusion argument and its limits.** Discuss shared regional
   wage targets, donor/target economic shocks, overlapping reporting areas,
   common shocks across instruments, and estimated-weight uncertainty. Use
   primary methodological sources; the shock-level identification discussion
   in [Borusyak, Hull, and Jaravel](https://arxiv.org/abs/1806.01221) is a useful
   starting point, but establish whether its conditions fit this construction.
   Predetermined loadings and entropy balancing alone do not establish validity.

### Implementation requirements

- Implement at least the fixed-preperiod-loading construction and retain the
  current moving-wage construction as an explicitly labeled comparator. Select
  the candidate on identification, admissible information, and feasibility;
  first-stage strength is evidence to report, not permission to optimize the
  preferred specification after seeing second-stage results.
- Consider `sum_c a[g,c,t-1] * b[c]` and its fixed-reference-centered version
  `sum_c (a[g,c,t-1] - a0[g,c]) * b[c]`. Verify numerical equivalence after
  county fixed effects only under the actual fixed mapping, support, and sample.
  A difference from a time-varying prior need not have that equivalence.
  Unweighted changes in a normalized distribution sum to zero; assigning every
  donor the same constant can likewise eliminate identifying variation.
- Define baseline support and missing-data rules without selecting donors on
  policy-period outcomes or future wage availability. Retain target-area overlap
  protection and audit any evolving mapping or eligibility. Do not claim that
  all variation is from weights if changing coverage contributes to it.
- Preserve heterogeneity in fixed loadings, consistent real units, and `t-1`
  policy timing. Once baseline real loadings are fixed, do not re-deflate them
  each year in a way that reintroduces another changing multiplier.
- Add explicit construction/version metadata for loadings, baseline window,
  wage source, shock definition, calibration, support, and any trend fit.
  Invalidate only affected caches. Rebuild stages 05 onward for new instrument
  construction, and stage 04 plus consumers if calibration changes; trace any
  further dependency rather than reusing incompatible artifacts.
- Keep the two calibration families and controlled/uncontrolled comparisons
  interpretable. Preserve the original outcome registry, controls, fixed effects,
  normalization, and inference as the comparison baseline unless a separately
  justified revision is required. Give new variants unambiguous identifiers;
  do not silently relabel old fields or promote a favorable column.
- Compare variants on identical observations for each outcome, with additional
  labeled native-sample estimates when useful. Report coverage losses, first
  stages, partial fit, correctly labeled cluster-robust strength diagnostics,
  reduced forms, 2SLS, uncertainty, and cluster counts. Include appropriate
  weak-IV-robust inference where feasible and identify the dependence assumptions
  behind it; retain weak or unbounded results.
- Test the decomposition identity, fixed-loadings invariance, year alignment,
  normalized weights, support rules, and cache metadata. Perturb postbaseline
  donor wages while holding recovered weights fixed: the frozen-loading
  instrument must not change. Separately test/reveal recalibration's response
  to those wages. Check that future data cannot alter earlier admissible fits.

### Deliverables and completion evidence

Deliver executable branch changes, regenerated compatible artifacts, meaningful
tests, updated branch documentation, and
`outputs/reports/panel_iv_weight_shocks.md`. The report must show the original
equation, component decomposition, candidate comparison, selected construction,
Hamilton feasibility/implementation, calibration caveats, identical-sample
results, and remaining threats. Link every empirical claim to generated evidence.

A valid frozen-loading implementation can still produce a weak instrument.
Report that outcome accurately. Complete the shared documentation/grounding
integration described below before claiming all required checks pass.

## 6. Goal 2: a credible, more informative AEWR-border design

### User objective and success criterion

The old county-pair design around AEWR-region borders lacked power and had large
standard errors. Develop and implement a better border-based design.

“Works” means a defensible identifying comparison with transparent assumptions,
valid inference, and precision that can distinguish economically meaningful
effects. It does not mean obtaining `p < 0.05`, the expected sign, or a smaller
standard error by assuming away dependence. A precise null can be informative;
a significant but invalid comparison does not satisfy the goal.

### Start from the actual existing branch

Read the branch instructions, README, canonical design page, `design.json`, and
stages 01--06, together with the scope snapshot's remaining context.

The current branch already implements a screened disjoint one-ring primary
geography, border-core/two-ring/stacked sensitivities, and a bounded audit. Its
registry is `border_ring_experiment_v5` at dispatch preparation. It uses current
real AEWR, H-2A certifications per fixed 2011 farm employment, county and
border-component-year fixed effects, and border-pair inference. It already
compares CRV1, WCR11-FNW, CV3J, and a labeled pair-by-region two-way sensitivity.
Verify the current executable state. Do not propose widening to one ring or
adding a wild bootstrap as if those features were absent.

### Required diagnosis before redesign

1. Reproduce the current primary model and its audit when local inputs permit.
   Trace retained outputs to the current code, sample, and inference settings.
   Inventory missing inputs before launching the full branch.
2. Decompose the precision problem: residualized AEWR variation, number and
   effective number of independent policy contrasts, cluster leverage, outcome
   noise and concentration, missingness, singletons, sample loss, and overlap.
   Use the existing support, influence, leave-one-pair-out, and exact FWL
   decomposition outputs. Diagnose which problem is actually binding.
3. Identify the assignment and dependence structure. Several border pairs can
   share the same AEWR region and region-year shock. More county rows or repeated
   memberships do not create new independent policy assignments. Evaluate whether
   the current pair-level inference and its existing sensitivity are adequate.
4. Audit geography and counterfactual comparability: connected support across
   actual AEWR boundaries, predetermined agricultural relevance and similarity,
   preperiod outcomes and covariates, policy endogeneity, state policies,
   differential trends, sorting, anticipation, and cross-border spillovers.
   County aggregates and adjacency do not by themselves supply a continuous
   distance-to-border running variable or identify a pointwise spatial RD.
5. Define a substantively meaningful effect scale before comparing new treatment
   coefficients. If the user has not supplied a threshold, give a transparent
   range in the outcome's actual units and mark it as an assumption, rather than
   tuning it to the observed estimate. Estimate minimum detectable effects and
   power using a declared significance level, target power, and credible cluster
   dependence. Assess rejection size as well as power in simulations. Label
   assumed DGPs and limited preperiod information; do not use observed-effect
   “post hoc power” as evidence of a better design.

### Candidate redesigns to investigate

Use primary econometric and institutional sources to compare a bounded menu.
The [Cattaneo, Idrobo, and Titiunik RD monograph](https://arxiv.org/abs/2301.08958)
is a starting reference for checking which RD framework the available geography
and treatment can support. Verify the exact method ultimately implemented.

- **Better local pooling and comparison weights.** Examine whether predetermined
  matching, disjoint assignment, boundary-segment aggregation, or disciplined
  local expansion can reduce noise while preserving a credible local population.
  Compare against all existing ring variants and account for repeated counties.
- **Changes across boundaries over time.** Assess a border-panel or
  difference-in-discontinuities formulation that uses changes in the wage gap
  across the same boundary. State the counterfactual trend restriction and the
  continuous-treatment estimand. Do not introduce region-year fixed effects that
  absorb all AEWR variation and then treat the dropped treatment as identified.
- **Predetermined precision adjustment.** Consider parsimonious preperiod outcome
  histories or covariates, with appropriate held-out or cluster-aware fitting if
  estimating an adjustment. Show the estimand and sample implications. Avoid
  post-treatment controls and selection on policy-period H-2A participation.
- **Predetermined exposure or policy bite.** Assess whether baseline exposure
  interacting with the cross-border wage change answers a useful question.
  This changes the treatment/estimand and adds assumptions; label it explicitly
  and retain the direct-wage benchmark.
- **Independent support or finer geography.** Determine whether existing local
  data support more defensible border segments, years, or finer spatial units.
  If useful improvements require unavailable data or policy variation, specify
  exactly what is missing. Do not manufacture precision by duplicating rows,
  relabeling existing counties as new clusters, or using unrestricted national
  comparisons while calling the result a border discontinuity.

These are candidates, not predetermined winners. Before running new
policy-outcome comparisons, record a bounded candidate registry and selection
rule based on identification, baseline balance/support, and declared power
calculations. Existing policy outcomes have already been examined: describe this
as a redesign informed by previous results, not an untouched preregistration.
Retain all evaluated candidates and failures. Separate exploratory comparisons
from any genuinely held-out confirmation.

### Implementation and inference requirements

- Select the strongest feasible design on the declared criteria and implement
  it through the branch's producer, geography, panel, estimator, diagnostics,
  and validator. Version `design.json` and make its role assignments explicit.
  The request authorizes revising the old primary design, but not selecting
  the new primary because it has the smallest p-value.
- Preserve the old model and its credibility audit as labeled benchmarks. When
  changing bandwidths, screens, years, treatment, weights, outcome scale, fixed
  effects, or clustering, explain why the new choice improves the research
  design and how it changes the target population or identifying assumption.
- Use dependence and finite-sample procedures appropriate to the revised
  assignment structure. Diagnose shared-region and spatial dependence rather
  than counting border pairs as automatically independent. Do not downgrade
  clustering to reduce standard errors or claim that a bootstrap creates
  independent information. Randomization inference requires a justified
  assignment/permutation mechanism. Preserve the existing inference methods as
  comparisons where applicable and explain any method that no longer applies.
- Report treatment support and cluster concentration, MDE/power under stated
  assumptions, effect estimates and confidence sets, preperiod/placebo results,
  balance, leave-one-policy-unit-out influence, and relevant geography/weight
  sensitivities. A nonsignificant pretrend test with low power is not proof of
  parallel trends. Retain adverse diagnostics in the final report.
- Explain whether any precision improvement comes from residual noise reduction,
  more treatment variation, different populations/estimands, or a different
  inferential assumption. Decompose these changes with comparable-sample and
  comparable-estimand checks where possible.
- Keep county-year panels unique by `county_fips, year`. If a stacked design
  requires memberships, store them separately with an explicit expanded key,
  account for duplicate exposure/weighting, and validate both levels. Preserve
  geographic identifiers as strings and use the shared normalizers.
- Verify the new estimator with meaningful numerical checks, including FWL or
  equivalent aggregation where applicable, geography/support invariants, and
  inference behavior under an appropriate bounded simulation. Preserve declared
  seeds and resource limits; keep new analysis entry points as Marimo sources
  where Python is used.

### Deliverables and completion evidence

Deliver the implemented design and benchmark, reproducible registry and code,
generated results and diagnostics, updated local instructions/documentation,
and `outputs/reports/border_redesign.md`. The report must answer:

- Why was the original design imprecise?
- What changed, and what causal parameter does the new design identify?
- What assumptions and independent policy variation support it?
- How much did credible precision improve, in comparable units?
- What do the placebo, support, influence, and dependence checks imply?
- Does the result meet the declared economically meaningful precision standard?

If no feasible candidate has both credible identification and useful precision,
finish the bounded audit and implemented comparisons, retain that finding, and
state what additional data or policy variation would be needed. Label the
empirical objective unresolved; do not say that the redesign “works” or mark a
required empirical outcome achieved merely because the search budget ended.
Missing inputs and infeasibility must be distinguished from unfinished coding.

## 7. Validation, grounding, and final integration

### Branch validation

Each lead must follow all applicable `AGENTS.md` checks and record the exact
commands and outcomes. Begin with:

```bash
git status --short
UV_CACHE_DIR=/tmp/h2a-uv-cache uv run --no-sync python scripts/agent_docs.py verify
```

Session A's scope and runner-order checks:

```bash
UV_CACHE_DIR=/tmp/h2a-uv-cache uv run --no-sync python scripts/agent_docs.py snapshot --scope code/designs/panel_iv
DRY_RUN=1 ./scripts/run_panel_iv.sh
```

Session B's corresponding checks:

```bash
UV_CACHE_DIR=/tmp/h2a-uv-cache uv run --no-sync python scripts/agent_docs.py snapshot --scope code/designs/border_discontinuity
DRY_RUN=1 ./scripts/run_border_discontinuity.sh
```

For analysis changes, run the narrowest affected entry points first and the
affected branch runner when its local inputs are available, under the common
heavy-job lock. The border branch offers
`H2A_BORDER_INPUTS_ONLY=1 ./scripts/run_border_discontinuity.sh` for constructing
and validating inputs; that mode does not establish estimation success.

Run R entry points with `Rscript --vanilla`. Run ordinary Python-module Ruff
checks only on changed files through `uv run --no-sync`. Export each changed
Marimo app to a temporary file under `/tmp` before executing it; the existing
runner helper already implements that workflow. Run relevant design tests and
artifact validators. Separate new failures from the recorded baseline.

### Shared documentation and grounding

Each branch owner updates its own design page and README. Each records the
exact shared-contract and assumption changes it needs in its own status file,
including source anchors and revised empirical meaning.

Session A is the sole writer for `agent/assumptions.toml`, shared canonical
pages, and the generated `static/grounding-manifest.json` and `static/llms.txt`.
The latter two must be produced by `agent_docs.py generate`, never hand-edited.

Grounding can temporarily fail while either branch edits an anchored region.
Record that failure and stop relying on generated context until verification
passes; continue using executable sources and the reviewed canonical documents.
Do not accept digests automatically, bypass checks, or treat expected shared
drift as a reason to abandon otherwise independent implementation.

When both branches are ready, B records `ready-for-integration`, releases its
branch files and artifacts, and completes/stops outstanding subagent writes
and background commands. A does the same before integration. B remains available
for requested corrections but makes no uncoordinated changes during this phase.

Session A then:

1. Reviews both reports, scoped diffs, changed assumptions, and validation logs
   against the working baseline. Checks that each output still has one producer
   and shared inputs were preserved or explicitly coordinated.
2. Integrates cross-design contracts and assumption updates. Arranges any
   additional edits to B-owned files through B or an acknowledged handoff.
3. Reviews each changed grounded source region and its linked page. Applies
   `agent_docs.py accept-drift --document <page> --anchor <anchor>` and its
   `--write` form only after substantive prose/context review. Coordinate with
   B before accepting a digest in B's page. Do not edit only a digest to satisfy
   the machine check.
4. Generates context and runs the final checks below, sequentially with respect
   to other writers. Match additional syntax/analysis checks to the actual edits;
   a dry run is not a substitute for required analysis validation.

```bash
UV_CACHE_DIR=/tmp/h2a-uv-cache uv run --no-sync python scripts/agent_docs.py generate
UV_CACHE_DIR=/tmp/h2a-uv-cache uv run --no-sync python scripts/agent_docs.py verify
UV_CACHE_DIR=/tmp/h2a-uv-cache uv run --no-sync python scripts/test_agent_docs.py
zola check --skip-external-links
zola build
```

If shell runners changed:

```bash
bash -n scripts/*.sh
```

If R changed:

```bash
Rscript --vanilla -e 'for (f in list.files("code", pattern = "\\.R$", recursive = TRUE, full.names = TRUE)) parse(file = f)'
```

If orchestration changed:

```bash
DRY_RUN=1 ./scripts/run_all.sh
```

Run no real `scripts/run_all.sh`, source refresh, credentialed acquisition, or
exhaustive Mundlak–Chamberlain queue as part of this dispatch. Do not add border
results to the manuscript or feed a new elasticity into welfare automatically.

A records the final integration and checks in `A.md`; B reads that record and
records its final status in `B.md`. Both final responses identify implemented
changes, exact validation, empirical findings, remaining limitations, and the
two retained reports. Distinguish branch implementation, integrated validation,
and empirical adequacy. Do not mark required work complete while it remains
pending. Leave the combined changes uncommitted.

## 8. Monitor, pause, and resume

Use these commands in the relevant session:

| Command | Use |
| --- | --- |
| `/goal` | Inspect that session's objective and state |
| `/goal pause` | Pause its persistent goal |
| `/goal resume` | Resume its goal |
| `/agent` | Inspect its subagents |
| `/status` | Inspect its configuration and usage |

Pausing a goal is not evidence that every launched process or subagent has
stopped. Before handing off file ownership, ask the lead to account for and
quiesce its writers and active analysis commands.

If a terminal closes, run `codex resume` from the repository root and choose
the named session. Read the existing coordination files, inspect `/goal`, and
resume if needed; do not initialize a second copy of the same goal against
the same owned files. See the [official CLI command reference](https://learn.chatgpt.com/docs/developer-commands).

To request status without changing a goal, send:

```text
Report the current design decision, evidence gathered, active subagents and
commands, remaining work, and any request waiting on the other session.
Update your coordination file and continue pursuing the existing goal.
```
