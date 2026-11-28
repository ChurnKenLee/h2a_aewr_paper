# Manuscript instructions

- The manuscript presents selected results; it is not executable design truth.
  Confirm every number, table, figure, and claim against owning code and
  retained outputs.
- Do not hand-edit included generated tables or figures. Regenerate them from
  the owning branch (`code/designs/policy_2025`, `code/welfare`,
  `code/designs/bite`, `code/designs/panel_iv`, or `code/descriptives`).
- The restructured manuscript (`draft/draft1.tex`, built with `tectonic` using
  `natbib` and `draft/h2a_paper.bib`) centers on the 2025 statewide AEWR reform
  (`code/designs/policy_2025`: offered wages, Track 1 complete FY2026 Q4 county
  quantities, five-panel mechanism heterogeneity, and exploratory interim QCEW
  earnings) and certification-space envelope-theorem producer incidence bounds
  (`code/welfare`). Pre-2025 county-panel designs (`code/designs/bite`,
  `code/designs/border_discontinuity`, `code/designs/panel_iv`, and
  `code/designs/mundlak_chamberlain`) appear strictly as power-ceiling context
  and appendix evidence; legacy `code/designs/did` tables are retired from the
  manuscript.
- Causal caveats, post-inspection disclosures (`POLICY2025-001`), Holm
  multiplicity adjustments, support limits, unresolved job-level legal floors,
  realized-hours scaling ($\rho$), and placebo evidence are substantive and
  cannot be removed merely for narrative smoothness.
- A changed description of sample, treatment, inference, or result requires a
  corresponding executable and canonical-design review.
- Authorship provenance: the author intends to write the entire manuscript by
  hand. All LLM-drafted placeholder prose, headings, title, abstract, equation
  exposition, and figure captions in `draft/draft1.tex` are explicitly
  delimited by `llmdraft` / `\llmcaption` / `\llminline` (and source comments)
  and BibTeX provenance headers in `draft/h2a_paper.bib`. Never remove those
  markers from LLM-generated text or insert unmarked LLM prose into `draft/`.
