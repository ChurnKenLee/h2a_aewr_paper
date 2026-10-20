+++
title = "H-2A prediction contract"
description = "Canonical prediction cutoff, model specification, and panel semantics."

[extra]
scopes = ["code/paths.R", "code/b01_derived", "code/c02_build"]
+++

Every shared and design-specific panel uses one source-controlled prediction
cutoff and model specification. Changing either setting requires rebuilding the
prediction artifact and every downstream panel.

{{ <grounding path="code/paths.R" anchor="prediction-selection" sha256="2a1d5d5019ebb0bd1f18d6a1342d81b6b5050741cbf10f0ce0061c17ed963192" /> }}

The prediction is a static county propensity. Each compatible model is scored
once per county; it is not a rolling annual prediction. The shared panel
requires one valid prediction per county, verifies the configured cutoff and
model specification, and defines the predicted share using fixed 2011 farm
employment.

The same shared-panel contract block independently validates and fills the
reviewed FY2008--2024 worksite-business v2 provenance. That linkage metadata
does not enter prediction fitting, scoring, or cutoff selection.

{{ <grounding path="code/c02_build/01_build_county_panel.R" anchor="shared-panel-contract" sha256="b8fd6cf733f2ea700c01ab53c1b5d582073d588ee0707dff590b17387ffd2677" /> }}

The selected values are metadata carried with the artifact. A stale or mixed
model specification is rejected rather than silently repaired downstream.
