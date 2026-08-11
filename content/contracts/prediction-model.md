+++
title = "H-2A prediction contract"
description = "Canonical prediction cutoff, model specification, and panel semantics."

[extra]
scopes = ["code/paths.R", "code/b01_derived", "code/c02_build"]
+++

Every shared and design-specific panel uses one source-controlled prediction
cutoff and model specification. Changing either setting requires rebuilding the
prediction artifact and every downstream panel.

{{ grounding(path="code/paths.R", anchor="prediction-selection", sha256="2a1d5d5019ebb0bd1f18d6a1342d81b6b5050741cbf10f0ce0061c17ed963192") }}

The prediction is a static county propensity. Each compatible model is scored
once per county; it is not a rolling annual prediction. The shared panel
requires one valid prediction per county, verifies the configured cutoff and
model specification, and defines the predicted share using fixed 2011 farm
employment.

{{ grounding(path="code/c02_build/01_build_county_panel.R", anchor="shared-panel-contract", sha256="ffb30f884e3403e1fd5c2b88d13d20c21eace1d1e1df0957fe7ac8c24b55aada") }}

The selected values are metadata carried with the artifact. A stale or mixed
model specification is rejected rather than silently repaired downstream.
