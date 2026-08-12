"""Compile the bounded or explicitly exhaustive version-4 registry."""

from __future__ import annotations

import os
import sys
from pathlib import Path

BRANCH_DIR = Path(__file__).resolve().parent
if str(BRANCH_DIR) not in sys.path:
    sys.path.insert(0, str(BRANCH_DIR))

from mcw.design import DEFAULT_STAGE, specification_registry
from mcw.io import REGISTRY_PATH, atomic_write_frame


def main() -> None:
    stage = os.getenv("MC_SPEC_STAGE", DEFAULT_STAGE)
    registry = specification_registry(stage)
    selected = os.getenv("MC_SPEC_IDS")
    if selected:
        identifiers = [value.strip() for value in selected.split(",") if value.strip()]
        unknown = sorted(
            set(identifiers).difference(registry["specification_id"].to_list())
        )
        if unknown:
            raise ValueError(f"MC_SPEC_IDS contains unknown identifiers: {unknown}")
        registry = registry.filter(registry["specification_id"].is_in(identifiers))
    limit = os.getenv("MC_SPEC_MAX")
    if limit:
        registry = registry.head(int(limit))
    if registry.is_empty():
        raise ValueError("The selected MCW specification registry is empty.")
    atomic_write_frame(registry, REGISTRY_PATH)
    print(f"Wrote {registry.height:,} MCW v4 {stage} specification records.")


if __name__ == "__main__":
    main()
