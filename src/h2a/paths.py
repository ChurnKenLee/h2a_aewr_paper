# src/h2a/paths.py

import os
from pathlib import Path

import pyprojroot
from dotenv import load_dotenv


def find_project_root() -> Path:
    # pyprojroot finds the repo root from the calling context / project markers.
    root = Path(pyprojroot.find_root(criterion="pyproject.toml")).resolve()

    # Load .env from the project root, independent of Marimo's temp cwd.
    load_dotenv(root / ".env")

    # Optional override, useful only for unusual machine setups.
    env_root = os.getenv("H2A_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    return root


ROOT = find_project_root()

CODE = ROOT / "code"
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERMEDIATE = DATA / "intermediate"
PROCESSED = DATA / "processed"

OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
LOGS = OUTPUTS / "logs"
