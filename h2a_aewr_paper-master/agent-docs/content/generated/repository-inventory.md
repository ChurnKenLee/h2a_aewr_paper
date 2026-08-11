+++
title = "Repository inventory"
description = "Watched files, language/file types, top-level ownership, and AGENTS layering."
weight = 1
+++

> [!NOTE]
> Generated file. Change repository sources or `scripts/agent_grounding.py`, then regenerate.

Grounding input digest: `e31af360d1b7f4e86eab859c378b4d3fb04fe6f19c042477ed3d46de045608ea`.

Watched repository files: **291**. Generated projection files: **9**.

## File types

| Type | Count |
|---|---:|
| `.md` | 64 |
| `.r` | 60 |
| `.py` | 43 |
| `.pdf` | 32 |
| `.tex` | 28 |
| `.sh` | 12 |
| `.html` | 10 |
| `.nix` | 8 |
| `.toml` | 5 |
| `.txt` | 5 |
| `.lock` | 3 |
| `.docx` | 2 |
| `.gitignore` | 2 |
| `.yaml` | 2 |
| `.yml` | 2 |
| `.Rprofile` | 1 |
| `.bib` | 1 |
| `.bibtex` | 1 |
| `.codex` | 1 |
| `.envrc` | 1 |
| `.example` | 1 |
| `.here` | 1 |
| `.json` | 1 |
| `.log` | 1 |
| `.python-version` | 1 |
| `.rclone-ignore` | 1 |
| `.scss` | 1 |
| `[none]` | 1 |

## Top-level ownership

| Path | Watched files |
|---|---:|
| `code` | 115 |
| `outputs` | 32 |
| `agent-docs` | 27 |
| `markdowns` | 25 |
| `papers` | 24 |
| `scripts` | 16 |
| `src` | 10 |
| `nix` | 7 |
| `draft` | 6 |
| `documentation` | 5 |
| `renv` | 3 |
| `.github` | 2 |
| `.Rprofile` | 1 |
| `.codex` | 1 |
| `.env.example` | 1 |
| `.envrc` | 1 |
| `.gitignore` | 1 |
| `.here` | 1 |
| `.python-version` | 1 |
| `.rclone-ignore` | 1 |
| `AGENTS.md` | 1 |
| `README.md` | 1 |
| `Rplots.pdf` | 1 |
| `devenv.lock` | 1 |
| `devenv.nix` | 1 |
| `devenv.yaml` | 1 |
| `main.py` | 1 |
| `pyproject.toml` | 1 |
| `renv.lock` | 1 |
| `snowflake.log` | 1 |
| `uv.lock` | 1 |

## Instruction layers

| File | Bytes |
|---|---:|
| `AGENTS.md` | 8783 |
| `agent-docs/AGENTS.md` | 1761 |
| `code/AGENTS.md` | 1154 |
| `code/a01_sources/AGENTS.md` | 1043 |
| `code/b01_derived/AGENTS.md` | 1041 |
| `code/c00_shared/AGENTS.md` | 562 |
| `code/c01_clean/AGENTS.md` | 668 |
| `code/c02_build/AGENTS.md` | 781 |
| `code/designs/AGENTS.md` | 942 |
| `code/designs/did/AGENTS.md` | 884 |
| `code/designs/mundlak_chamberlain/AGENTS.md` | 1318 |
| `code/designs/panel_iv/AGENTS.md` | 1069 |
| `documentation/AGENTS.md` | 706 |
| `draft/AGENTS.md` | 757 |
| `outputs/AGENTS.md` | 663 |
| `scripts/AGENTS.md` | 1179 |

The largest discovered instruction chain is **12197 bytes** at `code/designs/mundlak_chamberlain`: `AGENTS.md` → `code/AGENTS.md` → `code/designs/AGENTS.md` → `code/designs/mundlak_chamberlain/AGENTS.md`. The Codex default project-document budget is 32768 bytes.
