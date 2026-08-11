+++
title = "Runtime locks"
description = "Python, R, Nix/devenv, and key analytical package versions."
weight = 3
+++

> [!NOTE]
> Generated file. Change repository sources or `scripts/agent_grounding.py`, then regenerate.

## Language contracts

- `.python-version`: `3.14`
- `pyproject.toml` requirement: `>=3.14`
- uv lock package records: **395**
- R version: `4.6.0`
- renv package records: **143**

## Key Python packages

| Package | Locked version(s) |
|---|---|
| `jax` | 0.10.1 |
| `numpy` | 2.4.6 |
| `pandas` | 3.0.5 |
| `polars` | 1.43.2 |
| `pyarrow` | 25.0.0 |
| `pyfixest` | 0.60.0 |
| `scipy` | 1.18.0 |
| `torch` | 2.13.0, 2.13.0+cu132 |

## Key R packages

| Package | Locked version |
|---|---|
| `arrow` | 25.0.0 |
| `fixest` | 0.14.2 |
| `here` | 1.0.2 |
| `renv` | 1.2.3 |
| `tidyverse` | 2.0.0 |

## Nix inputs

| Input | Revision |
|---|---|
| `nixpkgs` | `b7c2ada94fe99c15b0dbcf4d11fd7850b957a436` |
| `devenv` | `34a93eec7e71e64e9da1819a684d12f92331d638` |

The lockfiles define intended resolution; installed tools and external drivers still need runtime checks. Do not regenerate a lock merely because a newer release exists.
