#!/usr/bin/env python3
"""Generate and verify coding-agent grounding for the H-2A AEWR repository."""

import argparse
import ast
import collections
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
import urllib.parse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "agent-docs"
GENERATED = DOCS / "content" / "generated"
STATIC = DOCS / "static"
DEFAULT_AGENT_LIMIT = 32 * 1024
SNIPPET_CATALOG = DOCS / "snippets.toml"
SNIPPET_PAGE = GENERATED / "code-grounding.md"
SNIPPET_INDEX = STATIC / "grounding-snippets.json"
FENCE_CLASSIFICATIONS = ("illustrative", "pseudocode", "expected-output")

GENERATED_PATHS = (
    GENERATED / "repository-inventory.md",
    GENERATED / "pipeline-contracts.md",
    GENERATED / "runtime-locks.md",
    GENERATED / "assumptions.md",
    SNIPPET_PAGE,
    GENERATED / "drift-report.md",
    STATIC / "grounding-manifest.json",
    SNIPPET_INDEX,
    STATIC / "llms.txt",
)

IGNORED_DIR_NAMES = {
    ".devenv",
    ".direnv",
    ".git",
    ".venv",
    "__marimo__",
    "__pycache__",
    "data",
    "library",
    "public",
}

KEY_PYTHON_PACKAGES = (
    "jax",
    "numpy",
    "pandas",
    "polars",
    "pyarrow",
    "pyfixest",
    "scipy",
    "torch",
)

KEY_R_PACKAGES = (
    "arrow",
    "fixest",
    "here",
    "renv",
    "tidyverse",
)


def relative(path):
    return path.relative_to(ROOT).as_posix()


def display_path(path):
    try:
        return relative(path)
    except ValueError:
        return str(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def should_ignore(path):
    rel = path.relative_to(ROOT)
    parts = rel.parts
    if any(part in IGNORED_DIR_NAMES for part in parts):
        return True
    if parts[:3] == ("agent-docs", "content", "generated"):
        return True
    if parts[:2] == ("agent-docs", "static"):
        return True
    if parts[:2] in (("renv", "sandbox"), ("renv", "staging")):
        return True
    return False


def watched_files():
    files = []
    for path in ROOT.rglob("*"):
        if path.is_file() and not path.is_symlink() and not should_ignore(path):
            files.append(path)
    return sorted(files, key=relative)


def watched_manifest():
    entries = [
        {
            "path": relative(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in watched_files()
    ]
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(entry["path"].encode())
        digest.update(b"\0")
        digest.update(entry["sha256"].encode())
        digest.update(b"\n")
    return entries, digest.hexdigest()


def read_toml(path):
    with path.open("rb") as stream:
        return tomllib.load(stream)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def git_output(*args):
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def repository_revision():
    return {
        "commit": git_output("rev-parse", "HEAD"),
        "branch": git_output("branch", "--show-current"),
        "dirty": bool(git_output("status", "--porcelain")),
    }


def extension_label(path):
    if path.name.startswith(".") and path.suffix == "":
        return path.name
    return path.suffix.lower() or "[none]"


def inventory_facts():
    files = watched_files()
    extensions = collections.Counter(extension_label(path) for path in files)
    top_levels = collections.Counter(relative(path).split("/", 1)[0] for path in files)
    agent_files = sorted(ROOT.rglob("AGENTS.md"), key=relative)
    generated_existing = [path for path in GENERATED_PATHS if path.exists()]
    return {
        "files": files,
        "extensions": extensions,
        "top_levels": top_levels,
        "agents": agent_files,
        "generated_existing": generated_existing,
    }


def expand_runner_globs(text):
    expanded = []
    for match in re.finditer(r"for\s+script\s+in\s+([^;]+);\s*do", text):
        expression = match.group(1).strip()
        if "$" not in expression and "*" in expression:
            expanded.extend(relative(path) for path in sorted(ROOT.glob(expression)))
    return expanded


def runner_facts():
    runners = {}
    for path in sorted((ROOT / "scripts").glob("run_*.sh")):
        if path.name == "run_tests.sh":
            continue
        text = path.read_text(encoding="utf-8")
        variables = {
            match.group(1): match.group(2)
            for match in re.finditer(
                r"^([A-Za-z_][A-Za-z0-9_]*)=([^\s#]+)", text, re.MULTILINE
            )
        }
        steps = []
        for match in re.finditer(r"\brun_step\s+([^\s\\]+)", text):
            token = match.group(1).strip("\"'")
            if token in ("$script", "${script}"):
                continue
            if token.startswith("$"):
                token = variables.get(token.lstrip("${").rstrip("}"), token)
            if token.startswith("code/"):
                steps.append(token)
        steps.extend(expand_runner_globs(text))
        calls = []
        for match in re.finditer(r"(?:\$SCRIPT_DIR|\$\{SCRIPT_DIR\})/(run_[\w]+\.sh)", text):
            calls.append(f"scripts/{match.group(1)}")
        runners[relative(path)] = {
            "steps": list(dict.fromkeys(steps)),
            "calls": list(dict.fromkeys(calls)),
        }
    return runners


def active_agent_chain(scope):
    scope_path = scope if scope.is_dir() else scope.parent
    try:
        scope_rel = scope_path.resolve().relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"scope escapes repository: {scope}") from error
    chain = []
    cursor = ROOT
    root_agent = ROOT / "AGENTS.md"
    if root_agent.exists():
        chain.append(root_agent)
    for part in scope_rel.parts:
        cursor /= part
        candidate = cursor / "AGENTS.md"
        if candidate.exists() and candidate not in chain:
            chain.append(candidate)
    return chain


def assumption_records():
    data = read_toml(DOCS / "assumptions.toml")
    return data.get("assumptions", [])


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def snippet_catalog():
    data = read_toml(SNIPPET_CATALOG)
    if data.get("schema_version") != 1:
        raise ValueError("agent-docs/snippets.toml must have schema_version = 1")
    return data.get("groups", []), data.get("snippets", [])


def source_lines(path):
    return path.read_text(encoding="utf-8").splitlines(keepends=True)


def normalized_excerpt(lines):
    return "".join(lines).rstrip() + "\n"


def one_exact_line(lines, target, label):
    matches = [index for index, line in enumerate(lines) if line.rstrip("\r\n") == target]
    if len(matches) != 1:
        raise ValueError(f"{label} must match exactly one line; found {len(matches)}")
    return matches[0]


def braced_symbol_excerpt(lines, pattern, label):
    matches = [
        index
        for index, line in enumerate(lines)
        if re.match(pattern, line.rstrip("\r\n"))
    ]
    if len(matches) != 1:
        raise ValueError(f"{label} must match exactly one symbol; found {len(matches)}")
    start = matches[0]
    depth = 0
    opened = False
    quote = None
    escaped = False
    for line_index in range(start, len(lines)):
        comment = False
        for character in lines[line_index]:
            if comment:
                continue
            if escaped:
                escaped = False
                continue
            if quote is not None:
                if character == "\\":
                    escaped = True
                elif character == quote:
                    quote = None
                continue
            if character in ("'", '"'):
                quote = character
            elif character == "#":
                comment = True
            elif character == "{":
                depth += 1
                opened = True
            elif character == "}":
                depth -= 1
                if depth < 0:
                    raise ValueError(f"{label} has an unmatched closing brace")
                if opened and depth == 0:
                    return normalized_excerpt(lines[start : line_index + 1])
    raise ValueError(f"{label} has no balanced function body")


def extract_snippet(record):
    snippet_id = record.get("id", "<missing-id>")
    rel_path = record.get("source", "")
    if not rel_path:
        raise ValueError(f"{snippet_id}: missing source")
    path = (ROOT / rel_path).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"{snippet_id}: source escapes repository") from error
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{snippet_id}: source is not a regular file: {rel_path}")
    if should_ignore(path):
        raise ValueError(f"{snippet_id}: source is ignored/generated: {rel_path}")

    selector = record.get("selector", {})
    kind = selector.get("kind")
    lines = source_lines(path)
    text = "".join(lines)
    if kind == "whole_file":
        excerpt = normalized_excerpt(lines)
    elif kind == "between":
        start = one_exact_line(lines, selector.get("start", ""), f"{snippet_id} start")
        end = one_exact_line(lines, selector.get("end", ""), f"{snippet_id} end")
        if end <= start:
            raise ValueError(f"{snippet_id}: end selector does not follow start selector")
        first = start if selector.get("include_start", True) else start + 1
        last = end + 1 if selector.get("include_end", False) else end
        excerpt = normalized_excerpt(lines[first:last])
    elif kind == "from_line":
        start = one_exact_line(lines, selector.get("start", ""), f"{snippet_id} start")
        first = start if selector.get("include_start", True) else start + 1
        excerpt = normalized_excerpt(lines[first:])
    elif kind == "python_symbol":
        symbol = selector.get("symbol", "")
        tree = ast.parse(text, filename=rel_path)
        matches = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == symbol
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{snippet_id}: Python symbol {symbol!r} must be unique; found {len(matches)}"
            )
        node = matches[0]
        start = min(
            [node.lineno]
            + [decorator.lineno for decorator in getattr(node, "decorator_list", [])]
        )
        excerpt = normalized_excerpt(lines[start - 1 : node.end_lineno])
    elif kind == "r_function":
        symbol = re.escape(selector.get("symbol", ""))
        excerpt = braced_symbol_excerpt(
            lines,
            rf"^\s*{symbol}\s*<-\s*function\b",
            f"{snippet_id} R function",
        )
    elif kind == "shell_function":
        symbol = re.escape(selector.get("symbol", ""))
        excerpt = braced_symbol_excerpt(
            lines,
            rf"^\s*(?:function\s+)?{symbol}\s*(?:\(\s*\))?\s*\{{",
            f"{snippet_id} shell function",
        )
    else:
        raise ValueError(f"{snippet_id}: unsupported selector kind {kind!r}")

    if not excerpt.strip():
        raise ValueError(f"{snippet_id}: selector produced an empty excerpt")
    if "```" in excerpt:
        raise ValueError(f"{snippet_id}: excerpt contains a Markdown fence")
    return {
        **record,
        "path": path,
        "excerpt": excerpt,
        "excerpt_sha256": text_sha256(excerpt),
        "source_sha256": sha256(path),
    }


def selector_description(selector):
    kind = selector["kind"]
    if kind == "whole_file":
        return "whole file"
    if kind == "from_line":
        return f"from exact line {selector['start']!r} to EOF"
    if kind == "between":
        return f"exact lines {selector['start']!r} … {selector['end']!r}"
    return f"{kind.replace('_', ' ')} {selector['symbol']!r}"


def validate_snippet_syntax(snippet):
    validation = snippet.get("validation", "text")
    excerpt = snippet["excerpt"]
    if validation == "text":
        return None
    if validation == "python-parse":
        ast.parse(excerpt, filename=f"snippet:{snippet['id']}")
        return None
    if validation == "toml-parse":
        tomllib.loads(excerpt)
        return None
    if validation == "json-parse":
        json.loads(excerpt)
        return None
    commands = {
        "bash-parse": ["bash", "-n"],
        "r-parse": ["Rscript", "--vanilla", "-e", "parse(file = stdin())"],
        "nix-parse": ["nix-instantiate", "--parse", "-"],
    }
    if validation not in commands:
        raise ValueError(f"unknown snippet validation {validation!r}")
    tool = commands[validation][0]
    if not shutil.which(tool):
        return f"{tool} unavailable; skipped {validation} for {snippet['id']}"
    result = subprocess.run(
        commands[validation],
        input=excerpt,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise ValueError(f"{validation} failed for {snippet['id']}: {detail}")
    return None


def snippet_digest_error(snippet):
    expected = snippet.get("expected_sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        return f"{snippet['id']}: expected_sha256 is not a lowercase SHA-256"
    if snippet["excerpt_sha256"] != expected:
        return (
            f"{snippet['id']}: source-linked excerpt drifted; review with "
            f"`python scripts/agent_grounding.py accept-snippet-drift --id {snippet['id']}`"
        )
    return None


def validate_snippets():
    errors = []
    warnings = []
    extracted = []
    try:
        groups, records = snippet_catalog()
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        return [f"snippet catalog invalid: {error}"], warnings, [], []
    group_ids = [group.get("id") for group in groups]
    duplicate_groups = sorted(
        key for key, count in collections.Counter(group_ids).items() if count > 1
    )
    if duplicate_groups:
        errors.append("duplicate snippet group IDs: " + ", ".join(duplicate_groups))
    ids = [record.get("id", "<missing-id>") for record in records]
    duplicate_ids = sorted(key for key, count in collections.Counter(ids).items() if count > 1)
    if duplicate_ids:
        errors.append("duplicate snippet IDs: " + ", ".join(duplicate_ids))
    for record in records:
        snippet_id = record.get("id", "<missing-id>")
        missing = [
            field
            for field in ("id", "group", "title", "purpose", "source", "language", "selector", "validation", "expected_sha256")
            if field not in record
        ]
        if missing:
            errors.append(f"{snippet_id}: missing fields: {', '.join(missing)}")
            continue
        if record["group"] not in group_ids:
            errors.append(f"{snippet_id}: unknown group {record['group']!r}")
        try:
            snippet = extract_snippet(record)
            digest_error = snippet_digest_error(snippet)
            if digest_error:
                errors.append(digest_error)
            warning = validate_snippet_syntax(snippet)
            if warning:
                if os.environ.get("STRICT_TOOLING") == "1":
                    errors.append(warning)
                else:
                    warnings.append(warning)
            extracted.append(snippet)
        except (SyntaxError, ValueError, OSError, tomllib.TOMLDecodeError, json.JSONDecodeError) as error:
            errors.append(f"{snippet_id}: {error}")
    if len(records) < 30:
        errors.append(f"snippet catalog is not extensive enough: {len(records)} registered; require at least 30")
    return errors, warnings, groups, extracted


def grounding_markdown_files():
    paths = [DOCS / "AGENTS.md", DOCS / "README.md"]
    paths.extend(ROOT.rglob("AGENTS.md"))
    paths.extend(markdown_files_for_local_link_check())
    paths.extend(
        path
        for path in DOCS.glob("content/**/*.md")
        if path != SNIPPET_PAGE
    )
    return sorted(set(path for path in paths if path.exists()), key=relative)


def validate_literal_grounding_fences():
    errors = []
    for path in grounding_markdown_files():
        errors.extend(validate_literal_fences_in_path(path))
    return errors


def validate_literal_fences_in_path(path):
    errors = []
    annotation = re.compile(
        r"^<!-- grounding-fence: (" + "|".join(FENCE_CLASSIFICATIONS) + r") -->$"
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    active = None
    for index, line in enumerate(lines):
        match = re.match(r"^\s*(`{3,}|~{3,})(.*)$", line)
        if not match:
            continue
        marker, info = match.groups()
        if active:
            if marker[0] == active[0] and len(marker) >= active[1]:
                active = None
            continue
        language = info.strip().split(None, 1)[0] if info.strip() else ""
        if not language:
            errors.append(f"{display_path(path)}:{index + 1}: grounding fence has no language")
        previous = index - 1
        while previous >= 0 and not lines[previous].strip():
            previous -= 1
        if previous < 0 or not annotation.fullmatch(lines[previous].strip()):
            allowed = ", ".join(FENCE_CLASSIFICATIONS)
            errors.append(
                f"{display_path(path)}:{index + 1}: literal grounding fence is unclassified; "
                f"precede it with <!-- grounding-fence: CLASS --> where CLASS is {allowed}"
            )
        active = (marker[0], len(marker))
    if active:
        errors.append(f"{display_path(path)}: unclosed Markdown fence")
    return errors


def validate_assumptions(records):
    errors = []
    ids = []
    for record in records:
        assumption_id = record.get("id", "<missing-id>")
        ids.append(assumption_id)
        checks = record.get("checks", [])
        if not checks:
            errors.append(f"{assumption_id}: no executable checks")
        for index, check in enumerate(checks, start=1):
            rel_path = check.get("path")
            if not rel_path:
                errors.append(f"{assumption_id} check {index}: missing path")
                continue
            path = ROOT / rel_path
            if check.get("exists") is True:
                if not path.exists():
                    errors.append(f"{assumption_id}: required path missing: {rel_path}")
                continue
            if not path.exists():
                errors.append(f"{assumption_id}: check source missing: {rel_path}")
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if check.get("allow_empty") and not text:
                continue
            if "contains" in check and check["contains"] not in text:
                errors.append(
                    f"{assumption_id}: {rel_path} no longer contains {check['contains']!r}"
                )
            if "not_contains" in check and check["not_contains"] in text:
                errors.append(
                    f"{assumption_id}: {rel_path} unexpectedly contains {check['not_contains']!r}"
                )
            if "regex" in check and not re.search(check["regex"], text, re.DOTALL):
                errors.append(
                    f"{assumption_id}: {rel_path} no longer matches /{check['regex']}/"
                )
    duplicates = sorted(key for key, count in collections.Counter(ids).items() if count > 1)
    if duplicates:
        errors.append("duplicate assumption IDs: " + ", ".join(duplicates))
    return errors


def validate_agents():
    errors = []
    agents = sorted(ROOT.rglob("AGENTS.md"), key=relative)
    if not agents or agents[0] != ROOT / "AGENTS.md":
        errors.append("root AGENTS.md is missing")
    overrides = sorted(ROOT.rglob("AGENTS.override.md"), key=relative)
    if overrides:
        errors.append(
            "checked-in AGENTS.override.md files obscure normal instructions: "
            + ", ".join(relative(path) for path in overrides)
        )
    worst = (0, None, [])
    directories = {path.parent for path in watched_files()}
    directories.update(path.parent for path in agents)
    for directory in directories:
        chain = active_agent_chain(directory)
        size = sum(path.stat().st_size for path in chain)
        if size > worst[0]:
            worst = (size, directory, chain)
        if size > DEFAULT_AGENT_LIMIT:
            errors.append(
                f"AGENTS chain exceeds {DEFAULT_AGENT_LIMIT} bytes at {relative(directory)}: "
                + " -> ".join(relative(path) for path in chain)
            )
    return errors, worst


def validate_runners(runners):
    errors = []
    for runner, contract in runners.items():
        for target in contract["steps"] + contract["calls"]:
            if not (ROOT / target).exists():
                errors.append(f"{runner}: missing target {target}")
    docs = [ROOT / "README.md", ROOT / "scripts" / "README.md"]
    for path in docs:
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"(?:\./)?scripts/(run_[A-Za-z0-9_]+\.sh)", text):
            target = ROOT / "scripts" / match.group(1)
            if not target.exists():
                errors.append(f"{relative(path)} references missing {relative(target)}")
    return errors


def markdown_files_for_local_link_check():
    paths = [ROOT / "README.md", ROOT / "scripts" / "README.md"]
    paths.extend(ROOT.glob("code/**/README.md"))
    paths.extend(ROOT.rglob("AGENTS.md"))
    paths.append(DOCS / "README.md")
    return sorted(set(paths), key=relative)


def validate_local_markdown_links():
    errors = []
    link_pattern = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    for path in markdown_files_for_local_link_check():
        text = path.read_text(encoding="utf-8")
        for match in link_pattern.finditer(text):
            target = match.group(1).strip().split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:", "@/")):
                continue
            target = urllib.parse.unquote(target.split(" ", 1)[0].strip("<>"))
            candidate = (path.parent / target).resolve()
            try:
                candidate.relative_to(ROOT.resolve())
            except ValueError:
                errors.append(f"{relative(path)} link escapes repository: {target}")
                continue
            if not candidate.exists():
                errors.append(f"{relative(path)} has broken local link: {target}")
    return errors


def validate_source_catalog():
    errors = []
    path = ROOT / "documentation" / "raw_data_sources.yaml"
    text = path.read_text(encoding="utf-8")
    for required in ("schema_version:", "sources:", "crosswalks:"):
        if required not in text:
            errors.append(f"raw_data_sources.yaml missing {required}")
    ids = re.findall(r"^\s+- id:\s*([^\s#]+)", text, re.MULTILINE)
    if len(ids) < 10:
        errors.append("raw_data_sources.yaml has implausibly few source/crosswalk IDs")
    duplicates = sorted(key for key, count in collections.Counter(ids).items() if count > 1)
    if duplicates:
        errors.append("raw_data_sources.yaml duplicate IDs: " + ", ".join(duplicates))
    if shutil.which("yq"):
        result = subprocess.run(
            ["yq", "eval", ".", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            errors.append("raw_data_sources.yaml is invalid YAML: " + result.stderr.strip())
    return errors, ids


def base_checks():
    runners = runner_facts()
    records = assumption_records()
    snippet_errors, snippet_warnings, snippet_groups, snippets = validate_snippets()
    errors = []
    errors.extend(validate_assumptions(records))
    errors.extend(snippet_errors)
    errors.extend(validate_literal_grounding_fences())
    agent_errors, worst_chain = validate_agents()
    errors.extend(agent_errors)
    errors.extend(validate_runners(runners))
    errors.extend(validate_local_markdown_links())
    catalog_errors, source_ids = validate_source_catalog()
    errors.extend(catalog_errors)

    warnings = list(snippet_warnings)
    if not (ROOT / "data").exists():
        warnings.append("The repository snapshot has no data/ directory; empirical stages were not validated.")
    if (ROOT / "main.py").exists() and "Hello from" in (ROOT / "main.py").read_text(
        encoding="utf-8", errors="replace"
    ):
        warnings.append("main.py is still a scaffold and is not a supported pipeline entry point.")
    if (ROOT / "code" / ".codex").exists() and (ROOT / "code" / ".codex").stat().st_size == 0:
        warnings.append("code/.codex is an empty regular file; it has no instruction/configuration effect.")
    if (ROOT / "snowflake.log").exists():
        warnings.append("snowflake.log is retained repository noise, not a pipeline log contract.")
    proof_link = ROOT / "ccv_symlink.lean"
    if proof_link.is_symlink() and not proof_link.exists():
        warnings.append(
            "ccv_symlink.lean is a broken developer-local symlink; it cannot ground the CCV implementation."
        )
    warnings.append("The two documentation/*.docx files are historical proposals, not active implementation contracts.")

    return {
        "errors": errors,
        "warnings": warnings,
        "runners": runners,
        "assumptions": records,
        "snippet_groups": snippet_groups,
        "snippets": snippets,
        "worst_chain": worst_chain,
        "source_ids": source_ids,
    }


def page_front_matter(title, description, weight):
    return (
        "+++\n"
        f'title = {json.dumps(title)}\n'
        f'description = {json.dumps(description)}\n'
        f"weight = {weight}\n"
        "+++\n\n"
        "> [!NOTE]\n"
        "> Generated file. Change repository sources or `scripts/agent_grounding.py`, then regenerate.\n\n"
    )


def render_inventory(manifest_entries, input_digest, facts, worst_chain):
    lines = [
        page_front_matter(
            "Repository inventory",
            "Watched files, language/file types, top-level ownership, and AGENTS layering.",
            1,
        ),
        f"Grounding input digest: `{input_digest}`.\n\n",
        f"Watched repository files: **{len(manifest_entries)}**. "
        f"Generated projection files: **{len(GENERATED_PATHS)}**.\n\n",
        "## File types\n\n| Type | Count |\n|---|---:|\n",
    ]
    for extension, count in sorted(facts["extensions"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{extension}` | {count} |\n")
    lines.append("\n## Top-level ownership\n\n| Path | Watched files |\n|---|---:|\n")
    for name, count in sorted(facts["top_levels"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{name}` | {count} |\n")
    lines.append("\n## Instruction layers\n\n| File | Bytes |\n|---|---:|\n")
    for path in facts["agents"]:
        lines.append(f"| `{relative(path)}` | {path.stat().st_size} |\n")
    size, directory, chain = worst_chain
    lines.extend(
        [
            "\nThe largest discovered instruction chain is ",
            f"**{size} bytes** at `{relative(directory)}`: ",
            " → ".join(f"`{relative(path)}`" for path in chain),
            f". The Codex default project-document budget is {DEFAULT_AGENT_LIMIT} bytes.\n",
        ]
    )
    return "".join(lines)


def render_pipeline(runners):
    lines = [
        page_front_matter(
            "Pipeline contracts",
            "Runner-to-step projection generated from the supported shell entry points.",
            2,
        ),
        "## Runner summary\n\n| Runner | Direct steps | Child runners |\n|---|---:|---:|\n",
    ]
    for runner, contract in runners.items():
        lines.append(f"| `{runner}` | {len(contract['steps'])} | {len(contract['calls'])} |\n")
    for runner, contract in runners.items():
        lines.append(f"\n## `{runner}`\n\n")
        if contract["calls"]:
            lines.append("Calls, in source order:\n\n")
            lines.extend(f"1. `{target}`\n" for target in contract["calls"])
        if contract["steps"]:
            lines.append("Direct `run_step` targets, in execution order:\n\n")
            lines.extend(f"1. `{target}`\n" for target in contract["steps"])
        if not contract["steps"] and not contract["calls"]:
            lines.append("No statically resolved child runner or `run_step` target.\n")
    return "".join(lines)


def unique_package_versions(lock, name):
    versions = sorted({package["version"] for package in lock.get("package", []) if package.get("name") == name})
    return ", ".join(versions) if versions else "not locked"


def render_runtime_locks():
    pyproject = read_toml(ROOT / "pyproject.toml")
    uv_lock = read_toml(ROOT / "uv.lock")
    renv_lock = read_json(ROOT / "renv.lock")
    devenv_lock = read_json(ROOT / "devenv.lock")
    python_pin = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
    project_requires = pyproject["project"]["requires-python"]
    uv_packages = uv_lock.get("package", [])
    r_packages = renv_lock.get("Packages", {})
    nodes = devenv_lock.get("nodes", {})
    lines = [
        page_front_matter(
            "Runtime locks",
            "Python, R, Nix/devenv, and key analytical package versions.",
            3,
        ),
        "## Language contracts\n\n",
        f"- `.python-version`: `{python_pin}`\n",
        f"- `pyproject.toml` requirement: `{project_requires}`\n",
        f"- uv lock package records: **{len(uv_packages)}**\n",
        f"- R version: `{renv_lock.get('R', {}).get('Version', 'unknown')}`\n",
        f"- renv package records: **{len(r_packages)}**\n",
        "\n## Key Python packages\n\n| Package | Locked version(s) |\n|---|---|\n",
    ]
    for name in KEY_PYTHON_PACKAGES:
        lines.append(f"| `{name}` | {unique_package_versions(uv_lock, name)} |\n")
    lines.append("\n## Key R packages\n\n| Package | Locked version |\n|---|---|\n")
    for name in KEY_R_PACKAGES:
        lines.append(f"| `{name}` | {r_packages.get(name, {}).get('Version', 'not locked')} |\n")
    lines.append("\n## Nix inputs\n\n| Input | Revision |\n|---|---|\n")
    for name in ("nixpkgs", "devenv"):
        revision = nodes.get(name, {}).get("locked", {}).get("rev", "unknown")
        lines.append(f"| `{name}` | `{revision}` |\n")
    lines.extend(
        [
            "\nThe lockfiles define intended resolution; installed tools and external drivers still need runtime checks. ",
            "Do not regenerate a lock merely because a newer release exists.\n",
        ]
    )
    return "".join(lines)


def render_assumptions(records):
    lines = [
        page_front_matter(
            "Assumption registry",
            "Curated implementation assumptions tied to executable source checks.",
            4,
        ),
        "Every record below passed its checks when this page was generated. A passing check shows that code and the declared statement still agree syntactically; it does not prove the research assumption.\n\n",
        "| ID | Status | Risk | Owner | Statement | Source |\n|---|---|---|---|---|---|\n",
    ]
    for record in records:
        statement = record["statement"].replace("|", "\\|")
        lines.append(
            f"| `{record['id']}` | {record['status']} | {record['risk']} | "
            f"{record['owner']} | {statement} | `{record['source']}` |\n"
        )
    lines.append("\n## Review triggers\n\n")
    for record in records:
        lines.append(f"- **{record['id']}** — {record['review_when']}\n")
    return "".join(lines)


def render_code_grounding(groups, snippets):
    lines = [
        page_front_matter(
            "Source-linked code grounding",
            "Authoritative excerpts re-extracted from repository sources and rejected on unreviewed drift.",
            5,
        ),
        "These are not copied examples. Every fence is deterministically extracted from the named source, "
        "syntax-checked according to its registry rule, and compared with a reviewed excerpt SHA-256. "
        "Ordinary generation refuses changed excerpts; acceptance requires the explicit "
        "`accept-snippet-drift` review command. The complete machine-readable projection is "
        "`agent-docs/static/grounding-snippets.json`.\n\n",
        f"Registered source-linked snippets: **{len(snippets)}** across **{len(groups)}** groups.\n\n",
    ]
    by_group = collections.defaultdict(list)
    for snippet in snippets:
        by_group[snippet["group"]].append(snippet)
    for group in groups:
        group_id = group["id"]
        lines.append(f"## {group['title']}\n\n{group['description']}\n\n")
        for snippet in by_group[group_id]:
            selector = selector_description(snippet["selector"])
            lines.extend(
                [
                    f"### {snippet['title']}\n\n",
                    f"{snippet['purpose']}\n\n",
                    f"- Snippet ID: `{snippet['id']}`\n",
                    f"- Source: `{snippet['source']}`\n",
                    f"- Stable selector: `{selector}`\n",
                    f"- Source-file SHA-256: `{snippet['source_sha256']}`\n",
                    f"- Extracted-text SHA-256: `{snippet['excerpt_sha256']}`\n",
                    f"- Validation: `{snippet['validation']}`\n\n",
                    f"<!-- grounding-snippet:{snippet['id']} excerpt-sha256={snippet['excerpt_sha256']} -->\n",
                    f"```{snippet['language']}\n",
                    snippet["excerpt"],
                    "```\n\n",
                ]
            )
    return "".join(lines)


def render_snippet_index(groups, snippets):
    payload = {
        "schema_version": 1,
        "catalog": relative(SNIPPET_CATALOG),
        "groups": groups,
        "snippets": [
            {
                "id": snippet["id"],
                "group": snippet["group"],
                "title": snippet["title"],
                "purpose": snippet["purpose"],
                "source": snippet["source"],
                "selector": snippet["selector"],
                "language": snippet["language"],
                "validation": snippet["validation"],
                "source_sha256": snippet["source_sha256"],
                "excerpt_sha256": snippet["excerpt_sha256"],
            }
            for snippet in snippets
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_drift_report(checks):
    lines = [
        page_front_matter(
            "Drift report",
            "Current static warnings and the checks used to reject stale grounding.",
            6,
        ),
        "## Verification result\n\n",
        f"Assumption records checked: **{len(checks['assumptions'])}**. "
        f"Source/catalog IDs indexed: **{len(checks['source_ids'])}**. "
        f"Runner contracts inspected: **{len(checks['runners'])}**. "
        f"Source-linked code snippets verified: **{len(checks['snippets'])}**.\n\n",
    ]
    if checks["errors"]:
        lines.append("Generation-time errors existed:\n\n")
        lines.extend(f"- {error}\n" for error in checks["errors"])
    else:
        lines.append("All enforced repository and assumption checks passed at generation time.\n")
    lines.append("\n## Explicit warnings and boundaries\n\n")
    lines.extend(f"- {warning}\n" for warning in checks["warnings"])
    lines.extend(
        [
            "\n## What makes verification fail\n\n",
            "- Any watched file hash changes without regeneration.\n",
            "- A runner or README references a missing script.\n",
            "- A curated assumption no longer matches its named source.\n",
            "- A registered source excerpt differs from its reviewed SHA-256.\n",
            "- A grounding fence is literal, unclassified, or missing a language.\n",
            "- A Python, R, Bash, Nix, TOML, or JSON snippet fails its configured parser.\n",
            "- The source catalog loses required structure or unique IDs.\n",
            "- An AGENTS chain exceeds the default discovery budget.\n",
            "- A checked local Markdown link breaks.\n",
            "- A generated page or machine-readable manifest differs from its deterministic projection.\n",
        ]
    )
    return "".join(lines)


def render_manifest_json(entries, input_digest, checks):
    payload = {
        "schema_version": 2,
        "input_digest": input_digest,
        "watched_file_count": len(entries),
        "assumption_count": len(checks["assumptions"]),
        "runner_count": len(checks["runners"]),
        "snippet_count": len(checks["snippets"]),
        "files": entries,
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_llms_txt(input_digest):
    return f"""# H-2A AEWR coding-agent guide

Freshness command: python scripts/agent_grounding.py verify
Grounding input digest: {input_digest}

Start:
- /operating-protocol/
- /architecture/authority-map/
- /architecture/pipeline/
- /contracts/data-geography/
- /contracts/research-integrity/
- /contracts/historical-documents/
- /contracts/reproducibility/

Designs:
- /designs/did/
- /designs/panel-iv/
- /designs/mundlak-chamberlain/

Generated facts:
- /generated/repository-inventory/
- /generated/pipeline-contracts/
- /generated/runtime-locks/
- /generated/assumptions/
- /generated/code-grounding/
- /generated/drift-report/

Rules:
- Executable code, supported runners, lockfiles, and checked contracts outrank prose.
- Generated facts are invalid when freshness verification fails.
- Authoritative code fences are source-derived and invalid if their reviewed excerpt digest drifts.
- The two documentation Word files are historical proposals, not current implementation contracts.
- Static or dry-run checks do not establish empirical pipeline completion.
- Never change a research design or suppress diagnostics merely to make validation pass.
"""


def expected_outputs(checks=None):
    checks = checks or base_checks()
    entries, input_digest = watched_manifest()
    facts = inventory_facts()
    outputs = {
        GENERATED / "repository-inventory.md": render_inventory(
            entries, input_digest, facts, checks["worst_chain"]
        ),
        GENERATED / "pipeline-contracts.md": render_pipeline(checks["runners"]),
        GENERATED / "runtime-locks.md": render_runtime_locks(),
        GENERATED / "assumptions.md": render_assumptions(checks["assumptions"]),
        SNIPPET_PAGE: render_code_grounding(
            checks["snippet_groups"], checks["snippets"]
        ),
        GENERATED / "drift-report.md": render_drift_report(checks),
        STATIC / "grounding-manifest.json": render_manifest_json(
            entries, input_digest, checks
        ),
        SNIPPET_INDEX: render_snippet_index(
            checks["snippet_groups"], checks["snippets"]
        ),
        STATIC / "llms.txt": render_llms_txt(input_digest),
    }
    return outputs


def write_outputs(outputs):
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def verify_generated(outputs):
    errors = []
    for path, expected in outputs.items():
        if not path.exists():
            errors.append(f"generated grounding missing: {relative(path)}")
            continue
        actual = path.read_text(encoding="utf-8")
        if actual != expected:
            errors.append(
                f"generated grounding stale: {relative(path)}; run `python scripts/agent_grounding.py generate`"
            )
    return errors


def command_generate(_args):
    checks = base_checks()
    if checks["errors"]:
        for error in checks["errors"]:
            print(f"ERROR: {error}", file=sys.stderr)
        print("Refusing to generate from an inconsistent repository.", file=sys.stderr)
        return 1
    outputs = expected_outputs(checks)
    write_outputs(outputs)
    print(f"Generated {len(outputs)} grounding artifacts.")
    for warning in checks["warnings"]:
        print(f"WARNING: {warning}")
    return 0


def command_verify(_args):
    checks = base_checks()
    errors = list(checks["errors"])
    errors.extend(verify_generated(expected_outputs(checks)))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        f"Grounding verified: {len(checks['assumptions'])} assumptions, "
        f"{len(checks['runners'])} runner contracts, {len(checks['snippets'])} source-linked snippets, "
        f"{len(watched_files())} watched files."
    )
    for warning in checks["warnings"]:
        print(f"WARNING: {warning}")
    return 0


def relevant_pages(scope_rel):
    pages = [
        "agent-docs/content/operating-protocol.md",
        "agent-docs/content/architecture/authority-map.md",
    ]
    if scope_rel.startswith("code/a01_sources"):
        pages.append("agent-docs/content/contracts/data-geography.md")
    elif scope_rel.startswith("code/b01_derived"):
        pages.extend(
            [
                "agent-docs/content/architecture/pipeline.md",
                "agent-docs/content/contracts/reproducibility.md",
            ]
        )
    elif scope_rel.startswith("code/designs/did"):
        pages.append("agent-docs/content/designs/did.md")
    elif scope_rel.startswith("code/designs/panel_iv"):
        pages.append("agent-docs/content/designs/panel-iv.md")
    elif scope_rel.startswith("code/designs/mundlak_chamberlain"):
        pages.append("agent-docs/content/designs/mundlak-chamberlain.md")
    elif scope_rel.startswith(("code/c00_shared", "code/c01_clean", "code/c02_build")):
        pages.extend(
            [
                "agent-docs/content/architecture/pipeline.md",
                "agent-docs/content/contracts/data-geography.md",
            ]
        )
    elif scope_rel.startswith("documentation"):
        pages.append("agent-docs/content/contracts/historical-documents.md")
    elif scope_rel.startswith("draft") or scope_rel.startswith("outputs"):
        pages.append("agent-docs/content/contracts/research-integrity.md")
    elif scope_rel.startswith(("agent-docs", "scripts")):
        pages.extend(
            [
                "agent-docs/content/architecture/pipeline.md",
                "agent-docs/content/generated/drift-report.md",
            ]
        )
    else:
        pages.append("agent-docs/content/architecture/pipeline.md")
    pages.append("agent-docs/content/generated/assumptions.md")
    pages.append("agent-docs/content/generated/code-grounding.md")
    return list(dict.fromkeys(pages))


def relevant_snippet_ids(scope_rel, snippets):
    groups = {"execution-runtime"}
    if scope_rel.startswith(("code/a01_sources", "code/b01_derived", "code/c00_shared", "code/c01_clean", "code/c02_build", "src/h2a")):
        groups.add("geography-data")
    if scope_rel.startswith("code/designs/did"):
        groups.add("did")
    if scope_rel.startswith("code/designs/panel_iv"):
        groups.add("panel-iv")
    if scope_rel.startswith("code/designs/mundlak_chamberlain"):
        groups.add("mundlak-chamberlain")
    if scope_rel.startswith(("agent-docs", "scripts")):
        groups.update({"geography-data", "did", "panel-iv", "mundlak-chamberlain"})
    return [snippet["id"] for snippet in snippets if snippet.get("group") in groups]


def command_snapshot(args):
    scope = (ROOT / args.scope).resolve() if not Path(args.scope).is_absolute() else Path(args.scope).resolve()
    try:
        scope_rel = scope.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        print(f"ERROR: scope escapes repository: {args.scope}", file=sys.stderr)
        return 2
    if not scope.exists():
        print(f"ERROR: scope does not exist: {scope_rel}", file=sys.stderr)
        return 2
    checks = base_checks()
    revision = repository_revision()
    chain = active_agent_chain(scope)
    payload = {
        "repository": str(ROOT),
        "scope": scope_rel,
        "revision": revision,
        "active_agents": [relative(path) for path in chain],
        "read_next": relevant_pages(scope_rel),
        "grounded_snippets": relevant_snippet_ids(scope_rel, checks["snippets"]),
        "grounding_errors": checks["errors"],
        "boundaries": checks["warnings"],
    }
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1 if checks["errors"] else 0
    print(f"# Grounding snapshot: {scope_rel}\n")
    if revision["commit"]:
        print(f"- Commit: `{revision['commit']}`")
        print(f"- Branch: `{revision['branch'] or '(detached)'}`")
        print(f"- Dirty worktree: `{revision['dirty']}`")
    else:
        print("- Git metadata: unavailable (for example, a downloaded source archive)")
    print("\n## Active AGENTS chain\n")
    for path in chain:
        print(f"1. `{relative(path)}` ({path.stat().st_size} bytes)")
    print("\n## Read next\n")
    for path in relevant_pages(scope_rel):
        print(f"- `{path}`")
    print("\n## Grounded code excerpts\n")
    for snippet_id in relevant_snippet_ids(scope_rel, checks["snippets"]):
        print(f"- `{snippet_id}`")
    if checks["errors"]:
        print("\n## Grounding errors\n")
        for error in checks["errors"]:
            print(f"- {error}")
    print("\n## Boundaries\n")
    for warning in checks["warnings"]:
        print(f"- {warning}")
    return 1 if checks["errors"] else 0


def command_query(args):
    terms = [term.casefold() for term in args.terms]
    candidates = set(ROOT.rglob("AGENTS.md"))
    candidates.update(DOCS.glob("content/**/*.md"))
    candidates.update(ROOT.glob("code/**/README.md"))
    candidates.update((ROOT / name for name in ("README.md", "scripts/README.md")))
    matches = 0
    for path in sorted((path for path in candidates if path.exists()), key=relative):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            folded = line.casefold()
            if all(term in folded for term in terms):
                print(f"{relative(path)}:{lineno}:{line.strip()}")
                matches += 1
                if matches >= args.limit:
                    return 0
    if matches == 0:
        print("No agent-documentation matches.", file=sys.stderr)
        return 1
    return 0


def rendered_snippet_excerpt(snippet_id):
    if not SNIPPET_PAGE.exists():
        return None
    text = SNIPPET_PAGE.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"<!-- grounding-snippet:{re.escape(snippet_id)} excerpt-sha256=[0-9a-f]{{64}} -->\n"
        rf"\x60{{3}}[^\n]+\n(.*?)\x60{{3}}",
        re.DOTALL,
    )
    match = pattern.search(text)
    return match.group(1) if match else None


def replace_reviewed_hash(catalog_text, snippet_id, new_hash):
    section = re.compile(
        r'(?ms)(^\[\[snippets\]\]\n(?:(?!^\[\[snippets\]\]).)*?'
        rf'^id = {re.escape(json.dumps(snippet_id))}\n'
        r'(?:(?!^\[\[snippets\]\]).)*?^expected_sha256 = ")([0-9a-f]*)("$)'
    )
    updated, count = section.subn(rf"\g<1>{new_hash}\g<3>", catalog_text)
    if count != 1:
        raise ValueError(f"could not uniquely update expected hash for {snippet_id}")
    return updated


def command_accept_snippet_drift(args):
    try:
        _groups, records = snippet_catalog()
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        print(f"ERROR: snippet catalog invalid: {error}", file=sys.stderr)
        return 2
    selected_ids = [record["id"] for record in records] if args.all else args.ids
    if not selected_ids:
        print("ERROR: select --id ID (repeatable) or --all", file=sys.stderr)
        return 2
    by_id = {record["id"]: record for record in records}
    unknown = sorted(set(selected_ids) - set(by_id))
    if unknown:
        print("ERROR: unknown snippet IDs: " + ", ".join(unknown), file=sys.stderr)
        return 2
    catalog_text = SNIPPET_CATALOG.read_text(encoding="utf-8")
    changes = 0
    for snippet_id in dict.fromkeys(selected_ids):
        try:
            snippet = extract_snippet(by_id[snippet_id])
            warning = validate_snippet_syntax(snippet)
            if warning:
                print(f"WARNING: {warning}", file=sys.stderr)
        except (SyntaxError, ValueError, OSError, tomllib.TOMLDecodeError, json.JSONDecodeError) as error:
            print(f"ERROR: {snippet_id}: {error}", file=sys.stderr)
            return 1
        old_hash = by_id[snippet_id].get("expected_sha256", "")
        new_hash = snippet["excerpt_sha256"]
        if old_hash == new_hash:
            print(f"UNCHANGED: {snippet_id} {new_hash}")
            continue
        previous = rendered_snippet_excerpt(snippet_id)
        print(f"REVIEW: {snippet_id} {old_hash or '(unset)'} -> {new_hash}")
        if previous is None:
            print("  No prior generated excerpt is available; inspect the source selection directly.")
        else:
            diff = difflib.unified_diff(
                previous.splitlines(),
                snippet["excerpt"].splitlines(),
                fromfile=f"reviewed/{snippet_id}",
                tofile=f"current/{snippet_id}",
                lineterm="",
            )
            for line in diff:
                print(line)
        if args.write:
            catalog_text = replace_reviewed_hash(catalog_text, snippet_id, new_hash)
            changes += 1
    if args.write and changes:
        SNIPPET_CATALOG.write_text(catalog_text, encoding="utf-8")
        print(
            f"Accepted {changes} reviewed snippet digest(s). Run "
            "python scripts/agent_grounding.py generate next."
        )
    elif changes == 0:
        print("No snippet digest changes to accept.")
    elif not args.write:
        print("Review only: rerun with --write after approving the displayed drift.")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Regenerate deterministic agent grounding")
    generate.set_defaults(func=command_generate)

    verify = subparsers.add_parser("verify", help="Verify contracts and generated freshness")
    verify.set_defaults(func=command_verify)

    snapshot = subparsers.add_parser("snapshot", help="Print scoped starting context")
    snapshot.add_argument("--scope", required=True, help="Repository-relative target path")
    snapshot.add_argument("--format", choices=("markdown", "json"), default="markdown")
    snapshot.set_defaults(func=command_snapshot)

    query = subparsers.add_parser("query", help="Search agent instructions and documentation")
    query.add_argument("terms", nargs="+", help="Terms that must all occur on a matching line")
    query.add_argument("--limit", type=int, default=100)
    query.set_defaults(func=command_query)

    accept = subparsers.add_parser(
        "accept-snippet-drift",
        help="Review and explicitly accept changed source-linked excerpt digests",
    )
    selection = accept.add_mutually_exclusive_group(required=True)
    selection.add_argument("--id", dest="ids", action="append", default=[])
    selection.add_argument("--all", action="store_true")
    accept.add_argument(
        "--write",
        action="store_true",
        help="Update reviewed digests after displaying the old/new excerpt diff",
    )
    accept.set_defaults(func=command_accept_snippet_drift)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
