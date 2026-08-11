#!/usr/bin/env python3
"""Build and verify source-grounded context for humans and coding agents."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT / "content"
STATIC = ROOT / "static"
MANIFEST = STATIC / "grounding-manifest.json"
LLMS = STATIC / "llms.txt"
ASSUMPTIONS = ROOT / "agent" / "assumptions.toml"
SCHEMA_VERSION = 1

SHORTCODE_RE = re.compile(r"{{\s*grounding\((?P<args>[^{}]*)\)\s*}}")
ATTRIBUTE_RE = re.compile(r'(?P<name>path|anchor|sha256)\s*=\s*"(?P<value>[^"]*)"')
SHA_ATTRIBUTE_RE = re.compile(r'(?P<prefix>sha256\s*=\s*")[^"]*(?P<suffix>")')
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

AGENT_ROOTS = ("code", "content", "documentation", "draft", "outputs", "scripts", "src")


@dataclass(frozen=True)
class Document:
    path: str
    title: str
    description: str
    route: str
    scopes: tuple[str, ...]
    prose_sha256: str


@dataclass(frozen=True)
class GroundingReference:
    id: str
    document: str
    document_title: str
    source: str
    anchor: str
    expected_sha256: str
    current_sha256: str
    excerpt: str
    call_start: int
    call_end: int


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalized_document_text(text: str) -> str:
    """Ignore reviewed digest values when deciding whether prose changed."""

    def normalize(match: re.Match[str]) -> str:
        call = match.group(0)
        return SHA_ATTRIBUTE_RE.sub(r'\g<prefix>REVIEWED_DIGEST\g<suffix>', call)

    return SHORTCODE_RE.sub(normalize, text)


def split_front_matter(path: Path, text: str) -> tuple[dict, str]:
    if not text.startswith("+++\n"):
        raise ValueError(f"{path.relative_to(ROOT)}: missing TOML front matter")
    try:
        raw_front_matter, body = text[4:].split("\n+++\n", 1)
    except ValueError as error:
        raise ValueError(f"{path.relative_to(ROOT)}: unterminated TOML front matter") from error
    try:
        return tomllib.loads(raw_front_matter), body
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"{path.relative_to(ROOT)}: invalid TOML front matter: {error}") from error


def document_route(relative_path: str) -> str:
    path = Path(relative_path).relative_to("content")
    if path.name == "_index.md":
        parent = path.parent.as_posix()
        return "/" if parent == "." else f"/{parent}/"
    return f"/{path.with_suffix('').as_posix()}/"


def safe_repository_path(relative_path: str) -> Path:
    candidate = (ROOT / relative_path).resolve()
    try:
        candidate.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"path escapes repository root: {relative_path}") from error
    return candidate


def extract_region(source: str, source_path: str, anchor: str) -> str:
    start_marker = f"# docs-ground:start {anchor}"
    end_marker = f"# docs-ground:end {anchor}"
    if source.count(start_marker) != 1:
        raise ValueError(f"expected exactly one start marker for {source_path}#{anchor}")
    if source.count(end_marker) != 1:
        raise ValueError(f"expected exactly one end marker for {source_path}#{anchor}")
    before_end, _separator, _after_end = source.partition(end_marker)
    _before_start, separator, after_start = before_end.partition(start_marker)
    if not separator:
        raise ValueError(f"end marker occurs before start marker for {source_path}#{anchor}")
    return after_start.strip()


def parse_documents() -> tuple[list[Document], list[GroundingReference], list[str]]:
    documents: list[Document] = []
    references: list[GroundingReference] = []
    errors: list[str] = []
    seen_ids: set[str] = set()

    for path in sorted(CONTENT.rglob("*.md")):
        if path.name == "AGENTS.md":
            continue
        relative_path = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        try:
            front_matter, _body = split_front_matter(path, text)
        except ValueError as error:
            errors.append(str(error))
            continue

        title = front_matter.get("title")
        if not isinstance(title, str) or not title.strip():
            errors.append(f"{relative_path}: front matter needs a nonempty title")
            title = relative_path
        description = front_matter.get("description", "")
        if not isinstance(description, str):
            errors.append(f"{relative_path}: description must be a string")
            description = ""
        extra = front_matter.get("extra", {})
        scopes = extra.get("scopes", []) if isinstance(extra, dict) else []
        if not isinstance(scopes, list) or not all(isinstance(scope, str) for scope in scopes):
            errors.append(f"{relative_path}: extra.scopes must be an array of repository paths")
            scopes = []
        for scope in scopes:
            try:
                safe_repository_path(scope)
            except ValueError as error:
                errors.append(f"{relative_path}: invalid scope: {error}")

        document = Document(
            path=relative_path,
            title=title.strip(),
            description=description.strip(),
            route=document_route(relative_path),
            scopes=tuple(scopes),
            prose_sha256=sha256_text(normalized_document_text(text)),
        )
        documents.append(document)

        for match in SHORTCODE_RE.finditer(text):
            attributes = {item.group("name"): item.group("value") for item in ATTRIBUTE_RE.finditer(match.group("args"))}
            missing = sorted({"path", "anchor", "sha256"} - attributes.keys())
            if missing:
                errors.append(f"{relative_path}: grounding shortcode missing {', '.join(missing)}")
                continue
            source_path = attributes["path"]
            anchor = attributes["anchor"]
            expected_sha256 = attributes["sha256"]
            reference_id = f"{relative_path}::{source_path}#{anchor}"
            if reference_id in seen_ids:
                errors.append(f"{relative_path}: duplicate grounding reference {source_path}#{anchor}")
                continue
            seen_ids.add(reference_id)
            if not SHA256_RE.fullmatch(expected_sha256):
                errors.append(
                    f"{relative_path}: {source_path}#{anchor} has an invalid reviewed SHA-256: "
                    f"{expected_sha256}"
                )
            try:
                source_file = safe_repository_path(source_path)
                source = source_file.read_text(encoding="utf-8")
                excerpt = extract_region(source, source_path, anchor)
            except (OSError, UnicodeError, ValueError) as error:
                errors.append(f"{relative_path}: {error}")
                continue
            current_sha256 = sha256_text(excerpt)
            if expected_sha256 != current_sha256:
                errors.append(
                    f"{relative_path}: stale documentation for {source_path}#{anchor}; "
                    f"review with `python scripts/agent_docs.py accept-drift "
                    f"--document {relative_path} --anchor {anchor}`"
                )
            references.append(
                GroundingReference(
                    id=reference_id,
                    document=relative_path,
                    document_title=document.title,
                    source=source_path,
                    anchor=anchor,
                    expected_sha256=expected_sha256,
                    current_sha256=current_sha256,
                    excerpt=excerpt,
                    call_start=match.start(),
                    call_end=match.end(),
                )
            )

    return documents, references, errors


def validate_assumptions() -> tuple[list[dict], list[str]]:
    errors: list[str] = []
    if not ASSUMPTIONS.is_file():
        return [], [f"missing assumption registry: {ASSUMPTIONS.relative_to(ROOT)}"]
    try:
        data = tomllib.loads(ASSUMPTIONS.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        return [], [f"cannot read assumption registry: {error}"]
    if data.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"{ASSUMPTIONS.relative_to(ROOT)} must have schema_version = {SCHEMA_VERSION}")

    assumptions = data.get("assumptions", [])
    if not isinstance(assumptions, list):
        return [], errors + [f"{ASSUMPTIONS.relative_to(ROOT)}: assumptions must be an array"]
    seen_ids: set[str] = set()
    required = {"id", "risk", "owner", "statement", "source", "review_when", "scopes", "checks"}
    for assumption in assumptions:
        if not isinstance(assumption, dict):
            errors.append(f"{ASSUMPTIONS.relative_to(ROOT)}: every assumption must be a table")
            continue
        assumption_id = assumption.get("id", "<missing-id>")
        missing = sorted(required - assumption.keys())
        if missing:
            errors.append(f"{assumption_id}: missing assumption fields: {', '.join(missing)}")
            continue
        if assumption_id in seen_ids:
            errors.append(f"duplicate assumption id: {assumption_id}")
        seen_ids.add(assumption_id)
        if assumption.get("risk") not in {"critical", "high", "medium", "low"}:
            errors.append(f"{assumption_id}: risk must be critical, high, medium, or low")
        scopes = assumption.get("scopes")
        if not isinstance(scopes, list) or not scopes or not all(isinstance(scope, str) for scope in scopes):
            errors.append(f"{assumption_id}: scopes must be a nonempty array of repository paths")
        checks = assumption.get("checks")
        if not isinstance(checks, list) or not checks:
            errors.append(f"{assumption_id}: checks must be a nonempty array")
            continue
        for index, check in enumerate(checks, start=1):
            if not isinstance(check, dict) or not isinstance(check.get("path"), str):
                errors.append(f"{assumption_id} check {index}: path is required")
                continue
            operators = [operator for operator in ("contains", "regex", "not_contains") if operator in check]
            if len(operators) != 1:
                errors.append(f"{assumption_id} check {index}: specify exactly one text operator")
                continue
            try:
                checked_path = safe_repository_path(check["path"])
                checked_text = checked_path.read_text(encoding="utf-8")
            except (OSError, UnicodeError, ValueError) as error:
                errors.append(f"{assumption_id} check {index}: {error}")
                continue
            operator = operators[0]
            expected = check[operator]
            if not isinstance(expected, str):
                errors.append(f"{assumption_id} check {index}: {operator} must be a string")
            elif operator == "contains" and expected not in checked_text:
                errors.append(f"{assumption_id} check {index}: {check['path']} is missing required text")
            elif operator == "not_contains" and expected in checked_text:
                errors.append(f"{assumption_id} check {index}: {check['path']} contains forbidden text")
            elif operator == "regex":
                try:
                    matched = re.search(expected, checked_text, flags=re.MULTILINE | re.DOTALL)
                except re.error as error:
                    errors.append(f"{assumption_id} check {index}: invalid regex: {error}")
                else:
                    if not matched:
                        errors.append(f"{assumption_id} check {index}: {check['path']} does not match required regex")
    return assumptions, errors


def discover_agent_files() -> list[Path]:
    paths = [ROOT / "AGENTS.md"]
    for directory in AGENT_ROOTS:
        root = ROOT / directory
        if root.exists():
            paths.extend(root.rglob("AGENTS.md"))
    return sorted({path for path in paths if path.is_file()}, key=lambda path: path.relative_to(ROOT).as_posix())


def manifest_data(
    documents: list[Document],
    references: list[GroundingReference],
    assumptions: list[dict],
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_note": "Generated by scripts/agent_docs.py; edit sources, documentation, or AGENTS.md instead.",
        "agent_instructions": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_text(path.read_text(encoding="utf-8")),
                "text": path.read_text(encoding="utf-8"),
            }
            for path in discover_agent_files()
        ],
        "assumptions": assumptions,
        "documents": [
            {
                "path": document.path,
                "route": document.route,
                "title": document.title,
                "description": document.description,
                "scopes": list(document.scopes),
                "prose_sha256": document.prose_sha256,
            }
            for document in documents
        ],
        "grounding": [
            {
                "id": reference.id,
                "document": reference.document,
                "document_title": reference.document_title,
                "source": reference.source,
                "anchor": reference.anchor,
                "sha256": reference.current_sha256,
                "excerpt": reference.excerpt,
            }
            for reference in references
        ],
    }


def render_manifest(
    documents: list[Document],
    references: list[GroundingReference],
    assumptions: list[dict],
) -> str:
    return json.dumps(
        manifest_data(documents, references, assumptions),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def render_llms(
    documents: list[Document],
    references: list[GroundingReference],
    assumptions: list[dict],
) -> str:
    references_by_document: dict[str, list[GroundingReference]] = {}
    for reference in references:
        references_by_document.setdefault(reference.document, []).append(reference)

    lines = [
        "# H-2A Paper agent documentation",
        "",
        "> Grounded technical context for coding agents working on the empirical pipeline.",
        "",
        (
            "Start with the repository `AGENTS.md`, then run "
            "`python scripts/agent_docs.py snapshot --scope <target-path>`."
        ),
        (
            "The machine-readable instruction and reviewed-source bundle is "
            "[`/grounding-manifest.json`](/grounding-manifest.json)."
        ),
        "",
        "## Canonical pages",
        "",
    ]
    for document in documents:
        description = f" — {document.description}" if document.description else ""
        lines.append(f"- [{document.title}]({document.route}){description}")
        for reference in references_by_document.get(document.path, []):
            lines.append(f"  - Grounded in `{reference.source}#{reference.anchor}`")
    lines.extend(["", "## Declared high-risk assumptions", ""])
    for assumption in assumptions:
        lines.append(
            f"- `{assumption['id']}` ({assumption['risk']}, {assumption['owner']}) — "
            f"{assumption['statement']}"
        )
    lines.extend(
        [
            "",
            "## Grounding rule",
            "",
            (
                "A matching SHA-256 means the named source region has not changed since the linked page was reviewed. "
                "It does not independently prove the prose or empirical claim."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def expected_outputs(
    documents: list[Document],
    references: list[GroundingReference],
    assumptions: list[dict],
) -> dict[Path, str]:
    return {
        MANIFEST: render_manifest(documents, references, assumptions),
        LLMS: render_llms(documents, references, assumptions),
    }


def generated_output_errors(outputs: dict[Path, str]) -> list[str]:
    errors = []
    for path, expected in outputs.items():
        relative_path = path.relative_to(ROOT).as_posix()
        if not path.exists():
            errors.append(f"generated agent context missing: {relative_path}; run `python scripts/agent_docs.py generate`")
        elif path.read_text(encoding="utf-8") != expected:
            errors.append(f"generated agent context stale: {relative_path}; run `python scripts/agent_docs.py generate`")
    return errors


def write_outputs(outputs: dict[Path, str]) -> None:
    for path, rendered in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")


def verify_repository() -> tuple[list[Document], list[GroundingReference], list[dict], list[str]]:
    documents, references, errors = parse_documents()
    assumptions, assumption_errors = validate_assumptions()
    errors.extend(assumption_errors)
    errors.extend(generated_output_errors(expected_outputs(documents, references, assumptions)))
    return documents, references, assumptions, errors


# docs-ground:start agent-docs-verification
def command_generate(_args: argparse.Namespace) -> int:
    documents, references, errors = parse_documents()
    assumptions, assumption_errors = validate_assumptions()
    errors.extend(assumption_errors)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print("Refusing to generate context while source grounding is invalid.", file=sys.stderr)
        return 1
    outputs = expected_outputs(documents, references, assumptions)
    write_outputs(outputs)
    print(f"Generated {len(outputs)} agent-context artifacts with {len(references)} grounded references.")
    return 0


def command_verify(_args: argparse.Namespace) -> int:
    documents, references, assumptions, errors = verify_repository()
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        f"Agent documentation verified: {len(documents)} pages, "
        f"{len(references)} grounded references, {len(assumptions)} assumptions, "
        f"{len(discover_agent_files())} AGENTS files."
    )
    return 0
# docs-ground:end agent-docs-verification


def paths_overlap(left: str, right: str) -> bool:
    left = left.strip("/") or "."
    right = right.strip("/") or "."
    return left == right or left.startswith(f"{right}/") or right.startswith(f"{left}/")


def active_agent_files(scope: str) -> list[Path]:
    target = safe_repository_path(scope)
    directory = target if target.is_dir() else target.parent
    if not target.exists() and not scope.endswith("/"):
        directory = target.parent
    relative_directory = directory.relative_to(ROOT)
    candidates = [ROOT / "AGENTS.md"]
    cursor = ROOT
    for part in relative_directory.parts:
        cursor /= part
        candidates.append(cursor / "AGENTS.md")
    return [path for path in candidates if path.is_file()]


def nearest_readmes(scope: str) -> list[Path]:
    target = safe_repository_path(scope)
    cursor = target if target.is_dir() else target.parent
    if not target.exists() and not scope.endswith("/"):
        cursor = target.parent
    found = []
    while True:
        readme = cursor / "README.md"
        if readme.is_file():
            found.append(readme)
        if cursor == ROOT:
            break
        cursor = cursor.parent
    return list(reversed(found))


def git_context() -> dict:
    try:
        top_level = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if Path(top_level).resolve() != ROOT:
            return {"available": False, "warning": f"Git root is {top_level}, not {ROOT}"}
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "branch", "--show-current"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
            ).stdout
        )
        return {"available": True, "commit": commit, "branch": branch, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"available": False, "warning": str(error)}


# docs-ground:start agent-context-snapshot
def snapshot_data(scope: str) -> tuple[dict, list[str]]:
    safe_repository_path(scope)
    documents, references, assumptions, errors = verify_repository()
    relevant_documents = []
    for document in documents:
        source_match = any(
            reference.document == document.path and paths_overlap(scope, reference.source)
            for reference in references
        )
        scope_match = any(paths_overlap(scope, declared_scope) for declared_scope in document.scopes)
        if document.path == "content/_index.md" or source_match or scope_match:
            relevant_documents.append(document)
    data = {
        "scope": scope,
        "git": git_context(),
        "verified": not errors,
        "errors": errors,
        "active_agents": [path.relative_to(ROOT).as_posix() for path in active_agent_files(scope)],
        "readmes": [path.relative_to(ROOT).as_posix() for path in nearest_readmes(scope)],
        "documents": [
            {
                "path": document.path,
                "title": document.title,
                "route": document.route,
            }
            for document in relevant_documents
        ],
        "assumptions": [
            {
                "id": assumption["id"],
                "risk": assumption["risk"],
                "owner": assumption["owner"],
                "statement": assumption["statement"],
                "source": assumption["source"],
                "review_when": assumption["review_when"],
            }
            for assumption in assumptions
            if any(paths_overlap(scope, declared_scope) for declared_scope in assumption["scopes"])
        ],
        "grounding": [
            {
                "document": reference.document,
                "source": reference.source,
                "anchor": reference.anchor,
                "sha256": reference.current_sha256,
            }
            for reference in references
            if paths_overlap(scope, reference.source)
        ],
    }
    return data, errors


def command_snapshot(args: argparse.Namespace) -> int:
    try:
        data, errors = snapshot_data(args.scope)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if args.format == "json":
        print(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))
        return 1 if errors else 0

    print(f"# Agent context: {data['scope']}\n")
    git = data["git"]
    if git.get("available"):
        print(f"- Commit: `{git['commit']}`")
        print(f"- Branch: `{git['branch']}`")
        print(f"- Dirty worktree: `{git['dirty']}`")
    else:
        print(f"- Git context unavailable: {git.get('warning', 'unknown error')}")
    print(f"- Grounding verified: `{data['verified']}`")

    print("\n## Active instructions\n")
    for path in data["active_agents"]:
        print(f"- `{path}`")
    print("\n## Operational context\n")
    for path in data["readmes"]:
        print(f"- `{path}`")
    print("\n## Canonical documentation\n")
    for document in data["documents"]:
        print(f"- `{document['path']}` — {document['title']}")
    if data["assumptions"]:
        print("\n## Relevant declared assumptions\n")
        for assumption in data["assumptions"]:
            print(f"- `{assumption['id']}` ({assumption['risk']}): {assumption['statement']}")
    if data["grounding"]:
        print("\n## Directly grounded source regions\n")
        for reference in data["grounding"]:
            print(f"- `{reference['source']}#{reference['anchor']}` -> `{reference['document']}`")
    if errors:
        print("\n## Grounding errors\n")
        for error in errors:
            print(f"- {error}")
    return 1 if errors else 0
# docs-ground:end agent-context-snapshot


def load_previous_manifest() -> dict:
    if not MANIFEST.is_file():
        raise ValueError("grounding manifest is missing; establish valid initial hashes and run generate first")
    try:
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {MANIFEST.relative_to(ROOT)}: {error}") from error


# docs-ground:start agent-drift-review
def command_accept_drift(args: argparse.Namespace) -> int:
    document_path = Path(args.document).as_posix()
    documents, references, _errors = parse_documents()
    matches = [
        reference
        for reference in references
        if reference.document == document_path and reference.anchor == args.anchor
    ]
    if len(matches) != 1:
        print(
            f"ERROR: expected one grounding reference in {document_path} with anchor {args.anchor}; "
            f"found {len(matches)}",
            file=sys.stderr,
        )
        return 1
    reference = matches[0]
    if reference.current_sha256 == reference.expected_sha256:
        print(f"No source drift for {reference.source}#{reference.anchor}.")
        return 0

    try:
        previous = load_previous_manifest()
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    previous_references = {item["id"]: item for item in previous.get("grounding", [])}
    previous_documents = {item["path"]: item for item in previous.get("documents", [])}
    old_reference = previous_references.get(reference.id)
    old_document = previous_documents.get(reference.document)
    if old_reference is None or old_document is None:
        print(f"ERROR: no previously reviewed context exists for {reference.id}", file=sys.stderr)
        return 1

    diff = difflib.unified_diff(
        old_reference["excerpt"].splitlines(),
        reference.excerpt.splitlines(),
        fromfile=f"reviewed/{reference.source}#{reference.anchor}",
        tofile=f"current/{reference.source}#{reference.anchor}",
        lineterm="",
    )
    print("\n".join(diff) or "The selected excerpt text is unchanged; inspect marker placement.")

    current_document = next(document for document in documents if document.path == reference.document)
    prose_changed = current_document.prose_sha256 != old_document.get("prose_sha256")
    print(f"\nDocumentation prose changed since review: {prose_changed}")
    print(f"Current source SHA-256: {reference.current_sha256}")
    if not args.write:
        print(
            "Review the source diff and update the documentation prose. Then rerun this command with --write."
        )
        return 0
    if not prose_changed:
        print(
            "ERROR: refusing to accept source drift because the documentation prose and context are unchanged. "
            "Update the page's explanation before accepting the new digest.",
            file=sys.stderr,
        )
        return 1

    document_file = safe_repository_path(reference.document)
    text = document_file.read_text(encoding="utf-8")
    call = text[reference.call_start : reference.call_end]
    updated_call = SHA_ATTRIBUTE_RE.sub(
        rf"\g<prefix>{reference.current_sha256}\g<suffix>",
        call,
        count=1,
    )
    updated_text = text[: reference.call_start] + updated_call + text[reference.call_end :]
    document_file.write_text(updated_text, encoding="utf-8")
    print(f"Accepted reviewed digest in {reference.document}.")

    refreshed_documents, refreshed_references, refreshed_errors = parse_documents()
    if refreshed_errors:
        for error in refreshed_errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print("Other grounding drift remains; accept each affected page before regenerating context.", file=sys.stderr)
        return 1
    assumptions, assumption_errors = validate_assumptions()
    if assumption_errors:
        for error in assumption_errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    write_outputs(expected_outputs(refreshed_documents, refreshed_references, assumptions))
    print("Regenerated machine-readable agent context.")
    return 0
# docs-ground:end agent-drift-review


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate reviewed machine-readable agent context")
    generate.set_defaults(func=command_generate)

    verify = subparsers.add_parser("verify", help="verify grounding and generated context freshness")
    verify.set_defaults(func=command_verify)

    snapshot = subparsers.add_parser("snapshot", help="print instructions and documentation for a target scope")
    snapshot.add_argument("--scope", required=True, help="repository-relative file or directory")
    snapshot.add_argument("--format", choices=("markdown", "json"), default="markdown")
    snapshot.set_defaults(func=command_snapshot)

    accept = subparsers.add_parser("accept-drift", help="review and explicitly accept one changed source region")
    accept.add_argument("--document", required=True, help="grounded Markdown page under content/")
    accept.add_argument("--anchor", required=True, help="docs-ground anchor name")
    accept.add_argument("--write", action="store_true", help="record the reviewed digest after prose changed")
    accept.set_defaults(func=command_accept_drift)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
