#!/usr/bin/env python3
"""Focused regression tests for LLM-oriented documentation grounding."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent_docs


class AgentDocumentationTests(unittest.TestCase):
    def test_extraction_matches_zola_shortcode_contract(self):
        source = (
            "before\n"
            "# docs-ground:start demo\n"
            "  reviewed value  \n"
            "# docs-ground:end demo\n"
            "after\n"
        )
        self.assertEqual(agent_docs.extract_region(source, "demo.py", "demo"), "reviewed value")

    def test_duplicate_markers_fail_closed(self):
        source = (
            "# docs-ground:start demo\nfirst\n# docs-ground:end demo\n"
            "# docs-ground:start demo\nsecond\n"
        )
        with self.assertRaisesRegex(ValueError, "exactly one start marker"):
            agent_docs.extract_region(source, "demo.py", "demo")

    def test_hash_only_update_is_not_a_prose_change(self):
        old = '{{ grounding(path="x.py", anchor="demo", sha256="' + "a" * 64 + '") }}\nClaim.\n'
        new_hash = '{{ grounding(path="x.py", anchor="demo", sha256="' + "b" * 64 + '") }}\nClaim.\n'
        new_prose = '{{ grounding(path="x.py", anchor="demo", sha256="' + "b" * 64 + '") }}\nRevised claim.\n'
        self.assertEqual(
            agent_docs.sha256_text(agent_docs.normalized_document_text(old)),
            agent_docs.sha256_text(agent_docs.normalized_document_text(new_hash)),
        )
        self.assertNotEqual(
            agent_docs.sha256_text(agent_docs.normalized_document_text(old)),
            agent_docs.sha256_text(agent_docs.normalized_document_text(new_prose)),
        )

    def test_repository_grounding_is_valid(self):
        documents, references, errors = agent_docs.parse_documents()
        self.assertEqual(errors, [])
        self.assertGreaterEqual(len(documents), 15)
        self.assertGreaterEqual(len(references), 13)

    def test_generated_context_is_current(self):
        documents, references, errors = agent_docs.parse_documents()
        self.assertEqual(errors, [])
        assumptions, assumption_errors = agent_docs.validate_assumptions()
        self.assertEqual(assumption_errors, [])
        self.assertEqual(
            agent_docs.generated_output_errors(
                agent_docs.expected_outputs(documents, references, assumptions)
            ),
            [],
        )

    def test_assumptions_are_checked_and_scope_aware(self):
        assumptions, errors = agent_docs.validate_assumptions()
        self.assertEqual(errors, [])
        self.assertGreaterEqual(len(assumptions), 8)
        panel_iv = next(assumption for assumption in assumptions if assumption["id"] == "IV-001")
        self.assertIn("code/designs/panel_iv", panel_iv["scopes"])

    def test_scoped_agent_chain_is_discovered(self):
        paths = [
            path.relative_to(agent_docs.ROOT).as_posix()
            for path in agent_docs.active_agent_files("code/designs/panel_iv/design.R")
        ]
        self.assertEqual(
            paths,
            ["AGENTS.md", "code/AGENTS.md", "code/designs/AGENTS.md", "code/designs/panel_iv/AGENTS.md"],
        )


if __name__ == "__main__":
    unittest.main()
