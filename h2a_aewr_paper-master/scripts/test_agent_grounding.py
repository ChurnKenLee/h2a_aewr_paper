#!/usr/bin/env python3
"""Focused regression tests for source-linked coding-agent grounding."""

import tempfile
import unittest
from pathlib import Path

import agent_grounding as grounding


class SourceLinkedSnippetTests(unittest.TestCase):
    def test_catalog_is_extensive_extracted_and_digest_locked(self):
        errors, _warnings, groups, snippets = grounding.validate_snippets()
        self.assertEqual(errors, [])
        self.assertGreaterEqual(len(groups), 5)
        self.assertGreaterEqual(len(snippets), 30)
        self.assertEqual(len({snippet["id"] for snippet in snippets}), len(snippets))
        for snippet in snippets:
            self.assertEqual(snippet["excerpt_sha256"], snippet["expected_sha256"])
            self.assertEqual(len(snippet["source_sha256"]), 64)
            self.assertTrue(snippet["excerpt"].endswith("\n"))

    def test_braced_symbol_ignores_strings_and_comments(self):
        lines = [
            "demo <- function(value) {\n",
            '  text <- "} is data, not structure"\n',
            "  # } is a comment\n",
            "  if (value) {\n",
            "    1\n",
            "  }\n",
            "}\n",
            "after <- 2\n",
        ]
        excerpt = grounding.braced_symbol_excerpt(
            lines,
            r"^\s*demo\s*<-\s*function\b",
            "demo",
        )
        self.assertIn('text <- "} is data, not structure"', excerpt)
        self.assertTrue(excerpt.endswith("}\n"))
        self.assertNotIn("after <- 2", excerpt)

    def test_unclassified_literal_fence_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guide.md"
            path.write_text("\x60\x60\x60python\nprint('stale')\n\x60\x60\x60\n", encoding="utf-8")
            errors = grounding.validate_literal_fences_in_path(path)
        self.assertTrue(any("unclassified" in error for error in errors))

    def test_classified_literal_fence_is_non_authoritative(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guide.md"
            path.write_text(
                "<!-- grounding-fence: illustrative -->\n"
                "\x60\x60\x60python\n"
                "print('example')\n"
                "\x60\x60\x60\n",
                encoding="utf-8",
            )
            errors = grounding.validate_literal_fences_in_path(path)
        self.assertEqual(errors, [])

    def test_reviewed_hash_update_targets_one_registry_record(self):
        source = (
            "[[snippets]]\n"
            'id = "one"\n'
            'expected_sha256 = "' + "a" * 64 + '"\n\n'
            "[[snippets]]\n"
            'id = "two"\n'
            'expected_sha256 = "' + "b" * 64 + '"\n'
        )
        updated = grounding.replace_reviewed_hash(source, "two", "c" * 64)
        self.assertIn('id = "one"\nexpected_sha256 = "' + "a" * 64, updated)
        self.assertIn('id = "two"\nexpected_sha256 = "' + "c" * 64, updated)

    def test_changed_excerpt_digest_is_an_error(self):
        snippet = {
            "id": "drift-test",
            "expected_sha256": "a" * 64,
            "excerpt_sha256": "b" * 64,
        }
        error = grounding.snippet_digest_error(snippet)
        self.assertIn("source-linked excerpt drifted", error)
        self.assertIn("accept-snippet-drift --id drift-test", error)

    def test_rendered_page_contains_one_fence_per_snippet(self):
        errors, _warnings, groups, snippets = grounding.validate_snippets()
        self.assertEqual(errors, [])
        rendered = grounding.render_code_grounding(groups, snippets)
        self.assertEqual(rendered.count("<!-- grounding-snippet:"), len(snippets))
        self.assertEqual(rendered.count("Extracted-text SHA-256"), len(snippets))


if __name__ == "__main__":
    unittest.main()
