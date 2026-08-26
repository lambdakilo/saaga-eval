"""Contamination detection must fire on real leaks and stay quiet otherwise.

A checker that flags everything gets switched off, and a checker that flags
nothing is decoration. Both failure modes are tested here.
"""

from __future__ import annotations

from saaga_eval.contamination import (
    Severity,
    blocking,
    check_instance,
    patch_introduced_symbols,
    scan,
    failing_test_identifiers,
)

GOLD_PATCH = """\
diff --git a/pkg/resolver.py b/pkg/resolver.py
--- a/pkg/resolver.py
+++ b/pkg/resolver.py
@@ -10,6 +10,12 @@ class Resolver:
     def resolve(self, name):
-        return self._cache[name]
+        return self._lookup_with_fallback(name)
+
+    def _lookup_with_fallback(self, name):
+        try:
+            return self._cache[name]
+        except KeyError:
+            return None
"""


def test_introduced_symbols_finds_new_definition():
    assert "_lookup_with_fallback" in patch_introduced_symbols(GOLD_PATCH)


def test_edited_existing_function_is_not_introduced():
    """`resolve` is edited, not introduced, so pre-fix docs may name it."""
    assert "resolve" not in patch_introduced_symbols(GOLD_PATCH)


def test_short_and_generic_names_are_ignored():
    patch = """\
+def run(self):
+    pass
+def config(self):
+    pass
"""
    assert patch_introduced_symbols(patch) == set()


def test_clean_corpus_produces_no_findings():
    corpus = {
        "saaga-docs/concepts/resolution.md": (
            "The Resolver class exposes `resolve`, which returns a cached entry."
        )
    }
    assert check_instance(corpus, GOLD_PATCH, []) == []


def test_leaked_symbol_is_blocking():
    corpus = {
        "saaga-docs/concepts/resolution.md": (
            "Resolution falls back via `_lookup_with_fallback` when the key is absent."
        )
    }
    findings = check_instance(corpus, GOLD_PATCH, [])
    assert blocking(findings)
    assert findings[0].severity is Severity.BLOCKING
    assert findings[0].token == "_lookup_with_fallback"
    assert findings[0].line_number == 1


def test_test_names_are_extracted_and_flagged():
    names = failing_test_identifiers(["tests/test_resolver.py::TestResolver::test_missing_key_returns_none"])
    assert names == {"test_missing_key_returns_none"}

    corpus = {"saaga-docs/features/resolution.md": "See test_missing_key_returns_none."}
    findings = scan(corpus, set(), names)
    assert [f.kind for f in findings] == ["test-name"]


def test_parametrised_test_ids_are_normalised():
    assert failing_test_identifiers(["a.py::test_widget_flow[case-3]"]) == {"test_widget_flow"}


def test_substring_matches_do_not_count():
    """`\\b` anchoring keeps a longer identifier from tripping a shorter one."""
    corpus = {"doc.md": "the _lookup_with_fallback_v2 helper is unrelated"}
    findings = scan(corpus, {"_lookup_with_fallback"}, set())
    assert findings == []
