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


TEST_PATCH = """\
diff --git a/tests/test_models.py b/tests/test_models.py
--- a/tests/test_models.py
+++ b/tests/test_models.py
@@ -1,2 +1,8 @@
+def test_chatmessage_from_dict_role_conversion():
+    assert True
+
+def _build_fixture():
+    return None
"""


def test_test_patch_symbols_keeps_only_test_functions():
    """AGENTbench ships pr_test_patch rather than a fail_to_pass list."""
    from saaga_eval.contamination import test_patch_symbols

    assert test_patch_symbols(TEST_PATCH) == {"test_chatmessage_from_dict_role_conversion"}


def test_check_instance_accepts_a_test_patch():
    corpus = {"saaga-docs/x.md": "covered by test_chatmessage_from_dict_role_conversion"}
    findings = check_instance(corpus, GOLD_PATCH, test_patch=TEST_PATCH)
    assert [f.kind for f in findings] == ["test-name"]


def test_test_patch_helper_additions_are_not_markers():
    """A helper the test patch adds is not distinctive enough to flag on."""
    corpus = {"saaga-docs/x.md": "see _build_fixture for setup"}
    assert check_instance(corpus, GOLD_PATCH, test_patch=TEST_PATCH) == []


DUNDER_PATCH = """\
+++ b/pkg/usage.py
@@
+    def __post_init__(self):
+        self.total = self.a + self.b
+    def reconcile_token_ledger(self):
+        return self.total
"""


def test_dunders_are_not_contamination_markers():
    """Protocol names recur in every codebase; adding one proves nothing."""
    symbols = patch_introduced_symbols(DUNDER_PATCH)
    assert "__post_init__" not in symbols
    assert "reconcile_token_ledger" in symbols


def test_docs_mentioning_dunders_are_not_flagged():
    """The real false positive: an audit of a smolagents corpus reported only these."""
    corpus = {
        "saaga-docs/concepts/token-accounting.md":
            "`TokenUsage.__post_init__()` automatically calculates total_tokens",
        "saaga-docs/ARCHITECTURE.md": "`BaseTool` exposes name and __call__",
    }
    assert check_instance(corpus, DUNDER_PATCH, []) == []


def test_a_real_leak_alongside_dunders_still_fires():
    """Suppressing dunders must not suppress the signal next to them."""
    corpus = {"saaga-docs/x.md": "__post_init__ then reconcile_token_ledger runs"}
    findings = check_instance(corpus, DUNDER_PATCH, [])
    assert [f.token for f in findings] == ["reconcile_token_ledger"]


CLEAN_ADDITION = """\
+++ b/widgets/core.py
@@
+def compute_widget_checksum(data):
+    return sum(data)
"""


WORD_PATCH = """\
+++ b/pkg/retry.py
@@
+class Retrying:
+    pass
"""


def test_prose_use_of_a_word_like_symbol_is_review_not_blocking():
    """Real case: a class named Retrying vs the English word."""
    corpus = {"saaga-docs/patterns/error-handling.md":
              "Retrying invalid API keys won't help"}
    findings = check_instance(corpus, WORD_PATCH, [])
    assert [f.severity for f in findings] == [Severity.REVIEW]
    assert blocking(findings) == []


def test_code_context_still_blocks():
    for line in ("see `Retrying` for details",
                 "call Retrying(max=3)",
                 "from pkg.retry import Retrying",
                 "handler = Retrying"):
        findings = check_instance({"d.md": line}, WORD_PATCH, [])
        assert blocking(findings), f"should block on: {line}"


def test_leaked_symbol_in_backticks_remains_blocking():
    corpus = {"saaga-docs/x.md": "Resolution uses `_lookup_with_fallback` here."}
    assert blocking(check_instance(corpus, GOLD_PATCH, []))


def test_composed_identifiers_block_even_in_prose():
    """compute_widget_checksum cannot occur as English; prose is still a leak."""
    corpus = {"saaga-docs/x.md": "The helper compute_widget_checksum sums the payload."}
    assert blocking(check_instance(corpus, CLEAN_ADDITION, []))


def test_camelcase_identifiers_block_even_in_prose():
    patch = "+++ b/m.py\n+class TokenUsage:\n+    pass\n"
    corpus = {"saaga-docs/x.md": "TokenUsage tracks totals across a run."}
    assert blocking(check_instance(corpus, patch, []))


def test_markdown_bold_colon_is_not_an_assignment():
    """Verbatim from a real corpus; the colon belongs to markdown, not code."""
    line = "- **Retry is not always right**: Retrying invalid API keys won't help"
    findings = check_instance({"saaga-docs/patterns/error-handling.md": line}, WORD_PATCH, [])
    assert [f.severity for f in findings] == [Severity.REVIEW]
    assert blocking(findings) == []


def test_real_assignment_of_a_word_like_symbol_still_blocks():
    assert blocking(check_instance({"d.md": "handler = Retrying"}, WORD_PATCH, []))
    assert blocking(check_instance({"d.md": "policy: Retrying = build()"}, WORD_PATCH, []))
