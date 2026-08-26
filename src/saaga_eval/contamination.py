"""Detect leakage of gold-patch content into a saaga documentation corpus.

Why this is not optional
------------------------
`saaga init` reads a whole repository to write its documentation. If it runs at
a commit that already contains an instance's fix, the resulting docs can name
the fixed symbol, describe the new behaviour, or -- worst case -- restate the
patch. The agent then "solves" the task by reading the answer out of the
documentation, and saaga wins the benchmark without helping anyone.

That failure is silent and it flatters the tool being measured, which is the
worst combination: nobody reviewing the result would believe it, and they would
be right not to.

What counts as contamination
----------------------------
Not every shared identifier. A corpus is *supposed* to describe code that exists
at the base commit -- that is its entire job. The signal is narrower:

    a symbol that the gold patch INTRODUCES, appearing in documentation
    generated before that patch was applied

so this module diffs the patch against itself (added lines minus removed lines)
and only flags symbols that are genuinely new, plus the fail-to-pass test names,
which should never appear in pre-fix documentation at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

# Definition sites worth tracking. Attribute and variable assignments are
# deliberately excluded -- too noisy to carry signal.
_DEF_PATTERN = re.compile(
    r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)",
)

# Identifiers this short, or this generic, match everywhere and only produce
# noise. A real leak virtually always involves a distinctive name.
_MIN_SYMBOL_LENGTH = 5
_STOPLIST = frozenset(
    {
        "setup",
        "teardown",
        "value",
        "result",
        "index",
        "items",
        "param",
        "params",
        "error",
        "errors",
        "config",
        "client",
        "server",
        "parse",
        "build",
        "create",
        "delete",
        "update",
        "handle",
        "process",
        "render",
        "format",
        "encode",
        "decode",
    }
)


class Severity(str, Enum):
    """How much a finding should stop the run."""

    BLOCKING = "blocking"  # drop the instance; the corpus names the answer
    REVIEW = "review"      # plausible leak, needs a human look


@dataclass(frozen=True)
class Finding:
    """One piece of gold-patch content located inside the corpus."""

    severity: Severity
    kind: str          # "introduced-symbol" | "test-name"
    token: str
    doc_path: str
    line_number: int
    line: str

    def format(self) -> str:
        return (
            f"[{self.severity.value}] {self.kind} {self.token!r} "
            f"in {self.doc_path}:{self.line_number}\n    {self.line.strip()}"
        )


def _definitions(lines: list[str]) -> set[str]:
    found = set()
    for line in lines:
        match = _DEF_PATTERN.match(line)
        if match:
            found.add(match.group(1))
    return found


def patch_introduced_symbols(patch: str) -> set[str]:
    """Symbols the gold patch adds and does not merely move.

    Definitions present on both sides of the diff are excluded: a patch that
    edits the body of an existing function should not make that function's name
    a contamination marker, because pre-fix docs may legitimately describe it.
    """
    added_lines: list[str] = []
    removed_lines: list[str] = []

    for raw in patch.splitlines():
        if raw.startswith("+++") or raw.startswith("---"):
            continue
        if raw.startswith("+"):
            added_lines.append(raw[1:])
        elif raw.startswith("-"):
            removed_lines.append(raw[1:])

    introduced = _definitions(added_lines) - _definitions(removed_lines)

    return {
        symbol
        for symbol in introduced
        if len(symbol) >= _MIN_SYMBOL_LENGTH and symbol.lower() not in _STOPLIST
    }


def failing_test_identifiers(fail_to_pass: list[str]) -> set[str]:
    """Bare test-function names from AGENTbench FAIL_TO_PASS entries.

    Entries look like ``tests/test_mod.py::TestCase::test_thing``; the final
    component is the discriminating part.
    """
    names = set()
    for entry in fail_to_pass or []:
        tail = entry.split("::")[-1].strip()
        tail = re.sub(r"\[.*\]$", "", tail)  # strip pytest parametrisation
        if tail and len(tail) >= _MIN_SYMBOL_LENGTH:
            names.add(tail)
    return names


def scan(
    corpus_texts: dict[str, str],
    introduced_symbols: set[str],
    test_names: set[str],
) -> list[Finding]:
    """Search a corpus for introduced symbols and test names.

    Matching is whole-word and case-sensitive: documentation prose that happens
    to contain a symbol as a substring of a longer identifier is not a leak.
    """
    findings: list[Finding] = []

    targets: list[tuple[str, str, Severity]] = [
        *((token, "introduced-symbol", Severity.BLOCKING) for token in introduced_symbols),
        *((token, "test-name", Severity.BLOCKING) for token in test_names),
    ]
    if not targets:
        return findings

    compiled = [
        (re.compile(rf"\b{re.escape(token)}\b"), token, kind, severity)
        for token, kind, severity in targets
    ]

    for doc_path, text in sorted(corpus_texts.items()):
        for line_number, line in enumerate(text.splitlines(), start=1):
            for pattern, token, kind, severity in compiled:
                if pattern.search(line):
                    findings.append(
                        Finding(
                            severity=severity,
                            kind=kind,
                            token=token,
                            doc_path=doc_path,
                            line_number=line_number,
                            line=line,
                        )
                    )

    return findings


def test_patch_symbols(test_patch: str) -> set[str]:
    """Test functions a PR's test patch introduces.

    AGENTbench has no `fail_to_pass` column; it ships `pr_test_patch`, the test
    changes alone. Tests added by the fix are the strongest contamination
    marker there is -- a corpus generated before the fix has no legitimate
    reason to name one.

    Only `test`-prefixed definitions are kept, so a helper the test patch also
    adds does not widen the net.
    """
    return {
        symbol
        for symbol in patch_introduced_symbols(test_patch)
        if symbol.lower().startswith("test")
    }


def check_instance(
    corpus_texts: dict[str, str],
    patch: str,
    fail_to_pass: list[str] | None = None,
    test_patch: str | None = None,
) -> list[Finding]:
    """Full contamination check for one instance against one corpus.

    `fail_to_pass` and `test_patch` are alternative sources for the same
    signal: SWE-bench-style datasets carry the former, AGENTbench the latter.
    """
    test_names = failing_test_identifiers(fail_to_pass or [])
    if test_patch:
        test_names |= test_patch_symbols(test_patch)

    return scan(corpus_texts, patch_introduced_symbols(patch), test_names)


def blocking(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.severity is Severity.BLOCKING]
