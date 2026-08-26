#!/usr/bin/env python3
"""Report the two design contrasts from AGENTbench analyze.py CSV output.

Run AGENTbench's own `analyze.py` per arm first -- it produces the CSV this
reads. Then::

    python scripts/analyze_2x2.py \\
        --baseline out/baseline.csv --saaga out/saaga.csv \\
        --stripped-baseline out/stripped.csv --saaga-substitution out/sub.csv

Partial input is fine: with only C and D it reports the substitution contrast
alone, which is what a pilot produces.

This prints intervals, not verdicts. The threshold that would change saaga's
design is left open in PREREGISTRATION.md for the maintainer to set, and a rule
chosen after seeing the numbers is not a rule.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from saaga_eval.analysis import core_contrasts, load_arm, paired_metric  # noqa: E402

ARM_FLAGS = {
    "baseline": "--baseline",
    "saaga": "--saaga",
    "stripped_baseline": "--stripped-baseline",
    "saaga_substitution": "--saaga-substitution",
}

MECHANISM_METRICS = [
    ("steps_first_read", "steps to first gold-patch file read", "fewer is better"),
    ("steps", "total steps", "fewer is better"),
    ("errors", "tool errors", "fewer is better"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for arm, flag in ARM_FLAGS.items():
        parser.add_argument(flag, dest=arm, type=Path, help=f"analyze.py CSV for arm {arm}")
    args = parser.parse_args()

    arms = {}
    for arm in ARM_FLAGS:
        path = getattr(args, arm)
        if path:
            arms[arm] = load_arm(path)
            print(f"loaded {arm:<22} {len(arms[arm]):>4} instances  ({path})")

    if len(arms) < 2:
        raise SystemExit("\nNeed at least two arms to compare.")

    contrasts = core_contrasts(arms)
    if not contrasts:
        raise SystemExit(
            "\nNo comparable pair. B-A needs baseline+saaga; "
            "D-C needs stripped_baseline+saaga_substitution."
        )

    print("\n" + "=" * 68)
    print("RESOLVE RATE (primary outcome)")
    print("=" * 68)
    for contrast in contrasts:
        print("\n" + contrast.render())

    print("\n" + "=" * 68)
    print("MECHANISM (where an effect would surface first)")
    print("=" * 68)

    pairs = [
        ("saaga", "baseline", "B - A"),
        ("saaga_substitution", "stripped_baseline", "D - C"),
    ]
    for treatment, control, label in pairs:
        if treatment not in arms or control not in arms:
            continue
        print(f"\n{label}")
        for attribute, title, direction in MECHANISM_METRICS:
            outcome = paired_metric(arms[treatment], arms[control], attribute)
            if outcome is None:
                print(f"  {title:<38} no comparable instances")
                continue
            mean, low, high, n = outcome
            crosses = low <= 0 <= high
            verdict = "no detectable effect" if crosses else (
                "saaga better" if mean < 0 else "saaga worse"
            )
            print(
                f"  {title:<38} {mean:+7.2f}  95% CI [{low:+.2f}, {high:+.2f}]  "
                f"n={n:<4} {verdict}  ({direction})"
            )

    print("\n" + "-" * 68)
    print(
        "Intervals crossing zero mean the data does not distinguish the arms at\n"
        "this sample size. That is a result, not a missing one -- both published\n"
        "studies report exactly that for resolve rate.\n"
        "Apply the pre-registered threshold; do not pick one now."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
