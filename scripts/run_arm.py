#!/usr/bin/env python3
"""Run one arm of the 2x2 through AGENTbench's harness.

Integrates without forking AGENTbench, using two hooks it already has:

1. `generate.py` builds its planner config as
   ``deepcopy(ALL_PLAN_CONFIGS[plan_type])``, and `ALL_PLAN_CONFIGS` is a plain
   dict -- so registering an entry before the call is enough.
2. `agentbench.planners.get_planner_class` resolves an unknown spec as a dotted
   import path (``_PLANNER_MAPPING.get(spec, spec)``), so the entry can point
   straight at `saaga_eval.planner.SaagaPlanner`.

Prerequisites::

    git clone https://github.com/eth-sri/agentbench vendor/agentbench
    pip install -e vendor/agentbench
    pip install -e .

Example -- smoke test on a free endpoint, two instances, one worker::

    python scripts/run_arm.py --arm saaga_substitution \\
        --generator miniswe --exec-model glm-4.6 \\
        --slice-spec ":2" --workers 1

Example -- a real pilot arm on Sonnet::

    python scripts/run_arm.py --arm saaga --generator claude \\
        --exec-model claude-sonnet-5 --run-id 0 --workers 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from saaga_eval.arms import ARMS, get_arm, planner_config  # noqa: E402


def register_saaga_planner(corpus_root: str) -> None:
    """Add saaga's plan config to AGENTbench's registry, in place."""
    try:
        from configs import plan_constants
    except ImportError as exc:  # pragma: no cover - environment guidance
        raise SystemExit(
            "Could not import AGENTbench's `configs` package.\n"
            "  git clone https://github.com/eth-sri/agentbench vendor/agentbench\n"
            "  pip install -e vendor/agentbench"
        ) from exc

    for arm in ARMS.values():
        config = planner_config(arm, corpus_root=corpus_root)
        if config is not None:
            plan_constants.ALL_PLAN_CONFIGS[arm.plan_type] = config


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arm", required=True, choices=sorted(ARMS), help="Which cell of the design")
    parser.add_argument("--generator", default="miniswe", help="Agent scaffold (miniswe, claude, codex, ...)")
    parser.add_argument("--exec-model", required=True, help="Model that solves the tasks")
    parser.add_argument("--dataset-name", default="eth-sri/agentbench")
    parser.add_argument("--benchmark", default="agentbench")
    parser.add_argument("--output-dir", default="output/saaga_eval")
    parser.add_argument("--corpus-root", default="corpora")
    parser.add_argument("--run-id", type=int, default=0, help="Seed index; vary it for repeated trials")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--slice-spec", default=None, help='e.g. ":2" to smoke-test two instances')
    parser.add_argument("--filter-spec", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved plan and exit")
    args = parser.parse_args()

    arm = get_arm(args.arm)
    register_saaga_planner(args.corpus_root)

    # Arms differ only where the design says they differ; everything else is held
    # constant so a difference in outcome is attributable to the arm.
    call = dict(
        plan_type=arm.plan_type,
        generator=args.generator,
        exec_model=args.exec_model,
        plan_generator=args.generator,
        plan_model=args.exec_model,
        benchmark=args.benchmark,
        dataset_name=args.dataset_name,
        output_dir=f"{args.output_dir}/{arm.key}",
        workers=args.workers,
        run_id=args.run_id,
        remove_docs=arm.remove_docs,
    )
    if args.slice_spec:
        call["slice_spec"] = args.slice_spec
    if args.filter_spec:
        call["filter_spec"] = args.filter_spec

    print(f"Arm {arm.key}: {arm.label}")
    print(f"  why: {arm.rationale}")
    print(f"  plan_type={arm.plan_type}  remove_docs={arm.remove_docs}")
    for key, value in sorted(call.items()):
        print(f"    {key} = {value!r}")

    if args.dry_run:
        return 0

    try:
        from scripts.agentbench.run_harness.generate import main as generate_main
    except ImportError:
        sys.path.insert(0, str(ROOT / "vendor" / "agentbench"))
        from scripts.agentbench.run_harness.generate import main as generate_main

    generate_main(**call)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
