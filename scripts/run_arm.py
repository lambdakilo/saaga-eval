#!/usr/bin/env python3
"""Run one arm of the 2x2 through AGENTbench's harness.

Integrates without forking AGENTbench, using hooks it already has:

1. `generate.py` builds its configs by key lookup into plain dicts --
   ``ALL_PLAN_CONFIGS[plan_type]``, ``ALL_MODEL_CONFIGS[exec_model]`` -- so
   registering entries before the call is enough.
2. `agentbench.planners.get_planner_class` resolves an unknown spec as a dotted
   import path (``_PLANNER_MAPPING.get(spec, spec)``), so a plan config can
   point straight at `saaga_eval.planner.SaagaPlanner`.

Prerequisites::

    git clone https://github.com/eth-sri/agentbench vendor/agentbench
    pip install -e vendor/agentbench
    pip install -e .

Smoke run on NVIDIA NIM's free tier, two instances, one worker::

    export NVIDIA_NIM_API_KEY=nvapi-...
    python scripts/run_arm.py --arm saaga_substitution \\
        --exec-model nim:zai/glm-5.2 --slice-spec ":2" --workers 1

A real pilot arm on Sonnet::

    python scripts/run_arm.py --arm saaga --generator claude_code \\
        --exec-model sonnet-4-5 --run-id 0 --workers 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor" / "agentbench"))
sys.path.insert(0, str(ROOT / "vendor" / "agentbench" / "src"))

from saaga_eval.arms import ARMS, Arm, get_arm, planner_config  # noqa: E402
from saaga_eval.models import maybe_register  # noqa: E402

# Planners that generate a context file need a model to generate it with.
# Everything in the 2x2 either installs a prebuilt corpus or installs nothing,
# so passing a plan model would start a second, entirely idle model server.
_PLANNERS_NEEDING_A_MODEL = {"claude_planner", "codex_planner", "qwen_planner", "gemini_planner"}


def _configs():
    try:
        from configs import plan_constants
        from configs.generator_constants import ALL_GENERATOR_CONFIGS
    except ImportError as exc:  # pragma: no cover - environment guidance
        raise SystemExit(
            "Could not import AGENTbench's `configs` package.\n"
            "  git clone https://github.com/eth-sri/agentbench vendor/agentbench\n"
            "  pip install -e vendor/agentbench"
        ) from exc
    return plan_constants, ALL_GENERATOR_CONFIGS


def register_saaga_planner(plan_constants, corpus_root: str) -> None:
    """Add saaga's plan configs to AGENTbench's registry, in place."""
    for arm in ARMS.values():
        config = planner_config(arm, corpus_root=corpus_root)
        if config is not None:
            plan_constants.ALL_PLAN_CONFIGS[arm.plan_type] = config


def build_call(arm: Arm, args, exec_model: str) -> dict:
    """Assemble generate.main kwargs.

    Everything not named by the design is held constant across arms, so a
    difference in outcome is attributable to the arm rather than the setup.
    """
    call = dict(
        plan_type=arm.plan_type,
        exec_model=exec_model,
        generator=args.generator,
        benchmark=args.benchmark,
        dataset_name=args.dataset_name,
        output_dir=f"{args.output_dir}/{arm.key}",
        workers=args.workers,
        run_id=args.run_id,
        remove_docs=arm.remove_docs,
        slice_spec=args.slice_spec,
    )

    # Leaving plan_model None makes generate.py reuse exec_model as a label
    # without starting a second model server on port+1.
    if arm.plan_type in _PLANNERS_NEEDING_A_MODEL:
        call["plan_model"] = args.plan_model or exec_model

    if arm.has_saaga:
        call["plan_args"] = {"corpus_root": args.corpus_root}
    if args.filter_spec:
        call["filter_spec"] = args.filter_spec
    return call


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument(
        "--exec-model",
        required=True,
        help="AGENTbench model key, or nim:<model_id>, or "
        "openai-compat:<base_url>::<model_id> to auto-register",
    )
    parser.add_argument("--generator", default="miniswe_agents", help="Agent scaffold key")
    parser.add_argument("--plan-model", default=None, help="Calibration arms only")
    parser.add_argument("--dataset-name", default="eth-sri/agentbench")
    parser.add_argument("--benchmark", default="agentbench")
    parser.add_argument("--output-dir", default="output/saaga_eval")
    parser.add_argument("--corpus-root", default="corpora")
    parser.add_argument("--run-id", type=int, default=0, help="Seed index; vary for repeated trials")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--slice-spec", default="0:200", help='e.g. ":2" to smoke-test two instances')
    parser.add_argument("--filter-spec", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print the config, run nothing")
    args = parser.parse_args()

    arm = get_arm(args.arm)
    plan_constants, all_generators = _configs()

    if args.generator not in all_generators:
        raise SystemExit(
            f"Unknown generator {args.generator!r}. Available: {', '.join(sorted(all_generators))}"
        )

    register_saaga_planner(plan_constants, args.corpus_root)
    exec_model = maybe_register(args.exec_model)
    call = build_call(arm, args, exec_model)

    print(f"Arm {arm.key}: {arm.label}")
    print(f"  why: {arm.rationale}")
    for key, value in sorted(call.items()):
        print(f"    {key} = {value!r}")

    if arm.has_saaga:
        corpus_root = Path(args.corpus_root)
        if not corpus_root.exists():
            print(
                f"\n  note: {corpus_root}/ does not exist yet. This arm needs a corpus:\n"
                f"        python scripts/build_corpus.py --repo <owner/name> --backend claude"
            )

    if args.dry_run:
        return 0

    from scripts.agentbench.run_harness.generate import main as generate_main

    generate_main(**call)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
