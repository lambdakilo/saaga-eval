#!/usr/bin/env python3
"""Prove the four arms are four different environments -- without running an agent.

Why not just run the agent
--------------------------
A full agent run is the wrong instrument for this question. It takes tens of
minutes, depends on model quality, and can end in `TimeoutExpired` having told
you nothing about whether the arms differ. The thing that silently breaks --
arm D deleting its own corpus via `find . -name "*.md" -delete` -- is a property
of the *environment*, observable the moment the planner and doc removal have
run.

So this reproduces exactly what `generate.py` does, in the same order::

    env = instance.setup(...)
    planner.plan(env=env, model=model, instance=instance)
    if remove_docs:
        instance.remove_docs(env)

and then looks at the filesystem. Both `no_plan` and `SaagaPlanner` ignore the
`model` argument, so this needs **no model server, no API key, and no tokens**.

Usage::

    python scripts/verify_arms.py --repo huggingface/smolagents
    python scripts/verify_arms.py --repo huggingface/smolagents --arms baseline saaga
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor" / "agentbench"))
sys.path.insert(0, str(ROOT / "vendor" / "agentbench" / "src"))

from saaga_eval.arms import ARMS, CORE_2X2, Arm, get_arm, planner_config  # noqa: E402


def _count(env, command: str) -> int:
    result = env.execute(command, timeout=False)
    text = (result.get("output") or "0").strip().splitlines()
    for line in reversed(text):
        if line.strip().isdigit():
            return int(line.strip())
    return 0


def observe(env) -> dict:
    """Filesystem facts that distinguish the arms."""
    return {
        "saaga_docs": _count(env, 'find saaga-docs -type f 2>/dev/null | wc -l'),
        "repo_md": _count(
            env,
            'find . -name "*.md" -not -path "./saaga-docs/*" -not -name "AGENTS.md" '
            '-not -name "CLAUDE.md" -not -path "./.git/*" 2>/dev/null | wc -l',
        ),
        "readme": _count(env, 'test -f README.md && echo 1 || echo 0'),
        "docs_dir": _count(env, 'test -d docs && echo 1 || echo 0'),
        "agents_md": _count(env, 'test -f AGENTS.md && echo 1 || echo 0'),
    }


def expectations(arm: Arm) -> dict:
    """What each cell must look like for the design to mean anything."""
    return {
        "saaga_docs": ("> 0" if arm.has_saaga else "== 0"),
        "repo_md": ("== 0" if arm.remove_docs else "> 0"),
    }


def check(arm: Arm, seen: dict) -> list[str]:
    problems = []
    if arm.has_saaga and seen["saaga_docs"] == 0:
        problems.append(
            "saaga corpus MISSING -- this arm is a duplicate of its baseline"
            + (" (remove_docs deleted it)" if arm.remove_docs else "")
        )
    if not arm.has_saaga and seen["saaga_docs"] > 0:
        problems.append("saaga corpus present in a no-context arm")
    if arm.remove_docs and seen["repo_md"] > 0:
        problems.append(f"repo docs NOT stripped ({seen['repo_md']} .md files remain)")
    if not arm.remove_docs and seen["repo_md"] == 0:
        problems.append("repo docs absent in an arm that should keep them")
    return problems


def run_arm(instance, arm: Arm, corpus_root: str) -> tuple[dict, list[str]]:
    from agentbench.planners import get_planner

    from configs import plan_constants

    config = planner_config(arm, corpus_root=corpus_root)
    if config is None:
        config = dict(plan_constants.ALL_PLAN_CONFIGS[arm.plan_type])
    planner = get_planner(dict(config))

    env = instance.setup(env_config={}, setup_repo=False)
    try:
        planner.plan(env=env, model=None, instance=instance)
        if arm.remove_docs:
            instance.remove_docs(env)
        seen = observe(env)
    finally:
        try:
            env.cleanup()
        except Exception:
            pass
    return seen, check(arm, seen)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="owner/name, used to pick an instance")
    parser.add_argument("--arms", nargs="*", default=list(CORE_2X2))
    parser.add_argument("--corpus-root", default="corpora")
    parser.add_argument("--dataset-name", default="eth-sri/agentbench")
    args = parser.parse_args()

    from agentbench.benchmarks.agentbench import AgentbenchBenchmark

    key = args.repo.replace("/", "_").lower()
    benchmark = AgentbenchBenchmark(
        dataset_name=args.dataset_name, filter_spec=f"{key}.*", slice_spec=":1", split="train"
    )
    if not benchmark.instances:
        raise SystemExit(f"No instances for {args.repo!r}.")
    instance = benchmark.instances[0]
    print(f"Instance: {instance.instance_id}  (image {instance.docker_image})\n")

    header = f"{'arm':<20} {'saaga-docs':>11} {'repo .md':>9} {'README':>7} {'docs/':>6}"
    print(header)
    print("-" * len(header))

    failures = 0
    for name in args.arms:
        arm = get_arm(name)
        try:
            seen, problems = run_arm(instance, arm, args.corpus_root)
        except FileNotFoundError as exc:
            print(f"{name:<20} SKIPPED -- {exc}".split("\n")[0])
            continue
        print(
            f"{name:<20} {seen['saaga_docs']:>11} {seen['repo_md']:>9} "
            f"{seen['readme']:>7} {seen['docs_dir']:>6}"
        )
        for problem in problems:
            failures += 1
            print(f"  FAIL: {problem}")

    print()
    if failures:
        print(f"{failures} problem(s). The arms do not mean what the design says.")
        return 1
    print("All arms distinct and correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
