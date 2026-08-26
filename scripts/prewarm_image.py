#!/usr/bin/env python3
"""Pre-build a setup-complete Docker image so each run doesn't reinstall the world.

The problem
-----------
AGENTbench runs an instance's `setup_commands` inside every fresh container.
For huggingface/smolagents those are::

    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -e .[test]

and the `[test]` extras pull the full ML stack -- torch alone is a 526 MB
wheel. That reliably outruns the 1800s per-exec ceiling in
`AgentBenchInstance.setup`, so the instance dies before the agent is ever
invoked (visible as `TimeoutExpired` with `$0.0000` cost).

Paying it once per repository instead of once per run turns roughly 25 minutes
of downloading into a cached image layer. Across a full 2x2 pilot -- 16
instances x 4 arms x 3 seeds -- that is the difference between about 80 hours
of setup and about 25 minutes.

Note also that `source venv/bin/activate` is inert here: each command is a
separate `docker exec`, so the activation is discarded and everything installs
against system Python. Reproduced faithfully rather than fixed, because the
benchmark's own runs behave this way and a pre-warmed image must match what
`setup_repo=True` would have produced.

Usage::

    python scripts/prewarm_image.py --repo huggingface/smolagents
    python scripts/run_arm.py --arm baseline --exec-model ... --prewarmed
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor" / "agentbench"))
sys.path.insert(0, str(ROOT / "vendor" / "agentbench" / "src"))

PREWARM_PREFIX = "saaga-eval/prewarmed"


def image_tag(repo: str) -> str:
    return f"{PREWARM_PREFIX}:{repo.replace('/', '_').lower()}"


def docker(*args: str, timeout: int | None = None, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args],
        capture_output=capture,
        text=True,
        timeout=timeout,
    )


def image_exists(tag: str) -> bool:
    return docker("image", "inspect", tag).returncode == 0


def prewarm(instance, tag: str, per_command_timeout: int, keep_going: bool) -> bool:
    """Run setup_commands in a container and commit the result."""
    base = instance.docker_image
    print(f"base image : {base}")
    print(f"target tag : {tag}")

    started = docker("run", "-d", "--rm", "-w", "/testbed", base, "sleep", "7200")
    if started.returncode != 0:
        raise SystemExit(f"Could not start container: {started.stderr.strip()}")
    container = started.stdout.strip()
    print(f"container  : {container[:12]}\n")

    try:
        for command in instance.setup_commands:
            print(f"  running: {command}")
            start = time.time()
            try:
                result = docker(
                    "exec", "-w", "/testbed", container, "bash", "-lc", command,
                    timeout=per_command_timeout,
                )
                rc = result.returncode
            except subprocess.TimeoutExpired:
                rc = 124
            elapsed = time.time() - start
            status = "ok" if rc == 0 else f"rc={rc}" + (" (timeout)" if rc == 124 else "")
            print(f"    [{elapsed:6.1f}s] {status}")
            if rc != 0 and not keep_going:
                raise SystemExit(
                    f"\nSetup command failed: {command}\n"
                    "Raise --per-command-timeout, or pass --keep-going to commit anyway."
                )

        print("\n  committing ...")
        committed = docker("commit", container, tag)
        if committed.returncode != 0:
            raise SystemExit(f"Commit failed: {committed.stderr.strip()}")
    finally:
        docker("kill", container)

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="owner/name")
    parser.add_argument("--dataset-name", default="eth-sri/agentbench")
    parser.add_argument("--per-command-timeout", type=int, default=3600,
                        help="Per setup command; the default is deliberately above AGENTbench's 1800s")
    parser.add_argument("--force", action="store_true", help="Rebuild even if the image exists")
    parser.add_argument("--keep-going", action="store_true", help="Commit even if a setup command fails")
    args = parser.parse_args()

    tag = image_tag(args.repo)
    if image_exists(tag) and not args.force:
        print(f"{tag} already exists. Use --force to rebuild.")
        return 0

    from agentbench.benchmarks.agentbench import AgentbenchBenchmark

    key = args.repo.replace("/", "_").lower()
    benchmark = AgentbenchBenchmark(
        dataset_name=args.dataset_name, filter_spec=f"{key}.*", slice_spec=":1", split="train"
    )
    if not benchmark.instances:
        raise SystemExit(f"No instances for {args.repo!r}")

    started = time.time()
    prewarm(benchmark.instances[0], tag, args.per_command_timeout, args.keep_going)
    print(f"\nBuilt {tag} in {(time.time() - started) / 60:.1f} min")
    print("Every later run of this repo reuses it instead of reinstalling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
