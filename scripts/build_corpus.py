#!/usr/bin/env python3
"""Build one repository's saaga documentation corpus, on the host.

Runs outside Docker on purpose. `saaga init` drives an agent CLI, and the
`claude` backend authenticates through whatever `claude` is already logged into
-- including a Pro/Max subscription. saaga itself never reads `ANTHROPIC_API_KEY`;
it shells out. So corpus building can run on a subscription while the benchmark
runs on API credit.

Choosing the commit is the whole ballgame
-----------------------------------------
Documentation generated at a commit that already contains an instance's fix can
name the fix, and the agent then reads the answer out of the docs. This script
defaults to the **earliest base commit** across the repository's instances, so
no instance's solution has landed yet, and verifies that choice with
`git merge-base --is-ancestor` rather than trusting dataset ordering.

Usage::

    python scripts/build_corpus.py --repo jlowin/fastmcp \\
        --instances-json instances/jlowin_fastmcp.json \\
        --backend claude

    python scripts/build_corpus.py --repo acme/widgets --base-commit abc123
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from saaga_eval.contamination import blocking, check_instance  # noqa: E402
from saaga_eval.corpus import (  # noqa: E402
    CorpusMeta,
    SAAGA_ARTIFACTS,
    pack_corpus,
    read_archive_text,
    save_corpus,
)


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> str:
    result = subprocess.run(
        cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True
    )
    if check and result.returncode != 0:
        raise SystemExit(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n{result.stderr.strip()}"
        )
    return result.stdout


def is_ancestor(repo_dir: Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=str(repo_dir),
        capture_output=True,
    )
    return result.returncode == 0


def earliest_commit(repo_dir: Path, commits: list[str]) -> str:
    """The commit that is an ancestor of every other. Fails if none is.

    A repository whose instances sit on divergent branches has no single safe
    commit, and silently picking one would produce a corpus that is ahead of
    some instances. Better to stop and let a human decide.
    """
    unique = sorted(set(commits))
    if len(unique) == 1:
        return unique[0]

    for candidate in unique:
        others = [c for c in unique if c != candidate]
        if all(is_ancestor(repo_dir, candidate, other) for other in others):
            return candidate

    raise SystemExit(
        "No single commit is an ancestor of all instance base commits "
        f"({len(unique)} distinct). Build per-branch corpora, or pass "
        "--base-commit explicitly and document the choice."
    )


def load_instances(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data.get("instances", [])


def saaga_version() -> str:
    try:
        return run(["saaga", "--version"], check=False).strip() or "unknown"
    except FileNotFoundError:
        raise SystemExit("`saaga` not found on PATH. Install it: npm i -g @wonna/saaga")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="owner/name, e.g. jlowin/fastmcp")
    parser.add_argument("--base-commit", help="Pin explicitly; otherwise derived from --instances-json")
    parser.add_argument("--instances-json", type=Path, help="Instances for this repo (commit choice + audit)")
    parser.add_argument("--corpus-root", type=Path, default=Path("corpora"))
    parser.add_argument("--workdir", type=Path, default=Path("build"))
    parser.add_argument("--backend", default="claude", choices=["claude", "cursor", "copilot"])
    parser.add_argument("--doc-model", help="Passed to saaga as --model high=<model>")
    parser.add_argument("--keep-checkout", action="store_true", help="Leave the checkout for inspection")
    parser.add_argument("--skip-init", action="store_true", help="Repack an existing checkout without rerunning saaga")
    args = parser.parse_args()

    instances = load_instances(args.instances_json) if args.instances_json else []
    checkout = args.workdir / args.repo.replace("/", "_")

    if not args.skip_init:
        if checkout.exists():
            shutil.rmtree(checkout)
        checkout.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning {args.repo} ...")
        run(["git", "clone", f"https://github.com/{args.repo}.git", str(checkout)])

    base_commit = args.base_commit
    if not base_commit:
        commits = [i["base_commit"] for i in instances if i.get("base_commit")]
        if not commits:
            raise SystemExit("Need --base-commit or --instances-json containing base_commit values.")
        base_commit = earliest_commit(checkout, commits)
        print(f"Earliest base commit across {len(commits)} instance(s): {base_commit}")

    run(["git", "checkout", "--detach", base_commit], cwd=checkout)

    if not args.skip_init:
        cmd = ["saaga", "init", "--backend", args.backend]
        if args.doc_model:
            cmd += ["--model", f"high={args.doc_model}"]
        print(f"Running: {' '.join(cmd)}\n(this takes hours -- saaga's own README calls init the heavy one)")
        result = subprocess.run(cmd, cwd=str(checkout))
        if result.returncode != 0:
            raise SystemExit(f"`saaga init` failed with exit code {result.returncode}")

    archive = pack_corpus(checkout)
    meta = CorpusMeta(
        repo=args.repo,
        base_commit=base_commit,
        saaga_version=saaga_version(),
        doc_model=args.doc_model or "backend-default",
        backend=args.backend,
        artifacts=SAAGA_ARTIFACTS,
    )
    target = save_corpus(args.corpus_root, meta, archive)
    print(f"Saved corpus -> {target} ({len(archive) / 1024:.1f} KiB gzipped)")

    if instances:
        texts = read_archive_text(archive)
        contaminated = 0
        for instance in instances:
            findings = blocking(
                check_instance(
                    texts,
                    instance.get("patch", ""),
                    instance.get("fail_to_pass") or instance.get("FAIL_TO_PASS") or [],
                )
            )
            if findings:
                contaminated += 1
                print(f"\nCONTAMINATED {instance.get('instance_id', '?')}:")
                for finding in findings[:3]:
                    print("  " + finding.format().replace("\n", "\n  "))
        print(
            f"\nContamination audit: {contaminated}/{len(instances)} instance(s) affected."
            + ("  Exclude them or rebuild at an earlier commit." if contaminated else "  Clean.")
        )

    if not args.keep_checkout and not args.skip_init:
        shutil.rmtree(checkout, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
