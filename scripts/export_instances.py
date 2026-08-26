#!/usr/bin/env python3
"""Export AGENTbench instances to per-repository JSON for corpus building.

`build_corpus.py` needs two things per repository: the base commits of its
instances (to pick a commit no fix has landed at yet) and the patches (to audit
the finished corpus for leakage). Both live in the HuggingFace dataset.

The dataset's column names are not the ones a reader would guess:

    base_sha        not base_commit
    repo            the underscore key ("huggingface_smolagents"), which is what
                    CLEANUP_COMMANDS and the corpus directories are keyed on
    base_repo       the clonable "owner/name" form
    patch           the full PR diff
    clean_pr_patch  the solution with test changes removed
    pr_test_patch   the test changes alone -- the source of new test names

Usage::

    python scripts/export_instances.py --out instances/
    python scripts/export_instances.py --repo huggingface/smolagents
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

PARQUET_URL = (
    "https://huggingface.co/datasets/eth-sri/agentbench/resolve/"
    "refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)

# Columns actually needed downstream. Pulling the whole row would carry
# test_file_contents, which is large and useless here.
COLUMNS = (
    "instance_id",
    "repo",
    "base_repo",
    "base_sha",
    "patch",
    "clean_pr_patch",
    "pr_test_patch",
    "docker_image",
    "test_commands",
)


def repo_key(repo: str) -> str:
    """Idempotent: accepts "owner/name" or an already-underscored key."""
    return repo.replace("/", "_").lower()


def download(cache: Path) -> Path:
    if cache.exists():
        return cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading AGENTbench parquet -> {cache}")
    urllib.request.urlretrieve(PARQUET_URL, cache)
    return cache


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("instances"))
    parser.add_argument("--repo", help="Export only this repo (owner/name)")
    parser.add_argument("--cache", type=Path, default=Path("build/agentbench.parquet"))
    args = parser.parse_args()

    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit("pyarrow is required: pip install pyarrow")

    table = pq.read_table(download(args.cache), columns=list(COLUMNS))
    rows = table.to_pylist()

    grouped: dict[str, list[dict]] = defaultdict(list)
    wanted = repo_key(args.repo) if args.repo else None
    for row in rows:
        # --repo accepts either form; the dataset stores both.
        if wanted and repo_key(row["base_repo"]) != wanted:
            continue
        # Normalise to the names build_corpus.py and the contamination checks use.
        grouped[row["base_repo"]].append(
            {
                "instance_id": row["instance_id"],
                # owner/name for cloning; repo_key for corpus + cleanup lookup.
                "repo": row["base_repo"],
                "repo_key": row["repo"],
                "base_commit": row["base_sha"],
                "patch": row.get("clean_pr_patch") or row.get("patch") or "",
                "test_patch": row.get("pr_test_patch") or "",
                "docker_image": row.get("docker_image"),
                "test_commands": row.get("test_commands"),
            }
        )

    if not grouped:
        available = sorted({r["base_repo"] for r in rows})
        raise SystemExit(
            f"No instances matched {args.repo!r}.\nAvailable:\n  " + "\n  ".join(available)
        )

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"\n{'REPO':<36} {'N':>3}  FILE")
    for repo, instances in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
        path = args.out / f"{repo_key(repo)}.json"
        path.write_text(json.dumps(instances, indent=2), encoding="utf-8")
        print(f"{repo:<36} {len(instances):>3}  {path}")

    print(f"\nTotal: {sum(len(v) for v in grouped.values())} instances across {len(grouped)} repo(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
