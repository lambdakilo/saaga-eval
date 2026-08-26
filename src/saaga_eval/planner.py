"""An AGENTbench planner that installs a saaga documentation corpus.

Registers without patching AGENTbench. `agentbench.planners.get_planner_class`
resolves an unknown spec as a dotted import path::

    full_path = _PLANNER_MAPPING.get(spec, spec)

so ``"planner_class": "saaga_eval.planner.SaagaPlanner"`` is enough.

How this differs from `InitPlanner`
-----------------------------------
`InitPlanner` generates one AGENTS.md per instance, inline, during the run.
saaga generates a whole documentation tree per *repository*, out of band, over
hours. So this planner never generates anything at plan time: it installs a
corpus that `scripts/build_corpus.py` produced earlier, and refuses to run if
that corpus is missing or contaminated.

The `remove_docs` hazard
------------------------
`generate.py` calls the planner first and strips documentation second::

    planner.plan(env=env, model=model, instance=instance)
    if remove_docs:
        instance.remove_docs(env)

`AgentBenchInstance.remove_docs` backs up only ``AGENTS.md`` and ``CLAUDE.md``
before running per-repo cleanup commands that include
``find . -name "*.md" -delete``. Every saaga corpus is Markdown, so in the
docs-stripped saaga arm the corpus would be deleted while the rules block that
points at it survives -- an arm that looks healthy and measures nothing.

`_protect_corpus` wraps `instance.remove_docs` to stash and restore the corpus
around that cleanup, so the arm means what it claims: saaga's documentation
present, the repository's own documentation gone.
"""

from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from saaga_eval.contamination import blocking, check_instance
from saaga_eval.corpus import archive_members, install_into_env, load_corpus, read_archive_text

logger = logging.getLogger("saaga_eval.planner")

_STASH_DIR = "/tmp/saaga_corpus_stash"


@dataclass
class SaagaPlannerConfig:
    """Configuration for `SaagaPlanner`.

    `generator_config`, `model_config` and `plan_model` are accepted because the
    harness injects them into every planner config; saaga does not use them,
    since the corpus is built ahead of time on the host.
    """

    corpus_root: str = "corpora"
    fail_on_contamination: bool = True
    storage_dir: str = "output/plans"
    plan_model: str = "saaga"
    generator_config: dict[str, Any] = field(default_factory=dict)
    model_config: dict[str, Any] = field(default_factory=dict)


class SaagaPlanner:
    """Installs a prebuilt saaga corpus into an instance's container."""

    def __init__(self, **kwargs: Any) -> None:
        known = {f for f in SaagaPlannerConfig.__dataclass_fields__}
        unknown = set(kwargs) - known
        if unknown:
            logger.debug("Ignoring unused planner kwargs: %s", ", ".join(sorted(unknown)))
        self.config = SaagaPlannerConfig(**{k: v for k, v in kwargs.items() if k in known})

    def get_name(self) -> str:
        return "saaga"

    def get_template_vars(self) -> dict[str, Any]:
        return {}

    def plan(self, env, model, instance) -> None:
        repo = instance.repo
        instance_id = instance.instance_id

        archive, meta = load_corpus(Path(self.config.corpus_root), repo)
        self._assert_clean(archive, instance)

        # Drop the repository's own context files so the only agent-facing
        # instructions are saaga's. Matches what every other planner does.
        if hasattr(instance, "remove_agents_md_files"):
            instance.remove_agents_md_files(env)

        install_into_env(env, archive)
        members = archive_members(archive)
        logger.info(
            "Installed saaga corpus for %s (%d files, base commit %s) into %s",
            repo,
            len(members),
            meta.get("base_commit", "unknown"),
            instance_id,
        )

        self._protect_corpus(instance, members)

    def update_plan(self, **kwargs: Any) -> None:
        pass

    def _assert_clean(self, archive: bytes, instance) -> None:
        """Refuse to run an instance whose answer is sitting in the corpus."""
        findings = check_instance(
            corpus_texts=read_archive_text(archive),
            patch=getattr(instance, "patch", "") or "",
            fail_to_pass=_fail_to_pass(instance),
        )
        hard = blocking(findings)
        if not hard:
            return

        report = "\n".join(f.format() for f in hard[:10])
        message = (
            f"Contaminated corpus for {instance.instance_id}: the documentation "
            f"names {len(hard)} symbol(s) introduced by the gold patch.\n{report}"
        )
        if self.config.fail_on_contamination:
            raise RuntimeError(message)
        logger.warning(message)

    @staticmethod
    def _protect_corpus(instance, members: list[str]) -> None:
        """Make the corpus survive `instance.remove_docs` in the stripped arm."""
        original = getattr(instance, "remove_docs", None)
        if original is None or getattr(original, "_saaga_wrapped", False):
            return

        # Restore whole top-level entries; tar recreates the tree beneath them.
        roots = sorted({member.split("/", 1)[0] for member in members})
        quoted = " ".join(shlex.quote(root) for root in roots)

        def remove_docs_preserving_corpus(env, *args: Any, **kwargs: Any):
            # Stash only what is actually on disk. The planner may have been
            # bound to this instance during an earlier arm that installed a
            # corpus into a *different* container, and a stale wrapper must not
            # break an arm that legitimately has no corpus to preserve.
            present = [
                root
                for root in roots
                if env.execute(
                    f"test -e {shlex.quote(root)}", timeout=False
                ).get("returncode", 1) == 0
            ]
            if not present:
                logger.debug("No saaga corpus present; removing docs unprotected")
                return original(env, *args, **kwargs)

            env.execute(f"mkdir -p {_STASH_DIR}", timeout=False)
            stashed = env.execute(
                "tar czf %s/corpus.tgz %s"
                % (_STASH_DIR, " ".join(shlex.quote(p) for p in present)),
                timeout=False,
            )
            if stashed.get("returncode", 1) != 0:
                raise RuntimeError(
                    "Could not stash saaga corpus before doc removal: "
                    f"{stashed.get('output', '').strip()}"
                )

            result = original(env, *args, **kwargs)

            restored = env.execute(f"tar xzf {_STASH_DIR}/corpus.tgz -C .", timeout=False)
            if restored.get("returncode", 1) != 0:
                raise RuntimeError(
                    "Could not restore saaga corpus after doc removal: "
                    f"{restored.get('output', '').strip()}"
                )
            env.execute(f"rm -rf {_STASH_DIR}", timeout=False)
            logger.info("Preserved saaga corpus (%s) across remove_docs", ", ".join(roots))
            return result

        remove_docs_preserving_corpus._saaga_wrapped = True
        instance.remove_docs = remove_docs_preserving_corpus


def _fail_to_pass(instance) -> list[str]:
    """Best-effort read of an instance's fail-to-pass tests across shapes."""
    for attr in ("fail_to_pass", "FAIL_TO_PASS"):
        value = getattr(instance, attr, None)
        if value:
            return list(value) if not isinstance(value, str) else _maybe_json_list(value)
    return []


def _maybe_json_list(value: str) -> list[str]:
    import json

    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return [value]
    return parsed if isinstance(parsed, list) else [value]
