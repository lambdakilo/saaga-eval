"""End-to-end behaviour of the saaga planner against a real filesystem.

The test that matters most here is `test_corpus_survives_remove_docs`: without
the wrapper it guards, the docs-stripped saaga arm silently measures an empty
documentation tree and reports a clean null.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from saaga_eval.corpus import CorpusMeta, install_into_env, load_corpus, pack_corpus, save_corpus
from saaga_eval.planner import SaagaPlanner

REPO = "acme/widgets"

CLEAN_PATCH = """\
diff --git a/widgets/core.py b/widgets/core.py
--- a/widgets/core.py
+++ b/widgets/core.py
@@ -1,3 +1,6 @@
+def compute_widget_checksum(data):
+    return sum(data)
"""


class FakeInstance:
    """Stands in for `AgentBenchInstance` with the attributes the planner uses."""

    def __init__(self, patch: str = "", fail_to_pass: list[str] | None = None) -> None:
        self.instance_id = "acme__widgets-1"
        self.repo = REPO
        self.patch = patch
        self.fail_to_pass = fail_to_pass or []
        self.removed_agents_md = False

    def remove_agents_md_files(self, env) -> None:
        self.removed_agents_md = True
        env.execute(r'find . -type f \( -name "AGENTS.md" -o -name "CLAUDE.md" \) -delete')

    def remove_docs(self, env) -> None:
        """Mirrors AGENTbench: preserve only the two root files, delete all .md."""
        agents = env.read_file("AGENTS.md")
        claude = env.read_file("CLAUDE.md")
        env.execute('find . -name "*.md" -delete')
        if agents:
            env.write_file("AGENTS.md", agents)
        if claude:
            env.write_file("CLAUDE.md", claude)


@pytest.fixture
def corpus_root(tmp_path: Path, saaga_checkout: Path) -> Path:
    root = tmp_path / "corpora"
    archive = pack_corpus(saaga_checkout)
    save_corpus(
        root,
        CorpusMeta(
            repo=REPO,
            base_commit="deadbeef",
            saaga_version="1.0.0-alpha.6",
            doc_model="claude-opus-5",
            backend="claude",
            artifacts=("saaga-docs", "AGENTS.md", "CLAUDE.md", ".saagarules"),
        ),
        archive,
    )
    return root


def test_pack_and_load_round_trip(corpus_root: Path):
    archive, meta = load_corpus(corpus_root, REPO)
    assert archive
    assert meta["base_commit"] == "deadbeef"
    assert meta["doc_model"] == "claude-opus-5"


def test_missing_corpus_fails_loudly(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No saaga corpus"):
        load_corpus(tmp_path / "empty", REPO)


def test_pack_rejects_a_checkout_with_no_saaga_output(tmp_path: Path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="Did `saaga init` fail"):
        pack_corpus(empty)


def test_install_reassembles_chunked_upload(env, corpus_root: Path):
    archive, _ = load_corpus(corpus_root, REPO)
    install_into_env(env, archive)

    assert (env.cwd / "saaga-docs" / "concepts" / "scope.md").exists()
    assert (env.cwd / "AGENTS.md").exists()
    assert (env.cwd / ".saagarules").exists()
    assert not (env.cwd / "/tmp/saaga_corpus.tar.gz.b64").exists()


def test_plan_installs_corpus_and_clears_repo_context_files(env, corpus_root: Path):
    instance = FakeInstance(patch=CLEAN_PATCH)
    (env.cwd / "AGENTS.md").write_text("stale repo file", encoding="utf-8")

    SaagaPlanner(corpus_root=str(corpus_root)).plan(env=env, model=None, instance=instance)

    assert instance.removed_agents_md
    assert (env.cwd / "saaga-docs" / "INDEX.md").exists()
    assert "Read saaga-docs first" in (env.cwd / "AGENTS.md").read_text()


def test_corpus_survives_remove_docs(env, corpus_root: Path):
    """Arm D: repository docs gone, saaga corpus intact.

    Without the planner's wrapper, `find . -name "*.md" -delete` removes the
    whole corpus and leaves a rules block pointing at nothing.
    """
    instance = FakeInstance(patch=CLEAN_PATCH)
    (env.cwd / "README.md").write_text("repo readme", encoding="utf-8")

    planner = SaagaPlanner(corpus_root=str(corpus_root))
    planner.plan(env=env, model=None, instance=instance)
    instance.remove_docs(env)

    assert not (env.cwd / "README.md").exists(), "repo docs should be stripped"
    assert (env.cwd / "saaga-docs" / "INDEX.md").exists(), "corpus must survive"
    assert (env.cwd / "saaga-docs" / "concepts" / "scope.md").exists()
    assert (env.cwd / "AGENTS.md").exists()


def test_unwrapped_remove_docs_would_destroy_the_corpus(env, corpus_root: Path):
    """Documents the hazard the wrapper exists to prevent."""
    archive, _ = load_corpus(corpus_root, REPO)
    install_into_env(env, archive)

    FakeInstance().remove_docs(env)  # no planner, so no protection

    assert not (env.cwd / "saaga-docs" / "INDEX.md").exists()
    assert (env.cwd / "AGENTS.md").exists(), "the misleading part: rules survive"


def test_contaminated_corpus_aborts_the_instance(env, corpus_root: Path, saaga_checkout: Path):
    leaked = "The helper compute_widget_checksum sums the payload.\n"
    (saaga_checkout / "saaga-docs" / "concepts" / "scope.md").write_text(leaked, encoding="utf-8")
    save_corpus(
        corpus_root,
        CorpusMeta(REPO, "deadbeef", "1.0.0-alpha.6", "claude-opus-5", "claude", ()),
        pack_corpus(saaga_checkout),
    )

    with pytest.raises(RuntimeError, match="Contaminated corpus"):
        SaagaPlanner(corpus_root=str(corpus_root)).plan(
            env=env, model=None, instance=FakeInstance(patch=CLEAN_PATCH)
        )


def test_contamination_can_be_downgraded_to_a_warning(env, corpus_root: Path, saaga_checkout: Path):
    (saaga_checkout / "saaga-docs" / "INDEX.md").write_text(
        "compute_widget_checksum\n", encoding="utf-8"
    )
    save_corpus(
        corpus_root,
        CorpusMeta(REPO, "deadbeef", "1.0.0-alpha.6", "claude-opus-5", "claude", ()),
        pack_corpus(saaga_checkout),
    )

    planner = SaagaPlanner(corpus_root=str(corpus_root), fail_on_contamination=False)
    planner.plan(env=env, model=None, instance=FakeInstance(patch=CLEAN_PATCH))
    assert (env.cwd / "saaga-docs" / "INDEX.md").exists()


def test_planner_tolerates_harness_injected_kwargs(corpus_root: Path):
    """`generate.py` injects generator/model config into every planner config."""
    planner = SaagaPlanner(
        corpus_root=str(corpus_root),
        storage_dir="output/plans",
        plan_model="unused",
        generator_config={"name": "miniswe"},
        model_config={"model": "glm-5.2"},
        prompt_type="ignored-by-saaga",
    )
    assert planner.config.corpus_root == str(corpus_root)


def test_meta_records_the_saaga_commit(tmp_path, saaga_checkout):
    """Version alone cannot identify the build; alpha releases reuse it."""
    import json

    root = tmp_path / "corpora"
    save_corpus(
        root,
        CorpusMeta(
            repo=REPO,
            base_commit="deadbeef",
            saaga_version="1.0.0-alpha.6",
            doc_model="haiku",
            backend="claude",
            artifacts=("saaga-docs",),
            saaga_commit="b9c902f-dirty",
        ),
        pack_corpus(saaga_checkout),
    )
    meta = json.loads((root / "acme_widgets" / "meta.json").read_text())
    assert meta["saaga_commit"] == "b9c902f-dirty"
    assert meta["doc_model"] == "haiku"


def test_meta_commit_is_optional_for_registry_installs():
    """An npm install has no commit; that absence is itself worth recording."""
    meta = CorpusMeta(REPO, "deadbeef", "1.0.0-alpha.6", "haiku", "claude", ())
    assert meta.saaga_commit is None
