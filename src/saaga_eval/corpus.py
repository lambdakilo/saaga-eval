"""Build, cache, and install saaga documentation corpora.

A saaga corpus is expensive: `saaga init` drives a multi-phase agent pipeline
over a whole repository and takes hours. It is therefore built **once per
repository** (on the host, at a pinned base commit) and reused across every
AGENTbench instance belonging to that repository. That is what keeps 12 inits
covering 138 instances instead of 138 inits.

Layout on the host::

    corpora/
      <repo_key>/
        corpus.tar.gz   # saaga-docs/, AGENTS.md, CLAUDE.md, .saagarules
        meta.json       # base commit, saaga version, doc-generation model

`meta.json` is not decoration: the base commit is the contamination control,
and the doc-generation model is a variable that must be reported separately
from the model that solves the tasks.
"""

from __future__ import annotations

import base64
import io
import json
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path

# Everything `saaga init` writes into a project that the agent should see.
# Kept explicit rather than globbed so an unexpected saaga output shows up as a
# missing-file error instead of being silently omitted from the corpus.
SAAGA_ARTIFACTS = (
    "saaga-docs",
    "AGENTS.md",
    "CLAUDE.md",
    ".saagarules",
)

# Chunk size for base64 upload. Docker exec argv limits bite well before this,
# but the environment falls back to stdin for long commands; 48 KiB per chunk
# stays comfortably inside both paths.
_CHUNK_BYTES = 48 * 1024


@dataclass(frozen=True)
class CorpusMeta:
    """Provenance for one repository's corpus. Written alongside the tarball."""

    repo: str
    base_commit: str
    saaga_version: str
    doc_model: str
    backend: str
    artifacts: tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def repo_key(repo: str) -> str:
    """Normalise ``owner/name`` to AGENTbench's ``owner_name`` convention.

    Mirrors `AgentBenchInstance.remove_docs`, which does
    ``self.repo.replace("/", "_").lower()`` to look up cleanup commands. Keeping
    the same key means a corpus directory and a cleanup entry always agree.
    """
    return repo.replace("/", "_").lower()


def corpus_dir(root: Path, repo: str) -> Path:
    return Path(root) / repo_key(repo)


def pack_corpus(source: Path, artifacts: tuple[str, ...] = SAAGA_ARTIFACTS) -> bytes:
    """Tar+gzip saaga's outputs from a checkout. Missing artifacts are skipped.

    Returns the archive bytes. Raises if *nothing* was found, which is the
    signal that `saaga init` did not actually produce a corpus.
    """
    source = Path(source)
    buf = io.BytesIO()
    found: list[str] = []

    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in artifacts:
            path = source / name
            if not path.exists():
                continue
            tar.add(path, arcname=name)
            found.append(name)

    if not found:
        raise FileNotFoundError(
            f"No saaga artifacts found under {source}. Expected any of: "
            f"{', '.join(artifacts)}. Did `saaga init` fail?"
        )

    return buf.getvalue()


def save_corpus(root: Path, meta: CorpusMeta, archive: bytes) -> Path:
    """Persist a packed corpus plus its provenance."""
    target = corpus_dir(root, meta.repo)
    target.mkdir(parents=True, exist_ok=True)
    (target / "corpus.tar.gz").write_bytes(archive)
    (target / "meta.json").write_text(meta.to_json(), encoding="utf-8")
    return target


def load_corpus(root: Path, repo: str) -> tuple[bytes, dict]:
    """Load a repository's corpus and metadata, or fail loudly.

    Failing loudly matters: a silently-missing corpus would turn a saaga arm
    into a duplicate of the baseline arm and quietly produce a null result.
    """
    target = corpus_dir(root, repo)
    archive_path = target / "corpus.tar.gz"
    meta_path = target / "meta.json"

    if not archive_path.exists():
        raise FileNotFoundError(
            f"No saaga corpus for {repo!r} at {archive_path}. "
            f"Build it first: python scripts/build_corpus.py --repo {repo}"
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return archive_path.read_bytes(), meta


def archive_members(archive: bytes) -> list[str]:
    """List the paths inside a packed corpus (used by contamination checks)."""
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        return [m.name for m in tar.getmembers() if m.isfile()]


def read_archive_text(archive: bytes) -> dict[str, str]:
    """Extract every text member of a corpus as ``{path: contents}``."""
    out: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            try:
                out[member.name] = handle.read().decode("utf-8")
            except UnicodeDecodeError:
                continue  # binary asset; contamination checks only scan text
    return out


def install_into_env(env, archive: bytes, remote_tmp: str = "/tmp/saaga_corpus.tar.gz") -> None:
    """Upload a corpus into a running container and unpack it over the repo.

    Uploaded in base64 chunks because a corpus is far larger than a single
    shell command can carry. Each chunk is appended, so a truncated upload
    fails at `tar` rather than silently installing a partial corpus.
    """
    encoded = base64.b64encode(archive).decode("ascii")
    b64_remote = f"{remote_tmp}.b64"

    env.execute(f"rm -f {b64_remote} {remote_tmp}", timeout=False)

    for start in range(0, len(encoded), _CHUNK_BYTES):
        chunk = encoded[start : start + _CHUNK_BYTES]
        env.execute(f"printf '%s' '{chunk}' >> {b64_remote}", timeout=False)

    env.execute(f"base64 -d {b64_remote} > {remote_tmp}", timeout=False)
    result = env.execute(f"tar xzf {remote_tmp} -C .", timeout=False)
    if result.get("returncode", 1) != 0:
        raise RuntimeError(
            f"Failed to unpack saaga corpus in container: {result.get('output', '').strip()}"
        )

    env.execute(f"rm -f {b64_remote} {remote_tmp}", timeout=False)
