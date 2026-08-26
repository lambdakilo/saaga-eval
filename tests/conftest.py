"""A container stand-in that runs commands for real in a temp directory.

Mocking `env.execute` would let the base64 chunking and the tar round-trip pass
without ever being exercised -- exactly the parts most likely to break on a
large corpus. This runs the same shell commands the Docker environment would.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


class LocalEnv:
    """Implements enough of AGENTbench's `Environment` protocol to test against."""

    def __init__(self, cwd: Path) -> None:
        self.cwd = Path(cwd)
        self.commands: list[str] = []

    def execute(self, command: str, cwd: str = "", timeout=True) -> dict:
        self.commands.append(command)
        completed = subprocess.run(
            ["bash", "-c", command],
            cwd=str(self.cwd),
            capture_output=True,
            text=True,
        )
        return {
            "returncode": completed.returncode,
            "output": completed.stdout + completed.stderr,
        }

    def read_file(self, path: str) -> str:
        target = self.cwd / path
        return target.read_text(encoding="utf-8") if target.exists() else ""

    def write_file(self, path: str, contents: str) -> None:
        target = self.cwd / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents, encoding="utf-8")


@pytest.fixture
def env(tmp_path: Path) -> LocalEnv:
    workdir = tmp_path / "testbed"
    workdir.mkdir()
    return LocalEnv(workdir)


@pytest.fixture
def saaga_checkout(tmp_path: Path) -> Path:
    """A checkout that looks like one `saaga init` has already run against."""
    root = tmp_path / "checkout"
    (root / "saaga-docs" / "concepts").mkdir(parents=True)
    (root / "saaga-docs" / "patterns").mkdir(parents=True)

    (root / "AGENTS.md").write_text("# Rules\nRead saaga-docs first.\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("# Rules\nRead saaga-docs first.\n", encoding="utf-8")
    (root / ".saagarules").write_text("docs-first\n", encoding="utf-8")
    (root / "saaga-docs" / "INDEX.md").write_text("# Index\n", encoding="utf-8")
    (root / "saaga-docs" / "concepts" / "scope.md").write_text(
        "Scope carries values between steps.\n", encoding="utf-8"
    )
    (root / "saaga-docs" / "patterns" / "backends.md").write_text(
        "Adding a backend means implementing the adapter.\n", encoding="utf-8"
    )
    return root
