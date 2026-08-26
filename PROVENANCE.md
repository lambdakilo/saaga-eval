# Provenance

What exactly was evaluated. A benchmark result is only as traceable as the
artefact that produced it, and saaga is alpha software whose version string does
not uniquely identify a build.

## saaga is pinned to a locally built, security-audited checkout

saaga was **not** installed from the npm registry. It was built from a local
checkout and installed from there:

```bash
cd /path/to/saaga
pnpm install --frozen-lockfile
pnpm build
npm i -g .
```

| | |
| --- | --- |
| Version | `1.0.0-alpha.6` |
| Commit | `b9c902f` |
| Source | local checkout, not `npm i -g @wonna/saaga` |
| Backend | `claude` (shells out to the `claude` CLI) |

### Why

The local checkout is the tree that was **read and security-audited before any
of this work started** — its permission model, backend adapters, sandbox probes,
and the closed script registry that keeps arbitrary shell out of its flow DSL.
The registry artefact is a different thing: same version string, separately
built, not reviewed.

Since this project runs saaga against twelve third-party repositories and feeds
the output to coding agents, evaluating the reviewed code rather than an
unreviewed download was the conservative choice. It also removes a variable —
if a result looks strange, "which build was that?" has an exact answer.

The tradeoff is that `1.0.0-alpha.6 @ b9c902f` may not be identical to
`1.0.0-alpha.6` on npm. Anyone reproducing this should build from that commit
rather than installing from the registry, and any comparison against a registry
install should be reported as a different configuration rather than a rerun.

Verification that the audited tree is sound: `pnpm install --frozen-lockfile`
executed **no** dependency build scripts, and saaga's own suite passes — 48 test
files, 473 tests.

## Recorded per corpus

`scripts/build_corpus.py` writes a `meta.json` beside every corpus tarball:

```json
{
  "repo": "huggingface/smolagents",
  "base_commit": "<earliest instance base commit>",
  "saaga_version": "1.0.0-alpha.6",
  "saaga_commit": "b9c902f...",
  "doc_model": "haiku",
  "backend": "claude",
  "artifacts": ["saaga-docs", "AGENTS.md", "CLAUDE.md", ".saagarules"]
}
```

`saaga_commit` exists because `saaga_version` is not sufficient: a local
checkout and the published release both report `1.0.0-alpha.6`. It is resolved
by following the `saaga` shim to its real path and asking git, and carries a
`-dirty` suffix when the checkout has uncommitted changes. A registry install
has no commit, and that absence is recorded as `null` rather than guessed at.

## Doc-generation model is a separate variable

The model saaga uses to *write* documentation is independent of the model that
*solves* tasks. Both are recorded; only the solver is held constant across arms.

Corpora built with `--doc-model haiku` exist to prove the pipeline runs
end-to-end at low cost, and to stay inside a Claude Pro plan's rolling usage
window, which a heavier model can exhaust mid-init. **A Haiku-generated corpus
is plumbing evidence, not evidence about saaga.** Any result that would inform a
decision about saaga needs a corpus built with the model a real user would use.

If `saaga init` itself fails under a cheap model, that is a result about saaga
on that model — not a harness defect. Report the two separately.

## Benchmark substrate

| | |
| --- | --- |
| Harness | [`eth-sri/agentbench`](https://github.com/eth-sri/agentbench), MIT, vendored unmodified |
| Dataset | `eth-sri/agentbench` on HuggingFace, 138 instances, 12 repositories |
| Integration | no fork; registered through `ALL_PLAN_CONFIGS` and the dotted-path planner resolver |

Keeping upstream unmodified means their fixes remain available and the diff a
reviewer has to read is only this repository.
