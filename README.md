# saaga-eval

Does AI-generated repository documentation actually help coding agents?

[saaga](https://github.com/wonna-fi/saaga) pre-generates a structured documentation
corpus (`concepts/`, `patterns/`, `features/`) and installs always-on rules telling
agents to read those docs before reading source. This repository measures whether
that works, using [`eth-sri/agentbench`](https://github.com/eth-sri/agentbench) —
138 instances across 12 recent, niche Python repositories — as the substrate.

It exists because the question has already been studied twice, with results that
do not obviously favour saaga's approach, and one specific gap that nobody has
tested.

## What the literature already says

**Gloaguen et al., [_Evaluating AGENTS.md_](https://arxiv.org/abs/2602.11988)**
(ETH Zurich / LogicStar.ai) built AGENTbench precisely to avoid the objection that
benchmark repositories are memorised, then found that context files
*"do not generally improve task success rates, while increasing inference cost by
over 20% on average."* LLM-generated files scored ~2% **worse**; developer-written
files ~4% better. They single out repository overviews — roughly what saaga
produces — as *"not helpful"*, and recommend omitting LLM-generated context files.

**Lulla et al., [_On the Impact of AGENTS.md Files on Efficiency_](https://arxiv.org/abs/2601.20404)**
(ICSE JAWs 2026) ran 124 paired pull requests and found the opposite sign on
efficiency: **−28.64% median runtime, −16.58% output tokens** — but explicitly
*"comparable task completion."*

They disagree about cost and agree about correctness: neither found a real
task-success gain. So the prior going in is unfavourable, and this harness is
built to be able to report that cleanly.

## The gap this measures

Both papers tested a **single flat context file**. saaga ships a large indexed
corpus *and* a rules block instructing agents to treat it as authoritative and
source as a second resort. That is a **substitution** claim, not an addition
claim — and the ETH ablation is suggestive: with the repository's own
documentation removed, LLM-generated context files *"not only consistently
improve performance by 2.7% on average, but also outperform developer-written
documentation."*

If saaga's rules block reproduces that condition without deleting anything, it
should show up as a gain in the stripped cell. Nobody has tested it.

## Design

|                     | repo docs present | repo docs stripped |
| ------------------- | ----------------- | ------------------ |
| **no context file** | **A** baseline    | **C** stripped baseline |
| **saaga corpus**    | **B** saaga       | **D** substitution |

- **B − A** — saaga as an *addition*. Literature predicts ≈ −2% success, +20% cost.
- **D − C** — saaga as a *substitution*. The untested claim.

**C is not optional.** Without a stripped baseline, a gain in D cannot be
separated from the effect of deleting the repository's documentation.

Two calibration arms (`init_calibration`, `human_calibration`) reproduce the
original paper's arms. If those do not roughly reproduce published direction on
your model, the pipeline is wrong and no saaga number from it should be believed.

See [PREREGISTRATION.md](PREREGISTRATION.md) for the decision rule — agree it
before running, not after.

## A hazard worth knowing about

`generate.py` plans first and strips documentation second:

```python
planner.plan(env=env, model=model, instance=instance)
if remove_docs:
    instance.remove_docs(env)
```

`AgentBenchInstance.remove_docs` preserves only `AGENTS.md` and `CLAUDE.md`, then
runs per-repo cleanup that includes `find . -name "*.md" -delete`. Every saaga
corpus is Markdown. Left alone, **arm D would delete the entire corpus and keep
the rules block pointing at files that no longer exist** — an arm that looks
healthy and measures nothing.

`SaagaPlanner._protect_corpus` wraps `remove_docs` to stash and restore the
corpus around that cleanup. `tests/test_planner.py` covers both the fix and the
hazard it prevents.

## Setup

```bash
git clone https://github.com/eth-sri/agentbench vendor/agentbench
python -m venv .venv && source .venv/bin/activate
pip install -e vendor/agentbench
pip install -e ".[dev]"
```

Docker is required — AGENTbench provisions a container per instance.

### 1. Build a corpus (once per repository)

`saaga init` runs on the **host**, not in Docker, at a pinned base commit. saaga
shells out to an agent CLI and never reads `ANTHROPIC_API_KEY`, so with
`--backend claude` this runs on an existing Claude Code login — including a
Pro/Max subscription. Only the benchmark runs need API credit.

```bash
python scripts/build_corpus.py --repo jlowin/fastmcp \
    --instances-json instances/jlowin_fastmcp.json \
    --backend claude
```

The commit choice is the contamination control: the script defaults to the
**earliest** base commit across that repository's instances, verified with
`git merge-base --is-ancestor`, so no instance's fix has landed yet. It then
audits the corpus against every instance and names any that leak.

Expect hours — saaga's own README calls `init` "the heaviest command".

### 2. Run an arm

```bash
python scripts/run_arm.py --arm saaga_substitution \
    --generator claude --exec-model claude-sonnet-5 \
    --run-id 0 --workers 4
```

`--dry-run` prints the resolved configuration without spending anything.

### 3. Smoke test first

```bash
./scripts/smoke_test.sh
```

Asserts on mechanics, never pass rates. Results from a cheap model do not
transfer to an expensive one, but "are these four arms actually four different
experiments?" transfers perfectly — and that is the failure you want to find
before the spend, not after.

For a free smoke run, point `--generator miniswe` at any OpenAI-compatible
endpoint (NVIDIA NIM, vLLM, OpenRouter). mini-SWE-agent takes a `base_url`
directly, which is a much shorter path than configuring a vendor CLI. Keep
`--workers` at 1–2 if the endpoint is rate-limited.

## Cost

Rough, order-of-magnitude, at Sonnet 5 rates ($3/$15 per MTok):

| Scope | Runs | Estimate |
| ----- | ---- | -------- |
| Smoke test (2 instances, 2 arms, free endpoint) | 4 | ~$0 |
| Pilot (1 repo, 4 arms, 3 seeds) | ~144 | **~$150–250** |
| Full study (12 repos, 4 arms, 3 seeds) | ~1,656 | **~$1.5–4k** |

Plus one `saaga init` per repository. Seeds matter more than model tier here: the
effect under test is a couple of percent, and the original study
**sampled each instance once** (*"We sample completions for each agent once"*),
which is underpowered against agentic run-to-run variance. Multi-seed replication
is a real methodological improvement independent of what it finds about saaga.

## Status

Working: the planner adapter, the 2×2 arm registry, corpus build/pack/install,
contamination detection, and the arm-D corpus protection — all covered by tests
(`pytest`, 18 passing).

Not done: no corpus has been built, no arm has been run, and no result exists.
Nothing here claims anything about saaga yet.

## Integration approach

No fork of AGENTbench. Two existing hooks carry it:

- `get_planner_class` resolves unknown specs as dotted import paths
  (`_PLANNER_MAPPING.get(spec, spec)`), so `saaga_eval.planner.SaagaPlanner`
  registers itself.
- `generate.py` reads `ALL_PLAN_CONFIGS[plan_type]`, a plain dict that
  `scripts/run_arm.py` adds an entry to at import time.

Upstream stays a pinned, unmodified dependency, which keeps this reviewable and
keeps their fixes available.

## Licence

MIT, matching AGENTbench.
