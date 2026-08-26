# Pre-registration

**Status: draft. Not agreed with saaga's maintainer yet.**

This document exists to be settled *before* any run. The prior in the literature
is unfavourable to saaga's approach, which means an unflattering result is
likely — and a result nobody agreed to in advance gets relitigated on
methodology instead of acted on. Fixing the decision rule up front is what makes
the outcome usable either way.

Anything below marked **[OPEN]** needs the maintainer's input.

## Hypotheses

**H1 (addition).** Adding a saaga corpus to a repository that already has
documentation does not improve task success, and increases cost.

> Directional prediction: `B − A` ≈ −2% success, +20% cost.
> Source: Gloaguen et al., arXiv 2602.11988.

**H2 (substitution).** When saaga's corpus is the only documentation present, it
improves task success relative to having no documentation at all.

> Directional prediction: `D − C` > 0.
> Source: the ETH ablation, where LLM-generated files improved results by 2.7%
> and beat developer-written files once repository docs were removed.

H2 is the reason this study is worth running. H1 is close to a replication and
mainly serves to show the harness reproduces known results.

## Primary outcomes

1. **Resolve rate** — fraction of instances passing fail-to-pass tests.
2. **Cost per instance** — total inference cost, input tokens included.

Both are reported for every arm. Cost is not secondary: the ETH paper's headline
finding was a cost regression with no success gain, and a harness that reports
only pass/fail cannot detect that.

## Secondary outcomes

- Wall-clock runtime per instance (comparability with Lulla et al.)
- Output tokens per instance
- **Steps to first gold-patch file read** — AGENTbench already computes this as
  `number_steps_first_read`, so it costs nothing to collect. It is the most
  direct probe of the "agents waste turns rediscovering the codebase" claim,
  and the measure most likely to move even if resolve rate does not. Given that
  both published studies found no resolve-rate effect, **this is where a real
  saaga effect would show up first**, and it should be read as the primary
  mechanism outcome rather than a nice-to-have.

  Note it is undefined when the agent never opens a gold-patch file. Those
  instances are excluded from the paired comparison rather than imputed —
  see `src/saaga_eval/analysis.py`.

## Decision rule **[OPEN]**

Proposed, subject to the maintainer's agreement:

| Outcome | Reading |
| ------- | ------- |
| `D − C` ≥ +3pp, non-overlapping 95% bootstrap CI | Substitution hypothesis supported |
| `D − C` within ±3pp | No detectable effect; report as null |
| `D − C` ≤ −3pp | saaga's corpus underperforms no documentation |
| `B − A` ≤ −2pp **and** cost +15% or worse | Replicates the published addition penalty |

±3pp is a placeholder chosen to be detectable at the planned sample size, not a
claim about what matters practically. **The maintainer should set the threshold
they would actually act on** — if a 3pp gain would not change saaga's design or
its README, the threshold is wrong.

## Sampling

- **Instances:** all 138 AGENTbench instances, minus any excluded by the
  contamination audit (exclusions listed with reasons before results are read).
- **Seeds:** 3 runs per instance per arm. The original study used one; that is
  underpowered for a single-digit effect.
- **Pilot:** one repository first, all four arms, full seed count — to validate
  plumbing and produce an effect estimate before committing to the full grid.

## Held constant

- Solver model and scaffold, identical across arms
- Turn and token budgets, identical across arms — an arm with more compute wins
  for reasons unrelated to documentation
- Container image, base commit, and test command per instance
- One corpus per repository, built once and reused across that repository's
  instances

## Reported separately, not held constant

- **Doc-generation model** — the model saaga uses to write documentation is a
  different variable from the model that solves tasks. A corpus written by Opus 5
  and evaluated on Sonnet 5 is a legitimate configuration, but it must be
  labelled as one.
- **saaga version** — alpha, and its own README warns re-initialisation will be
  needed. Recorded in each corpus's `meta.json`.

## Contamination controls

1. Corpus built at the **earliest** base commit across the repository's
   instances, verified with `git merge-base --is-ancestor`.
2. Automated audit: symbols *introduced* by each gold patch, plus fail-to-pass
   test names, must not appear in the corpus. Blocking by default.
3. Git history scrubbed by the harness (`_clean_git_history`).
4. **[OPEN]** Network egress during solve. Containers currently run
   `--network=host`, so an agent can fetch the upstream fix from GitHub. This
   affects all arms equally but adds variance. Proposal: block egress during the
   solve phase.

## Publication commitment

The result is published as-is, including a null or a result unfavourable to
saaga. Instance exclusions and their reasons are published alongside. Raw
trajectories and per-instance metrics are released so the analysis can be
checked independently.

## Open questions for the maintainer **[OPEN]**

1. Is `--rule-targets agentsmd,claude` with a full `saaga-docs/` tree the
   configuration you would want measured, or is there a preferred setup?
2. Which model should generate the documentation? saaga's `--model high=` makes
   this a free parameter and it plausibly matters more than the solver model.
3. Does the ETH finding that *repository overviews are "not helpful"* change how
   you would want the corpus structured for this test?
4. What effect size would actually change your mind — in either direction?
