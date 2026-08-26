"""The 2x2 experimental design, plus calibration arms.

The question is not "does saaga help?" but *which of two different claims* is
true, because they have opposite priors in the literature:

                         | repo docs present | repo docs stripped
    ---------------------+-------------------+--------------------
    no context file      | A  baseline       | C  stripped_baseline
    saaga corpus         | B  saaga          | D  saaga_substitution

* **B - A** measures saaga as an *addition*. Gloaguen et al. (arXiv 2602.11988)
  found LLM-generated context files here cost 20-23% more and score ~2% worse.
* **D - C** measures saaga as a *substitution*. That paper's ablation found
  LLM-generated files help by 2.7% and beat developer-written ones *when they
  are the only documentation present*. saaga's rules block instructs agents to
  treat its docs as authoritative and source as a second resort, which is a
  substitution claim. Nobody has tested it.

C is not optional. Without a stripped baseline, a gain in D cannot be separated
from the effect of deleting the repository's documentation.

Calibration arms
----------------
`init_calibration` and `human_calibration` reproduce the original paper's arms.
Running them on the same model as the saaga arms answers "is our harness
behaving like theirs?" before anyone spends money on the full grid. If these do
not roughly reproduce published direction, the pipeline is wrong and no saaga
result from it should be believed.
"""

from __future__ import annotations

from dataclasses import dataclass

SAAGA_PLANNER_CLASS = "saaga_eval.planner.SaagaPlanner"


@dataclass(frozen=True)
class Arm:
    """One cell of the design."""

    key: str
    label: str
    plan_type: str
    remove_docs: bool
    rationale: str

    @property
    def has_saaga(self) -> bool:
        return self.plan_type == "saaga_planner"

    def cli_flags(self) -> list[str]:
        flags = ["--plan_type", self.plan_type]
        if self.remove_docs:
            flags.append("--remove_docs")
        return flags


ARMS: dict[str, Arm] = {
    "baseline": Arm(
        key="baseline",
        label="A - no context file, repo docs intact",
        plan_type="no_plan",
        remove_docs=False,
        rationale="Reference point for B. The repository as a contributor finds it.",
    ),
    "saaga": Arm(
        key="saaga",
        label="B - saaga corpus, repo docs intact",
        plan_type="saaga_planner",
        remove_docs=False,
        rationale="Realistic deployment: saaga added on top of existing docs.",
    ),
    "stripped_baseline": Arm(
        key="stripped_baseline",
        label="C - no context file, repo docs stripped",
        plan_type="no_plan",
        remove_docs=True,
        rationale="Isolates the effect of removing docs, so D-C attributes gains to saaga.",
    ),
    "saaga_substitution": Arm(
        key="saaga_substitution",
        label="D - saaga corpus, repo docs stripped",
        plan_type="saaga_planner",
        remove_docs=True,
        rationale="The untested claim: saaga's docs as the single source of truth.",
    ),
    "init_calibration": Arm(
        key="init_calibration",
        label="Calibration - one-shot LLM AGENTS.md (paper's arm)",
        plan_type="claude_planner",
        remove_docs=False,
        rationale="Harness sanity check against published results. Not part of the 2x2.",
    ),
    "human_calibration": Arm(
        key="human_calibration",
        label="Calibration - developer-committed context file (paper's arm)",
        plan_type="human_planner",
        remove_docs=False,
        rationale="Harness sanity check against published results. Not part of the 2x2.",
    ),
}

CORE_2X2 = ("baseline", "saaga", "stripped_baseline", "saaga_substitution")


def planner_config(arm: Arm, corpus_root: str = "corpora") -> dict | None:
    """Planner config override for saaga arms; None means use AGENTbench's own.

    Returned as a dotted path so AGENTbench resolves it through its existing
    `_PLANNER_MAPPING.get(spec, spec)` fallback -- no fork required.
    """
    if not arm.has_saaga:
        return None
    return {
        "planner_class": SAAGA_PLANNER_CLASS,
        "corpus_root": corpus_root,
        "fail_on_contamination": True,
        "storage_dir": "output/plans",
    }


def get_arm(key: str) -> Arm:
    try:
        return ARMS[key]
    except KeyError:
        raise SystemExit(
            f"Unknown arm {key!r}. Available: {', '.join(ARMS)}"
        ) from None
