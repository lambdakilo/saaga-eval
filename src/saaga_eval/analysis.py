"""Paired analysis of the 2x2, implementing the pre-registered decision rule.

Why paired
----------
Instances differ enormously in difficulty. Comparing arm means treats an easy
instance in one arm and a hard one in the other as equivalent evidence, which
wastes most of the statistical power available. Every arm runs the *same*
instances, so the comparison is within-instance: for each instance, did the
treatment change the outcome? That is what `paired_difference` computes.

Why bootstrap rather than a t-test
----------------------------------
The per-instance outcome is binary (resolved / not), the differences are
therefore in {-1, 0, +1}, and n is 138 at most -- often 16 in a pilot. A normal
approximation is a poor fit at that size. Bootstrapping the paired differences
makes no distributional assumption and degrades gracefully on small samples.

What this deliberately does not do
----------------------------------
It does not pick a threshold. `PREREGISTRATION.md` leaves the effect size that
would change saaga's design open for the maintainer to set, and a decision rule
invented after seeing results is not a decision rule.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

# Fixed so a rerun of the analysis reproduces the same interval. The seed is an
# analysis parameter, not a result; changing it should move CI bounds only in
# the last digits.
_BOOTSTRAP_SEED = 20260826
_BOOTSTRAP_ITERATIONS = 10_000


@dataclass(frozen=True)
class InstanceResult:
    """One instance under one arm."""

    instance_id: str
    resolved: bool
    cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    runtime_s: float = 0.0
    status: str = ""

    @property
    def usable(self) -> bool:
        """Whether this run produced a real verdict.

        A harness failure -- container death, an agent command outrunning the
        1800s docker-exec ceiling -- is not evidence that the task was
        unsolvable. Counting it as an unresolved instance would silently
        penalise whichever arm happened to hit more infrastructure trouble.
        """
        return self.status.lower() not in {
            "timeoutexpired",
            "ratelimiterror",
            "error",
            "setupfailed",
        }


@dataclass(frozen=True)
class Comparison:
    """A paired contrast between two arms."""

    treatment: str
    control: str
    n_paired: int
    n_dropped: int
    treatment_resolved: int
    control_resolved: int
    delta_pp: float
    ci_low_pp: float
    ci_high_pp: float
    cost_ratio: float | None

    @property
    def crosses_zero(self) -> bool:
        return self.ci_low_pp <= 0.0 <= self.ci_high_pp

    def render(self) -> str:
        direction = "no detectable effect" if self.crosses_zero else (
            "favours treatment" if self.delta_pp > 0 else "favours control"
        )
        cost = (
            f"  cost x{self.cost_ratio:.2f}" if self.cost_ratio is not None else "  cost n/a"
        )
        dropped = f"  ({self.n_dropped} dropped)" if self.n_dropped else ""
        return (
            f"{self.treatment} - {self.control}\n"
            f"  resolved   {self.treatment_resolved}/{self.n_paired} vs "
            f"{self.control_resolved}/{self.n_paired}{dropped}\n"
            f"  delta      {self.delta_pp:+.1f}pp  "
            f"95% CI [{self.ci_low_pp:+.1f}, {self.ci_high_pp:+.1f}]  -> {direction}\n"
            f"{cost}"
        )


def bootstrap_ci(
    differences: list[float],
    iterations: int = _BOOTSTRAP_ITERATIONS,
    seed: int = _BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of paired differences."""
    if not differences:
        return (0.0, 0.0)

    rng = random.Random(seed)
    n = len(differences)
    means = []
    for _ in range(iterations):
        sample = [differences[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return (means[int(0.025 * iterations)], means[int(0.975 * iterations) - 1])


def paired_difference(
    treatment: dict[str, InstanceResult],
    control: dict[str, InstanceResult],
    treatment_name: str,
    control_name: str,
) -> Comparison:
    """Compare two arms on the instances both resolved a verdict for."""
    shared = sorted(set(treatment) & set(control))
    usable = [i for i in shared if treatment[i].usable and control[i].usable]
    dropped = len(shared) - len(usable)

    differences = [
        float(treatment[i].resolved) - float(control[i].resolved) for i in usable
    ]
    mean = (sum(differences) / len(differences)) if differences else 0.0
    low, high = bootstrap_ci(differences)

    treatment_cost = sum(treatment[i].cost for i in usable)
    control_cost = sum(control[i].cost for i in usable)
    ratio = (treatment_cost / control_cost) if control_cost > 0 else None

    return Comparison(
        treatment=treatment_name,
        control=control_name,
        n_paired=len(usable),
        n_dropped=dropped,
        treatment_resolved=sum(1 for i in usable if treatment[i].resolved),
        control_resolved=sum(1 for i in usable if control[i].resolved),
        delta_pp=mean * 100,
        ci_low_pp=low * 100,
        ci_high_pp=high * 100,
        cost_ratio=ratio,
    )


def load_arm(report_path: Path) -> dict[str, InstanceResult]:
    """Read an arm's per-instance results from AGENTbench report JSON.

    Tolerant of shape: the harness writes per-instance `report.json` files and
    an aggregate, and field names have moved between versions. Anything
    unparseable is skipped rather than silently scored as unresolved.
    """
    raw = json.loads(Path(report_path).read_text(encoding="utf-8"))
    rows = raw.values() if isinstance(raw, dict) else raw

    results: dict[str, InstanceResult] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        instance_id = row.get("instance_id") or row.get("id")
        if not instance_id:
            continue
        results[instance_id] = InstanceResult(
            instance_id=instance_id,
            resolved=bool(row.get("resolved", row.get("passed", False))),
            cost=float(row.get("cost", 0.0) or 0.0),
            input_tokens=int(row.get("input_tokens", 0) or 0),
            output_tokens=int(row.get("output_tokens", 0) or 0),
            runtime_s=float(row.get("runtime", 0.0) or 0.0),
            status=str(row.get("status", "") or ""),
        )
    return results


def core_contrasts(arms: dict[str, dict[str, InstanceResult]]) -> list[Comparison]:
    """The two contrasts the design exists to produce.

    B - A  saaga as an addition     (literature predicts negative, +20% cost)
    D - C  saaga as a substitution  (the untested claim)
    """
    contrasts = []
    if "saaga" in arms and "baseline" in arms:
        contrasts.append(paired_difference(arms["saaga"], arms["baseline"], "saaga (B)", "baseline (A)"))
    if "saaga_substitution" in arms and "stripped_baseline" in arms:
        contrasts.append(
            paired_difference(
                arms["saaga_substitution"],
                arms["stripped_baseline"],
                "saaga_substitution (D)",
                "stripped_baseline (C)",
            )
        )
    return contrasts
