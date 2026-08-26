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
    """One instance under one arm.

    `steps_first_read` is AGENTbench's `number_steps_first_read`: how many
    steps the agent took before first opening a file that the gold patch
    touches. It is the most direct measure of the claim saaga actually makes --
    that agents waste turns rediscovering a codebase -- and it can move even
    when resolve rate does not, which both published studies found it doesn't.
    Treat it as the primary *mechanism* outcome.
    """

    instance_id: str
    resolved: bool
    cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    runtime_s: float = 0.0
    status: str = ""
    steps: int | None = None
    steps_first_read: int | None = None
    errors: int = 0
    sys_prompt_size: int = 0

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


def _truthy(value) -> bool:
    """CSV round-trips booleans as text; `bool("False")` is True."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "t"}


def _number(value, cast=float, default=None):
    text = str(value).strip() if value is not None else ""
    if text in {"", "none", "None", "nan", "NaN"}:
        return default
    try:
        return cast(float(text))
    except (TypeError, ValueError):
        return default


def load_arm(path: Path) -> dict[str, InstanceResult]:
    """Read one arm's per-instance results from `analyze.py` CSV output.

    Columns come from AGENTbench's `CSV_COLUMNS`. A JSON file is accepted too,
    for hand-built fixtures. Unparseable rows are skipped rather than silently
    scored as unresolved -- a missing row is missing evidence, not a failure.

    When several rows share an instance_id (repeated seeds via `--run-id`), the
    last one wins here; aggregate seeds with `pool_seeds` instead of relying on
    this.
    """
    path = Path(path)
    if path.suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = list(raw.values()) if isinstance(raw, dict) else list(raw)
    else:
        import csv

        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

    results: dict[str, InstanceResult] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        instance_id = row.get("instance_id") or row.get("id")
        if not instance_id:
            continue
        results[instance_id] = InstanceResult(
            instance_id=instance_id,
            resolved=_truthy(row.get("resolved", row.get("passed", False))),
            cost=_number(row.get("execution_cost", row.get("cost")), float, 0.0) or 0.0,
            input_tokens=_number(row.get("prompt_tokens_first_read"), int, 0) or 0,
            output_tokens=_number(row.get("completion_tokens_first_read"), int, 0) or 0,
            status=str(row.get("status", "") or ""),
            steps=_number(row.get("number_steps"), int),
            steps_first_read=_number(row.get("number_steps_first_read"), int),
            errors=_number(row.get("number_errors"), int, 0) or 0,
            sys_prompt_size=_number(row.get("sys_prompt_size"), int, 0) or 0,
        )
    return results


def paired_metric(
    treatment: dict[str, InstanceResult],
    control: dict[str, InstanceResult],
    attribute: str,
) -> tuple[float, float, float, int] | None:
    """Paired mean difference on a continuous metric, e.g. `steps_first_read`.

    Returns ``(mean_delta, ci_low, ci_high, n)``. Instances where either arm
    lacks the metric are excluded: `number_steps_first_read` is absent when the
    agent never opened a gold-patch file at all, and imputing a value there
    would invent data.
    """
    shared = sorted(set(treatment) & set(control))
    differences = []
    for key in shared:
        left, right = treatment[key], control[key]
        if not (left.usable and right.usable):
            continue
        a, b = getattr(left, attribute), getattr(right, attribute)
        if a is None or b is None:
            continue
        differences.append(float(a) - float(b))

    if not differences:
        return None

    mean = sum(differences) / len(differences)
    low, high = bootstrap_ci(differences)
    return (mean, low, high, len(differences))


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
