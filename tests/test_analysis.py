"""The analysis must not manufacture an effect, and must not hide one.

Both failure modes are cheap to introduce and expensive to notice later, so
each is pinned here with a constructed dataset whose right answer is known.
"""

from __future__ import annotations

from saaga_eval.analysis import (
    InstanceResult,
    bootstrap_ci,
    core_contrasts,
    paired_difference,
)


def arm(spec: dict[str, bool], cost: float = 1.0, status: str = "") -> dict[str, InstanceResult]:
    return {
        key: InstanceResult(instance_id=key, resolved=value, cost=cost, status=status)
        for key, value in spec.items()
    }


def test_identical_arms_report_no_effect():
    same = {f"i{n}": n % 2 == 0 for n in range(20)}
    result = paired_difference(arm(same), arm(same), "t", "c")
    assert result.delta_pp == 0.0
    assert result.crosses_zero


def test_uniform_improvement_is_detected_and_excludes_zero():
    control = arm({f"i{n}": False for n in range(30)})
    treatment = arm({f"i{n}": True for n in range(30)})
    result = paired_difference(treatment, control, "t", "c")
    assert result.delta_pp == 100.0
    assert not result.crosses_zero
    assert result.ci_low_pp > 0


def test_small_effect_on_a_small_sample_stays_inconclusive():
    """One flip in sixteen must not read as a finding."""
    control = arm({f"i{n}": False for n in range(16)})
    flipped = {f"i{n}": n == 0 for n in range(16)}
    result = paired_difference(arm(flipped), control, "t", "c")
    assert result.delta_pp == 6.25
    assert result.crosses_zero, "a single flip at n=16 is not evidence"


def test_harness_failures_are_dropped_not_scored_as_failures():
    """A container timeout is not proof the task was unsolvable."""
    control = arm({"a": True, "b": True})
    treatment = {
        "a": InstanceResult("a", resolved=True),
        "b": InstanceResult("b", resolved=False, status="TimeoutExpired"),
    }
    result = paired_difference(treatment, control, "t", "c")
    assert result.n_paired == 1
    assert result.n_dropped == 1
    assert result.delta_pp == 0.0, "the dropped instance must not count against treatment"


def test_unpaired_instances_are_ignored():
    control = arm({"a": True, "b": False})
    treatment = arm({"a": True, "c": True})
    result = paired_difference(treatment, control, "t", "c")
    assert result.n_paired == 1


def test_cost_ratio_uses_only_paired_instances():
    control = {
        "a": InstanceResult("a", True, cost=1.0),
        "b": InstanceResult("b", True, cost=100.0, status="TimeoutExpired"),
    }
    treatment = {
        "a": InstanceResult("a", True, cost=2.0),
        "b": InstanceResult("b", True, cost=1.0),
    }
    result = paired_difference(treatment, control, "t", "c")
    assert result.cost_ratio == 2.0, "the dropped instance's cost must not leak in"


def test_bootstrap_is_deterministic():
    diffs = [1.0, 0.0, -1.0, 1.0, 0.0] * 6
    assert bootstrap_ci(diffs) == bootstrap_ci(diffs)


def test_bootstrap_handles_an_empty_sample():
    assert bootstrap_ci([]) == (0.0, 0.0)


def test_core_contrasts_produces_both_design_comparisons():
    data = {name: arm({f"i{n}": n % 3 == 0 for n in range(12)}) for name in
            ("baseline", "saaga", "stripped_baseline", "saaga_substitution")}
    contrasts = core_contrasts(data)
    assert len(contrasts) == 2
    assert contrasts[0].treatment.startswith("saaga (B)")
    assert contrasts[1].treatment.startswith("saaga_substitution (D)")


def test_core_contrasts_skips_missing_arms():
    """A pilot that has only run C and D should still analyse D - C."""
    data = {
        "stripped_baseline": arm({"i0": False}),
        "saaga_substitution": arm({"i0": True}),
    }
    contrasts = core_contrasts(data)
    assert len(contrasts) == 1
    assert "substitution" in contrasts[0].treatment
