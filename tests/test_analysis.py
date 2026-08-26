"""The analysis must not manufacture an effect, and must not hide one.

Both failure modes are cheap to introduce and expensive to notice later, so
each is pinned here with a constructed dataset whose right answer is known.
"""

from __future__ import annotations

from saaga_eval.analysis import (
    load_arm,
    paired_metric,
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


# --- CSV loading against AGENTbench's real column schema -------------------

CSV_HEADER = (
    "instance_id,resolved,execution_cost,number_steps,number_steps_first_read,"
    "number_errors,sys_prompt_size,plan_type,run_id\n"
)


def write_csv(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text(CSV_HEADER + "".join(rows), encoding="utf-8")
    return path


def test_csv_booleans_are_parsed_not_coerced(tmp_path):
    """bool("False") is True -- the trap this guards against."""
    path = write_csv(tmp_path, "a.csv", [
        "i0,False,0.5,10,3,0,1200,no_plan,0\n",
        "i1,True,0.7,12,4,1,1200,no_plan,0\n",
    ])
    arm = load_arm(path)
    assert arm["i0"].resolved is False
    assert arm["i1"].resolved is True


def test_csv_maps_real_column_names(tmp_path):
    path = write_csv(tmp_path, "a.csv", ["i0,True,1.25,20,7,2,900,saaga_planner,0\n"])
    result = load_arm(path)["i0"]
    assert result.cost == 1.25
    assert result.steps == 20
    assert result.steps_first_read == 7
    assert result.errors == 2
    assert result.sys_prompt_size == 900


def test_missing_first_read_is_none_not_zero(tmp_path):
    """Absent means the agent never opened a gold-patch file, not 'zero steps'."""
    path = write_csv(tmp_path, "a.csv", ["i0,False,0.1,30,,0,900,no_plan,0\n"])
    assert load_arm(path)["i0"].steps_first_read is None


def test_paired_metric_excludes_instances_missing_the_metric(tmp_path):
    control = write_csv(tmp_path, "c.csv", [
        "i0,True,1.0,20,10,0,900,no_plan,0\n",
        "i1,True,1.0,20,,0,900,no_plan,0\n",
    ])
    treatment = write_csv(tmp_path, "t.csv", [
        "i0,True,1.0,15,4,0,900,saaga_planner,0\n",
        "i1,True,1.0,15,5,0,900,saaga_planner,0\n",
    ])
    out = paired_metric(load_arm(treatment), load_arm(control), "steps_first_read")
    assert out is not None
    mean, _low, _high, n = out
    assert n == 1, "i1 has no control value and must be excluded"
    assert mean == -6.0


def test_paired_metric_returns_none_when_nothing_is_comparable(tmp_path):
    control = write_csv(tmp_path, "c.csv", ["i0,True,1.0,20,,0,900,no_plan,0\n"])
    treatment = write_csv(tmp_path, "t.csv", ["i0,True,1.0,15,,0,900,saaga_planner,0\n"])
    assert paired_metric(load_arm(treatment), load_arm(control), "steps_first_read") is None
