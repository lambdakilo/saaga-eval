"""Integration against a real AGENTbench install.

Skipped when the vendored harness is absent, so the unit suite stays runnable
without Docker or a checkout. When it is present these are the tests that catch
an upstream change breaking the unforked integration -- the failure mode that
would otherwise show up only after a corpus had been built and money spent.
"""

from __future__ import annotations

import pytest

configs = pytest.importorskip("configs", reason="AGENTbench not installed (pip install -e vendor/agentbench)")

from saaga_eval.arms import ARMS, CORE_2X2, get_arm, planner_config  # noqa: E402
from saaga_eval.models import openai_compatible_model  # noqa: E402


@pytest.fixture(autouse=True)
def registered():
    """Register saaga's plan configs the way `run_arm.py` does."""
    from configs import plan_constants

    original = dict(plan_constants.ALL_PLAN_CONFIGS)
    for arm in ARMS.values():
        config = planner_config(arm)
        if config is not None:
            plan_constants.ALL_PLAN_CONFIGS[arm.plan_type] = config
    yield plan_constants
    plan_constants.ALL_PLAN_CONFIGS.clear()
    plan_constants.ALL_PLAN_CONFIGS.update(original)


def test_dotted_path_resolves_without_a_fork(registered):
    """`_PLANNER_MAPPING.get(spec, spec)` is what makes this work."""
    from agentbench.planners import get_planner_class

    from saaga_eval.planner import SaagaPlanner

    assert get_planner_class("saaga_eval.planner.SaagaPlanner") is SaagaPlanner


def test_get_planner_builds_saaga_planner_from_registry(registered):
    from agentbench.planners import get_planner

    from saaga_eval.planner import SaagaPlanner

    # Mirror the keys generate.py injects into every planner config.
    config = dict(registered.ALL_PLAN_CONFIGS["saaga_planner"])
    config["plan_model"] = "nim:zai/glm-5.2"
    config["generator_config"] = {"generator_class": "miniswe_agents", "step_limit": 200}
    config["storage_dir"] = "output/plans/eth-sri_agentbench/miniswe_agents"

    planner = get_planner(config)
    assert isinstance(planner, SaagaPlanner)
    assert planner.config.corpus_root == "corpora"
    assert planner.config.fail_on_contamination is True


def test_every_arm_plan_type_is_resolvable(registered):
    """Including the calibration arms, which use AGENTbench's own planners."""
    from agentbench.planners import get_planner_class

    for key, arm in ARMS.items():
        assert arm.plan_type in registered.ALL_PLAN_CONFIGS, f"{key} unregistered"
        spec = registered.ALL_PLAN_CONFIGS[arm.plan_type]["planner_class"]
        assert get_planner_class(spec) is not None


def test_generator_default_is_a_real_registry_key():
    """`miniswe`, the obvious guess, is not the key -- `miniswe_agents` is."""
    from configs.generator_constants import ALL_GENERATOR_CONFIGS

    assert "miniswe_agents" in ALL_GENERATOR_CONFIGS
    assert "miniswe" not in ALL_GENERATOR_CONFIGS


def test_core_arms_map_onto_distinct_harness_calls(registered):
    """Four cells must reach generate.py as four different configurations."""
    signatures = {
        (get_arm(key).plan_type, get_arm(key).remove_docs) for key in CORE_2X2
    }
    assert len(signatures) == len(CORE_2X2)


def test_nim_model_registers_into_the_model_registry():
    from configs import model_constants

    from saaga_eval.models import register

    key = register("nim:test/model", "test/model", api_key="anything")
    try:
        assert key in model_constants.ALL_MODEL_CONFIGS
        entry = model_constants.ALL_MODEL_CONFIGS[key]
        assert entry["model_name"] == "openai/test/model"
        assert entry["api_base"].endswith("/v1")
        assert "test/model" in model_constants.MODEL_PRICES, "missing price breaks cost analysis"
    finally:
        model_constants.ALL_MODEL_CONFIGS.pop(key, None)
        model_constants.MODEL_PRICES.pop("test/model", None)


def test_openai_compatible_config_shape_matches_vendored_examples():
    """Same shape as AGENTbench's own vLLM entries (model_name/api_base/model_kwargs)."""
    from configs.model_constants import MODEL_QWEN3_30B_CODER

    built = openai_compatible_model("zai/glm-5.2", "https://example.invalid/v1", "k")
    assert set(built) <= set(MODEL_QWEN3_30B_CODER) | {"api_base"}
    assert built["model_kwargs"]["drop_params"] is True
