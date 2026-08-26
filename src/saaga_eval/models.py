"""Register OpenAI-compatible endpoints (NVIDIA NIM, vLLM, OpenRouter) with AGENTbench.

AGENTbench resolves models by *registry key*, not by model name::

    config["model"] = deepcopy(ALL_MODEL_CONFIGS[exec_model])

so an endpoint it has never heard of has to be added to that dict first. This
module does that at import time, without forking, the same way
`scripts/run_arm.py` registers the saaga planner.

Why this exists
---------------
The smoke test needs a model that costs nothing, because its job is to prove the
four arms are four different experiments -- not to measure anything about saaga.
Any OpenAI-compatible endpoint works for that, and NVIDIA NIM has a free tier.

Results from a smoke model transfer nothing to the real run. What transfers is
whether the plumbing is correct, and that is model-independent.

Routing note
------------
Models are registered under LiteLLM's ``openai/`` prefix with an explicit
``api_base`` rather than a provider-specific prefix. Any endpoint that speaks
the OpenAI chat-completions API then works, whether or not LiteLLM ships a
dedicated provider entry for it.
"""

from __future__ import annotations

import os

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

# NIM free tier bills nothing, so cost columns from a smoke run are zeros by
# construction. Registered anyway: the harness looks up prices per model and a
# missing entry makes the cost analysis silently useless.
_FREE = (0.0, 0.0, 0.0, 0.0)


def nim_api_key() -> str | None:
    """NIM key from either common environment variable name."""
    return os.getenv("NVIDIA_NIM_API_KEY") or os.getenv("NVIDIA_API_KEY")


def openai_compatible_model(
    model_id: str,
    base_url: str,
    api_key: str | None,
    temperature: float = 0.0,
    max_completion_tokens: int | None = None,
) -> dict:
    """Build an AGENTbench model config for any OpenAI-compatible endpoint."""
    kwargs: dict = {
        "drop_params": True,  # endpoints vary in what sampling params they accept
        "temperature": temperature,
        "api_key": api_key or "anything",
        "stream": False,
    }
    if max_completion_tokens:
        kwargs["max_completion_tokens"] = max_completion_tokens

    return {
        "model_name": f"openai/{model_id}",
        "api_base": base_url,
        "model_kwargs": kwargs,
    }


def register(
    key: str,
    model_id: str,
    base_url: str = NIM_BASE_URL,
    api_key: str | None = None,
    price: tuple[float, float, float, float] = _FREE,
    **kwargs,
) -> str:
    """Add a model to AGENTbench's registries. Returns the key to pass as exec_model."""
    try:
        from configs import model_constants
    except ImportError as exc:  # pragma: no cover - environment guidance
        raise SystemExit(
            "Could not import AGENTbench's `configs` package.\n"
            "  git clone https://github.com/eth-sri/agentbench vendor/agentbench\n"
            "  pip install -e vendor/agentbench"
        ) from exc

    model_constants.ALL_MODEL_CONFIGS[key] = openai_compatible_model(
        model_id=model_id,
        base_url=base_url,
        api_key=api_key if api_key is not None else nim_api_key(),
        **kwargs,
    )
    model_constants.MODEL_PRICES.setdefault(model_id, price)
    return key


def register_nim(model_id: str) -> str:
    """Register a NIM-hosted model, keyed ``nim:<model_id>``.

    Called from `run_arm.py` whenever `--exec-model` starts with ``nim:``, so a
    model id can be passed straight through without this file having to know it
    in advance. NIM model ids look like ``zai/glm-5.2`` or
    ``meta/llama-3.3-70b-instruct``; copy the exact string from
    https://build.nvidia.com.
    """
    if not nim_api_key():
        raise SystemExit(
            "NVIDIA_NIM_API_KEY (or NVIDIA_API_KEY) is not set.\n"
            "Create a free key at https://build.nvidia.com, then:\n"
            "  export NVIDIA_NIM_API_KEY=nvapi-..."
        )
    return register(key=f"nim:{model_id}", model_id=model_id, base_url=NIM_BASE_URL)


def maybe_register(exec_model: str) -> str:
    """Resolve `--exec-model`, auto-registering ``nim:``/``openai-compat:`` forms.

    ``nim:zai/glm-5.2``                       -> NIM free tier
    ``openai-compat:<base_url>::<model_id>``  -> any other endpoint (vLLM, LM Studio)
    anything else                             -> assumed already in the registry
    """
    if exec_model.startswith("nim:"):
        return register_nim(exec_model.split(":", 1)[1])

    if exec_model.startswith("openai-compat:"):
        spec = exec_model.split(":", 1)[1]
        if "::" not in spec:
            raise SystemExit(
                "Expected --exec-model openai-compat:<base_url>::<model_id>, "
                f"got {exec_model!r}"
            )
        base_url, model_id = spec.split("::", 1)
        return register(
            key=exec_model,
            model_id=model_id,
            base_url=base_url,
            api_key=os.getenv("OPENAI_COMPAT_API_KEY", "anything"),
        )

    return exec_model
