from __future__ import annotations as _annotations

from . import ModelProfile


class ZaiModelProfile(ModelProfile, total=False):
    """Profile for Z.AI (Zhipu AI) GLM models."""

    zai_supports_reasoning_effort: bool
    """Whether the model accepts a per-request `reasoning_effort` level (GLM-5.2)."""


_REASONING_EFFORT_MODEL_PREFIXES = ('glm-5.2',)
"""Model name prefixes for GLM models that accept the per-request `reasoning_effort` parameter.

GLM-5.2 introduced per-request reasoning effort. Add released models here as they gain support (like the
OpenAI profile's enumerated `gpt-5.x` set — concrete ids, not a derived "and newer"). On earlier GLM models
the effort levels collapse to thinking on/off.
"""


def zai_model_profile(model_name: str) -> ModelProfile | None:
    """The model profile for ZAI (Zhipu AI) GLM models, matched by Z.AI's native `glm-*` ids.

    Marks thinking-capable models (`glm-5`, `glm-4.7`, `glm-4.6`, `glm-4.5`) via `supports_thinking=True`.
    This includes the `glm-4.6v` and `glm-4.5v` vision models, which also support thinking mode per the
    Z.AI docs. GLM-5.2 additionally accepts a per-request reasoning effort level, flagged via
    `zai_supports_reasoning_effort=True`.

    The provider-specific request/response shape (e.g. the `reasoning_content` field used by Z.AI's API)
    is configured in `ZaiProvider.model_profile()` rather than here. Providers that serve GLM models under
    a different id scheme (e.g. Cerebras's `zai-glm-*`, which doesn't match the `glm-*` prefixes above)
    configure thinking support in their own `model_profile()`.
    """
    model_lower = model_name.lower()
    thinking_prefixes = ('glm-5', 'glm-4.7', 'glm-4.6', 'glm-4.5')
    if not model_lower.startswith(thinking_prefixes):
        return None
    return ZaiModelProfile(
        supports_thinking=True,
        zai_supports_reasoning_effort=model_lower.startswith(_REASONING_EFFORT_MODEL_PREFIXES),
    )
