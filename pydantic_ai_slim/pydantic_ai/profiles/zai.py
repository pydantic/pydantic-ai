from __future__ import annotations as _annotations

from collections.abc import Mapping

from . import ModelProfile


class ZaiModelProfile(ModelProfile, total=False):
    """Profile for Z.AI (Zhipu AI) GLM models."""

    zai_supports_reasoning_effort: bool
    """Whether the model accepts a per-request `reasoning_effort` level (GLM-5.2 and GLM-5.3)."""

    zai_reasoning_effort_mapping: Mapping[str, str]
    """Substitutions applied to the unified thinking effort level before it is sent as `reasoning_effort`.

    Levels not in the mapping are forwarded unchanged. GLM-5.2 accepts all unified levels, so it needs no
    mapping; GLM-5.3 only accepts `low`/`high`/`max` (per the Z.AI docs and the error message returned when
    disabling thinking on it), so its other unified levels are mapped.
    """


_REASONING_EFFORT_MODEL_PREFIXES = ('glm-5.3', 'glm-5.2')
"""Model name prefixes for GLM models that accept the per-request `reasoning_effort` parameter.

GLM-5.2 introduced per-request reasoning effort. Add released models here as they gain support (like the
OpenAI profile's enumerated `gpt-5.x` set — concrete ids, not a derived "and newer"). On earlier GLM models
the effort levels collapse to thinking on/off.
"""

_ALWAYS_THINKING_MODEL_PREFIXES = ('glm-5.3',)
"""Model name prefixes for GLM models that always reason and reject `thinking.type: 'disabled'` (GLM-5.3).

Like `_REASONING_EFFORT_MODEL_PREFIXES`, list concrete released ids only.
"""

_GLM_5_3_REASONING_EFFORT_MAPPING = {'minimal': 'low', 'medium': 'high', 'xhigh': 'max'}
"""GLM-5.3 only accepts `reasoning_effort` values `low`, `high`, and `max` (per the Z.AI docs, and the
'please use low, high, or max' text of the error Z.AI returns for `thinking.type: 'disabled'` on the model),
so the unified levels it doesn't accept map to the nearest supported one — the same fallback approach as
Gemini's `MINIMAL` -> `LOW`. `xhigh` maps to `max`, which has no unified equivalent."""


def zai_model_profile(model_name: str) -> ModelProfile | None:
    """The model profile for ZAI (Zhipu AI) GLM models, matched by Z.AI's native `glm-*` ids.

    Marks thinking-capable models (`glm-5`, `glm-4.7`, `glm-4.6`, `glm-4.5`) via `supports_thinking=True`.
    This includes the `glm-4.6v` and `glm-4.5v` vision models, which also support thinking mode per the
    Z.AI docs. GLM-5.2 and GLM-5.3 additionally accept a per-request reasoning effort level, flagged via
    `zai_supports_reasoning_effort=True`. GLM-5.3 always reasons and cannot disable thinking, flagged via
    `thinking_always_enabled=True`.

    The provider-specific request/response shape (e.g. the `reasoning_content` field used by Z.AI's API)
    is configured in `ZaiProvider.model_profile()` rather than here. Providers that serve GLM models under
    a different id scheme (e.g. Cerebras's `zai-glm-*`, which doesn't match the `glm-*` prefixes above)
    configure thinking support in their own `model_profile()`.
    """
    model_lower = model_name.lower()
    thinking_prefixes = ('glm-5', 'glm-4.7', 'glm-4.6', 'glm-4.5')
    if not model_lower.startswith(thinking_prefixes):
        return None
    profile = ZaiModelProfile(
        supports_thinking=True,
        thinking_always_enabled=model_lower.startswith(_ALWAYS_THINKING_MODEL_PREFIXES),
        zai_supports_reasoning_effort=model_lower.startswith(_REASONING_EFFORT_MODEL_PREFIXES),
    )
    if model_lower.startswith('glm-5.3'):
        profile['zai_reasoning_effort_mapping'] = _GLM_5_3_REASONING_EFFORT_MAPPING
    return profile
