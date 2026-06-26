from __future__ import annotations as _annotations

from . import ModelProfile


def zai_model_profile(model_name: str) -> ModelProfile | None:
    """The model profile for ZAI (Zhipu AI) GLM models, matched by Z.AI's native `glm-*` ids.

    Marks thinking-capable models (`glm-5`, `glm-4.7`, `glm-4.6`, `glm-4.5`)
    via `supports_thinking=True`. This includes the `glm-4.6v` and `glm-4.5v`
    vision models, which also support thinking mode per the Z.AI docs.

    The provider-specific request/response shape (e.g. the `reasoning_content` field
    used by Z.AI's API) is configured in `ZaiProvider.model_profile()` rather than here.
    Providers that serve GLM models under a different id scheme (e.g. Cerebras's
    `zai-glm-*`, which doesn't match the `glm-*` prefixes above) configure thinking
    support in their own `model_profile()`.
    """
    model_lower = model_name.lower()
    thinking_prefixes = ('glm-5', 'glm-4.7', 'glm-4.6', 'glm-4.5')
    if model_lower.startswith(thinking_prefixes):
        return ModelProfile(supports_thinking=True)
    return None
