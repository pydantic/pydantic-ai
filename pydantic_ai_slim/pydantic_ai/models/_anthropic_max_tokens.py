"""Cap `max_tokens` at a Claude model's maximum output, for `AnthropicModel` and Bedrock.

It's a module of its own because `models/bedrock.py` can't import from `models/anthropic/` without the `anthropic` SDK.
"""

from __future__ import annotations as _annotations

from ..settings import ModelSettings


def cap_max_tokens(model_settings: ModelSettings | None, max_output_tokens: int | None) -> ModelSettings | None:
    """Lower a `max_tokens` above the model's `anthropic_max_output_tokens` to it, as the API rejects it with a 400.

    The model can't generate more than its maximum anyway, so nothing is lost and nothing is warned about.
    """
    if model_settings is None or max_output_tokens is None:
        return model_settings
    max_tokens = model_settings.get('max_tokens')
    if max_tokens is None or max_tokens <= max_output_tokens:
        return model_settings
    return {**model_settings, 'max_tokens': max_output_tokens}
