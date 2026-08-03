from __future__ import annotations as _annotations

from . import ModelProfile


def moonshotai_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a MoonshotAI model."""
    # Kimi reasoning models (kimi-k2.5/k2.6/k2.7-code, …) accept reasoning_effort and emit
    # reasoning_content; the moonshot-v1/instruct models don't. `thinking_always_enabled` is left
    # to the direct provider, since the `reasoning_effort='none'` quirk is specific to the
    # `api.moonshot.ai` endpoint and this profile is also routed through OpenRouter and Heroku.
    #
    # Model names reach this profile in both dotted and hyphenated spellings depending on the
    # route (e.g. `kimi-k2.5` on the MoonshotAI endpoint, but `kimi-k2-5`/`kimi-k2-thinking` via
    # Heroku, Bedrock and OpenRouter). Normalize `-` to `.` so both spellings are recognised.
    is_reasoning = model_name.lower().replace('-', '.').startswith(
        ('kimi.k2.5', 'kimi.k2.6', 'kimi.k2.7', 'kimi.k2.thinking', 'kimi.k3', 'kimi.thinking')
    )
    return ModelProfile(
        ignore_streamed_leading_whitespace=True,
        supports_thinking=is_reasoning,
    )
