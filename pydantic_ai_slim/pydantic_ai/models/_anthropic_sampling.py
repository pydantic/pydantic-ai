"""The sampling settings Claude rejects in combination with the rest of a request, for `AnthropicModel` and Bedrock.

It's a module of its own because `models/bedrock.py` can't import from `models/anthropic/` without the `anthropic` SDK.
"""

from __future__ import annotations as _annotations

from typing import cast

from ..profiles import ModelProfile
from ..settings import ModelSettings


def drop_conflicting_sampling_settings(
    model_settings: ModelSettings, profile: ModelProfile, *, thinking: bool
) -> ModelSettings:
    """Drop the sampling settings the model rejects in combination with the rest of the request.

    With thinking, a model that sets `anthropic_restricts_sampling_with_thinking` takes `temperature` only at `1`,
    `top_p` only from `0.95`, and no `top_k`. A model that sets `anthropic_disallows_temperature_with_top_p` takes only
    one of `temperature` and `top_p`: `top_p` is dropped, as Anthropic recommends tuning `temperature`. Each is rejected
    with a 400 (verified live 2026-11-04, on both the Anthropic API and Bedrock). They're tuning knobs, so they're
    dropped silently.
    """
    conflicting: set[str] = set()
    temperature = model_settings.get('temperature')
    top_p = model_settings.get('top_p')
    if thinking and profile.get('anthropic_restricts_sampling_with_thinking', False):
        if temperature is not None and temperature != 1:
            conflicting.add('temperature')
        if model_settings.get('top_k') is not None:
            conflicting.add('top_k')
        if top_p is not None and top_p < 0.95:
            conflicting.add('top_p')
    if (
        profile.get('anthropic_disallows_temperature_with_top_p', False)
        and temperature is not None
        and top_p is not None
        and not conflicting & {'temperature', 'top_p'}
    ):
        conflicting.add('top_p')
    return cast(ModelSettings, {key: value for key, value in model_settings.items() if key not in conflicting})
