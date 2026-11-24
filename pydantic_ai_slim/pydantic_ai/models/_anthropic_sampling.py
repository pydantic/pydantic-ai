"""The sampling settings Claude rejects in combination with the rest of a request, for `AnthropicModel` and Bedrock.

It's a module of its own because `models/bedrock.py` can't import from `models/anthropic/` without the `anthropic` SDK.
"""

from __future__ import annotations as _annotations

from typing import Any, Literal

from .._utils import is_str_mapping
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
    conflicting: set[Literal['temperature', 'top_p', 'top_k']] = set()
    extra_body = model_settings.get('extra_body')
    extra_body_mapping = extra_body if is_str_mapping(extra_body) else None

    def _effective_setting(name: Literal['temperature', 'top_p', 'top_k']) -> Any:
        if extra_body_mapping is not None and name in extra_body_mapping:
            return extra_body_mapping[name]
        return model_settings.get(name)

    temperature = _effective_setting('temperature')
    top_p = _effective_setting('top_p')
    top_k = _effective_setting('top_k')
    if thinking and profile.get('anthropic_restricts_sampling_with_thinking', False):
        if temperature is not None and temperature != 1:
            conflicting.add('temperature')
        if top_k is not None:
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
    filtered: ModelSettings = {**model_settings}
    for setting in conflicting:
        filtered.pop(setting, None)
    if extra_body_mapping is not None and any(setting in extra_body_mapping for setting in conflicting):
        new_extra_body = {key: value for key, value in extra_body_mapping.items() if key not in conflicting}
        if new_extra_body:
            filtered['extra_body'] = new_extra_body
        else:
            filtered.pop('extra_body', None)
    return filtered
