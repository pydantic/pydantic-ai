"""The OpenAI model names, and the SDK names that are deprecated or never existed."""

from __future__ import annotations as _annotations

from typing import Literal

DEPRECATED_OPENAI_MODELS: frozenset[str] = frozenset(
    {
        # https://developers.openai.com/api/docs/deprecations#2025-11-18-chatgpt-4o-latest-snapshot
        'chatgpt-4o-latest',
        # https://developers.openai.com/api/docs/deprecations#2025-11-17-codex-mini-latest-model-snapshot
        'codex-mini-latest',
        # https://developers.openai.com/api/docs/deprecations#2023-11-06-chat-model-updates
        'gpt-3.5-turbo-0613',
        'gpt-3.5-turbo-16k-0613',
        # https://developers.openai.com/api/docs/deprecations#2025-09-26-legacy-gpt-model-snapshots
        'gpt-4-0125-preview',
        'gpt-4-1106-preview',
        'gpt-4-turbo-preview',
        # https://developers.openai.com/api/docs/deprecations#2024-06-06-gpt-4-32k-and-vision-preview-models
        'gpt-4-32k',
        'gpt-4-32k-0314',
        'gpt-4-32k-0613',
        'gpt-4-vision-preview',
        # https://developers.openai.com/api/docs/deprecations#2025-06-10-gpt-4o-audio-preview-2024-10-01
        'gpt-4o-audio-preview-2024-10-01',
        # Does not exist
        'gpt-5.1-mini',
        # https://developers.openai.com/api/docs/deprecations#2025-04-28-o1-preview-and-o1-mini
        'o1-mini',
        'o1-mini-2024-09-12',
        'o1-preview',
        'o1-preview-2024-09-12',
    }
)
"""Models that are deprecated or don't exist but are still present in the OpenAI SDK's type definitions."""


OpenAIModelName = (
    str
    | Literal[
        'gpt-6.1-sol',
        'gpt-6.1-terra',
        'gpt-6-luna',
        'gpt-6-astra',
        'gpt-5.6-sol',
        'gpt-5.6-terra',
        'gpt-5.6-luna',
    ]
)
"""Possible OpenAI model names.

The `Literal` lists the models to pick from, for autocomplete: per line, the newest model of the newest
generation and of the previous one. Any other model id the API accepts works too, as a `str`.
See [the OpenAI docs](https://platform.openai.com/docs/models) for a full list.

This is used for the OpenAI-compatible models of other providers too (ie, Ollama, Deepseek).
"""
