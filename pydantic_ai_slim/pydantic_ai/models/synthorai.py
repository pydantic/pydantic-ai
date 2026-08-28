"""Synthorai model implementation using OpenAI-compatible API."""

from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Literal

from ..profiles import ModelProfileSpec
from ..providers import Provider
from ..settings import ModelSettings

try:
    from openai import AsyncOpenAI

    from .openai import OpenAIChatModel
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the Synthorai model, '
        'you can use the `synthorai` optional group — `pip install "pydantic-ai-slim[synthorai]"`'
    ) from _import_error

__all__ = ('SynthoraiModel', 'SynthoraiModelName')

LatestSynthoraiModelNames = Literal[
    'claude-haiku-4-5',
    'claude-opus-5',
    'claude-sonnet-5',
    'deepseek-v4-flash',
    'deepseek-v4-pro',
    'gemini-3.5-flash',
    'gemini-3.7-flash',
    'glm-5.2',
    'gpt-5.6-luna',
    'gpt-5.6-sol',
    'gpt-5.6-terra',
    'kimi-k2.5',
    'kimi-k3',
    'minimax-m2.5',
    'qwen3.8-max',
]

SynthoraiModelName = str | LatestSynthoraiModelNames
"""Possible Synthorai model names.

Synthorai routes to models from several upstream providers and the list changes as
channels are added, so a few known ids are listed for autocompletion while any name is
allowed in the type hints. The current list is served at
[`https://synthorai.io/api/models`](https://synthorai.io/api/models), and what a given
key can reach is narrower than the catalog: `/v1/models` returns the models that key is
permitted to use.
"""


@dataclass(init=False)
class SynthoraiModel(OpenAIChatModel):
    """A model that uses Synthorai's OpenAI-compatible API."""

    def __init__(
        self,
        model_name: SynthoraiModelName,
        *,
        provider: Literal['synthorai'] | Provider[AsyncOpenAI] = 'synthorai',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize a Synthorai model.

        Args:
            model_name: The name of the model to use.
            provider: The provider to use. Defaults to `'synthorai'`.
            profile: The model profile to use. Defaults to a profile picked by the provider
                from the model id's family prefix.
            settings: Model-specific settings that will be used as defaults for this model.
        """
        super().__init__(model_name, provider=provider, profile=profile, settings=settings)
