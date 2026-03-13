"""Ollama model implementation using OpenAI-compatible API."""

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
        'Please install the `openai` package to use the Ollama model, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

__all__ = ('OllamaModel',)


@dataclass(init=False)
class OllamaModel(OpenAIChatModel):
    """A model that uses Ollama's OpenAI-compatible Chat Completions API.

    Ollama's `/v1/chat/completions` endpoint supports `response_format` with `json_schema` natively,
    so no payload remapping is needed. The provider profile sets `supports_json_schema_output=True`
    and `openai_supports_strict_tool_definition=False` to match Ollama's capabilities.

    Apart from `__init__`, all methods are inherited from the base class.
    """

    def __init__(
        self,
        model_name: str,
        *,
        provider: Literal['ollama'] | Provider[AsyncOpenAI] = 'ollama',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize an Ollama model.

        Args:
            model_name: The name of the Ollama model to use (e.g. `'qwen3'`, `'llama3.2'`).
            provider: The provider to use. Defaults to `'ollama'`.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            settings: Model-specific settings that will be used as defaults for this model.
        """
        super().__init__(model_name, provider=provider, profile=profile, settings=settings)
