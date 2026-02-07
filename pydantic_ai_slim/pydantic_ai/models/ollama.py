"""Ollama model implementation using OpenAI-compatible API."""

from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Any, Literal

from typing_extensions import override

from ..profiles import ModelProfileSpec
from ..providers import Provider
from ..settings import ModelSettings
from . import ModelRequestParameters

try:
    from openai import AsyncOpenAI
    from openai.types.chat import completion_create_params

    from .openai import OpenAIChatModel
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the Ollama model, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

__all__ = ('OllamaModel',)


@dataclass(init=False)
class OllamaModel(OpenAIChatModel):
    """A model that uses Ollama's OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        model_name: str,
        *,
        provider: Literal['ollama'] | Provider[AsyncOpenAI] = 'ollama',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        super().__init__(model_name, provider=provider, profile=profile, settings=settings)

    @override
    def _customize_request_payload(
        self,
        *,
        extra_body: dict[str, Any] | None,
        response_format: completion_create_params.ResponseFormat | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[dict[str, Any] | None, completion_create_params.ResponseFormat | None]:
        if response_format is not None and isinstance(response_format, dict):
            response_format_type = response_format.get('type')

            # Ollama uses a top-level `format` request field for structured output.
            # - For JSON schema output, `format` expects the raw JSON schema.
            # - For JSON object output, `format="json"`.
            if response_format_type == 'json_schema':
                js = response_format.get('json_schema')
                if isinstance(js, dict) and 'schema' in js:
                    extra_body = dict(extra_body or {})
                    extra_body['format'] = js['schema']
                    response_format = None
            elif response_format_type == 'json_object':
                extra_body = dict(extra_body or {})
                extra_body['format'] = 'json'
                response_format = None

        return extra_body, response_format
