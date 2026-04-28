"""Ollama model implementation using OpenAI-compatible API."""

from __future__ import annotations as _annotations

from dataclasses import dataclass, replace
from typing import Literal
from urllib.parse import urlparse

from ..profiles import ModelProfileSpec
from ..providers import Provider, infer_provider
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


def _routes_to_ollama_cloud(provider: Provider[AsyncOpenAI], model_name: str) -> bool:
    """Return whether this Ollama provider and model route through Ollama Cloud.

    Two cases are covered:

    - The provider's `base_url` is on `ollama.com`, meaning the request goes directly
      to Ollama Cloud.
    - The model name ends with the `-cloud` suffix, which a local Ollama daemon
      forwards to the same upstream.

    Ollama Cloud accepts `response_format` with `json_schema` without error but does
    not apply grammar-constrained decoding, so structured-output schemas are not
    actually enforced. See
    [pydantic-ai#4917](https://github.com/pydantic/pydantic-ai/issues/4917) and
    [ollama/ollama#12362](https://github.com/ollama/ollama/issues/12362).
    """
    hostname = urlparse(provider.base_url).hostname or ''
    return hostname == 'ollama.com' or hostname.endswith('.ollama.com') or model_name.endswith('-cloud')


@dataclass(init=False)
class OllamaModel(OpenAIChatModel):
    """A model that uses Ollama's OpenAI-compatible Chat Completions API.

    Self-hosted Ollama (v0.5.0+) honors `response_format` with `json_schema` via
    `llama.cpp`'s grammar-constrained decoder, so `NativeOutput` produces
    schema-valid output at generation time.

    Ollama Cloud currently accepts `response_format` with `json_schema` without
    error but does not enforce the schema upstream (see
    [pydantic-ai#4917](https://github.com/pydantic/pydantic-ai/issues/4917) and
    [ollama/ollama#12362](https://github.com/ollama/ollama/issues/12362)). When
    this model detects a Cloud path — either a `base_url` on `ollama.com` or a
    model name ending in `-cloud` — it disables `supports_json_schema_output`
    on the resolved profile. With that flag off,
    [`NativeOutput`][pydantic_ai.output.NativeOutput] raises a clear
    [`UserError`][pydantic_ai.exceptions.UserError] so users pick a mode that
    actually works on Cloud ([`ToolOutput`][pydantic_ai.output.ToolOutput] —
    the default — and [`PromptedOutput`][pydantic_ai.output.PromptedOutput] are
    both verified to work).

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
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name,
                adjusted to disable `supports_json_schema_output` when the request routes through Ollama Cloud.
            settings: Model-specific settings that will be used as defaults for this model.
        """
        if isinstance(provider, str):
            provider = infer_provider(provider)

        if profile is None and _routes_to_ollama_cloud(provider, model_name):
            base_profile = provider.model_profile(model_name)
            assert base_profile is not None  # OllamaProvider always returns a profile
            profile = replace(base_profile, supports_json_schema_output=False)

        super().__init__(model_name, provider=provider, profile=profile, settings=settings)
