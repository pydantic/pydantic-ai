from __future__ import annotations as _annotations

import os
from typing import overload

import httpx

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.groq import groq_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import openai_model_profile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.providers import Provider

try:
    from groq import AsyncGroq
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `groq` package to use the Groq provider, '
        'you can use the `groq` optional group — `pip install "pydantic-ai-slim[groq]"`'
    ) from _import_error


def groq_moonshotai_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for an MoonshotAI model used with the Groq provider."""
    return ModelProfile(supports_json_object_output=True, supports_json_schema_output=True).update(
        moonshotai_model_profile(model_name)
    )


def meta_groq_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Meta model used with the Groq provider."""
    if model_name in {'llama-4-maverick-17b-128e-instruct', 'llama-4-scout-17b-16e-instruct'}:
        return ModelProfile(supports_json_object_output=True, supports_json_schema_output=True).update(
            meta_model_profile(model_name)
        )
    else:
        return meta_model_profile(model_name)


class GroqProvider(Provider[AsyncGroq]):
    """Provider for Groq API."""

    @property
    def name(self) -> str:
        return 'groq'

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def client(self) -> AsyncGroq:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        prefix_to_profile = {
            'llama': meta_model_profile,
            'meta-llama/': meta_groq_model_profile,
            'gemma': google_model_profile,
            'qwen': qwen_model_profile,
            'deepseek': deepseek_model_profile,
            'mistral': mistral_model_profile,
            'moonshotai/': groq_moonshotai_model_profile,
            'compound-': groq_model_profile,
            'openai/': openai_model_profile,
        }

        for prefix, profile_func in prefix_to_profile.items():
            model_name = model_name.lower()
            if model_name.startswith(prefix):
                if prefix.endswith('/'):
                    model_name = model_name[len(prefix) :]
                return profile_func(model_name)

        return None

    @overload
    def __init__(self, *, groq_client: AsyncGroq | None = None) -> None: ...

    @overload
    def __init__(
        self, *, api_key: str | None = None, base_url: str | None = None, http_client: httpx.AsyncClient | None = None
    ) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        groq_client: AsyncGroq | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a new Groq provider.

        Args:
            api_key: The API key to use for authentication, if not provided, the `GROQ_API_KEY` environment variable
                will be used if available.
            base_url: The base url for the Groq requests. If not provided, the `GROQ_BASE_URL` environment variable
                will be used if available. Otherwise, defaults to Groq's base url.
            groq_client: An existing
                [`AsyncGroq`](https://github.com/groq/groq-python?tab=readme-ov-file#async-usage)
                client to use. If provided, `api_key` and `http_client` must be `None`.
            http_client: An existing `AsyncHTTPClient` to use for making HTTP requests.
        """
        if groq_client is not None:
            assert http_client is None, 'Cannot provide both `groq_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `groq_client` and `api_key`'
            assert base_url is None, 'Cannot provide both `groq_client` and `base_url`'
            self._client = groq_client
        else:
            api_key = api_key or os.getenv('GROQ_API_KEY')
            base_url = base_url or os.getenv('GROQ_BASE_URL', 'https://api.groq.com')

            if not api_key:
                raise UserError(
                    'Set the `GROQ_API_KEY` environment variable or pass it via `GroqProvider(api_key=...)`'
                    'to use the Groq provider.'
                )
            elif http_client is not None:
                self._client = AsyncGroq(base_url=base_url, api_key=api_key, http_client=http_client)
            else:
                http_client = cached_async_http_client(provider='groq')
                self._client = AsyncGroq(base_url=base_url, api_key=api_key, http_client=http_client)
