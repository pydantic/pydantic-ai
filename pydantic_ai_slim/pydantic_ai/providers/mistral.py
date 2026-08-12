from __future__ import annotations as _annotations

import os
from collections.abc import Callable
from typing import overload

import httpx
import httpx2

from pydantic_ai import ModelProfile
from pydantic_ai._http import create_httpx2_client, warn_if_legacy_httpx_client
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.providers import Provider

try:
    from mistralai.client import Mistral
except ImportError as e:
    raise ImportError(
        'Please install the `mistral` package to use the Mistral provider, '
        'you can use the `mistral` optional group — `pip install "pydantic-ai-slim[mistral]"`'
    ) from e

# Models with adjustable reasoning via `reasoning_effort` (opt-in, unlike always-on `magistral`):
# the Mistral Small 4 and Medium 3.5 families. Older `mistral-small-*` / `mistral-medium-*`
# snapshots (e.g. `mistral-small-2506`, `mistral-medium-2505`) don't support reasoning and are
# deliberately excluded; keep this set in sync with the Small/Medium family ids reporting
# `capabilities.reasoning` on the Mistral `/v1/models` API. The alias ids (`-latest`,
# `mistral-medium`, `mistral-medium-3`) resolve to a reasoning model on the public API; on
# private deployments they may point to an older non-reasoning snapshot.
# See https://docs.mistral.ai/capabilities/reasoning/.
#
# This lives on the provider, not the shared `mistral_model_profile`, because the set is validated
# against Mistral's La Plateforme API and only the native route can translate `thinking` to
# Mistral's 'high'/'none'. OpenAI-compatible routes (LiteLLM etc.) would emit OpenAI-style effort
# values ('medium', 'low') these models reject, so they must keep ignoring `thinking`.
_ADJUSTABLE_REASONING_MODELS = frozenset(
    {
        'mistral-small-latest',
        'mistral-small-2603',
        'mistral-medium-latest',
        'mistral-medium',
        'mistral-medium-3',
        'mistral-medium-3-5',
        'mistral-medium-3.5',
        'mistral-medium-2604',
    }
)

_MistralHTTPClient = httpx.AsyncClient | httpx2.AsyncClient


class MistralProvider(Provider[Mistral]):
    """Provider for Mistral API."""

    _own_http_client: _MistralHTTPClient | None = None
    _http_client_factory: Callable[[], _MistralHTTPClient] | None = None

    @property
    def name(self) -> str:
        return 'mistral'

    @property
    def base_url(self) -> str:
        return self.client.sdk_configuration.get_server_details()[0]

    @property
    def client(self) -> Mistral:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile:
        profile = mistral_model_profile(model_name)
        if profile is None and model_name in _ADJUSTABLE_REASONING_MODELS:
            profile = ModelProfile(supports_thinking=True, thinking_always_enabled=False)
        return merge_profile(profile, ModelProfile(supports_inline_system_prompts=True))

    @overload
    def __init__(self, *, mistral_client: Mistral | None = None) -> None: ...

    @overload
    def __init__(self, *, api_key: str | None = None, http_client: _MistralHTTPClient | None = None) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        mistral_client: Mistral | None = None,
        base_url: str | None = None,
        http_client: _MistralHTTPClient | None = None,
    ) -> None:
        """Create a new Mistral provider.

        Args:
            api_key: The API key to use for authentication, if not provided, the `MISTRAL_API_KEY` environment variable
                will be used if available.
            mistral_client: An existing `Mistral` client to use, if provided, `api_key` and `http_client` must be `None`.
            base_url: The base url for the Mistral requests.
            http_client: An existing `httpx2.AsyncClient` or legacy `httpx.AsyncClient` to use for making HTTP requests.
        """
        if mistral_client is not None:
            assert http_client is None, 'Cannot provide both `mistral_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `mistral_client` and `api_key`'
            assert base_url is None, 'Cannot provide both `mistral_client` and `base_url`'
            self._client = mistral_client
        else:
            api_key = api_key or os.getenv('MISTRAL_API_KEY')

            if not api_key:
                raise UserError(
                    'Set the `MISTRAL_API_KEY` environment variable or pass it via `MistralProvider(api_key=...)`'
                    ' to use the Mistral provider.'
                )
            if http_client is None:
                http_client = create_httpx2_client()
                self._own_http_client = http_client  # pyright: ignore[reportIncompatibleVariableOverride]
                self._http_client_factory = create_httpx2_client  # pyright: ignore[reportIncompatibleVariableOverride]
            else:
                warn_if_legacy_httpx_client(http_client, consumer='the Mistral provider', stacklevel=2)

            # Mistral's runtime-checkable client protocol accepts HTTPX2, but its annotations name legacy HTTPX types.
            self._client = Mistral(
                api_key=api_key,
                async_client=http_client,  # pyright: ignore[reportArgumentType]
                server_url=base_url,
            )

    # The generic Provider currently only knows the legacy HTTPX client type.
    def _set_http_client(self, http_client: _MistralHTTPClient) -> None:
        self._client.sdk_configuration.async_client = http_client  # pyright: ignore[reportAttributeAccessIssue]
