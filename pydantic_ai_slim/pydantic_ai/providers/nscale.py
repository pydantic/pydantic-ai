from __future__ import annotations as _annotations

import os
from typing import overload

import httpx

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import create_async_http_client
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import openai_model_profile
from pydantic_ai.providers import Provider
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the GitHub Models provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

class NscaleProvider(Provider[AsyncOpenAI]):
    """Provider for NScale Models API.

    NScale Models provides access to various AI models through an OpenAI-compatible API.
    See <https://docs.nscale.com/docs/ai-services/models> for more information.
    """

    @property
    def name(self) -> str:
        return 'nscale'

    @property
    def base_url(self) -> str:
        return 'https://inference.api.nscale.com/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        provider_to_profile = {
            'openai': openai_model_profile,
            'moonshortai': moonshotai_model_profile,
        }
        profile: ModelProfile | None = None

        provider, model_name = model_name.removeprefix('~').split('/', 1)

        if provider in provider_to_profile:
            model_name, *_ = model_name.split(':', 1)
        profile = provider_to_profile[provider](model_name)

        for prefix, profile_func in provider_to_profile.items():
            if model_name.startswith(prefix):
                profile = profile_func(model_name)
                break

        return OpenAIModelProfile(
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            openai_supports_tool_choice_required=False,
        ).update(profile)

    def __init__(
        self,
        base_url: str | None =None,
        service_token: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize Nscale provider.
            Args:

        Args:
            api_key: SambaNova API key. If not provided, reads from NSCALE_SERVICE_TOKEN env var.
            base_url: Custom API base URL. Defaults to https://inference.api.nscale.com/v1
            openai_client: Optional pre-configured OpenAI client
            http_client: Optional custom httpx.AsyncClient for making HTTP requests

        Raises:
            UserError: If API key is not provided and NSCALE_SERVICE_TOKEN env var is not set
        """
        if openai_client is not None:
            self._client = openai_client
            self._base_url = str(openai_client.base_url)
        else:
            # Get API key from parameter or environment
            service_token = service_token or os.getenv('NSCALE_SERVICE_TOKEN')
            if not service_token:
                raise UserError(
                    'Set the `NSCALE_SERVICE_TOKEN` environment variable or pass it via '
                    '`NScaleProvider(service_token=...)` to use the NScale provider.'
                )

            # Set base URL (default to NScale API endpoint)
            self._base_url = base_url or os.getenv('NSCALE_BASE_URL', 'https://inference.api.nscale.com/v1')

            if http_client is None:
                http_client = create_async_http_client()
                self._own_http_client = http_client
                self._http_client_factory = create_async_http_client
            self._client = AsyncOpenAI(base_url=self._base_url, api_key=service_token, http_client=http_client)
        


    def _set_http_client(self, http_client: httpx.AsyncClient) -> None:
        self._client._client = http_client  # pyright: ignore[reportPrivateUsage]
