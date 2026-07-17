from __future__ import annotations as _annotations

import os
from typing import Literal, overload

import httpx

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import create_async_http_client
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.openai import OpenAIModelProfile, openai_model_profile
from pydantic_ai.providers import Provider
from pydantic_ai.providers._bedrock_model_names import split_bedrock_model_id

try:
    from openai import AsyncBedrockOpenAI, AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the Bedrock Mantle dependencies to use the Bedrock Mantle provider, '
        'you can use the `bedrock-mantle` optional group — `pip install "pydantic-ai-slim[bedrock-mantle]"`'
    ) from _import_error

BedrockMantleInterface = Literal['chat', 'responses', 'openai-responses']
"""The OpenAI-compatible endpoint family a Bedrock Mantle model is served on.

- `'chat'`: Chat Completions at `/v1/chat/completions` (GPT-OSS Safeguard).
- `'responses'`: Responses at `/v1/responses` (GPT-OSS).
- `'openai-responses'`: Responses at `/openai/v1/responses`, the OpenAI-model-specific path (GPT-5.4+).
"""


class BedrockMantleModelProfile(OpenAIModelProfile, total=False):
    """Profile for OpenAI models served through Amazon Bedrock Mantle."""

    bedrock_mantle_interface: BedrockMantleInterface
    """Which Mantle endpoint family serves this model, selecting the model class and base URL."""


def bedrock_mantle_model_profile(model_name: str) -> ModelProfile:
    """Resolve the profile for an OpenAI model served through Bedrock Mantle."""
    provider, base_model_name = split_bedrock_model_id(model_name)
    if provider != 'openai':
        raise UserError(
            f'Model {model_name!r} is not an OpenAI model on Bedrock Mantle. '
            'Bedrock Mantle currently serves OpenAI models through the `bedrock-mantle:` prefix.'
        )
    if base_model_name.startswith('gpt-oss-safeguard'):
        interface: BedrockMantleInterface = 'chat'
    elif base_model_name.startswith('gpt-oss'):
        interface = 'responses'
    else:
        interface = 'openai-responses'
    # Mantle GPT-5.6 resets Responses tool-call IDs across separate responses; qualify them with the
    # response ID at ingestion so pydantic-ai keeps its history-wide-unique invariant. See #6536.
    response_scoped = interface == 'openai-responses' and base_model_name.startswith('gpt-5.6')
    return merge_profile(
        openai_model_profile(base_model_name),
        BedrockMantleModelProfile(
            bedrock_mantle_interface=interface,
            openai_responses_tool_call_ids_are_response_scoped=response_scoped,
            # Bedrock Mantle does not serve image output for these models, unlike the direct OpenAI API
            # the base profile is resolved from (per the AWS model cards). Without this override the
            # image-output guard is bypassed and the request fails with an opaque provider error.
            supports_image_output=False,
            supported_native_tools=frozenset(),
        ),
    )


class BedrockMantleProvider(Provider[AsyncOpenAI]):
    """Provider for the Amazon Bedrock Mantle OpenAI-compatible API."""

    @property
    def name(self) -> str:
        return 'bedrock-mantle'

    @property
    def base_url(self) -> str:
        return str(self._client.base_url)

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        return bedrock_mantle_model_profile(model_name)

    def _openai_client(self, interface: BedrockMantleInterface) -> AsyncOpenAI:
        """Return the OpenAI client for a Mantle endpoint family.

        GPT-5.x models are served on `/openai/v1`; GPT-OSS models on `/v1`. When the provider was
        configured with an explicit `base_url` (or `openai_client`), that endpoint is used for every
        interface.
        """
        if self._origin is None or interface == 'openai-responses':
            return self._client
        if self._standard_client is None:
            self._standard_client = self._client.with_options(base_url=f'{self._origin}/v1')
        return self._standard_client

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI) -> None: ...

    @overload
    def __init__(
        self,
        *,
        region_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        profile_name: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        region_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        profile_name: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a Bedrock Mantle provider.

        Args:
            region_name: The AWS region used to construct the default `bedrock-mantle.{region}.api.aws`
                origin. If not set, the `AWS_DEFAULT_REGION` or `AWS_REGION` environment variable is used.
            base_url: A complete Mantle base URL, used for every model on this provider. Prefer
                `region_name` for automatic per-model endpoint routing between `/v1` and `/openai/v1`.
            api_key: A Bedrock API key. If omitted, `AWS_BEARER_TOKEN_BEDROCK` is used. Use this or the
                `aws_*` credentials, not both.
            aws_access_key_id: The AWS access key ID for SigV4 authentication.
            aws_secret_access_key: The AWS secret access key for SigV4 authentication.
            aws_session_token: The AWS session token for SigV4 authentication.
            profile_name: The AWS profile name for SigV4 authentication.
            openai_client: An existing OpenAI client. If provided, no other argument may be set, and its
                endpoint is used for every model.
            http_client: An existing `httpx.AsyncClient` used to make requests.
        """
        self._origin: str | None = None
        self._standard_client: AsyncOpenAI | None = None

        if openai_client is not None:
            assert region_name is None, 'Cannot provide both `openai_client` and `region_name`'
            assert base_url is None, 'Cannot provide both `openai_client` and `base_url`'
            assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            assert (aws_access_key_id, aws_secret_access_key, aws_session_token, profile_name) == (
                None,
                None,
                None,
                None,
            ), 'Cannot provide both `openai_client` and AWS credentials'
            self._client = openai_client
            return

        api_key = api_key or os.getenv('AWS_BEARER_TOKEN_BEDROCK')
        region_name = region_name or os.getenv('AWS_DEFAULT_REGION') or os.getenv('AWS_REGION')

        if base_url is not None:
            base_url = base_url.rstrip('/')
        else:
            if not region_name:
                raise UserError(
                    'Set the `AWS_DEFAULT_REGION` or `AWS_REGION` environment variable, pass `region_name`, '
                    'or pass `base_url` to use a Bedrock Mantle model.'
                )
            self._origin = f'https://bedrock-mantle.{region_name}.api.aws'
            base_url = f'{self._origin}/openai/v1'

        if http_client is None:
            http_client = create_async_http_client()
            self._own_http_client = http_client
            self._http_client_factory = create_async_http_client

        self._client = AsyncBedrockOpenAI(
            api_key=api_key,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            aws_profile=profile_name,
            aws_region=region_name,
            base_url=base_url,
            http_client=http_client,
        )

    def _set_http_client(self, http_client: httpx.AsyncClient) -> None:
        for client in (self._client, self._standard_client):
            if client is not None:
                client._client = http_client  # pyright: ignore[reportPrivateUsage]
