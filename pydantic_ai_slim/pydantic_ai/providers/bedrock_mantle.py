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
    # Mantle GPT-5.x on the `/openai/v1` Responses endpoint resets tool-call IDs to `call_0` across
    # separate responses (verified live on 5.5 and 5.6); qualify them with the response ID at ingestion
    # so pydantic-ai keeps its history-wide-unique invariant. GPT-OSS on `/v1/responses` is unaffected
    # (globally-unique IDs), so this keys on the endpoint family rather than the model version — the
    # reset follows the endpoint, and a per-version allowlist would silently miss future GPT-5.x. See #6536.
    response_scoped = interface == 'openai-responses'
    return merge_profile(
        openai_model_profile(base_model_name),
        BedrockMantleModelProfile(
            bedrock_mantle_interface=interface,
            openai_responses_tool_call_ids_are_response_scoped=response_scoped,
            # Bedrock Mantle does not serve image output for these models, unlike the direct OpenAI API
            # the base profile is resolved from (per the AWS model cards). Without this override the
            # image-output guard is bypassed and the request fails with an opaque provider error.
            supports_image_output=False,
            # Bedrock Mantle proxies the OpenAI models but not OpenAI's server-hosted tools (web search,
            # code interpreter, file search, image generation, ...); the AWS Mantle docs document only
            # custom function calling, so the base profile's native tools are disabled to fail with a
            # clean UserError rather than an opaque provider error.
            supported_native_tools=frozenset(),
        ),
    )


def _mantle_origin(base_url: str) -> str:
    """Return the scheme+host origin of a Mantle base URL, stripping a trailing `/openai/v1` or `/v1`.

    Mantle serves its two OpenAI-compatible endpoint families as siblings under one origin, so the origin
    is enough to derive both `{origin}/v1` and `{origin}/openai/v1` regardless of which one a caller passed.
    """
    origin = base_url.rstrip('/')
    for suffix in ('/openai/v1', '/v1'):
        if origin.endswith(suffix):
            return origin[: -len(suffix)]
    return origin


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

    def _client_for_interface(self, interface: BedrockMantleInterface) -> AsyncOpenAI:
        """Return the client for a Mantle interface.

        The `openai-responses` interface (GPT-5.x) is served at `/openai/v1`; `chat` and `responses`
        (GPT-OSS) are served at `/v1`. Both clients are built eagerly in `__init__` and share transport
        and auth, so this is a pure lookup.
        """
        return self._client if interface == 'openai-responses' else self._v1_client

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
            base_url: A Mantle base URL. Its origin (scheme + host, with any `/openai/v1` or `/v1` suffix
                stripped) is used to route between `/v1` and `/openai/v1` per model, the same as `region_name`.
            api_key: A Bedrock API key. If omitted, `AWS_BEARER_TOKEN_BEDROCK` is used. Use this or the
                `aws_*` credentials, not both.
            aws_access_key_id: The AWS access key ID for SigV4 authentication.
            aws_secret_access_key: The AWS secret access key for SigV4 authentication.
            aws_session_token: The AWS session token for SigV4 authentication.
            profile_name: The AWS profile name for SigV4 authentication.
            openai_client: An existing OpenAI client. If provided, no other argument may be set; its base
                URL's origin is used to derive both the `/v1` and `/openai/v1` endpoints (preserving its
                auth and transport) so every interface routes correctly.
            http_client: An existing `httpx.AsyncClient` used to make requests.
        """
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
            base_client = openai_client
            origin = _mantle_origin(str(openai_client.base_url))
        else:
            api_key = api_key or os.getenv('AWS_BEARER_TOKEN_BEDROCK')
            region_name = region_name or os.getenv('AWS_DEFAULT_REGION') or os.getenv('AWS_REGION')

            if base_url is not None:
                origin = _mantle_origin(base_url)
            elif region_name:
                origin = f'https://bedrock-mantle.{region_name}.api.aws'
            else:
                raise UserError(
                    'Set the `AWS_DEFAULT_REGION` or `AWS_REGION` environment variable, pass `region_name`, '
                    'or pass `base_url` to use a Bedrock Mantle model.'
                )

            if http_client is None:
                http_client = create_async_http_client()
                self._own_http_client = http_client
                self._http_client_factory = create_async_http_client

            base_client = AsyncBedrockOpenAI(
                api_key=api_key,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                aws_profile=profile_name,
                aws_region=region_name,
                base_url=f'{origin}/openai/v1',
                http_client=http_client,
            )

        # `_client` (the default `.client`/`.base_url`) serves the `openai-responses` interface at
        # `/openai/v1` — the endpoint for GPT-5.x, the primary Mantle models. `_v1_client` serves `chat`
        # and `responses` (GPT-OSS) at `/v1`. Both derive from one base client via `with_options`, so they
        # share transport and auth; deriving from the origin means a user-supplied `base_url`/`openai_client`
        # still routes both interfaces correctly regardless of which sibling endpoint it named.
        self._client = base_client.with_options(base_url=f'{origin}/openai/v1')
        self._v1_client = base_client.with_options(base_url=f'{origin}/v1')

    def _set_http_client(self, http_client: httpx.AsyncClient) -> None:
        for client in (self._client, self._v1_client):
            client._client = http_client  # pyright: ignore[reportPrivateUsage]
