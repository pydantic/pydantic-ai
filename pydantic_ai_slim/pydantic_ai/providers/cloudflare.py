from __future__ import annotations as _annotations

import os
from typing import overload

import httpx

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.cerebras import cerebras_provider_model_profile
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.profiles.groq import groq_provider_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile, openai_model_profile
from pydantic_ai.profiles.perplexity import perplexity_model_profile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Cloudflare provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


class CloudflareProvider(Provider[AsyncOpenAI]):
    """Provider for Cloudflare AI Gateway API.

    Cloudflare AI Gateway provides a unified OpenAI-compatible endpoint that routes
    requests to various AI providers while adding features like caching, rate limiting,
    analytics, and logging.

    !!! note
        This provider uses Cloudflare's unified API endpoint for routing requests.
        For the full list of supported providers, see
        [Cloudflare's documentation](https://developers.cloudflare.com/ai-gateway/usage/chat-completion/#supported-providers).

    This provider looks for these environment variables if they are not provided as parameters:
    - account_id: `CLOUDFLARE_ACCOUNT_ID`
    - gateway_id: `CLOUDFLARE_GATEWAY_ID`
    - gateway_auth_token: `CLOUDFLARE_AI_GATEWAY_AUTH` (optional)

    There are three usage modes:

    1. User-managed keys with unauthenticated gateway:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.cloudflare import CloudflareProvider

        model = OpenAIChatModel(
            'openai/gpt-4o',
            provider=CloudflareProvider(
                account_id='your-account-id',
                gateway_id='your-gateway-id',
                api_key='your-openai-api-key',
            ),
        )
        agent = Agent(model)
        ```

    2. User-managed keys with authenticated gateway (API key + gateway authentication):
        ```python
        from pydantic_ai import Agent
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.cloudflare import CloudflareProvider

        model = OpenAIChatModel(
            'anthropic/claude-3-5-sonnet',
            provider=CloudflareProvider(
                account_id='your-account-id',
                gateway_id='your-gateway-id',
                api_key='your-openai-api-key',
                gateway_auth_token='your-gateway-token',
            ),
        )
        agent = Agent(model)
        ```

    3. CF-managed keys mode (use API keys stored in Cloudflare dashboard):
        ```python
        from pydantic_ai import Agent
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.cloudflare import CloudflareProvider

        model = OpenAIChatModel(
            'openai/gpt-4o',
            provider=CloudflareProvider(
                account_id='your-account-id',
                gateway_id='your-gateway-id',
                gateway_auth_token='your-gateway-token',
            ),
        )
        agent = Agent(model)
        ```
    """

    @property
    def name(self) -> str:
        return 'cloudflare'

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        """Return the model profile for the given model name.

        Model names should be in the format 'provider/model', e.g., 'openai/gpt-4o',
        'anthropic/claude-3-5-sonnet', 'groq/llama-3.3-70b-versatile'.

        For the full list of supported providers, see
        [Cloudflare's documentation](https://developers.cloudflare.com/ai-gateway/usage/chat-completion/#supported-providers).
        """
        provider_to_profile = {
            'anthropic': anthropic_model_profile,
            'openai': openai_model_profile,
            'groq': groq_provider_model_profile,
            'mistral': mistral_model_profile,
            'cohere': cohere_model_profile,
            'deepseek': deepseek_model_profile,
            'perplexity': perplexity_model_profile,
            'workers-ai': openai_model_profile,  # Cloudflare Workers AI uses OpenAI-compatible API
            'workersai': openai_model_profile,  # Alternative naming
            'google-ai-studio': google_model_profile,
            'grok': grok_model_profile,
            'xai': grok_model_profile,  # xai is an alias for grok
            'cerebras': cerebras_provider_model_profile,
        }

        profile = None

        try:
            provider, model_name = model_name.split('/', 1)
        except ValueError:
            raise UserError(f"Model name must be in 'provider/model' format, got: {model_name!r}")

        if provider in provider_to_profile:
            profile = provider_to_profile[provider](model_name)

        # As CloudflareProvider is always used with OpenAIChatModel, which used to unconditionally use OpenAIJsonSchemaTransformer,
        # we need to maintain that behavior unless json_schema_transformer is set explicitly
        return OpenAIModelProfile(
            json_schema_transformer=OpenAIJsonSchemaTransformer,
        ).update(profile)

    # Scenario 1: User-managed keys with unauthenticated gateway (api_key required)
    @overload
    def __init__(self, *, account_id: str, gateway_id: str, api_key: str) -> None: ...

    @overload
    def __init__(self, *, account_id: str, gateway_id: str, api_key: str, http_client: httpx.AsyncClient) -> None: ...

    # Scenario 2: User-managed keys with authenticated gateway (api_key + gateway_auth_token)
    @overload
    def __init__(self, *, account_id: str, gateway_id: str, api_key: str, gateway_auth_token: str) -> None: ...

    @overload
    def __init__(
        self,
        *,
        account_id: str,
        gateway_id: str,
        api_key: str,
        gateway_auth_token: str,
        http_client: httpx.AsyncClient,
    ) -> None: ...

    # Scenario 3: CF-managed keys with authenticated gateway (no api_key, gateway_auth_token required)
    @overload
    def __init__(self, *, account_id: str, gateway_id: str, gateway_auth_token: str) -> None: ...

    @overload
    def __init__(
        self,
        *,
        account_id: str,
        gateway_id: str,
        gateway_auth_token: str,
        http_client: httpx.AsyncClient,
    ) -> None: ...

    # Advanced: Pre-configured OpenAI client
    @overload
    def __init__(self, *, account_id: str, gateway_id: str, openai_client: AsyncOpenAI) -> None: ...

    def __init__(
        self,
        *,
        account_id: str | None = None,
        gateway_id: str | None = None,
        api_key: str | None = None,
        gateway_auth_token: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize the Cloudflare AI Gateway provider.

        Args:
            account_id: Your Cloudflare account ID. Can also be set via CLOUDFLARE_ACCOUNT_ID environment variable.
            gateway_id: Your Cloudflare AI Gateway ID. Can also be set via CLOUDFLARE_GATEWAY_ID environment variable.
            api_key: The API key for the upstream provider (OpenAI, Anthropic, etc.).
                - Required for user-managed mode
                - Omit this (along with providing gateway_auth_token) to use CF-managed keys mode
                - Optional when using the openai_client parameter (pre-configured client)
            gateway_auth_token: Authorization token for authenticated gateways.
                - Required for CF-managed keys mode (when api_key is omitted)
                - Optional for user-managed mode (provides additional gateway authentication)
                - Can also be set via CLOUDFLARE_AI_GATEWAY_AUTH environment variable
            openai_client: Optional pre-configured AsyncOpenAI client for advanced use cases.
            http_client: Optional HTTP client to use for requests.

        Raises:
            UserError: If configuration is invalid (e.g., neither api_key nor CF-managed keys mode is configured).
        """
        account_id = account_id or os.getenv('CLOUDFLARE_ACCOUNT_ID')
        gateway_id = gateway_id or os.getenv('CLOUDFLARE_GATEWAY_ID')

        if not account_id:
            raise UserError(
                'Set the `CLOUDFLARE_ACCOUNT_ID` environment variable '
                'or pass it via `CloudflareProvider(account_id=...)` to use the Cloudflare provider.'
            )

        if not gateway_id:
            raise UserError(
                'Set the `CLOUDFLARE_GATEWAY_ID` environment variable '
                'or pass it via `CloudflareProvider(gateway_id=...)` to use the Cloudflare provider.'
            )

        gateway_auth_token = gateway_auth_token or os.getenv('CLOUDFLARE_AI_GATEWAY_AUTH')

        # Detect CF-managed keys mode: no api_key provided + gateway_auth_token present + no pre-configured client
        use_cf_managed_keys = api_key is None and gateway_auth_token is not None and openai_client is None

        if use_cf_managed_keys:
            # CF-managed keys mode: use API keys stored in Cloudflare dashboard
            # Use empty string for AsyncOpenAI - this prevents the Authorization header from being sent
            api_key = ''
        elif api_key is None and openai_client is None:
            # Not using CF-managed keys, so api_key is required (unless using pre-configured openai_client)
            raise UserError(
                'You must provide an api_key for user-managed mode.\n'
                'To use API keys stored in your Cloudflare dashboard (CF-managed), omit api_key and provide gateway_auth_token instead.'
            )

        self._base_url = f'https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway_id}/compat'

        default_headers = {
            'http-referer': 'https://ai.pydantic.dev/',
            'x-title': 'pydantic-ai',
        }

        if gateway_auth_token:
            default_headers['cf-aig-authorization'] = gateway_auth_token

        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(
                base_url=self._base_url, api_key=api_key, http_client=http_client, default_headers=default_headers
            )
        else:
            http_client = cached_async_http_client(provider='cloudflare')
            self._client = AsyncOpenAI(
                base_url=self._base_url, api_key=api_key, http_client=http_client, default_headers=default_headers
            )
