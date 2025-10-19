from __future__ import annotations as _annotations

import os
from typing import overload

import httpx

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile, openai_model_profile
from pydantic_ai.providers import Provider

from .cerebras import CerebrasProvider
from .groq import GroqProvider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Cloudflare provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


def _groq_model_profile_cloudflare(model_name: str) -> ModelProfile | None:
    """Get the model profile for Groq models routed through Cloudflare's unified API.

    Cloudflare routes to Groq's OpenAI-compatible endpoint, so we use prefix matching
    similar to the native GroqProvider to determine the appropriate profile.
    """
    return GroqProvider().model_profile(model_name)


def _cerebras_model_profile_cloudflare(model_name: str) -> ModelProfile | None:
    """Get the model profile for Cerebras models routed through Cloudflare's unified API.

    Similar to the native CerebrasProvider, use prefix matching to determine profiles.
    """
    return CerebrasProvider().model_profile(model_name)


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
    - cf_aig_authorization: `CLOUDFLARE_AI_GATEWAY_AUTH` (optional)

    There are three usage modes:

    1. BYOK with unauthenticated gateway (bring your own API key):
        ```python
        from pydantic_ai import Agent
        from pydantic_ai.providers.cloudflare import CloudflareProvider

        provider = CloudflareProvider(
            account_id='your-account-id',
            gateway_id='your-gateway-id',
            api_key='your-openai-api-key'  # Your own provider API key
        )
        agent = Agent('openai/gpt-4o', provider=provider)
        ```

    2. BYOK with authenticated gateway (API key + gateway authentication):
        ```python
        provider = CloudflareProvider(
            account_id='your-account-id',
            gateway_id='your-gateway-id',
            api_key='your-openai-api-key',
            cf_aig_authorization='your-gateway-token'
        )
        agent = Agent('anthropic/claude-3-5-sonnet', provider=provider)
        ```

    3. Stored keys mode (use API keys stored in Cloudflare dashboard):
        ```python
        # Requires authenticated gateway - API keys are stored in your Cloudflare dashboard
        # Set use_gateway_keys=True and provide cf_aig_authorization
        provider = CloudflareProvider(
            account_id='your-account-id',
            gateway_id='your-gateway-id',
            cf_aig_authorization='your-gateway-token',
            use_gateway_keys=True
        )
        agent = Agent('openai/gpt-4o', provider=provider)
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
            'groq': _groq_model_profile_cloudflare,
            'mistral': mistral_model_profile,
            'cohere': cohere_model_profile,
            'deepseek': deepseek_model_profile,
            # NOTE: this would be the first support for perplexity in pydantic-ai
            # to remove this, the equivalent test in tests/providers/test_cloudflare.py::test_cloudflare_provider_model_profile would need to be removed
            'perplexity': openai_model_profile,  # Perplexity uses OpenAI-compatible API
            'workers-ai': openai_model_profile,  # Cloudflare Workers AI uses OpenAI-compatible API
            'workersai': openai_model_profile,  # Alternative naming
            'google-ai-studio': google_model_profile,
            'grok': grok_model_profile,
            'xai': grok_model_profile,  # xai is an alias for grok
            'cerebras': _cerebras_model_profile_cloudflare,
        }

        profile = None

        try:
            provider, model_name = model_name.split('/', 1)
        except ValueError:
            raise UserError(f"Model name must be in 'provider/model' format, got: {model_name!r}")

        if provider in provider_to_profile:
            profile = provider_to_profile[provider](model_name)
        # If provider is not recognized, profile remains None and we fall back to OpenAI-compatible behavior.
        # This matches VercelProvider's behavior of silently supporting unknown providers through the unified API.

        # As CloudflareProvider is always used with OpenAIChatModel, which used to unconditionally use OpenAIJsonSchemaTransformer,
        # we need to maintain that behavior unless json_schema_transformer is set explicitly
        return OpenAIModelProfile(
            json_schema_transformer=OpenAIJsonSchemaTransformer,
        ).update(profile)

    @staticmethod
    def _create_stored_keys_client(base_client: httpx.AsyncClient) -> httpx.AsyncClient:
        """Create an HTTP client that strips the Authorization header for stored keys mode.

        When using Cloudflare's stored keys feature (API keys stored in the dashboard),
        the Authorization header must NOT be sent. If sent, it takes precedence over
        the stored keys, breaking the feature.

        This wraps the base client with an event hook that removes the Authorization header.
        """

        async def strip_auth_header(request: httpx.Request) -> None:
            """Remove Authorization header so Cloudflare uses stored keys from dashboard."""
            if 'authorization' in request.headers:
                del request.headers['authorization']

        # Merge event hooks - preserve any existing hooks and add our strip_auth_header hook
        existing_hooks = base_client.event_hooks
        new_request_hooks = list(existing_hooks.get('request', []))
        new_request_hooks.append(strip_auth_header)

        merged_hooks = dict(existing_hooks)
        merged_hooks['request'] = new_request_hooks

        # Create new client based on the base client's configuration
        base_client._event_hooks = merged_hooks  # type: ignore[attr-defined]
        return base_client

    # Scenario 1: BYOK with unauthenticated gateway (api_key required)
    @overload
    def __init__(self, *, account_id: str, gateway_id: str, api_key: str) -> None: ...

    @overload
    def __init__(self, *, account_id: str, gateway_id: str, api_key: str, http_client: httpx.AsyncClient) -> None: ...

    # Scenario 2: BYOK with authenticated gateway (api_key + cf_aig_authorization)
    @overload
    def __init__(self, *, account_id: str, gateway_id: str, api_key: str, cf_aig_authorization: str) -> None: ...

    @overload
    def __init__(
        self,
        *,
        account_id: str,
        gateway_id: str,
        api_key: str,
        cf_aig_authorization: str,
        http_client: httpx.AsyncClient,
    ) -> None: ...

    # Scenario 3: Stored keys with authenticated gateway (use_gateway_keys=True, cf_aig_authorization required)
    @overload
    def __init__(
        self, *, account_id: str, gateway_id: str, cf_aig_authorization: str, use_gateway_keys: bool = True
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        account_id: str,
        gateway_id: str,
        cf_aig_authorization: str,
        http_client: httpx.AsyncClient,
        use_gateway_keys: bool = True,
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
        cf_aig_authorization: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
        use_gateway_keys: bool = False,
    ) -> None:
        """Initialize the Cloudflare AI Gateway provider.

        Args:
            account_id: Your Cloudflare account ID. Can also be set via CLOUDFLARE_ACCOUNT_ID environment variable.
            gateway_id: Your Cloudflare AI Gateway ID. Can also be set via CLOUDFLARE_GATEWAY_ID environment variable.
            api_key: The API key for the upstream provider (OpenAI, Anthropic, etc.).
                - Required for BYOK (bring your own key) mode (default)
                - Do NOT provide when use_gateway_keys=True (conflicts with stored keys mode)
                - Optional when using the openai_client parameter (pre-configured client)
            cf_aig_authorization: Authorization token for authenticated gateways.
                - Required when use_gateway_keys=True (stored keys mode)
                - Optional for BYOK mode (provides additional gateway authentication)
                - Can also be set via CLOUDFLARE_AI_GATEWAY_AUTH environment variable
            openai_client: Optional pre-configured AsyncOpenAI client for advanced use cases.
            http_client: Optional HTTP client to use for requests.
            use_gateway_keys: Whether to use API keys stored in your Cloudflare dashboard (default: False).
                - Set to True to use stored keys mode (requires cf_aig_authorization)
                - When True, do not provide api_key (they are mutually exclusive)
                - When False (default), you must provide api_key for BYOK mode

        Raises:
            UserError: If use_gateway_keys=True and api_key is also provided (conflicting configuration).
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

        cf_aig_authorization = cf_aig_authorization or os.getenv('CLOUDFLARE_AI_GATEWAY_AUTH')

        if use_gateway_keys:
            # Stored keys mode requires authenticated gateway
            if cf_aig_authorization is None:
                raise UserError(
                    'When use_gateway_keys=True, you must provide cf_aig_authorization.\n'
                    'Stored keys (API keys stored in Cloudflare dashboard) require an authenticated gateway.'
                )
            # Can't use both stored keys and provide your own api_key
            if api_key is not None:
                raise UserError(
                    'When use_gateway_keys=True, do not provide an api_key.\n'
                    'use_gateway_keys=True means using API keys stored in your Cloudflare dashboard,\n'
                    'which is incompatible with providing your own api_key (BYOK mode).'
                )
            # Use placeholder for AsyncOpenAI (required by AsyncOpenAI client) but we'll strip the Authorization header
            api_key = 'stored-keys-placeholder'
        elif api_key is None and openai_client is None:
            # Not using stored keys, so api_key is required (unless using pre-configured openai_client)
            raise UserError(
                'When use_gateway_keys=False (the default), you must provide an api_key for BYOK mode.\n'
                'To use API keys stored in your Cloudflare dashboard, set use_gateway_keys=True and provide cf_aig_authorization.'
            )

        self._base_url = f'https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway_id}/compat'

        default_headers = {
            'http-referer': 'https://ai.pydantic.dev/',
            'x-title': 'pydantic-ai',
        }

        if cf_aig_authorization:
            default_headers['cf-aig-authorization'] = cf_aig_authorization

        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            # If user provided http_client and we're in stored keys mode, we need to wrap it
            if use_gateway_keys:
                http_client = self._create_stored_keys_client(http_client)
            self._client = AsyncOpenAI(
                base_url=self._base_url, api_key=api_key, http_client=http_client, default_headers=default_headers
            )
        else:
            http_client = cached_async_http_client(provider='cloudflare')
            # In stored keys mode, wrap the client to strip Authorization header
            if use_gateway_keys:
                http_client = self._create_stored_keys_client(http_client)
            self._client = AsyncOpenAI(
                base_url=self._base_url, api_key=api_key, http_client=http_client, default_headers=default_headers
            )
