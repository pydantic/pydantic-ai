from __future__ import annotations as _annotations

import os

from pydantic_ai._http import AsyncHTTPClient, create_async_httpx2_client, warn_if_legacy_httpx_client
from pydantic_ai.providers import Provider, missing_api_key_error


class ElevenLabsProvider(Provider[AsyncHTTPClient]):
    """Provider for the ElevenLabs API.

    The provider's `client` is a plain async HTTP client: model classes make their ElevenLabs API calls with it,
    authenticated by the `api_key` and addressed to the `base_url` the provider exposes. (For
    [ElevenLabs Agents][pydantic_ai.realtime.elevenlabs_agents.ElevenLabsAgentsRealtimeModel], these are the REST
    calls around the agent WebSocket: fetching the agent configuration, minting a signed WebSocket URL, and
    optionally syncing tool definitions.)

    `base_url` doubles as the region switch: pass `https://api.eu.residency.elevenlabs.io` (or the
    India and Singapore hosts, `api.in.residency.elevenlabs.io` and `api.sg.residency.elevenlabs.io`) to keep
    both the REST calls and the derived WebSocket endpoint inside a data-residency region.
    """

    @property
    def name(self) -> str:
        return 'elevenlabs'

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def client(self) -> AsyncHTTPClient:
        return self._client

    @property
    def api_key(self) -> str:
        """The resolved API key, sent as the `xi-api-key` header on REST requests."""
        return self._api_key

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = 'https://api.elevenlabs.io',
        http_client: AsyncHTTPClient | None = None,
    ) -> None:
        """Create a new ElevenLabs provider.

        Args:
            api_key: The API key to use for authentication. If not provided, the `ELEVENLABS_API_KEY`
                environment variable will be used if available. The key is always required: even a
                public agent's conversation is preceded by REST preflight calls that authenticate
                with it.
            base_url: The base URL for the ElevenLabs API. Defaults to the global endpoint; pass a
                [data-residency host](https://elevenlabs.io/docs/product-guides/administration/data-residency)
                such as `https://api.eu.residency.elevenlabs.io` to pin a region.
            http_client: An existing async HTTP client to use. If not provided, a new one is created
                (and closed with the provider).
        """
        api_key = api_key or os.getenv('ELEVENLABS_API_KEY')
        if not api_key:
            raise missing_api_key_error(
                'Set the `ELEVENLABS_API_KEY` environment variable or pass it via '
                '`ElevenLabsProvider(api_key=...)` to use the ElevenLabs provider.'
            )
        self._api_key = api_key
        self._base_url = base_url.rstrip('/')
        if http_client is None:
            http_client = create_async_httpx2_client()
            self._own_http_client = http_client
            self._http_client_factory = create_async_httpx2_client
        else:
            warn_if_legacy_httpx_client(http_client, consumer='the ElevenLabs provider', stacklevel=2)
        self._client = http_client

    def _set_http_client(self, http_client: AsyncHTTPClient) -> None:
        # The HTTP client *is* the provider's client, so a re-created one replaces it directly.
        self._client = http_client
