from __future__ import annotations as _annotations

import os
from typing import TYPE_CHECKING

from pydantic_ai.exceptions import UserError
from pydantic_ai.providers import Provider, missing_api_key_error

if TYPE_CHECKING:
    from pydantic_ai.realtime import RealtimeModelProfile


class AzureVoiceLiveProvider(Provider[None]):
    """Provider for the Azure AI Voice Live API.

    Voice Live uses a raw WebSocket rather than an SDK client, so [`client`][pydantic_ai.providers.Provider.client]
    is `None`. The provider owns the endpoint, API version, and API key used for the WebSocket handshake.
    """

    @property
    def name(self) -> str:
        return 'azure-voicelive'

    @property
    def base_url(self) -> str:
        return self._endpoint

    @property
    def client(self) -> None:
        return None

    @property
    def api_version(self) -> str:
        """The Voice Live API version sent in the WebSocket URL."""
        return self._api_version

    @property
    def api_key(self) -> str:
        """The Azure resource key sent in the WebSocket `api-key` header."""
        return self._api_key

    @staticmethod
    def realtime_model_profile(model_name: str) -> RealtimeModelProfile:
        del model_name
        return {
            'supports_image_input': True,
            'supports_manual_turn_control': True,
            'supports_interruption': True,
            'supports_output_truncation': True,
            'supports_session_seeding': True,
            'supports_seeding_images': True,
            'supports_seeding_audio': True,
            'audio_input_sample_rate': 24000,
            'audio_output_sample_rate': 24000,
        }

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Create an Azure AI Voice Live provider.

        Args:
            endpoint: The Azure AI Voice Live resource endpoint. Defaults to
                `AZURE_VOICELIVE_ENDPOINT`.
            api_version: The Voice Live API version. Defaults to
                `AZURE_VOICELIVE_API_VERSION`.
            api_key: The Azure resource key. Defaults to `AZURE_VOICELIVE_API_KEY`.
        """
        endpoint = endpoint or os.getenv('AZURE_VOICELIVE_ENDPOINT')
        if not endpoint:
            raise UserError(
                'Set the `AZURE_VOICELIVE_ENDPOINT` environment variable or pass '
                '`endpoint=` to use the Azure AI Voice Live provider.'
            )
        api_version = api_version or os.getenv('AZURE_VOICELIVE_API_VERSION')
        if not api_version:
            raise UserError(
                'Set the `AZURE_VOICELIVE_API_VERSION` environment variable or pass '
                '`api_version=` to use the Azure AI Voice Live provider.'
            )
        api_key = api_key or os.getenv('AZURE_VOICELIVE_API_KEY')
        if not api_key:
            raise missing_api_key_error(
                'Set the `AZURE_VOICELIVE_API_KEY` environment variable or pass '
                '`api_key=` to use the Azure AI Voice Live provider.'
            )

        self._endpoint = endpoint.rstrip('/')
        self._api_version = api_version
        self._api_key = api_key


__all__ = ('AzureVoiceLiveProvider',)
