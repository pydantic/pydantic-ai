from __future__ import annotations as _annotations

from typing import Any

from pydantic_ai.profiles import ModelProfile
from pydantic_ai.providers import Provider


class OutlinesProvider(Provider[Any]):
    """Provider for Outlines API."""

    @property
    def name(self) -> str:
        """The provider name."""
        return 'outlines'

    @property
    def base_url(self) -> str:
        """The base URL for the provider API."""
        raise NotImplementedError(
            'The Outlines provider does not have a set base URL as it functions '
            + 'with a set of different underlying models.'
        )

    @property
    def client(self) -> Any:
        """The client for the provider."""
        raise NotImplementedError(
            'The Outlines provider does not have a set client as it functions '
            + 'with a set of different underlying models.'
        )

    def model_profile(self, model_name: str) -> ModelProfile | None:
        """The model profile for the named model, if available."""
        return ModelProfile(
            supports_tools=False,
            supports_json_schema_output=True,
            supports_json_object_output=True,
            default_structured_output_mode='native',
        )
