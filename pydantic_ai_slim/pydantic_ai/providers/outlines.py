from __future__ import annotations as _annotations

from typing import Any

from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles.outlines import outlines_model_profile
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
        raise NotImplementedError()

    @property
    def client(self) -> Any:
        """The client for the provider."""
        raise NotImplementedError()

    def model_profile(self, model_name: str) -> ModelProfile | None:
        """The model profile for the named model, if available."""
        return outlines_model_profile(model_name)
