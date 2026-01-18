from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Any, Literal

from typing_extensions import override

from ..profiles import ModelProfileSpec
from ..providers import Provider
from ..settings import ModelSettings
from ..usage import RequestUsage
from . import (
    OpenAIChatCompatibleProvider,
)
from .openai import OpenAIChatModel, OpenAIChatModelSettings, OpenAIStreamedResponse

try:
    from openai import AsyncOpenAI
    from openai.types import chat
except ImportError as _import_error:
    raise ImportError(
        'Please install `openai` to use the Databricks model, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


__all__ = ('DatabricksModel',)


class DatabricksModelSettings(OpenAIChatModelSettings, total=False):
    """Settings used for a Databricks model request."""


@dataclass(init=False)
class DatabricksModel(OpenAIChatModel):
    """A model that uses the Databricks API.

    This class subclasses `OpenAIChatModel` to handle Databricks-specific behavior,
    specifically the `content` field in responses which can be a list of content items
    instead of a string.
    """

    def __init__(
        self,
        model_name: str,
        *,
        provider: OpenAIChatCompatibleProvider
        | Literal['databricks', 'gateway']
        | Provider[AsyncOpenAI] = 'databricks',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        super().__init__(
            model_name=model_name,
            provider=provider,
            profile=profile,
            settings=settings,
        )

    @property
    def _streamed_response_cls(self) -> type[OpenAIStreamedResponse]:
        """Tell the model to use our custom class for streaming responses."""
        return DatabricksStreamedResponse

    @override
    def _map_usage(self, response: chat.ChatCompletion) -> RequestUsage:
        """Override usage mapping to handle Databricks' flat structure.

        Databricks returns `reasoning_tokens` at the top level of the usage object,
        whereas OpenAI places them inside `completion_tokens_details`.
        """
        if not response.usage:
            return RequestUsage()

        request_usage = RequestUsage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            details={},
        )

        reasoning_tokens = getattr(response.usage, 'reasoning_tokens', None)

        if reasoning_tokens is None and response.usage.model_extra:
            reasoning_tokens = response.usage.model_extra.get('reasoning_tokens')

        if reasoning_tokens is not None:
            request_usage.details['reasoning_tokens'] = reasoning_tokens

        return request_usage

    def _process_provider_details(self, response: chat.ChatCompletion) -> dict[str, Any] | None:
        """Capture Databricks-specific details.

        This ensures raw usage data (including unique Databricks fields)
        is available in `response.provider_details`.
        """
        details = super()._process_provider_details(response) or {}

        if response.usage:
            details['usage'] = response.usage.model_dump()

        if safety_id := getattr(response, 'safety_identifier', None):
            details['safety_identifier'] = safety_id

        return details


class DatabricksStreamedResponse(OpenAIStreamedResponse):
    """Custom streaming response to handle Databricks specific usage fields."""

    _internal_provider_details: dict[str, Any] | None = None

    def usage(self) -> RequestUsage:
        """Override usage mapping to handle Databricks' flat structure in streams."""
        if not self._usage:
            return RequestUsage()

        usage = RequestUsage(
            input_tokens=self._usage.prompt_tokens, output_tokens=self._usage.completion_tokens, requests=1, details={}
        )

        # Handle Databricks specific flat 'reasoning_tokens'
        reasoning_tokens = getattr(self._usage, 'reasoning_tokens', None)
        if reasoning_tokens is None and hasattr(self._usage, 'model_extra'):
            reasoning_tokens = (self._usage.model_extra or {}).get('reasoning_tokens')

        if reasoning_tokens is not None:
            usage.details['reasoning_tokens'] = reasoning_tokens

        return usage

    @property
    def provider_details(self) -> dict[str, Any]:
        """Ensure raw usage is included in provider details for streaming."""
        details = self._internal_provider_details or {}

        if self._usage:
            details['usage'] = self._usage

        return details

    @provider_details.setter
    def provider_details(self, value: dict[str, Any]) -> None:
        """Allow the base class to set provider_details."""
        self._internal_provider_details = value
