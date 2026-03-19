from __future__ import annotations as _annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Annotated, Any, Literal, cast

from openai.types import chat
from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from typing_extensions import override

from pydantic_ai import ModelResponseStreamEvent
from pydantic_ai.profiles import ModelProfileSpec
from pydantic_ai.profiles.databricks import DatabricksModelProfile

from ..providers import Provider
from ..settings import ModelSettings
from ..usage import RequestUsage
from . import OpenAIChatCompatibleProvider
from .openai import (
    OMIT,
    OpenAIChatModel,
    OpenAIChatModelSettings,
    OpenAIStreamedResponse,
)

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install `openai` to use the Databricks model, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


__all__ = ('DatabricksModel',)


class DatabricksTextContent(BaseModel):
    """Represents a text block in a Databricks response."""

    type: Literal['text']
    text: str


class DatabricksSummaryText(BaseModel):
    """Inner model for summary text blocks found in reasoning."""

    type: Literal['summary_text']
    text: str


class DatabricksReasoningContent(BaseModel):
    """Represents a reasoning block (e.g. R1 models)."""

    type: Literal['reasoning']
    text: str | None = None
    content: str | None = None
    summary: list[DatabricksSummaryText] | None = None

    def get_value(self) -> str:
        """Centralized logic to extract the actual content string."""
        if val := (self.text or self.content):
            return val
        if self.summary:
            return ''.join(s.text for s in self.summary)
        return ''


DatabricksContentBlock = Annotated[DatabricksTextContent | DatabricksReasoningContent, Field(discriminator='type')]


class DatabricksUsage(BaseModel):
    """Explicit definition of Databricks usage to satisfy type checkers."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: int | None = None


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
        provider: OpenAIChatCompatibleProvider | Literal['databricks'] | Provider[AsyncOpenAI] = 'databricks',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        super().__init__(
            model_name=model_name,
            provider=provider,
            profile=profile,
            settings=settings,
        )

    @override
    def _get_stream_options(self, model_settings: OpenAIChatModelSettings) -> chat.ChatCompletionStreamOptionsParam:
        profile = cast(DatabricksModelProfile, self.profile)
        if not profile.databricks_stream_options:
            return cast(chat.ChatCompletionStreamOptionsParam, OMIT)
        return super()._get_stream_options(model_settings)

    @property
    def _streamed_response_cls(self) -> type[OpenAIStreamedResponse]:
        """Tell the model to use our custom class for streaming responses."""
        return DatabricksStreamedResponse

    @override
    def _validate_completion(self, response: chat.ChatCompletion) -> chat.ChatCompletion:
        """Normalizes Databricks responses to the strict OpenAI schema."""
        data = response.model_dump(mode='json', warnings=False)

        # gemini 2.5 pro doesn't return a chat id
        if data.get('id') is None:
            data['id'] = 'databricks-placeholder-id'

        choices = data.get('choices', [])
        if not choices:
            return super()._validate_completion(response)

        message_payload = choices[0].get('message', {})
        raw_content = message_payload.get('content')

        if isinstance(raw_content, list):
            try:
                adapter = TypeAdapter(list[DatabricksContentBlock])
                parsed_blocks = adapter.validate_python(raw_content)

                text_parts = [b.text for b in parsed_blocks if isinstance(b, DatabricksTextContent)]
                full_text = ''.join(text_parts)

                reasoning_parts = [b.get_value() for b in parsed_blocks if isinstance(b, DatabricksReasoningContent)]
                full_reasoning = ''.join(reasoning_parts)

                message_payload['content'] = full_text

                if full_reasoning:
                    message_payload['reasoning_content'] = full_reasoning

            except ValidationError:
                pass

        return chat.ChatCompletion.model_validate(data)

    @override
    def _map_usage(self, response: chat.ChatCompletion) -> RequestUsage:
        """Override usage mapping to handle Databricks' flat structure.

        Databricks returns `reasoning_tokens` at the top level of the usage object,
        whereas OpenAI places them inside `completion_tokens_details`.
        """
        if not response.usage:
            return RequestUsage()

        usage_obj = cast(DatabricksUsage, response.usage)

        request_usage = RequestUsage(
            input_tokens=usage_obj.prompt_tokens,
            output_tokens=usage_obj.completion_tokens,
            details={},
        )

        reasoning_tokens = getattr(usage_obj, 'reasoning_tokens', None)

        if reasoning_tokens is None and hasattr(usage_obj, 'model_extra'):
            model_extra = getattr(usage_obj, 'model_extra', {})
            if model_extra:
                reasoning_tokens = cast(int | None, model_extra.get('reasoning_tokens'))

        if reasoning_tokens is not None:
            request_usage.details['reasoning_tokens'] = reasoning_tokens

        return request_usage

    def _process_provider_details(self, response: chat.ChatCompletion) -> dict[str, Any] | None:
        """Capture Databricks-specific details."""
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

        usage_obj = cast(DatabricksUsage, self._usage)

        usage = RequestUsage(
            input_tokens=usage_obj.prompt_tokens,
            output_tokens=usage_obj.completion_tokens,
            details={},
        )

        reasoning_tokens = getattr(usage_obj, 'reasoning_tokens', None)
        if reasoning_tokens is None:
            model_extra = getattr(usage_obj, 'model_extra', {}) or {}
            reasoning_tokens = model_extra.get('reasoning_tokens')

        if reasoning_tokens is not None:
            usage.details['reasoning_tokens'] = reasoning_tokens

        return usage

    @override
    def _map_part_delta(self, choice: chat.chat_completion_chunk.Choice) -> Iterable[ModelResponseStreamEvent]:
        """Override to handle Databricks' structured content using Pydantic validation."""
        content = choice.delta.content

        if isinstance(content, list):
            try:
                blocks = TypeAdapter(list[DatabricksContentBlock]).validate_python(content)

                for block in blocks:
                    if isinstance(block, DatabricksReasoningContent):
                        if val := block.get_value():
                            yield from self._parts_manager.handle_thinking_delta(
                                vendor_part_id='reasoning',
                                content=val,
                            )
                    elif isinstance(block, DatabricksTextContent):
                        yield from self._parts_manager.handle_text_delta(
                            vendor_part_id='content',
                            content=block.text,
                        )

            except ValidationError:
                pass

            yield from self._map_tool_call_delta(choice)

        else:
            yield from super()._map_part_delta(choice)

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
