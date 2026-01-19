from __future__ import annotations as _annotations

from dataclasses import dataclass, replace
from typing import Annotated, Any, Iterable, Literal

from openai import NOT_GIVEN, APIConnectionError, APIStatusError, AsyncStream
from openai.types import chat
from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from typing_extensions import override

from pydantic_ai import ModelAPIError, ModelHTTPError, ModelMessage, ModelResponse, ModelResponseStreamEvent
from pydantic_ai.profiles import ModelProfileSpec
from pydantic_ai.profiles.openai import OpenAIModelProfile

from ..profiles.databricks import DatabricksModelProfile
from ..providers import Provider
from ..settings import ModelSettings
from ..usage import RequestUsage
from . import OpenAIChatCompatibleProvider
from .openai import (
    OMIT,
    ModelRequestParameters,
    OpenAIChatModel,
    OpenAIChatModelSettings,
    OpenAIStreamedResponse,
    _check_azure_content_filter,
    get_user_agent,
)

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install `openai` to use the Databricks model, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


__all__ = ('DatabricksModel',)


class DbTextContent(BaseModel):
    """Represents a text block in a Databricks response."""

    type: Literal['text']
    text: str


class DbReasoningContent(BaseModel):
    """Represents a reasoning block (e.g. R1 models)."""

    type: Literal['reasoning']
    # Handle variations: sometimes 'text', sometimes 'content'
    text: str | None = None
    content: str | None = None

    def get_value(self) -> str:
        return self.text or self.content or ''


DbContentBlock = Annotated[DbTextContent | DbReasoningContent, Field(discriminator='type')]


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

    @override
    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: OpenAIChatModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> chat.ChatCompletion | AsyncStream[chat.ChatCompletionChunk] | ModelResponse:
        tools = self._get_tools(model_request_parameters)
        web_search_options = self._get_web_search_options(model_request_parameters)

        if not tools:
            tool_choice: Literal['none', 'required', 'auto'] | None = None
        elif not model_request_parameters.allow_text_output and self.profile.databricks_supports_tool_call_required:
            tool_choice = 'required'
        else:
            tool_choice = 'auto'

        openai_messages = await self._map_messages(messages, model_request_parameters)

        response_format: chat.completion_create_params.ResponseFormat | None = None
        if model_request_parameters.output_mode == 'native':
            output_object = model_request_parameters.output_object
            assert output_object is not None
            response_format = self._map_json_schema(output_object)
        elif (
            model_request_parameters.output_mode == 'prompted' and self.profile.supports_json_object_output
        ):  # pragma: no branch
            response_format = {'type': 'json_object'}

        # Fetch unsupported settings from the profile we set in __init__
        unsupported_model_settings = OpenAIModelProfile.from_profile(self.profile).openai_unsupported_model_settings

        # 1. Remove unsupported settings from the generic model_settings dict
        for setting in unsupported_model_settings:
            model_settings.pop(setting, None)

        if stream and self.profile.databricks_stream_options:
            stream_options = {'include_usage': True}
        else:
            stream_options = OMIT

        try:
            extra_headers = model_settings.get('extra_headers', {})
            extra_headers.setdefault('User-Agent', get_user_agent())
            return await self.client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                parallel_tool_calls=model_settings.get('parallel_tool_calls', OMIT),
                tools=tools or OMIT,
                tool_choice=tool_choice or OMIT,
                stream=stream,
                stream_options=stream_options,  # Pass the calculated variable
                stop=model_settings.get('stop_sequences', OMIT),
                max_completion_tokens=model_settings.get('max_tokens', OMIT),
                timeout=model_settings.get('timeout', NOT_GIVEN),
                response_format=response_format or OMIT,
                seed=model_settings.get('seed', OMIT),
                reasoning_effort=model_settings.get('openai_reasoning_effort', OMIT),
                user=model_settings.get('openai_user', OMIT),
                web_search_options=web_search_options or OMIT,
                service_tier=model_settings.get('openai_service_tier', OMIT),
                prediction=model_settings.get('openai_prediction', OMIT),
                temperature=model_settings.get('temperature', OMIT),
                top_p=model_settings.get('top_p', OMIT),
                presence_penalty=model_settings.get('presence_penalty', OMIT),
                frequency_penalty=model_settings.get('frequency_penalty', OMIT),
                logit_bias=model_settings.get('logit_bias', OMIT),
                logprobs=model_settings.get('openai_logprobs', OMIT),
                top_logprobs=model_settings.get('openai_top_logprobs', OMIT),
                prompt_cache_key=model_settings.get('openai_prompt_cache_key', OMIT),
                prompt_cache_retention=model_settings.get('openai_prompt_cache_retention', OMIT),
                extra_headers=extra_headers,
                extra_body=model_settings.get('extra_body'),
            )
        except APIStatusError as e:
            if model_response := _check_azure_content_filter(e, self.system, self.model_name):
                return model_response
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise  # pragma: lax no cover
        except APIConnectionError as e:
            raise ModelAPIError(model_name=self.model_name, message=e.message) from e

    @property
    def _streamed_response_cls(self) -> type[OpenAIStreamedResponse]:
        """Tell the model to use our custom class for streaming responses."""
        return DatabricksStreamedResponse

    @override
    def _validate_completion(self, response: chat.ChatCompletion) -> chat.ChatCompletion:
        """Normalizes Databricks responses to the strict OpenAI schema."""
        data = response.model_dump(mode='json', warnings=False)

        # For certain models, (gemini 2.5 pro) id was missing in response.
        if data.get('id') is None:
            data['id'] = 'databricks-placeholder-id'

        choices = data.get('choices', [])
        if not choices:
            return super()._validate_completion(response)

        message_payload = choices[0].get('message', {})
        raw_content = message_payload.get('content')

        if isinstance(raw_content, list):
            try:
                adapter = TypeAdapter(list[DbContentBlock])
                parsed_blocks = adapter.validate_python(raw_content)

                text_parts = [b.text for b in parsed_blocks if isinstance(b, DbTextContent)]
                full_text = ''.join(text_parts)

                reasoning_parts = [b.get_value() for b in parsed_blocks if isinstance(b, DbReasoningContent)]
                full_reasoning = ''.join(reasoning_parts)

                message_payload['content'] = full_text

                if full_reasoning:
                    message_payload['reasoning_content'] = full_reasoning

            except ValidationError:
                # Fallback: If the list contains unknown types (e.g., 'image'),
                # we leave it alone. The final validation below will raise a
                # descriptive error about the specific invalid fields.
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

    @override
    def _map_part_delta(self, choice: chat.chat_completion_chunk.Choice) -> Iterable[ModelResponseStreamEvent]:
        """Override to handle Databricks' structured content (list of blocks) instead of just plain strings."""
        content = choice.delta.content

        # Databricks specific: Content is a list of blocks (reasoning/text)
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue

                block_type = block.get('type')

                if block_type == 'reasoning':
                    val = block.get('text') or block.get('content')

                    # e.g. {'summary': [{'type': 'summary_text', 'text': 'The'}]}
                    if not val and (summary := block.get('summary')):
                        if isinstance(summary, list):
                            val = ''.join(
                                s.get('text', '')
                                for s in summary
                                if isinstance(s, dict) and s.get('type') == 'summary_text'
                            )

                    if val:
                        yield from self._parts_manager.handle_thinking_delta(
                            vendor_part_id='reasoning',
                            content=val,
                        )

                elif block_type == 'text':
                    val = block.get('text')
                    if val:
                        yield from self._parts_manager.handle_text_delta(
                            vendor_part_id='content',
                            content=val,
                        )

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
