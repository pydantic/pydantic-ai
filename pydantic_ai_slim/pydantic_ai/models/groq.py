from __future__ import annotations as _annotations

from collections.abc import AsyncIterable, AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, cast, overload

from pydantic import BaseModel, Json, ValidationError
from pydantic_core import from_json
from typing_extensions import assert_never

from .. import ModelHTTPError, UnexpectedModelBehavior, _utils, usage
from .._output import DEFAULT_OUTPUT_TOOL_NAME, OutputObjectDefinition
from .._run_context import RunContext
from .._thinking_part import split_content_into_text_and_thinking
from .._utils import generate_tool_call_id, guard_tool_call_id as _guard_tool_call_id, number_to_datetime
from ..builtin_tools import WebSearchTool
from ..exceptions import UserError
from ..messages import (
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    DocumentUrl,
    FilePart,
    FinishReason,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from ..profiles import ModelProfile, ModelProfileSpec
from ..profiles.groq import GroqModelProfile
from ..providers import Provider, infer_provider
from ..settings import ModelSettings
from ..tools import ToolDefinition
from . import (
    Model,
    ModelRequestParameters,
    StreamedResponse,
    check_allow_model_requests,
    get_user_agent,
)

try:
    from groq import NOT_GIVEN, APIError, APIStatusError, AsyncGroq, AsyncStream
    from groq.types import chat
    from groq.types.chat.chat_completion_content_part_image_param import ImageURL
    from groq.types.chat.chat_completion_message import ExecutedTool
except ImportError as _import_error:
    raise ImportError(
        'Please install `groq` to use the Groq model, '
        'you can use the `groq` optional group — `pip install "pydantic-ai-slim[groq]"`'
    ) from _import_error

ProductionGroqModelNames = Literal[
    'distil-whisper-large-v3-en',
    'gemma2-9b-it',
    'llama-3.3-70b-versatile',
    'llama-3.1-8b-instant',
    'llama-guard-3-8b',
    'llama3-70b-8192',
    'llama3-8b-8192',
    'whisper-large-v3',
    'whisper-large-v3-turbo',
]
"""Production Groq models from <https://console.groq.com/docs/models#production-models>."""

PreviewGroqModelNames = Literal[
    'playai-tts',
    'playai-tts-arabic',
    'qwen-qwq-32b',
    'mistral-saba-24b',
    'qwen-2.5-coder-32b',
    'qwen-2.5-32b',
    'deepseek-r1-distill-qwen-32b',
    'deepseek-r1-distill-llama-70b',
    'llama-3.3-70b-specdec',
    'llama-3.2-1b-preview',
    'llama-3.2-3b-preview',
    'llama-3.2-11b-vision-preview',
    'llama-3.2-90b-vision-preview',
    'moonshotai/kimi-k2-instruct',
]
"""Preview Groq models from <https://console.groq.com/docs/models#preview-models>."""

GroqModelName = str | ProductionGroqModelNames | PreviewGroqModelNames
"""Possible Groq model names.

Since Groq supports a variety of models and the list changes frequencly, we explicitly list the named models as of 2025-03-31
but allow any name in the type hints.

See <https://console.groq.com/docs/models> for an up to date date list of models and more details.
"""

_FINISH_REASON_MAP: dict[Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call'], FinishReason] = {
    'stop': 'stop',
    'length': 'length',
    'tool_calls': 'tool_call',
    'content_filter': 'content_filter',
    'function_call': 'tool_call',
}


class GroqModelSettings(ModelSettings, total=False):
    """Settings used for a Groq model request."""

    # ALL FIELDS MUST BE `groq_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    groq_reasoning_format: Literal['hidden', 'raw', 'parsed']
    """The format of the reasoning output.

    See [the Groq docs](https://console.groq.com/docs/reasoning#reasoning-format) for more details.
    """


@dataclass(init=False)
class GroqModel(Model):
    """A model that uses the Groq API.

    Internally, this uses the [Groq Python client](https://github.com/groq/groq-python) to interact with the API.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    client: AsyncGroq = field(repr=False)

    _model_name: GroqModelName = field(repr=False)
    _provider: Provider[AsyncGroq] = field(repr=False)

    def __init__(
        self,
        model_name: GroqModelName,
        *,
        provider: Literal['groq', 'gateway'] | Provider[AsyncGroq] = 'groq',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize a Groq model.

        Args:
            model_name: The name of the Groq model to use. List of model names available
                [here](https://console.groq.com/docs/models).
            provider: The provider to use for authentication and API access. Can be either the string
                'groq' or an instance of `Provider[AsyncGroq]`. If not provided, a new provider will be
                created using the other parameters.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            settings: Model-specific settings that will be used as defaults for this model.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider('gateway/groq' if provider == 'gateway' else provider)
        self._provider = provider
        self.client = provider.client

        super().__init__(settings=settings, profile=profile or provider.model_profile)

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def model_name(self) -> GroqModelName:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The model provider."""
        return self._provider.name

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        try:
            response = await self._completions_create(
                messages, False, cast(GroqModelSettings, model_settings or {}), model_request_parameters
            )
        except ModelHTTPError as e:
            if isinstance(e.body, dict):  # pragma: no branch
                # The Groq SDK tries to be helpful by raising an exception when generated tool arguments don't match the schema,
                # but we'd rather handle it ourselves so we can tell the model to retry the tool call.
                try:
                    error = _GroqToolUseFailedError.model_validate(e.body)  # pyright: ignore[reportUnknownMemberType]
                    tool_call_part = ToolCallPart(
                        tool_name=error.error.failed_generation.name,
                        args=error.error.failed_generation.arguments,
                    )
                    return ModelResponse(
                        parts=[tool_call_part],
                        model_name=e.model_name,
                        timestamp=_utils.now_utc(),
                        provider_name=self._provider.name,
                        finish_reason='error',
                    )
                except ValidationError:
                    pass
            raise
        model_response = self._process_response(response)
        return model_response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        response = await self._completions_create(
            messages, True, cast(GroqModelSettings, model_settings or {}), model_request_parameters
        )
        async with response:
            yield await self._process_streamed_response(response, model_request_parameters)

    @overload
    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[True],
        model_settings: GroqModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncStream[chat.ChatCompletionChunk]:
        pass

    @overload
    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[False],
        model_settings: GroqModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> chat.ChatCompletion:
        pass

    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: GroqModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> chat.ChatCompletion | AsyncStream[chat.ChatCompletionChunk]:
        tools = self._get_tools(model_request_parameters)
        tools += self._get_builtin_tools(model_request_parameters)
        if not tools:
            tool_choice: Literal['none', 'required', 'auto'] | None = None
        elif not model_request_parameters.allow_text_output:
            tool_choice = 'required'
        else:
            tool_choice = 'auto'

        groq_messages = self._map_messages(messages, model_request_parameters)

        response_format: chat.completion_create_params.ResponseFormat | None = None
        if model_request_parameters.output_mode == 'native':
            output_object = model_request_parameters.output_object
            assert output_object is not None
            response_format = self._map_json_schema(output_object)
        elif (
            model_request_parameters.output_mode == 'prompted'
            and not tools
            and self.profile.supports_json_object_output
        ):  # pragma: no branch
            response_format = {'type': 'json_object'}

        try:
            extra_headers = model_settings.get('extra_headers', {})
            extra_headers.setdefault('User-Agent', get_user_agent())
            return await self.client.chat.completions.create(
                model=self._model_name,
                messages=groq_messages,
                n=1,
                parallel_tool_calls=model_settings.get('parallel_tool_calls', NOT_GIVEN),
                tools=tools or NOT_GIVEN,
                tool_choice=tool_choice or NOT_GIVEN,
                stop=model_settings.get('stop_sequences', NOT_GIVEN),
                stream=stream,
                response_format=response_format or NOT_GIVEN,
                max_tokens=model_settings.get('max_tokens', NOT_GIVEN),
                temperature=model_settings.get('temperature', NOT_GIVEN),
                top_p=model_settings.get('top_p', NOT_GIVEN),
                timeout=model_settings.get('timeout', NOT_GIVEN),
                seed=model_settings.get('seed', NOT_GIVEN),
                presence_penalty=model_settings.get('presence_penalty', NOT_GIVEN),
                reasoning_format=model_settings.get('groq_reasoning_format', NOT_GIVEN),
                frequency_penalty=model_settings.get('frequency_penalty', NOT_GIVEN),
                logit_bias=model_settings.get('logit_bias', NOT_GIVEN),
                extra_headers=extra_headers,
                extra_body=model_settings.get('extra_body'),
            )
        except APIStatusError as e:
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise  # pragma: lax no cover

    def _process_response(self, response: chat.ChatCompletion) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        timestamp = number_to_datetime(response.created)
        choice = response.choices[0]
        items: list[ModelResponsePart] = []
        if choice.message.reasoning is not None:
            # NOTE: The `reasoning` field is only present if `groq_reasoning_format` is set to `parsed`.
            items.append(ThinkingPart(content=choice.message.reasoning))
        if choice.message.executed_tools:
            for tool in choice.message.executed_tools:
                call_part, return_part = _map_executed_tool(tool, self.system)
                if call_part and return_part:  # pragma: no branch
                    items.append(call_part)
                    items.append(return_part)
        if choice.message.content:
            # NOTE: The `<think>` tag is only present if `groq_reasoning_format` is set to `raw`.
            items.extend(split_content_into_text_and_thinking(choice.message.content, self.profile.thinking_tags))
        if choice.message.tool_calls is not None:
            for c in choice.message.tool_calls:
                items.append(ToolCallPart(tool_name=c.function.name, args=c.function.arguments, tool_call_id=c.id))

        raw_finish_reason = choice.finish_reason
        provider_details = {'finish_reason': raw_finish_reason}
        finish_reason = _FINISH_REASON_MAP.get(raw_finish_reason)
        return ModelResponse(
            parts=items,
            usage=_map_usage(response),
            model_name=response.model,
            timestamp=timestamp,
            provider_response_id=response.id,
            provider_name=self._provider.name,
            finish_reason=finish_reason,
            provider_details=provider_details,
        )

    async def _process_streamed_response(
        self, response: AsyncStream[chat.ChatCompletionChunk], model_request_parameters: ModelRequestParameters
    ) -> GroqStreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):
            raise UnexpectedModelBehavior(  # pragma: no cover
                'Streamed response ended without content or tool calls'
            )

        return GroqStreamedResponse(
            model_request_parameters=model_request_parameters,
            _response=peekable_response,
            _model_name=first_chunk.model,
            _model_profile=self.profile,
            _timestamp=number_to_datetime(first_chunk.created),
            _provider_name=self._provider.name,
        )

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[chat.ChatCompletionToolParam]:
        return [self._map_tool_definition(r) for r in model_request_parameters.tool_defs.values()]

    def _get_builtin_tools(
        self, model_request_parameters: ModelRequestParameters
    ) -> list[chat.ChatCompletionToolParam]:
        tools: list[chat.ChatCompletionToolParam] = []
        for tool in model_request_parameters.builtin_tools:
            if isinstance(tool, WebSearchTool):
                if not GroqModelProfile.from_profile(self.profile).groq_always_has_web_search_builtin_tool:
                    raise UserError('`WebSearchTool` is not supported by Groq')  # pragma: no cover
            else:
                raise UserError(
                    f'`{tool.__class__.__name__}` is not supported by `GroqModel`. If it should be, please file an issue.'
                )
        return tools

    def _map_messages(
        self, messages: list[ModelMessage], model_request_parameters: ModelRequestParameters
    ) -> list[chat.ChatCompletionMessageParam]:
        """Just maps a `pydantic_ai.Message` to a `groq.types.ChatCompletionMessageParam`."""
        groq_messages: list[chat.ChatCompletionMessageParam] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                groq_messages.extend(self._map_user_message(message))
            elif isinstance(message, ModelResponse):
                texts: list[str] = []
                tool_calls: list[chat.ChatCompletionMessageToolCallParam] = []
                for item in message.parts:
                    if isinstance(item, TextPart):
                        texts.append(item.content)
                    elif isinstance(item, ToolCallPart):
                        tool_calls.append(self._map_tool_call(item))
                    elif isinstance(item, ThinkingPart):
                        start_tag, end_tag = self.profile.thinking_tags
                        texts.append('\n'.join([start_tag, item.content, end_tag]))
                    elif isinstance(item, BuiltinToolCallPart | BuiltinToolReturnPart):  # pragma: no cover
                        # These are not currently sent back
                        pass
                    elif isinstance(item, FilePart):  # pragma: no cover
                        # Files generated by models are not sent back to models that don't themselves generate files.
                        pass
                    else:
                        assert_never(item)
                message_param = chat.ChatCompletionAssistantMessageParam(role='assistant')
                if texts:
                    # Note: model responses from this model should only have one text item, so the following
                    # shouldn't merge multiple texts into one unless you switch models between runs:
                    message_param['content'] = '\n\n'.join(texts)
                if tool_calls:
                    message_param['tool_calls'] = tool_calls
                groq_messages.append(message_param)
            else:
                assert_never(message)
        if instructions := self._get_instructions(messages, model_request_parameters):
            groq_messages.insert(0, chat.ChatCompletionSystemMessageParam(role='system', content=instructions))
        return groq_messages

    @staticmethod
    def _map_tool_call(t: ToolCallPart) -> chat.ChatCompletionMessageToolCallParam:
        return chat.ChatCompletionMessageToolCallParam(
            id=_guard_tool_call_id(t=t),
            type='function',
            function={'name': t.tool_name, 'arguments': t.args_as_json_str()},
        )

    @staticmethod
    def _map_tool_definition(f: ToolDefinition) -> chat.ChatCompletionToolParam:
        return {
            'type': 'function',
            'function': {
                'name': f.name,
                'description': f.description or '',
                'parameters': f.parameters_json_schema,
            },
        }

    def _map_json_schema(self, o: OutputObjectDefinition) -> chat.completion_create_params.ResponseFormat:
        response_format_param: chat.completion_create_params.ResponseFormatResponseFormatJsonSchema = {
            'type': 'json_schema',
            'json_schema': {
                'name': o.name or DEFAULT_OUTPUT_TOOL_NAME,
                'schema': o.json_schema,
                'strict': o.strict,
            },
        }
        if o.description:  # pragma: no branch
            response_format_param['json_schema']['description'] = o.description
        return response_format_param

    @classmethod
    def _map_user_message(cls, message: ModelRequest) -> Iterable[chat.ChatCompletionMessageParam]:
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                yield chat.ChatCompletionSystemMessageParam(role='system', content=part.content)
            elif isinstance(part, UserPromptPart):
                yield cls._map_user_prompt(part)
            elif isinstance(part, ToolReturnPart):
                yield chat.ChatCompletionToolMessageParam(
                    role='tool',
                    tool_call_id=_guard_tool_call_id(t=part),
                    content=part.model_response_str(),
                )
            elif isinstance(part, RetryPromptPart):  # pragma: no branch
                if part.tool_name is None:
                    yield chat.ChatCompletionUserMessageParam(  # pragma: no cover
                        role='user', content=part.model_response()
                    )
                else:
                    yield chat.ChatCompletionToolMessageParam(
                        role='tool',
                        tool_call_id=_guard_tool_call_id(t=part),
                        content=part.model_response(),
                    )

    @staticmethod
    def _map_user_prompt(part: UserPromptPart) -> chat.ChatCompletionUserMessageParam:
        content: str | list[chat.ChatCompletionContentPartParam]
        if isinstance(part.content, str):
            content = part.content
        else:
            content = []
            for item in part.content:
                if isinstance(item, str):
                    content.append(chat.ChatCompletionContentPartTextParam(text=item, type='text'))
                elif isinstance(item, ImageUrl):
                    image_url = ImageURL(url=item.url)
                    content.append(chat.ChatCompletionContentPartImageParam(image_url=image_url, type='image_url'))
                elif isinstance(item, BinaryContent):
                    if item.is_image:
                        image_url = ImageURL(url=item.data_uri)
                        content.append(chat.ChatCompletionContentPartImageParam(image_url=image_url, type='image_url'))
                    else:
                        raise RuntimeError('Only images are supported for binary content in Groq.')
                elif isinstance(item, DocumentUrl):  # pragma: no cover
                    raise RuntimeError('DocumentUrl is not supported in Groq.')
                else:  # pragma: no cover
                    raise RuntimeError(f'Unsupported content type: {type(item)}')

        return chat.ChatCompletionUserMessageParam(role='user', content=content)


@dataclass
class GroqStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for Groq models."""

    _model_name: GroqModelName
    _model_profile: ModelProfile
    _response: AsyncIterable[chat.ChatCompletionChunk]
    _timestamp: datetime
    _provider_name: str

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:  # noqa: C901
        try:
            executed_tool_call_id: str | None = None
            reasoning_index = 0
            reasoning = False
            async for chunk in self._response:
                self._usage += _map_usage(chunk)

                if chunk.id:  # pragma: no branch
                    self.provider_response_id = chunk.id

                try:
                    choice = chunk.choices[0]
                except IndexError:
                    continue

                if raw_finish_reason := choice.finish_reason:
                    self.provider_details = {'finish_reason': raw_finish_reason}
                    self.finish_reason = _FINISH_REASON_MAP.get(raw_finish_reason)

                if choice.delta.reasoning is not None:
                    if not reasoning:
                        reasoning_index += 1
                        reasoning = True

                    # NOTE: The `reasoning` field is only present if `groq_reasoning_format` is set to `parsed`.
                    yield self._parts_manager.handle_thinking_delta(
                        vendor_part_id=f'reasoning-{reasoning_index}', content=choice.delta.reasoning
                    )
                else:
                    reasoning = False

                if choice.delta.executed_tools:
                    for tool in choice.delta.executed_tools:
                        call_part, return_part = _map_executed_tool(
                            tool, self.provider_name, streaming=True, tool_call_id=executed_tool_call_id
                        )
                        if call_part:
                            executed_tool_call_id = call_part.tool_call_id
                            yield self._parts_manager.handle_part(
                                vendor_part_id=f'executed_tools-{tool.index}-call', part=call_part
                            )
                        if return_part:
                            executed_tool_call_id = None
                            yield self._parts_manager.handle_part(
                                vendor_part_id=f'executed_tools-{tool.index}-return', part=return_part
                            )

                # Handle the text part of the response
                content = choice.delta.content
                if content:
                    maybe_event = self._parts_manager.handle_text_delta(
                        vendor_part_id='content',
                        content=content,
                        thinking_tags=self._model_profile.thinking_tags,
                        ignore_leading_whitespace=self._model_profile.ignore_streamed_leading_whitespace,
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event

                # Handle the tool calls
                for dtc in choice.delta.tool_calls or []:
                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=dtc.index,
                        tool_name=dtc.function and dtc.function.name,
                        args=dtc.function and dtc.function.arguments,
                        tool_call_id=dtc.id,
                    )
                    if maybe_event is not None:
                        yield maybe_event
        except APIError as e:
            if isinstance(e.body, dict):  # pragma: no branch
                # The Groq SDK tries to be helpful by raising an exception when generated tool arguments don't match the schema,
                # but we'd rather handle it ourselves so we can tell the model to retry the tool call
                try:
                    error = _GroqToolUseFailedInnerError.model_validate(e.body)  # pyright: ignore[reportUnknownMemberType]
                    yield self._parts_manager.handle_tool_call_part(
                        vendor_part_id='tool_use_failed',
                        tool_name=error.failed_generation.name,
                        args=error.failed_generation.arguments,
                    )
                    return
                except ValidationError as e:  # pragma: no cover
                    pass
            raise  # pragma: no cover

    @property
    def model_name(self) -> GroqModelName:
        """Get the model name of the response."""
        return self._model_name

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self._provider_name

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._timestamp


def _map_usage(completion: chat.ChatCompletionChunk | chat.ChatCompletion) -> usage.RequestUsage:
    response_usage = None
    if isinstance(completion, chat.ChatCompletion):
        response_usage = completion.usage
    elif completion.x_groq is not None:
        response_usage = completion.x_groq.usage

    if response_usage is None:
        return usage.RequestUsage()

    return usage.RequestUsage(
        input_tokens=response_usage.prompt_tokens,
        output_tokens=response_usage.completion_tokens,
    )


class _GroqToolUseFailedGeneration(BaseModel):
    name: str
    arguments: dict[str, Any]


class _GroqToolUseFailedInnerError(BaseModel):
    message: str
    type: Literal['invalid_request_error']
    code: Literal['tool_use_failed']
    failed_generation: Json[_GroqToolUseFailedGeneration]


class _GroqToolUseFailedError(BaseModel):
    # The Groq SDK tries to be helpful by raising an exception when generated tool arguments don't match the schema,
    # but we'd rather handle it ourselves so we can tell the model to retry the tool call.
    # Example payload from `exception.body`:
    # {
    #     'error': {
    #         'message': "Tool call validation failed: tool call validation failed: parameters for tool get_something_by_name did not match schema: errors: [missing properties: 'name', additionalProperties 'foo' not allowed]",
    #         'type': 'invalid_request_error',
    #         'code': 'tool_use_failed',
    #         'failed_generation': '{"name": "get_something_by_name", "arguments": {\n  "foo": "bar"\n}}',
    #     }
    # }

    error: _GroqToolUseFailedInnerError


def _map_executed_tool(
    tool: ExecutedTool, provider_name: str, streaming: bool = False, tool_call_id: str | None = None
) -> tuple[BuiltinToolCallPart | None, BuiltinToolReturnPart | None]:
    if tool.type == 'search':
        if tool.search_results and (tool.search_results.images or tool.search_results.results):
            results = tool.search_results.model_dump(mode='json')
        else:
            results = tool.output

        tool_call_id = tool_call_id or generate_tool_call_id()
        call_part = BuiltinToolCallPart(
            tool_name=WebSearchTool.kind,
            args=from_json(tool.arguments),
            provider_name=provider_name,
            tool_call_id=tool_call_id,
        )
        return_part = BuiltinToolReturnPart(
            tool_name=WebSearchTool.kind,
            content=results,
            provider_name=provider_name,
            tool_call_id=tool_call_id,
        )

        if streaming:
            if results:
                return None, return_part
            else:
                return call_part, None
        else:
            return call_part, return_part
    else:  # pragma: no cover
        return None, None
