from __future__ import annotations as _annotations

import base64
import warnings
from collections.abc import AsyncIterable, AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Literal, Union, cast, overload

from typing_extensions import assert_never

from pydantic_ai.providers import Provider, infer_provider

from .. import ModelHTTPError, UnexpectedModelBehavior, _utils, usage
from .._utils import guard_tool_call_id as _guard_tool_call_id
from ..messages import (
    AudioUrl,
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from ..settings import ModelSettings
from ..tools import ToolDefinition
from . import (
    Model,
    ModelRequestParameters,
    StreamedResponse,
    cached_async_http_client,
    check_allow_model_requests,
    get_user_agent,
)

try:
    from openai import NOT_GIVEN, APIStatusError, AsyncOpenAI, AsyncStream, NotGiven
    from openai.types import ChatModel, chat, responses
    from openai.types.chat import (
        ChatCompletionChunk,
        ChatCompletionContentPartImageParam,
        ChatCompletionContentPartInputAudioParam,
        ChatCompletionContentPartParam,
        ChatCompletionContentPartTextParam,
    )
    from openai.types.chat.chat_completion_content_part_image_param import ImageURL
    from openai.types.chat.chat_completion_content_part_input_audio_param import InputAudio
    from openai.types.responses import ComputerToolParam, FileSearchToolParam, WebSearchToolParam
    from openai.types.responses.response_input_param import FunctionCallOutput, Message
    from openai.types.shared import ReasoningEffort
    from openai.types.shared_params import Reasoning
except ImportError as _import_error:
    raise ImportError(
        'Please install `openai` to use the OpenAI model, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

__all__ = (
    'OpenAIModel',
    'OpenAIResponsesModel',
    'OpenAIModelSettings',
    'OpenAIResponsesModelSettings',
    'OpenAIModelName',
)

OpenAIModelName = Union[str, ChatModel]
"""
Possible OpenAI model names.

Since OpenAI supports a variety of date-stamped models, we explicitly list the latest models but
allow any name in the type hints.
See [the OpenAI docs](https://platform.openai.com/docs/models) for a full list.

Using this more broad type for the model name instead of the ChatModel definition
allows this model to be used more easily with other model types (ie, Ollama, Deepseek).
"""

OpenAISystemPromptRole = Literal['system', 'developer', 'user']


class OpenAIModelSettings(ModelSettings, total=False):
    """Settings used for an OpenAI model request.

    ALL FIELDS MUST BE `openai_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    openai_reasoning_effort: ReasoningEffort
    """Constrains effort on reasoning for [reasoning models](https://platform.openai.com/docs/guides/reasoning).

    Currently supported values are `low`, `medium`, and `high`. Reducing reasoning effort can
    result in faster responses and fewer tokens used on reasoning in a response.
    """

    openai_user: str
    """A unique identifier representing the end-user, which can help OpenAI monitor and detect abuse.

    See [OpenAI's safety best practices](https://platform.openai.com/docs/guides/safety-best-practices#end-user-ids) for more details.
    """


class OpenAIResponsesModelSettings(OpenAIModelSettings, total=False):
    """Settings used for an OpenAI Responses model request.

    ALL FIELDS MUST BE `openai_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    openai_builtin_tools: Sequence[FileSearchToolParam | WebSearchToolParam | ComputerToolParam]
    """The provided OpenAI built-in tools to use.

    See [OpenAI's built-in tools](https://platform.openai.com/docs/guides/tools?api-mode=responses) for more details.
    """

    openai_reasoning_generate_summary: Literal['detailed', 'concise']
    """A summary of the reasoning performed by the model.

    This can be useful for debugging and understanding the model's reasoning process.
    One of `concise` or `detailed`.

    Check the [OpenAI Computer use documentation](https://platform.openai.com/docs/guides/tools-computer-use#1-send-a-request-to-the-model)
    for more details.
    """

    openai_truncation: Literal['disabled', 'auto']
    """The truncation strategy to use for the model response.

    It can be either:
    - `disabled` (default): If a model response will exceed the context window size for a model, the
        request will fail with a 400 error.
    - `auto`: If the context of this response and previous ones exceeds the model's context window size,
        the model will truncate the response to fit the context window by dropping input items in the
        middle of the conversation.
    """


@dataclass(init=False)
class OpenAIModel(Model):
    """A model that uses the OpenAI API.

    Internally, this uses the [OpenAI Python client](https://github.com/openai/openai-python) to interact with the API.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    client: AsyncOpenAI = field(repr=False)
    system_prompt_role: OpenAISystemPromptRole | None = field(default=None, repr=False)

    _model_name: OpenAIModelName = field(repr=False)
    _system: str = field(default='openai', repr=False)

    def __init__(
        self,
        model_name: OpenAIModelName,
        *,
        provider: Literal['openai', 'deepseek', 'azure'] | Provider[AsyncOpenAI] = 'openai',
        system_prompt_role: OpenAISystemPromptRole | None = None,
    ):
        """Initialize an OpenAI model.

        Args:
            model_name: The name of the OpenAI model to use. List of model names available
                [here](https://github.com/openai/openai-python/blob/v1.54.3/src/openai/types/chat_model.py#L7)
                (Unfortunately, despite being ask to do so, OpenAI do not provide `.inv` files for their API).
            provider: The provider to use. Defaults to `'openai'`.
            system_prompt_role: The role to use for the system prompt message. If not provided, defaults to `'system'`.
                In the future, this may be inferred from the model name.
        """
        self._model_name = model_name
        if isinstance(provider, str):
            provider = infer_provider(provider)
        self.client = provider.client
        self.system_prompt_role = system_prompt_role

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelResponse, usage.Usage]:
        check_allow_model_requests()
        response = await self._completions_create(
            messages, False, cast(OpenAIModelSettings, model_settings or {}), model_request_parameters
        )
        return self._process_response(response), _map_usage(response)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        check_allow_model_requests()
        response = await self._completions_create(
            messages, True, cast(OpenAIModelSettings, model_settings or {}), model_request_parameters
        )
        async with response:
            yield await self._process_streamed_response(response)

    def customize_request_parameters(self, model_request_parameters: ModelRequestParameters) -> ModelRequestParameters:
        return _customize_request_parameters(model_request_parameters)

    @property
    def model_name(self) -> OpenAIModelName:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The system / model provider."""
        return self._system

    @overload
    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[True],
        model_settings: OpenAIModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncStream[ChatCompletionChunk]: ...

    @overload
    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[False],
        model_settings: OpenAIModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> chat.ChatCompletion: ...

    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: OpenAIModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> chat.ChatCompletion | AsyncStream[ChatCompletionChunk]:
        tools = self._get_tools(model_request_parameters)

        # standalone function to make it easier to override
        if not tools:
            tool_choice: Literal['none', 'required', 'auto'] | None = None
        elif not model_request_parameters.allow_text_result:
            tool_choice = 'required'
        else:
            tool_choice = 'auto'

        openai_messages: list[chat.ChatCompletionMessageParam] = []
        for m in messages:
            async for msg in self._map_message(m):
                openai_messages.append(msg)

        try:
            return await self.client.chat.completions.create(
                model=self._model_name,
                messages=openai_messages,
                n=1,
                parallel_tool_calls=model_settings.get('parallel_tool_calls', NOT_GIVEN),
                tools=tools or NOT_GIVEN,
                tool_choice=tool_choice or NOT_GIVEN,
                stream=stream,
                stream_options={'include_usage': True} if stream else NOT_GIVEN,
                stop=model_settings.get('stop_sequences', NOT_GIVEN),
                max_completion_tokens=model_settings.get('max_tokens', NOT_GIVEN),
                temperature=model_settings.get('temperature', NOT_GIVEN),
                top_p=model_settings.get('top_p', NOT_GIVEN),
                timeout=model_settings.get('timeout', NOT_GIVEN),
                seed=model_settings.get('seed', NOT_GIVEN),
                presence_penalty=model_settings.get('presence_penalty', NOT_GIVEN),
                frequency_penalty=model_settings.get('frequency_penalty', NOT_GIVEN),
                logit_bias=model_settings.get('logit_bias', NOT_GIVEN),
                reasoning_effort=model_settings.get('openai_reasoning_effort', NOT_GIVEN),
                user=model_settings.get('openai_user', NOT_GIVEN),
                extra_headers={'User-Agent': get_user_agent()},
            )
        except APIStatusError as e:
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise

    def _process_response(self, response: chat.ChatCompletion) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        timestamp = datetime.fromtimestamp(response.created, tz=timezone.utc)
        choice = response.choices[0]
        items: list[ModelResponsePart] = []
        if choice.message.content is not None:
            items.append(TextPart(choice.message.content))
        if choice.message.tool_calls is not None:
            for c in choice.message.tool_calls:
                items.append(ToolCallPart(c.function.name, c.function.arguments, tool_call_id=c.id))
        return ModelResponse(items, model_name=response.model, timestamp=timestamp)

    async def _process_streamed_response(self, response: AsyncStream[ChatCompletionChunk]) -> OpenAIStreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):
            raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')

        return OpenAIStreamedResponse(
            _model_name=self._model_name,
            _response=peekable_response,
            _timestamp=datetime.fromtimestamp(first_chunk.created, tz=timezone.utc),
        )

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[chat.ChatCompletionToolParam]:
        tools = [self._map_tool_definition(r) for r in model_request_parameters.function_tools]
        if model_request_parameters.result_tools:
            tools += [self._map_tool_definition(r) for r in model_request_parameters.result_tools]
        return tools

    async def _map_message(self, message: ModelMessage) -> AsyncIterable[chat.ChatCompletionMessageParam]:
        """Just maps a `pydantic_ai.Message` to a `openai.types.ChatCompletionMessageParam`."""
        if isinstance(message, ModelRequest):
            async for item in self._map_user_message(message):
                yield item
        elif isinstance(message, ModelResponse):
            texts: list[str] = []
            tool_calls: list[chat.ChatCompletionMessageToolCallParam] = []
            for item in message.parts:
                if isinstance(item, TextPart):
                    texts.append(item.content)
                elif isinstance(item, ToolCallPart):
                    tool_calls.append(self._map_tool_call(item))
                else:
                    assert_never(item)
            message_param = chat.ChatCompletionAssistantMessageParam(role='assistant')
            if texts:
                # Note: model responses from this model should only have one text item, so the following
                # shouldn't merge multiple texts into one unless you switch models between runs:
                message_param['content'] = '\n\n'.join(texts)
            if tool_calls:
                message_param['tool_calls'] = tool_calls
            yield message_param
        else:
            assert_never(message)

    @staticmethod
    def _map_tool_call(t: ToolCallPart) -> chat.ChatCompletionMessageToolCallParam:
        return chat.ChatCompletionMessageToolCallParam(
            id=_guard_tool_call_id(t=t),
            type='function',
            function={'name': t.tool_name, 'arguments': t.args_as_json_str()},
        )

    @staticmethod
    def _map_tool_definition(f: ToolDefinition) -> chat.ChatCompletionToolParam:
        tool_param: chat.ChatCompletionToolParam = {
            'type': 'function',
            'function': {
                'name': f.name,
                'description': f.description,
                'parameters': f.parameters_json_schema,
            },
        }
        if f.strict:
            tool_param['function']['strict'] = f.strict
        return tool_param

    async def _map_user_message(self, message: ModelRequest) -> AsyncIterable[chat.ChatCompletionMessageParam]:
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                if self.system_prompt_role == 'developer':
                    yield chat.ChatCompletionDeveloperMessageParam(role='developer', content=part.content)
                elif self.system_prompt_role == 'user':
                    yield chat.ChatCompletionUserMessageParam(role='user', content=part.content)
                else:
                    yield chat.ChatCompletionSystemMessageParam(role='system', content=part.content)
            elif isinstance(part, UserPromptPart):
                yield await self._map_user_prompt(part)
            elif isinstance(part, ToolReturnPart):
                yield chat.ChatCompletionToolMessageParam(
                    role='tool',
                    tool_call_id=_guard_tool_call_id(t=part),
                    content=part.model_response_str(),
                )
            elif isinstance(part, RetryPromptPart):
                if part.tool_name is None:
                    yield chat.ChatCompletionUserMessageParam(role='user', content=part.model_response())
                else:
                    yield chat.ChatCompletionToolMessageParam(
                        role='tool',
                        tool_call_id=_guard_tool_call_id(t=part),
                        content=part.model_response(),
                    )
            else:
                assert_never(part)

    @staticmethod
    async def _map_user_prompt(part: UserPromptPart) -> chat.ChatCompletionUserMessageParam:
        content: str | list[ChatCompletionContentPartParam]
        if isinstance(part.content, str):
            content = part.content
        else:
            content = []
            for item in part.content:
                if isinstance(item, str):
                    content.append(ChatCompletionContentPartTextParam(text=item, type='text'))
                elif isinstance(item, ImageUrl):
                    image_url = ImageURL(url=item.url)
                    content.append(ChatCompletionContentPartImageParam(image_url=image_url, type='image_url'))
                elif isinstance(item, BinaryContent):
                    base64_encoded = base64.b64encode(item.data).decode('utf-8')
                    if item.is_image:
                        image_url = ImageURL(url=f'data:{item.media_type};base64,{base64_encoded}')
                        content.append(ChatCompletionContentPartImageParam(image_url=image_url, type='image_url'))
                    elif item.is_audio:
                        assert item.format in ('wav', 'mp3')
                        audio = InputAudio(data=base64_encoded, format=item.format)
                        content.append(ChatCompletionContentPartInputAudioParam(input_audio=audio, type='input_audio'))
                    else:  # pragma: no cover
                        raise RuntimeError(f'Unsupported binary content type: {item.media_type}')
                elif isinstance(item, AudioUrl):  # pragma: no cover
                    client = cached_async_http_client()
                    response = await client.get(item.url)
                    response.raise_for_status()
                    base64_encoded = base64.b64encode(response.content).decode('utf-8')
                    audio = InputAudio(data=base64_encoded, format=response.headers.get('content-type'))
                    content.append(ChatCompletionContentPartInputAudioParam(input_audio=audio, type='input_audio'))
                elif isinstance(item, DocumentUrl):  # pragma: no cover
                    raise NotImplementedError('DocumentUrl is not supported for OpenAI')
                    # The following implementation should have worked, but it seems we have the following error:
                    # pydantic_ai.exceptions.ModelHTTPError: status_code: 400, model_name: gpt-4o, body:
                    # {
                    #   'message': "Unknown parameter: 'messages[1].content[1].file.data'.",
                    #   'type': 'invalid_request_error',
                    #   'param': 'messages[1].content[1].file.data',
                    #   'code': 'unknown_parameter'
                    # }
                    #
                    # client = cached_async_http_client()
                    # response = await client.get(item.url)
                    # response.raise_for_status()
                    # base64_encoded = base64.b64encode(response.content).decode('utf-8')
                    # media_type = response.headers.get('content-type').split(';')[0]
                    # file_data = f'data:{media_type};base64,{base64_encoded}'
                    # file = File(file={'file_data': file_data, 'file_name': item.url, 'file_id': item.url}, type='file')
                    # content.append(file)
                else:
                    assert_never(item)
        return chat.ChatCompletionUserMessageParam(role='user', content=content)


@dataclass(init=False)
class OpenAIResponsesModel(Model):
    """A model that uses the OpenAI Responses API.

    The [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) is the
    new API for OpenAI models.

    The Responses API has built-in tools, that you can use instead of building your own:

    - [Web search](https://platform.openai.com/docs/guides/tools-web-search)
    - [File search](https://platform.openai.com/docs/guides/tools-file-search)
    - [Computer use](https://platform.openai.com/docs/guides/tools-computer-use)

    Use the `openai_builtin_tools` setting to add these tools to your model.

    If you are interested in the differences between the Responses API and the Chat Completions API,
    see the [OpenAI API docs](https://platform.openai.com/docs/guides/responses-vs-chat-completions).
    """

    client: AsyncOpenAI = field(repr=False)
    system_prompt_role: OpenAISystemPromptRole | None = field(default=None)

    _model_name: OpenAIModelName = field(repr=False)
    _system: str = field(default='openai', repr=False)

    def __init__(
        self,
        model_name: OpenAIModelName,
        *,
        provider: Literal['openai', 'deepseek', 'azure'] | Provider[AsyncOpenAI] = 'openai',
    ):
        """Initialize an OpenAI Responses model.

        Args:
            model_name: The name of the OpenAI model to use.
            provider: The provider to use. Defaults to `'openai'`.
        """
        self._model_name = model_name
        if isinstance(provider, str):
            provider = infer_provider(provider)
        self.client = provider.client

    @property
    def model_name(self) -> OpenAIModelName:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The system / model provider."""
        return self._system

    async def request(
        self,
        messages: list[ModelRequest | ModelResponse],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelResponse, usage.Usage]:
        check_allow_model_requests()
        response = await self._responses_create(
            messages, False, cast(OpenAIResponsesModelSettings, model_settings or {}), model_request_parameters
        )
        return self._process_response(response), _map_usage(response)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        check_allow_model_requests()
        response = await self._responses_create(
            messages, True, cast(OpenAIResponsesModelSettings, model_settings or {}), model_request_parameters
        )
        async with response:
            yield await self._process_streamed_response(response)

    def customize_request_parameters(self, model_request_parameters: ModelRequestParameters) -> ModelRequestParameters:
        return _customize_request_parameters(model_request_parameters)

    def _process_response(self, response: responses.Response) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        timestamp = datetime.fromtimestamp(response.created_at, tz=timezone.utc)
        items: list[ModelResponsePart] = []
        items.append(TextPart(response.output_text))
        for item in response.output:
            if item.type == 'function_call':
                items.append(ToolCallPart(item.name, item.arguments, tool_call_id=item.call_id))
        return ModelResponse(items, model_name=response.model, timestamp=timestamp)

    async def _process_streamed_response(
        self, response: AsyncStream[responses.ResponseStreamEvent]
    ) -> OpenAIResponsesStreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):  # pragma: no cover
            raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')

        assert isinstance(first_chunk, responses.ResponseCreatedEvent)
        return OpenAIResponsesStreamedResponse(
            _model_name=self._model_name,
            _response=peekable_response,
            _timestamp=datetime.fromtimestamp(first_chunk.response.created_at, tz=timezone.utc),
        )

    @overload
    async def _responses_create(
        self,
        messages: list[ModelRequest | ModelResponse],
        stream: Literal[False],
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> responses.Response: ...

    @overload
    async def _responses_create(
        self,
        messages: list[ModelRequest | ModelResponse],
        stream: Literal[True],
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncStream[responses.ResponseStreamEvent]: ...

    async def _responses_create(
        self,
        messages: list[ModelRequest | ModelResponse],
        stream: bool,
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> responses.Response | AsyncStream[responses.ResponseStreamEvent]:
        tools = self._get_tools(model_request_parameters)
        tools = list(model_settings.get('openai_builtin_tools', [])) + tools

        # standalone function to make it easier to override
        if not tools:
            tool_choice: Literal['none', 'required', 'auto'] | None = None
        elif not model_request_parameters.allow_text_result:
            tool_choice = 'required'
        else:
            tool_choice = 'auto'

        system_prompt, openai_messages = await self._map_message(messages)
        reasoning = self._get_reasoning(model_settings)

        try:
            return await self.client.responses.create(
                input=openai_messages,
                model=self._model_name,
                instructions=system_prompt,
                parallel_tool_calls=model_settings.get('parallel_tool_calls', NOT_GIVEN),
                tools=tools or NOT_GIVEN,
                tool_choice=tool_choice or NOT_GIVEN,
                max_output_tokens=model_settings.get('max_tokens', NOT_GIVEN),
                stream=stream,
                temperature=model_settings.get('temperature', NOT_GIVEN),
                top_p=model_settings.get('top_p', NOT_GIVEN),
                truncation=model_settings.get('openai_truncation', NOT_GIVEN),
                timeout=model_settings.get('timeout', NOT_GIVEN),
                reasoning=reasoning,
                user=model_settings.get('openai_user', NOT_GIVEN),
                extra_headers={'User-Agent': get_user_agent()},
            )
        except APIStatusError as e:
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise

    def _get_reasoning(self, model_settings: OpenAIResponsesModelSettings) -> Reasoning | NotGiven:
        reasoning_effort = model_settings.get('openai_reasoning_effort', None)
        reasoning_generate_summary = model_settings.get('openai_reasoning_generate_summary', None)

        if reasoning_effort is None and reasoning_generate_summary is None:
            return NOT_GIVEN
        return Reasoning(effort=reasoning_effort, generate_summary=reasoning_generate_summary)

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[responses.FunctionToolParam]:
        tools = [self._map_tool_definition(r) for r in model_request_parameters.function_tools]
        if model_request_parameters.result_tools:
            tools += [self._map_tool_definition(r) for r in model_request_parameters.result_tools]
        return tools

    @staticmethod
    def _map_tool_definition(f: ToolDefinition) -> responses.FunctionToolParam:
        return {
            'name': f.name,
            'parameters': f.parameters_json_schema,
            'type': 'function',
            'description': f.description,
            # NOTE: f.strict should already be a boolean thanks to customize_request_parameters
            'strict': f.strict or False,
        }

    async def _map_message(self, messages: list[ModelMessage]) -> tuple[str, list[responses.ResponseInputItemParam]]:
        """Just maps a `pydantic_ai.Message` to a `openai.types.responses.ResponseInputParam`."""
        system_prompt: str = ''
        openai_messages: list[responses.ResponseInputItemParam] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, SystemPromptPart):
                        system_prompt += part.content
                    elif isinstance(part, UserPromptPart):
                        openai_messages.append(await self._map_user_prompt(part))
                    elif isinstance(part, ToolReturnPart):
                        openai_messages.append(
                            FunctionCallOutput(
                                type='function_call_output',
                                call_id=_guard_tool_call_id(t=part),
                                output=part.model_response_str(),
                            )
                        )
                    elif isinstance(part, RetryPromptPart):
                        # TODO(Marcelo): How do we test this conditional branch?
                        if part.tool_name is None:  # pragma: no cover
                            openai_messages.append(
                                Message(role='user', content=[{'type': 'input_text', 'text': part.model_response()}])
                            )
                        else:
                            openai_messages.append(
                                FunctionCallOutput(
                                    type='function_call_output',
                                    call_id=_guard_tool_call_id(t=part),
                                    output=part.model_response(),
                                )
                            )
                    else:
                        assert_never(part)
            elif isinstance(message, ModelResponse):
                for item in message.parts:
                    if isinstance(item, TextPart):
                        openai_messages.append(responses.EasyInputMessageParam(role='assistant', content=item.content))
                    elif isinstance(item, ToolCallPart):
                        openai_messages.append(self._map_tool_call(item))
                    else:
                        assert_never(item)
            else:
                assert_never(message)
        return system_prompt, openai_messages

    @staticmethod
    def _map_tool_call(t: ToolCallPart) -> responses.ResponseFunctionToolCallParam:
        return responses.ResponseFunctionToolCallParam(
            arguments=t.args_as_json_str(),
            call_id=_guard_tool_call_id(t=t),
            name=t.tool_name,
            type='function_call',
        )

    @staticmethod
    async def _map_user_prompt(part: UserPromptPart) -> responses.EasyInputMessageParam:
        content: str | list[responses.ResponseInputContentParam]
        if isinstance(part.content, str):
            content = part.content
        else:
            content = []
            for item in part.content:
                if isinstance(item, str):
                    content.append(responses.ResponseInputTextParam(text=item, type='input_text'))
                elif isinstance(item, BinaryContent):
                    base64_encoded = base64.b64encode(item.data).decode('utf-8')
                    if item.is_image:
                        content.append(
                            responses.ResponseInputImageParam(
                                image_url=f'data:{item.media_type};base64,{base64_encoded}',
                                type='input_image',
                                detail='auto',
                            )
                        )
                    elif item.is_document:
                        content.append(
                            responses.ResponseInputFileParam(
                                type='input_file',
                                file_data=f'data:{item.media_type};base64,{base64_encoded}',
                                # NOTE: Type wise it's not necessary to include the filename, but it's required by the
                                # API itself. If we add empty string, the server sends a 500 error - which OpenAI needs
                                # to fix. In any case, we add a placeholder name.
                                filename=f'filename.{item.format}',
                            )
                        )
                    elif item.is_audio:
                        raise NotImplementedError('Audio as binary content is not supported for OpenAI Responses API.')
                    else:  # pragma: no cover
                        raise RuntimeError(f'Unsupported binary content type: {item.media_type}')
                elif isinstance(item, ImageUrl):
                    content.append(
                        responses.ResponseInputImageParam(image_url=item.url, type='input_image', detail='auto')
                    )
                elif isinstance(item, AudioUrl):  # pragma: no cover
                    client = cached_async_http_client()
                    response = await client.get(item.url)
                    response.raise_for_status()
                    base64_encoded = base64.b64encode(response.content).decode('utf-8')
                    content.append(
                        responses.ResponseInputFileParam(
                            type='input_file',
                            file_data=f'data:{item.media_type};base64,{base64_encoded}',
                        )
                    )
                elif isinstance(item, DocumentUrl):  # pragma: no cover
                    client = cached_async_http_client()
                    response = await client.get(item.url)
                    response.raise_for_status()
                    base64_encoded = base64.b64encode(response.content).decode('utf-8')
                    content.append(
                        responses.ResponseInputFileParam(
                            type='input_file',
                            file_data=f'data:{item.media_type};base64,{base64_encoded}',
                            filename=f'filename.{item.format}',
                        )
                    )
                else:
                    assert_never(item)
        return responses.EasyInputMessageParam(role='user', content=content)


@dataclass
class OpenAIStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for OpenAI models."""

    _model_name: OpenAIModelName
    _response: AsyncIterable[ChatCompletionChunk]
    _timestamp: datetime

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        async for chunk in self._response:
            self._usage += _map_usage(chunk)

            try:
                choice = chunk.choices[0]
            except IndexError:
                continue

            # Handle the text part of the response
            content = choice.delta.content
            if content is not None:
                yield self._parts_manager.handle_text_delta(vendor_part_id='content', content=content)

            for dtc in choice.delta.tool_calls or []:
                maybe_event = self._parts_manager.handle_tool_call_delta(
                    vendor_part_id=dtc.index,
                    tool_name=dtc.function and dtc.function.name,
                    args=dtc.function and dtc.function.arguments,
                    tool_call_id=dtc.id,
                )
                if maybe_event is not None:
                    yield maybe_event

    @property
    def model_name(self) -> OpenAIModelName:
        """Get the model name of the response."""
        return self._model_name

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._timestamp


@dataclass
class OpenAIResponsesStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for OpenAI Responses API."""

    _model_name: OpenAIModelName
    _response: AsyncIterable[responses.ResponseStreamEvent]
    _timestamp: datetime

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:  # noqa: C901
        async for chunk in self._response:
            if isinstance(chunk, responses.ResponseCompletedEvent):
                self._usage += _map_usage(chunk.response)

            elif isinstance(chunk, responses.ResponseContentPartAddedEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseContentPartDoneEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseCreatedEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseFailedEvent):  # pragma: no cover
                self._usage += _map_usage(chunk.response)

            elif isinstance(chunk, responses.ResponseFunctionCallArgumentsDeltaEvent):
                maybe_event = self._parts_manager.handle_tool_call_delta(
                    vendor_part_id=chunk.item_id,
                    tool_name=None,
                    args=chunk.delta,
                    tool_call_id=chunk.item_id,
                )
                if maybe_event is not None:
                    yield maybe_event

            elif isinstance(chunk, responses.ResponseFunctionCallArgumentsDoneEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseIncompleteEvent):  # pragma: no cover
                self._usage += _map_usage(chunk.response)

            elif isinstance(chunk, responses.ResponseInProgressEvent):
                self._usage += _map_usage(chunk.response)

            elif isinstance(chunk, responses.ResponseOutputItemAddedEvent):
                if isinstance(chunk.item, responses.ResponseFunctionToolCall):
                    yield self._parts_manager.handle_tool_call_part(
                        vendor_part_id=chunk.item.id,
                        tool_name=chunk.item.name,
                        args=chunk.item.arguments,
                        tool_call_id=chunk.item.id,
                    )

            elif isinstance(chunk, responses.ResponseOutputItemDoneEvent):
                # NOTE: We only need this if the tool call deltas don't include the final info.
                pass

            elif isinstance(chunk, responses.ResponseTextDeltaEvent):
                yield self._parts_manager.handle_text_delta(vendor_part_id=chunk.content_index, content=chunk.delta)

            elif isinstance(chunk, responses.ResponseTextDoneEvent):
                pass  # there's nothing we need to do here

            else:  # pragma: no cover
                warnings.warn(
                    f'Handling of this event type is not yet implemented. Please report on our GitHub: {chunk}',
                    UserWarning,
                )

    @property
    def model_name(self) -> OpenAIModelName:
        """Get the model name of the response."""
        return self._model_name

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._timestamp


def _map_usage(response: chat.ChatCompletion | ChatCompletionChunk | responses.Response) -> usage.Usage:
    response_usage = response.usage
    if response_usage is None:
        return usage.Usage()
    elif isinstance(response_usage, responses.ResponseUsage):
        details: dict[str, int] = {}
        return usage.Usage(
            request_tokens=response_usage.input_tokens,
            response_tokens=response_usage.output_tokens,
            total_tokens=response_usage.total_tokens,
            details={
                'reasoning_tokens': response_usage.output_tokens_details.reasoning_tokens,
                'cached_tokens': response_usage.input_tokens_details.cached_tokens,
            },
        )
    else:
        details = {}
        if response_usage.completion_tokens_details is not None:
            details.update(response_usage.completion_tokens_details.model_dump(exclude_none=True))
        if response_usage.prompt_tokens_details is not None:
            details.update(response_usage.prompt_tokens_details.model_dump(exclude_none=True))
        return usage.Usage(
            request_tokens=response_usage.prompt_tokens,
            response_tokens=response_usage.completion_tokens,
            total_tokens=response_usage.total_tokens,
            details=details,
        )


class _StrictSchemaHelper:
    def make_schema_strict(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Recursively handle the schema to make it compatible with OpenAI strict mode.

        See https://platform.openai.com/docs/guides/function-calling?api-mode=responses#strict-mode for more details,
        but this basically just requires:
        * `additionalProperties` must be set to false for each object in the parameters
        * all fields in properties must be marked as required
        """
        assert isinstance(schema, dict), 'Schema must be a dictionary, this is probably a bug'

        # Create a copy to avoid modifying the original schema
        schema = schema.copy()

        # Handle $defs
        if defs := schema.get('$defs'):
            schema['$defs'] = {k: self.make_schema_strict(v) for k, v in defs.items()}

        # Process schema based on its type
        schema_type = schema.get('type')
        if schema_type == 'object':
            # Handle object type by setting additionalProperties to false
            # and adding all properties to required list
            self._make_object_schema_strict(schema)
        elif schema_type == 'array':
            # Handle array types by processing their items
            if 'items' in schema:
                items: Any = schema['items']
                schema['items'] = self.make_schema_strict(items)
            if 'prefixItems' in schema:
                prefix_items: list[Any] = schema['prefixItems']
                schema['prefixItems'] = [self.make_schema_strict(item) for item in prefix_items]

        elif schema_type in {'string', 'number', 'integer', 'boolean', 'null'}:
            pass  # Primitive types need no special handling
        elif 'oneOf' in schema:
            schema['oneOf'] = [self.make_schema_strict(item) for item in schema['oneOf']]
        elif 'anyOf' in schema:
            schema['anyOf'] = [self.make_schema_strict(item) for item in schema['anyOf']]

        return schema

    def _make_object_schema_strict(self, schema: dict[str, Any]) -> None:
        schema['additionalProperties'] = False

        # Handle patternProperties; note this may not be compatible with strict mode but is included for completeness
        if 'patternProperties' in schema and isinstance(schema['patternProperties'], dict):
            pattern_props: dict[str, Any] = schema['patternProperties']
            schema['patternProperties'] = {str(k): self.make_schema_strict(v) for k, v in pattern_props.items()}

        # Handle properties — update their schemas recursively, and make all properties required
        if 'properties' in schema and isinstance(schema['properties'], dict):
            properties: dict[str, Any] = schema['properties']
            schema['properties'] = {k: self.make_schema_strict(v) for k, v in properties.items()}
            schema['required'] = list(properties.keys())

    def is_schema_strict(self, schema: dict[str, Any]) -> bool:
        """Check if the schema is strict-mode-compatible.

        A schema is compatible if:
        * `additionalProperties` is set to false for each object in the parameters
        * all fields in properties are marked as required

        See https://platform.openai.com/docs/guides/function-calling?api-mode=responses#strict-mode for more details.
        """
        assert isinstance(schema, dict), 'Schema must be a dictionary, this is probably a bug'

        # Note that checking the defs first is usually the fastest way to proceed, but
        # it makes it hard/impossible to hit coverage below, hence all the pragma no covers.
        # I still included the handling below because I'm not _confident_ those code paths can't be hit.
        if defs := schema.get('$defs'):
            if not all(self.is_schema_strict(v) for v in defs.values()):  # pragma: no branch
                return False

        schema_type = schema.get('type')
        if schema_type == 'object':
            if not self._is_object_schema_strict(schema):
                return False
        elif schema_type == 'array':
            if 'items' in schema:
                items: Any = schema['items']
                if not self.is_schema_strict(items):  # pragma: no cover
                    return False
            if 'prefixItems' in schema:
                prefix_items: list[Any] = schema['prefixItems']
                if not all(self.is_schema_strict(item) for item in prefix_items):  # pragma: no cover
                    return False
        elif schema_type in {'string', 'number', 'integer', 'boolean', 'null'}:
            pass
        elif 'oneOf' in schema:  # pragma: no cover
            if not all(self.is_schema_strict(item) for item in schema['oneOf']):
                return False

        elif 'anyOf' in schema:  # pragma: no cover
            if not all(self.is_schema_strict(item) for item in schema['anyOf']):
                return False

        return True

    def _is_object_schema_strict(self, schema: dict[str, Any]) -> bool:
        """Check if the schema is an object and has additionalProperties set to false."""
        if schema.get('additionalProperties') is not False:
            return False
        if 'properties' not in schema:  # pragma: no cover
            return False
        if 'required' not in schema:  # pragma: no cover
            return False

        for k, v in schema['properties'].items():
            if k not in schema['required']:
                return False
            if not self.is_schema_strict(v):  # pragma: no cover
                return False

        return True


def _customize_request_parameters(model_request_parameters: ModelRequestParameters) -> ModelRequestParameters:
    """Customize the request parameters for OpenAI models."""

    def _customize_tool_def(t: ToolDefinition):
        if t.strict is True:
            parameters_json_schema = _StrictSchemaHelper().make_schema_strict(t.parameters_json_schema)
            return replace(t, parameters_json_schema=parameters_json_schema)
        elif t.strict is None:
            strict = _StrictSchemaHelper().is_schema_strict(t.parameters_json_schema)
            return replace(t, strict=strict)
        return t

    return ModelRequestParameters(
        function_tools=[_customize_tool_def(tool) for tool in model_request_parameters.function_tools],
        allow_text_result=model_request_parameters.allow_text_result,
        result_tools=[_customize_tool_def(tool) for tool in model_request_parameters.result_tools],
    )
