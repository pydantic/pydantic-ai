from __future__ import annotations as _annotations

import base64
import warnings
from collections.abc import AsyncIterable, AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Union, cast, overload

from pydantic import ValidationError
from typing_extensions import assert_never

from pydantic_ai._thinking_part import split_content_into_text_and_thinking
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers import Provider, infer_provider

from .. import ModelHTTPError, UnexpectedModelBehavior, _utils, usage
from .._output import DEFAULT_OUTPUT_TOOL_NAME, OutputObjectDefinition
from .._utils import guard_tool_call_id as _guard_tool_call_id, now_utc as _now_utc, number_to_datetime
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
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)
from ..profiles import ModelProfileSpec
from ..settings import ModelSettings
from ..tools import ToolDefinition
from . import (
    Model,
    ModelRequestParameters,
    StreamedResponse,
    check_allow_model_requests,
    download_item,
    get_user_agent,
)

try:
    from openai import NOT_GIVEN, APIStatusError, AsyncOpenAI, AsyncStream, NotGiven
    from openai.types import AllModels, chat, responses
    from openai.types.chat import (
        ChatCompletionChunk,
        ChatCompletionContentPartImageParam,
        ChatCompletionContentPartInputAudioParam,
        ChatCompletionContentPartParam,
        ChatCompletionContentPartTextParam,
    )
    from openai.types.chat.chat_completion_content_part_image_param import ImageURL
    from openai.types.chat.chat_completion_content_part_input_audio_param import InputAudio
    from openai.types.chat.chat_completion_content_part_param import File, FileFile
    from openai.types.chat.chat_completion_prediction_content_param import ChatCompletionPredictionContentParam
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

OpenAIModelName = Union[str, AllModels]
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
    """Settings used for an OpenAI model request."""

    # ALL FIELDS MUST BE `openai_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    openai_reasoning_effort: ReasoningEffort
    """Constrains effort on reasoning for [reasoning models](https://platform.openai.com/docs/guides/reasoning).

    Currently supported values are `low`, `medium`, and `high`. Reducing reasoning effort can
    result in faster responses and fewer tokens used on reasoning in a response.
    """

    openai_logprobs: bool
    """Include log probabilities in the response."""

    openai_top_logprobs: int
    """Include log probabilities of the top n tokens in the response."""

    openai_user: str
    """A unique identifier representing the end-user, which can help OpenAI monitor and detect abuse.

    See [OpenAI's safety best practices](https://platform.openai.com/docs/guides/safety-best-practices#end-user-ids) for more details.
    """

    openai_service_tier: Literal['auto', 'default', 'flex']
    """The service tier to use for the model request.

    Currently supported values are `auto`, `default`, and `flex`.
    For more information, see [OpenAI's service tiers documentation](https://platform.openai.com/docs/api-reference/chat/object#chat/object-service_tier).
    """

    openai_prediction: ChatCompletionPredictionContentParam
    """Enables [predictive outputs](https://platform.openai.com/docs/guides/predicted-outputs).

    This feature is currently only supported for some OpenAI models.
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
    """Deprecated alias for `openai_reasoning_summary`."""

    openai_reasoning_summary: Literal['detailed', 'concise']
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
        provider: Literal[
            'openai',
            'deepseek',
            'azure',
            'openrouter',
            'moonshotai',
            'vercel',
            'grok',
            'fireworks',
            'together',
            'heroku',
            'github',
        ]
        | Provider[AsyncOpenAI] = 'openai',
        profile: ModelProfileSpec | None = None,
        system_prompt_role: OpenAISystemPromptRole | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize an OpenAI model.

        Args:
            model_name: The name of the OpenAI model to use. List of model names available
                [here](https://github.com/openai/openai-python/blob/v1.54.3/src/openai/types/chat_model.py#L7)
                (Unfortunately, despite being ask to do so, OpenAI do not provide `.inv` files for their API).
            provider: The provider to use. Defaults to `'openai'`.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            system_prompt_role: The role to use for the system prompt message. If not provided, defaults to `'system'`.
                In the future, this may be inferred from the model name.
            settings: Default model settings for this model instance.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider(provider)
        self.client = provider.client

        self.system_prompt_role = system_prompt_role

        super().__init__(settings=settings, profile=profile or provider.model_profile)

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        response = await self._completions_create(
            messages, False, cast(OpenAIModelSettings, model_settings or {}), model_request_parameters
        )
        model_response = self._process_response(response)
        model_response.usage.requests = 1
        return model_response

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
        if not tools:
            tool_choice: Literal['none', 'required', 'auto'] | None = None
        elif (
            not model_request_parameters.allow_text_output
            and OpenAIModelProfile.from_profile(self.profile).openai_supports_tool_choice_required
        ):
            tool_choice = 'required'
        else:
            tool_choice = 'auto'

        openai_messages = await self._map_messages(messages)

        response_format: chat.completion_create_params.ResponseFormat | None = None
        if model_request_parameters.output_mode == 'native':
            output_object = model_request_parameters.output_object
            assert output_object is not None
            response_format = self._map_json_schema(output_object)
        elif (
            model_request_parameters.output_mode == 'prompted' and self.profile.supports_json_object_output
        ):  # pragma: no branch
            response_format = {'type': 'json_object'}

        sampling_settings = (
            model_settings
            if OpenAIModelProfile.from_profile(self.profile).openai_supports_sampling_settings
            else OpenAIModelSettings()
        )

        try:
            extra_headers = model_settings.get('extra_headers', {})
            extra_headers.setdefault('User-Agent', get_user_agent())
            return await self.client.chat.completions.create(
                model=self._model_name,
                messages=openai_messages,
                parallel_tool_calls=model_settings.get('parallel_tool_calls', NOT_GIVEN),
                tools=tools or NOT_GIVEN,
                tool_choice=tool_choice or NOT_GIVEN,
                stream=stream,
                stream_options={'include_usage': True} if stream else NOT_GIVEN,
                stop=model_settings.get('stop_sequences', NOT_GIVEN),
                max_completion_tokens=model_settings.get('max_tokens', NOT_GIVEN),
                timeout=model_settings.get('timeout', NOT_GIVEN),
                response_format=response_format or NOT_GIVEN,
                seed=model_settings.get('seed', NOT_GIVEN),
                reasoning_effort=model_settings.get('openai_reasoning_effort', NOT_GIVEN),
                user=model_settings.get('openai_user', NOT_GIVEN),
                service_tier=model_settings.get('openai_service_tier', NOT_GIVEN),
                prediction=model_settings.get('openai_prediction', NOT_GIVEN),
                temperature=sampling_settings.get('temperature', NOT_GIVEN),
                top_p=sampling_settings.get('top_p', NOT_GIVEN),
                presence_penalty=sampling_settings.get('presence_penalty', NOT_GIVEN),
                frequency_penalty=sampling_settings.get('frequency_penalty', NOT_GIVEN),
                logit_bias=sampling_settings.get('logit_bias', NOT_GIVEN),
                logprobs=sampling_settings.get('openai_logprobs', NOT_GIVEN),
                top_logprobs=sampling_settings.get('openai_top_logprobs', NOT_GIVEN),
                extra_headers=extra_headers,
                extra_body=model_settings.get('extra_body'),
            )
        except APIStatusError as e:
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise  # pragma: no cover

    def _process_response(self, response: chat.ChatCompletion | str) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        # Although the OpenAI SDK claims to return a Pydantic model (`ChatCompletion`) from the chat completions function:
        # * it hasn't actually performed validation (presumably they're creating the model with `model_construct` or something?!)
        # * if the endpoint returns plain text, the return type is a string
        # Thus we validate it fully here.
        if not isinstance(response, chat.ChatCompletion):
            raise UnexpectedModelBehavior('Invalid response from OpenAI chat completions endpoint, expected JSON data')

        if response.created:
            timestamp = number_to_datetime(response.created)
        else:
            timestamp = _now_utc()
            response.created = int(timestamp.timestamp())

        try:
            response = chat.ChatCompletion.model_validate(response.model_dump())
        except ValidationError as e:
            raise UnexpectedModelBehavior(f'Invalid response from OpenAI chat completions endpoint: {e}') from e

        choice = response.choices[0]
        items: list[ModelResponsePart] = []
        # The `reasoning_content` is only present in DeepSeek models.
        if reasoning_content := getattr(choice.message, 'reasoning_content', None):
            items.append(ThinkingPart(content=reasoning_content))

        vendor_details: dict[str, Any] | None = None

        # Add logprobs to vendor_details if available
        if choice.logprobs is not None and choice.logprobs.content:
            # Convert logprobs to a serializable format
            vendor_details = {
                'logprobs': [
                    {
                        'token': lp.token,
                        'bytes': lp.bytes,
                        'logprob': lp.logprob,
                        'top_logprobs': [
                            {'token': tlp.token, 'bytes': tlp.bytes, 'logprob': tlp.logprob} for tlp in lp.top_logprobs
                        ],
                    }
                    for lp in choice.logprobs.content
                ],
            }

        if choice.message.content is not None:
            items.extend(split_content_into_text_and_thinking(choice.message.content))
        if choice.message.tool_calls is not None:
            for c in choice.message.tool_calls:
                part = ToolCallPart(c.function.name, c.function.arguments, tool_call_id=c.id)
                part.tool_call_id = _guard_tool_call_id(part)
                items.append(part)
        return ModelResponse(
            items,
            usage=_map_usage(response),
            model_name=response.model,
            timestamp=timestamp,
            vendor_details=vendor_details,
            vendor_id=response.id,
        )

    async def _process_streamed_response(self, response: AsyncStream[ChatCompletionChunk]) -> OpenAIStreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):
            raise UnexpectedModelBehavior(  # pragma: no cover
                'Streamed response ended without content or tool calls'
            )

        return OpenAIStreamedResponse(
            _model_name=self._model_name,
            _response=peekable_response,
            _timestamp=number_to_datetime(first_chunk.created),
        )

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[chat.ChatCompletionToolParam]:
        tools = [self._map_tool_definition(r) for r in model_request_parameters.function_tools]
        if model_request_parameters.output_tools:
            tools += [self._map_tool_definition(r) for r in model_request_parameters.output_tools]
        return tools

    async def _map_messages(self, messages: list[ModelMessage]) -> list[chat.ChatCompletionMessageParam]:
        """Just maps a `pydantic_ai.Message` to a `openai.types.ChatCompletionMessageParam`."""
        openai_messages: list[chat.ChatCompletionMessageParam] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                async for item in self._map_user_message(message):
                    openai_messages.append(item)
            elif isinstance(message, ModelResponse):
                texts: list[str] = []
                tool_calls: list[chat.ChatCompletionMessageToolCallParam] = []
                for item in message.parts:
                    if isinstance(item, TextPart):
                        texts.append(item.content)
                    elif isinstance(item, ThinkingPart):
                        # NOTE: We don't send ThinkingPart to the providers yet. If you are unsatisfied with this,
                        # please open an issue. The below code is the code to send thinking to the provider.
                        # texts.append(f'<think>\n{item.content}\n</think>')
                        pass
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
                openai_messages.append(message_param)
            else:
                assert_never(message)
        if instructions := self._get_instructions(messages):
            openai_messages.insert(0, chat.ChatCompletionSystemMessageParam(content=instructions, role='system'))
        return openai_messages

    @staticmethod
    def _map_tool_call(t: ToolCallPart) -> chat.ChatCompletionMessageToolCallParam:
        return chat.ChatCompletionMessageToolCallParam(
            id=_guard_tool_call_id(t=t),
            type='function',
            function={'name': t.tool_name, 'arguments': t.args_as_json_str()},
        )

    def _map_json_schema(self, o: OutputObjectDefinition) -> chat.completion_create_params.ResponseFormat:
        response_format_param: chat.completion_create_params.ResponseFormatJSONSchema = {  # pyright: ignore[reportPrivateImportUsage]
            'type': 'json_schema',
            'json_schema': {'name': o.name or DEFAULT_OUTPUT_TOOL_NAME, 'schema': o.json_schema, 'strict': True},
        }
        if o.description:
            response_format_param['json_schema']['description'] = o.description
        if OpenAIModelProfile.from_profile(self.profile).openai_supports_strict_tool_definition:  # pragma: no branch
            response_format_param['json_schema']['strict'] = o.strict
        return response_format_param

    def _map_tool_definition(self, f: ToolDefinition) -> chat.ChatCompletionToolParam:
        tool_param: chat.ChatCompletionToolParam = {
            'type': 'function',
            'function': {
                'name': f.name,
                'description': f.description or '',
                'parameters': f.parameters_json_schema,
            },
        }
        if f.strict and OpenAIModelProfile.from_profile(self.profile).openai_supports_strict_tool_definition:
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
                    yield chat.ChatCompletionUserMessageParam(  # pragma: no cover
                        role='user', content=part.model_response()
                    )
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
                    elif item.is_document:
                        content.append(
                            File(
                                file=FileFile(
                                    file_data=f'data:{item.media_type};base64,{base64_encoded}',
                                    filename=f'filename.{item.format}',
                                ),
                                type='file',
                            )
                        )
                    else:  # pragma: no cover
                        raise RuntimeError(f'Unsupported binary content type: {item.media_type}')
                elif isinstance(item, AudioUrl):
                    downloaded_item = await download_item(item, data_format='base64', type_format='extension')
                    assert downloaded_item['data_type'] in (
                        'wav',
                        'mp3',
                    ), f'Unsupported audio format: {downloaded_item["data_type"]}'
                    audio = InputAudio(data=downloaded_item['data'], format=downloaded_item['data_type'])
                    content.append(ChatCompletionContentPartInputAudioParam(input_audio=audio, type='input_audio'))
                elif isinstance(item, DocumentUrl):
                    downloaded_item = await download_item(item, data_format='base64_uri', type_format='extension')
                    file = File(
                        file=FileFile(
                            file_data=downloaded_item['data'], filename=f'filename.{downloaded_item["data_type"]}'
                        ),
                        type='file',
                    )
                    content.append(file)
                elif isinstance(item, VideoUrl):  # pragma: no cover
                    raise NotImplementedError('VideoUrl is not supported for OpenAI')
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
        provider: Literal['openai', 'deepseek', 'azure', 'openrouter', 'grok', 'fireworks', 'together']
        | Provider[AsyncOpenAI] = 'openai',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize an OpenAI Responses model.

        Args:
            model_name: The name of the OpenAI model to use.
            provider: The provider to use. Defaults to `'openai'`.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            settings: Default model settings for this model instance.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider(provider)
        self.client = provider.client

        super().__init__(settings=settings, profile=profile or provider.model_profile)

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
    ) -> ModelResponse:
        check_allow_model_requests()
        response = await self._responses_create(
            messages, False, cast(OpenAIResponsesModelSettings, model_settings or {}), model_request_parameters
        )
        return self._process_response(response)

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

    def _process_response(self, response: responses.Response) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        timestamp = number_to_datetime(response.created_at)
        items: list[ModelResponsePart] = []
        for item in response.output:
            if item.type == 'reasoning':
                for summary in item.summary:
                    # NOTE: We use the same id for all summaries because we can merge them on the round trip.
                    # The providers don't force the signature to be unique.
                    items.append(ThinkingPart(content=summary.text, id=item.id))
            elif item.type == 'message':
                for content in item.content:
                    if content.type == 'output_text':  # pragma: no branch
                        items.append(TextPart(content.text))
            elif item.type == 'function_call':
                items.append(ToolCallPart(item.name, item.arguments, tool_call_id=item.call_id))
        return ModelResponse(
            items,
            usage=_map_usage(response),
            model_name=response.model,
            vendor_id=response.id,
            timestamp=timestamp,
        )

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
            _timestamp=number_to_datetime(first_chunk.response.created_at),
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

        if not tools:
            tool_choice: Literal['none', 'required', 'auto'] | None = None
        elif not model_request_parameters.allow_text_output:
            tool_choice = 'required'
        else:
            tool_choice = 'auto'

        instructions, openai_messages = await self._map_messages(messages)
        reasoning = self._get_reasoning(model_settings)

        text: responses.ResponseTextConfigParam | None = None
        if model_request_parameters.output_mode == 'native':
            output_object = model_request_parameters.output_object
            assert output_object is not None
            text = {'format': self._map_json_schema(output_object)}
        elif (
            model_request_parameters.output_mode == 'prompted' and self.profile.supports_json_object_output
        ):  # pragma: no branch
            text = {'format': {'type': 'json_object'}}

            # Without this trick, we'd hit this error:
            # > Response input messages must contain the word 'json' in some form to use 'text.format' of type 'json_object'.
            # Apparently they're only checking input messages for "JSON", not instructions.
            assert isinstance(instructions, str)
            openai_messages.insert(0, responses.EasyInputMessageParam(role='system', content=instructions))
            instructions = NOT_GIVEN

        sampling_settings = (
            model_settings
            if OpenAIModelProfile.from_profile(self.profile).openai_supports_sampling_settings
            else OpenAIResponsesModelSettings()
        )

        try:
            extra_headers = model_settings.get('extra_headers', {})
            extra_headers.setdefault('User-Agent', get_user_agent())
            return await self.client.responses.create(
                input=openai_messages,
                model=self._model_name,
                instructions=instructions,
                parallel_tool_calls=model_settings.get('parallel_tool_calls', NOT_GIVEN),
                tools=tools or NOT_GIVEN,
                tool_choice=tool_choice or NOT_GIVEN,
                max_output_tokens=model_settings.get('max_tokens', NOT_GIVEN),
                stream=stream,
                temperature=sampling_settings.get('temperature', NOT_GIVEN),
                top_p=sampling_settings.get('top_p', NOT_GIVEN),
                truncation=model_settings.get('openai_truncation', NOT_GIVEN),
                timeout=model_settings.get('timeout', NOT_GIVEN),
                reasoning=reasoning,
                user=model_settings.get('openai_user', NOT_GIVEN),
                text=text or NOT_GIVEN,
                extra_headers=extra_headers,
                extra_body=model_settings.get('extra_body'),
            )
        except APIStatusError as e:
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise  # pragma: no cover

    def _get_reasoning(self, model_settings: OpenAIResponsesModelSettings) -> Reasoning | NotGiven:
        reasoning_effort = model_settings.get('openai_reasoning_effort', None)
        reasoning_summary = model_settings.get('openai_reasoning_summary', None)
        reasoning_generate_summary = model_settings.get('openai_reasoning_generate_summary', None)

        if reasoning_summary and reasoning_generate_summary:  # pragma: no cover
            raise ValueError('`openai_reasoning_summary` and `openai_reasoning_generate_summary` cannot both be set.')

        if reasoning_generate_summary is not None:  # pragma: no cover
            warnings.warn(
                '`openai_reasoning_generate_summary` is deprecated, use `openai_reasoning_summary` instead',
                DeprecationWarning,
            )
            reasoning_summary = reasoning_generate_summary

        if reasoning_effort is None and reasoning_summary is None:
            return NOT_GIVEN
        return Reasoning(effort=reasoning_effort, summary=reasoning_summary)

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[responses.FunctionToolParam]:
        tools = [self._map_tool_definition(r) for r in model_request_parameters.function_tools]
        if model_request_parameters.output_tools:
            tools += [self._map_tool_definition(r) for r in model_request_parameters.output_tools]
        return tools

    def _map_tool_definition(self, f: ToolDefinition) -> responses.FunctionToolParam:
        return {
            'name': f.name,
            'parameters': f.parameters_json_schema,
            'type': 'function',
            'description': f.description,
            'strict': bool(
                f.strict and OpenAIModelProfile.from_profile(self.profile).openai_supports_strict_tool_definition
            ),
        }

    async def _map_messages(
        self, messages: list[ModelMessage]
    ) -> tuple[str | NotGiven, list[responses.ResponseInputItemParam]]:
        """Just maps a `pydantic_ai.Message` to a `openai.types.responses.ResponseInputParam`."""
        openai_messages: list[responses.ResponseInputItemParam] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, SystemPromptPart):
                        openai_messages.append(responses.EasyInputMessageParam(role='system', content=part.content))
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
                # last_thinking_part_idx: int | None = None
                for item in message.parts:
                    if isinstance(item, TextPart):
                        openai_messages.append(responses.EasyInputMessageParam(role='assistant', content=item.content))
                    elif isinstance(item, ToolCallPart):
                        openai_messages.append(self._map_tool_call(item))
                    elif isinstance(item, ThinkingPart):
                        # NOTE: We don't send ThinkingPart to the providers yet. If you are unsatisfied with this,
                        # please open an issue. The below code is the code to send thinking to the provider.
                        # if last_thinking_part_idx is not None:
                        #     reasoning_item = cast(responses.ResponseReasoningItemParam, openai_messages[last_thinking_part_idx])  # fmt: skip
                        #     if item.id == reasoning_item['id']:
                        #         assert isinstance(reasoning_item['summary'], list)
                        #         reasoning_item['summary'].append(Summary(text=item.content, type='summary_text'))
                        #         continue
                        # last_thinking_part_idx = len(openai_messages)
                        # openai_messages.append(
                        #     responses.ResponseReasoningItemParam(
                        #         id=item.id or generate_tool_call_id(),
                        #         summary=[Summary(text=item.content, type='summary_text')],
                        #         type='reasoning',
                        #     )
                        # )
                        pass
                    else:
                        assert_never(item)
            else:
                assert_never(message)
        instructions = self._get_instructions(messages) or NOT_GIVEN
        return instructions, openai_messages

    @staticmethod
    def _map_tool_call(t: ToolCallPart) -> responses.ResponseFunctionToolCallParam:
        return responses.ResponseFunctionToolCallParam(
            arguments=t.args_as_json_str(),
            call_id=_guard_tool_call_id(t=t),
            name=t.tool_name,
            type='function_call',
        )

    def _map_json_schema(self, o: OutputObjectDefinition) -> responses.ResponseFormatTextJSONSchemaConfigParam:
        response_format_param: responses.ResponseFormatTextJSONSchemaConfigParam = {
            'type': 'json_schema',
            'name': o.name or DEFAULT_OUTPUT_TOOL_NAME,
            'schema': o.json_schema,
        }
        if o.description:
            response_format_param['description'] = o.description
        if OpenAIModelProfile.from_profile(self.profile).openai_supports_strict_tool_definition:  # pragma: no branch
            response_format_param['strict'] = o.strict
        return response_format_param

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
                    downloaded_item = await download_item(item, data_format='base64_uri', type_format='extension')
                    content.append(
                        responses.ResponseInputFileParam(
                            type='input_file',
                            file_data=downloaded_item['data'],
                            filename=f'filename.{downloaded_item["data_type"]}',
                        )
                    )
                elif isinstance(item, DocumentUrl):
                    downloaded_item = await download_item(item, data_format='base64_uri', type_format='extension')
                    content.append(
                        responses.ResponseInputFileParam(
                            type='input_file',
                            file_data=downloaded_item['data'],
                            filename=f'filename.{downloaded_item["data_type"]}',
                        )
                    )
                elif isinstance(item, VideoUrl):  # pragma: no cover
                    raise NotImplementedError('VideoUrl is not supported for OpenAI.')
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
            if content:
                maybe_event = self._parts_manager.handle_text_delta(
                    vendor_part_id='content', content=content, extract_think_tags=True
                )
                if maybe_event is not None:  # pragma: no branch
                    yield maybe_event

            # Handle reasoning part of the response, present in DeepSeek models
            if reasoning_content := getattr(choice.delta, 'reasoning_content', None):
                yield self._parts_manager.handle_thinking_delta(
                    vendor_part_id='reasoning_content', content=reasoning_content
                )

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
                    tool_call_id=None,
                )
                if maybe_event is not None:  # pragma: no branch
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
                        tool_call_id=chunk.item.call_id,
                    )
                elif isinstance(chunk.item, responses.ResponseReasoningItem):
                    content = chunk.item.summary[0].text if chunk.item.summary else ''
                    yield self._parts_manager.handle_thinking_delta(
                        vendor_part_id=chunk.item.id,
                        content=content,
                        signature=chunk.item.id,
                    )
                elif isinstance(chunk.item, responses.ResponseOutputMessage):
                    pass
                else:
                    warnings.warn(  # pragma: no cover
                        f'Handling of this item type is not yet implemented. Please report on our GitHub: {chunk}',
                        UserWarning,
                    )

            elif isinstance(chunk, responses.ResponseOutputItemDoneEvent):
                # NOTE: We only need this if the tool call deltas don't include the final info.
                pass

            elif isinstance(chunk, responses.ResponseReasoningSummaryPartAddedEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseReasoningSummaryPartDoneEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseReasoningSummaryTextDoneEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseReasoningSummaryTextDeltaEvent):
                yield self._parts_manager.handle_thinking_delta(
                    vendor_part_id=chunk.item_id,
                    content=chunk.delta,
                    signature=chunk.item_id,
                )

            elif isinstance(chunk, responses.ResponseTextDeltaEvent):
                maybe_event = self._parts_manager.handle_text_delta(
                    vendor_part_id=chunk.content_index, content=chunk.delta
                )
                if maybe_event is not None:  # pragma: no branch
                    yield maybe_event

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
        details: dict[str, int] = {
            key: value
            for key, value in response_usage.model_dump(
                exclude={'input_tokens', 'output_tokens', 'total_tokens'}
            ).items()
            if isinstance(value, int)
        }
        details['reasoning_tokens'] = response_usage.output_tokens_details.reasoning_tokens
        details['cached_tokens'] = response_usage.input_tokens_details.cached_tokens
        return usage.Usage(
            request_tokens=response_usage.input_tokens,
            response_tokens=response_usage.output_tokens,
            total_tokens=response_usage.total_tokens,
            details=details,
        )
    else:
        details = {
            key: value
            for key, value in response_usage.model_dump(
                exclude={'prompt_tokens', 'completion_tokens', 'total_tokens'}
            ).items()
            if isinstance(value, int)
        }
        if response_usage.completion_tokens_details is not None:
            details.update(response_usage.completion_tokens_details.model_dump(exclude_none=True))
        if response_usage.prompt_tokens_details is not None:
            details.update(response_usage.prompt_tokens_details.model_dump(exclude_none=True))
        return usage.Usage(
            requests=1,
            request_tokens=response_usage.prompt_tokens,
            response_tokens=response_usage.completion_tokens,
            total_tokens=response_usage.total_tokens,
            details=details,
        )
