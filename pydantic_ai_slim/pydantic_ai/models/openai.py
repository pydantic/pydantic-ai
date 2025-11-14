from __future__ import annotations as _annotations

import base64
import json
import warnings
from collections.abc import AsyncIterable, AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from pydantic import ValidationError
from pydantic_core import to_json
from typing_extensions import assert_never, deprecated

from .. import ModelHTTPError, UnexpectedModelBehavior, _utils, usage
from .._output import DEFAULT_OUTPUT_TOOL_NAME, OutputObjectDefinition
from .._run_context import RunContext
from .._thinking_part import split_content_into_text_and_thinking
from .._utils import guard_tool_call_id as _guard_tool_call_id, now_utc as _now_utc, number_to_datetime
from ..builtin_tools import CodeExecutionTool, ImageGenerationTool, MCPServerTool, WebSearchTool
from ..exceptions import UserError
from ..messages import (
    AudioUrl,
    BinaryContent,
    BinaryImage,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    CachePoint,
    DocumentUrl,
    FilePart,
    FinishReason,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    PartStartEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)
from ..profiles import ModelProfile, ModelProfileSpec
from ..profiles.openai import OpenAIModelProfile, OpenAISystemPromptRole
from ..providers import Provider, infer_provider
from ..settings import ModelSettings
from ..tools import ToolDefinition
from . import Model, ModelRequestParameters, StreamedResponse, check_allow_model_requests, download_item, get_user_agent

try:
    from openai import NOT_GIVEN, APIStatusError, AsyncOpenAI, AsyncStream
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
    from openai.types.chat.chat_completion_message_custom_tool_call import ChatCompletionMessageCustomToolCall
    from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall
    from openai.types.chat.chat_completion_message_function_tool_call_param import (
        ChatCompletionMessageFunctionToolCallParam,
    )
    from openai.types.chat.chat_completion_prediction_content_param import ChatCompletionPredictionContentParam
    from openai.types.chat.completion_create_params import (
        WebSearchOptions,
        WebSearchOptionsUserLocation,
        WebSearchOptionsUserLocationApproximate,
    )
    from openai.types.responses import ComputerToolParam, FileSearchToolParam, WebSearchToolParam
    from openai.types.responses.response_input_param import FunctionCallOutput, Message
    from openai.types.responses.response_reasoning_item_param import Summary
    from openai.types.responses.response_status import ResponseStatus
    from openai.types.shared import ReasoningEffort
    from openai.types.shared_params import Reasoning
except ImportError as _import_error:
    raise ImportError(
        'Please install `openai` to use the OpenAI model, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

if TYPE_CHECKING:
    from openai import Omit, omit

    OMIT = omit
else:
    # Backward compatibility with openai<2
    try:
        from openai import Omit, omit

        OMIT = omit
    except ImportError:  # pragma: lax no cover
        from openai import NOT_GIVEN, NotGiven

        OMIT = NOT_GIVEN
        Omit = NotGiven


__all__ = (
    'OpenAIModel',
    'OpenAIChatModel',
    'OpenAIResponsesModel',
    'OpenAIModelSettings',
    'OpenAIChatModelSettings',
    'OpenAIResponsesModelSettings',
    'OpenAIModelName',
)

OpenAIModelName = str | AllModels
"""
Possible OpenAI model names.

Since OpenAI supports a variety of date-stamped models, we explicitly list the latest models but
allow any name in the type hints.
See [the OpenAI docs](https://platform.openai.com/docs/models) for a full list.

Using this more broad type for the model name instead of the ChatModel definition
allows this model to be used more easily with other model types (ie, Ollama, Deepseek).
"""

MCP_SERVER_TOOL_CONNECTOR_URI_SCHEME: Literal['x-openai-connector'] = 'x-openai-connector'
"""
Prefix for OpenAI connector IDs. OpenAI supports either a URL or a connector ID when passing MCP configuration to a model,
by using that prefix like `x-openai-connector:<connector-id>` in a URL, you can pass a connector ID to a model.
"""

_CHAT_FINISH_REASON_MAP: dict[
    Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call'], FinishReason
] = {
    'stop': 'stop',
    'length': 'length',
    'tool_calls': 'tool_call',
    'content_filter': 'content_filter',
    'function_call': 'tool_call',
}

_RESPONSES_FINISH_REASON_MAP: dict[Literal['max_output_tokens', 'content_filter'] | ResponseStatus, FinishReason] = {
    'max_output_tokens': 'length',
    'content_filter': 'content_filter',
    'completed': 'stop',
    'cancelled': 'error',
    'failed': 'error',
}


class OpenAIChatModelSettings(ModelSettings, total=False):
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

    openai_service_tier: Literal['auto', 'default', 'flex', 'priority']
    """The service tier to use for the model request.

    Currently supported values are `auto`, `default`, `flex`, and `priority`.
    For more information, see [OpenAI's service tiers documentation](https://platform.openai.com/docs/api-reference/chat/object#chat/object-service_tier).
    """

    openai_prediction: ChatCompletionPredictionContentParam
    """Enables [predictive outputs](https://platform.openai.com/docs/guides/predicted-outputs).

    This feature is currently only supported for some OpenAI models.
    """


@deprecated('Use `OpenAIChatModelSettings` instead.')
class OpenAIModelSettings(OpenAIChatModelSettings, total=False):
    """Deprecated alias for `OpenAIChatModelSettings`."""


class OpenAIResponsesModelSettings(OpenAIChatModelSettings, total=False):
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

    Check the [OpenAI Reasoning documentation](https://platform.openai.com/docs/guides/reasoning?api-mode=responses#reasoning-summaries)
    for more details.
    """

    openai_send_reasoning_ids: bool
    """Whether to send the unique IDs of reasoning, text, and function call parts from the message history to the model. Enabled by default for reasoning models.

    This can result in errors like `"Item 'rs_123' of type 'reasoning' was provided without its required following item."`
    if the message history you're sending does not match exactly what was received from the Responses API in a previous response,
    for example if you're using a [history processor](../../message-history.md#processing-message-history).
    In that case, you'll want to disable this.
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

    openai_text_verbosity: Literal['low', 'medium', 'high']
    """Constrains the verbosity of the model's text response.

    Lower values will result in more concise responses, while higher values will
    result in more verbose responses. Currently supported values are `low`,
    `medium`, and `high`.
    """

    openai_previous_response_id: Literal['auto'] | str
    """The ID of a previous response from the model to use as the starting point for a continued conversation.

    When set to `'auto'`, the request automatically uses the most recent
    `provider_response_id` from the message history and omits earlier messages.

    This enables the model to use server-side conversation state and faithfully reference previous reasoning.
    See the [OpenAI Responses API documentation](https://platform.openai.com/docs/guides/reasoning#keeping-reasoning-items-in-context)
    for more information.
    """

    openai_include_code_execution_outputs: bool
    """Whether to include the code execution results in the response.

    Corresponds to the `code_interpreter_call.outputs` value of the `include` parameter in the Responses API.
    """

    openai_include_web_search_sources: bool
    """Whether to include the web search results in the response.

    Corresponds to the `web_search_call.action.sources` value of the `include` parameter in the Responses API.
    """


@dataclass(init=False)
class OpenAIChatModel(Model):
    """A model that uses the OpenAI API.

    Internally, this uses the [OpenAI Python client](https://github.com/openai/openai-python) to interact with the API.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    client: AsyncOpenAI = field(repr=False)

    _model_name: OpenAIModelName = field(repr=False)
    _provider: Provider[AsyncOpenAI] = field(repr=False)

    @overload
    def __init__(
        self,
        model_name: OpenAIModelName,
        *,
        provider: Literal[
            'azure',
            'deepseek',
            'cerebras',
            'fireworks',
            'github',
            'grok',
            'heroku',
            'moonshotai',
            'ollama',
            'openai',
            'openai-chat',
            'openrouter',
            'together',
            'vercel',
            'litellm',
            'nebius',
            'ovhcloud',
            'gateway',
        ]
        | Provider[AsyncOpenAI] = 'openai',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ) -> None: ...

    @deprecated('Set the `system_prompt_role` in the `OpenAIModelProfile` instead.')
    @overload
    def __init__(
        self,
        model_name: OpenAIModelName,
        *,
        provider: Literal[
            'azure',
            'deepseek',
            'cerebras',
            'fireworks',
            'github',
            'grok',
            'heroku',
            'moonshotai',
            'ollama',
            'openai',
            'openai-chat',
            'openrouter',
            'together',
            'vercel',
            'litellm',
            'nebius',
            'ovhcloud',
            'gateway',
        ]
        | Provider[AsyncOpenAI] = 'openai',
        profile: ModelProfileSpec | None = None,
        system_prompt_role: OpenAISystemPromptRole | None = None,
        settings: ModelSettings | None = None,
    ) -> None: ...

    def __init__(
        self,
        model_name: OpenAIModelName,
        *,
        provider: Literal[
            'azure',
            'deepseek',
            'cerebras',
            'fireworks',
            'github',
            'grok',
            'heroku',
            'moonshotai',
            'ollama',
            'openai',
            'openai-chat',
            'openrouter',
            'together',
            'vercel',
            'litellm',
            'nebius',
            'ovhcloud',
            'gateway',
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
            provider = infer_provider('gateway/openai' if provider == 'gateway' else provider)
        self._provider = provider
        self.client = provider.client

        super().__init__(settings=settings, profile=profile or provider.model_profile)

        if system_prompt_role is not None:
            self.profile = OpenAIModelProfile(openai_system_prompt_role=system_prompt_role).update(self.profile)

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def model_name(self) -> OpenAIModelName:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The model provider."""
        return self._provider.name

    @property
    @deprecated('Set the `system_prompt_role` in the `OpenAIModelProfile` instead.')
    def system_prompt_role(self) -> OpenAISystemPromptRole | None:
        return OpenAIModelProfile.from_profile(self.profile).openai_system_prompt_role

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
        response = await self._completions_create(
            messages, False, cast(OpenAIChatModelSettings, model_settings or {}), model_request_parameters
        )
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
            messages, True, cast(OpenAIChatModelSettings, model_settings or {}), model_request_parameters
        )
        async with response:
            yield await self._process_streamed_response(response, model_request_parameters)

    @overload
    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[True],
        model_settings: OpenAIChatModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncStream[ChatCompletionChunk]: ...

    @overload
    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[False],
        model_settings: OpenAIChatModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> chat.ChatCompletion: ...

    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: OpenAIChatModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> chat.ChatCompletion | AsyncStream[ChatCompletionChunk]:
        tools = self._get_tools(model_request_parameters)
        web_search_options = self._get_web_search_options(model_request_parameters)

        if not tools:
            tool_choice: Literal['none', 'required', 'auto'] | None = None
        elif (
            not model_request_parameters.allow_text_output
            and OpenAIModelProfile.from_profile(self.profile).openai_supports_tool_choice_required
        ):
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

        unsupported_model_settings = OpenAIModelProfile.from_profile(self.profile).openai_unsupported_model_settings
        for setting in unsupported_model_settings:
            model_settings.pop(setting, None)

        try:
            extra_headers = model_settings.get('extra_headers', {})
            extra_headers.setdefault('User-Agent', get_user_agent())
            return await self.client.chat.completions.create(
                model=self._model_name,
                messages=openai_messages,
                parallel_tool_calls=model_settings.get('parallel_tool_calls', OMIT),
                tools=tools or OMIT,
                tool_choice=tool_choice or OMIT,
                stream=stream,
                stream_options={'include_usage': True} if stream else OMIT,
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
                extra_headers=extra_headers,
                extra_body=model_settings.get('extra_body'),
            )
        except APIStatusError as e:
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise  # pragma: lax no cover

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

        # Workaround for local Ollama which sometimes returns a `None` finish reason.
        if response.choices and (choice := response.choices[0]) and choice.finish_reason is None:  # pyright: ignore[reportUnnecessaryComparison]
            choice.finish_reason = 'stop'

        try:
            response = chat.ChatCompletion.model_validate(response.model_dump())
        except ValidationError as e:
            raise UnexpectedModelBehavior(f'Invalid response from OpenAI chat completions endpoint: {e}') from e

        choice = response.choices[0]
        items: list[ModelResponsePart] = []

        # The `reasoning_content` field is only present in DeepSeek models.
        # https://api-docs.deepseek.com/guides/reasoning_model
        if reasoning_content := getattr(choice.message, 'reasoning_content', None):
            items.append(ThinkingPart(id='reasoning_content', content=reasoning_content, provider_name=self.system))

        # The `reasoning` field is only present in gpt-oss via Ollama and OpenRouter.
        # - https://cookbook.openai.com/articles/gpt-oss/handle-raw-cot#chat-completions-api
        # - https://openrouter.ai/docs/use-cases/reasoning-tokens#basic-usage-with-reasoning-tokens
        if reasoning := getattr(choice.message, 'reasoning', None):
            items.append(ThinkingPart(id='reasoning', content=reasoning, provider_name=self.system))

        # NOTE: We don't currently handle OpenRouter `reasoning_details`:
        # - https://openrouter.ai/docs/use-cases/reasoning-tokens#preserving-reasoning-blocks
        # If you need this, please file an issue.

        if choice.message.content:
            items.extend(
                (replace(part, id='content', provider_name=self.system) if isinstance(part, ThinkingPart) else part)
                for part in split_content_into_text_and_thinking(choice.message.content, self.profile.thinking_tags)
            )
        if choice.message.tool_calls is not None:
            for c in choice.message.tool_calls:
                if isinstance(c, ChatCompletionMessageFunctionToolCall):
                    part = ToolCallPart(c.function.name, c.function.arguments, tool_call_id=c.id)
                elif isinstance(c, ChatCompletionMessageCustomToolCall):  # pragma: no cover
                    # NOTE: Custom tool calls are not supported.
                    # See <https://github.com/pydantic/pydantic-ai/issues/2513> for more details.
                    raise RuntimeError('Custom tool calls are not supported')
                else:
                    assert_never(c)
                part.tool_call_id = _guard_tool_call_id(part)
                items.append(part)

        vendor_details: dict[str, Any] = {}

        # Add logprobs to vendor_details if available
        if choice.logprobs is not None and choice.logprobs.content:
            # Convert logprobs to a serializable format
            vendor_details['logprobs'] = [
                {
                    'token': lp.token,
                    'bytes': lp.bytes,
                    'logprob': lp.logprob,
                    'top_logprobs': [
                        {'token': tlp.token, 'bytes': tlp.bytes, 'logprob': tlp.logprob} for tlp in lp.top_logprobs
                    ],
                }
                for lp in choice.logprobs.content
            ]

        raw_finish_reason = choice.finish_reason
        vendor_details['finish_reason'] = raw_finish_reason
        finish_reason = _CHAT_FINISH_REASON_MAP.get(raw_finish_reason)

        return ModelResponse(
            parts=items,
            usage=_map_usage(response, self._provider.name, self._provider.base_url, self._model_name),
            model_name=response.model,
            timestamp=timestamp,
            provider_details=vendor_details or None,
            provider_response_id=response.id,
            provider_name=self._provider.name,
            finish_reason=finish_reason,
        )

    async def _process_streamed_response(
        self, response: AsyncStream[ChatCompletionChunk], model_request_parameters: ModelRequestParameters
    ) -> OpenAIStreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):
            raise UnexpectedModelBehavior(  # pragma: no cover
                'Streamed response ended without content or tool calls'
            )

        # When using Azure OpenAI and a content filter is enabled, the first chunk will contain a `''` model name,
        # so we set it from a later chunk in `OpenAIChatStreamedResponse`.
        model_name = first_chunk.model or self._model_name

        return OpenAIStreamedResponse(
            model_request_parameters=model_request_parameters,
            _model_name=model_name,
            _model_profile=self.profile,
            _response=peekable_response,
            _timestamp=number_to_datetime(first_chunk.created),
            _provider_name=self._provider.name,
            _provider_url=self._provider.base_url,
        )

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[chat.ChatCompletionToolParam]:
        return [self._map_tool_definition(r) for r in model_request_parameters.tool_defs.values()]

    def _get_web_search_options(self, model_request_parameters: ModelRequestParameters) -> WebSearchOptions | None:
        for tool in model_request_parameters.builtin_tools:
            if isinstance(tool, WebSearchTool):  # pragma: no branch
                if not OpenAIModelProfile.from_profile(self.profile).openai_chat_supports_web_search:
                    raise UserError(
                        f'WebSearchTool is not supported with `OpenAIChatModel` and model {self.model_name!r}. '
                        f'Please use `OpenAIResponsesModel` instead.'
                    )

                if tool.user_location:
                    return WebSearchOptions(
                        search_context_size=tool.search_context_size,
                        user_location=WebSearchOptionsUserLocation(
                            type='approximate',
                            approximate=WebSearchOptionsUserLocationApproximate(**tool.user_location),
                        ),
                    )
                return WebSearchOptions(search_context_size=tool.search_context_size)
            else:
                raise UserError(
                    f'`{tool.__class__.__name__}` is not supported by `OpenAIChatModel`. If it should be, please file an issue.'
                )

    async def _map_messages(
        self, messages: list[ModelMessage], model_request_parameters: ModelRequestParameters
    ) -> list[chat.ChatCompletionMessageParam]:
        """Just maps a `pydantic_ai.Message` to a `openai.types.ChatCompletionMessageParam`."""
        openai_messages: list[chat.ChatCompletionMessageParam] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                async for item in self._map_user_message(message):
                    openai_messages.append(item)
            elif isinstance(message, ModelResponse):
                texts: list[str] = []
                tool_calls: list[ChatCompletionMessageFunctionToolCallParam] = []
                for item in message.parts:
                    if isinstance(item, TextPart):
                        texts.append(item.content)
                    elif isinstance(item, ThinkingPart):
                        # NOTE: DeepSeek `reasoning_content` field should NOT be sent back per https://api-docs.deepseek.com/guides/reasoning_model,
                        # but we currently just send it in `<think>` tags anyway as we don't want DeepSeek-specific checks here.
                        # If you need this changed, please file an issue.
                        start_tag, end_tag = self.profile.thinking_tags
                        texts.append('\n'.join([start_tag, item.content, end_tag]))
                    elif isinstance(item, ToolCallPart):
                        tool_calls.append(self._map_tool_call(item))
                    # OpenAI doesn't return built-in tool calls
                    elif isinstance(item, BuiltinToolCallPart | BuiltinToolReturnPart):  # pragma: no cover
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
                else:
                    message_param['content'] = None
                if tool_calls:
                    message_param['tool_calls'] = tool_calls
                openai_messages.append(message_param)
            else:
                assert_never(message)
        if instructions := self._get_instructions(messages, model_request_parameters):
            openai_messages.insert(0, chat.ChatCompletionSystemMessageParam(content=instructions, role='system'))
        return openai_messages

    @staticmethod
    def _map_tool_call(t: ToolCallPart) -> ChatCompletionMessageFunctionToolCallParam:
        return ChatCompletionMessageFunctionToolCallParam(
            id=_guard_tool_call_id(t=t),
            type='function',
            function={'name': t.tool_name, 'arguments': t.args_as_json_str()},
        )

    def _map_json_schema(self, o: OutputObjectDefinition) -> chat.completion_create_params.ResponseFormat:
        response_format_param: chat.completion_create_params.ResponseFormatJSONSchema = {  # pyright: ignore[reportPrivateImportUsage]
            'type': 'json_schema',
            'json_schema': {'name': o.name or DEFAULT_OUTPUT_TOOL_NAME, 'schema': o.json_schema},
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
                system_prompt_role = OpenAIModelProfile.from_profile(self.profile).openai_system_prompt_role
                if system_prompt_role == 'developer':
                    yield chat.ChatCompletionDeveloperMessageParam(role='developer', content=part.content)
                elif system_prompt_role == 'user':
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

    async def _map_user_prompt(self, part: UserPromptPart) -> chat.ChatCompletionUserMessageParam:  # noqa: C901
        content: str | list[ChatCompletionContentPartParam]
        if isinstance(part.content, str):
            content = part.content
        else:
            content = []
            for item in part.content:
                if isinstance(item, str):
                    content.append(ChatCompletionContentPartTextParam(text=item, type='text'))
                elif isinstance(item, ImageUrl):
                    image_url: ImageURL = {'url': item.url}
                    if metadata := item.vendor_metadata:
                        image_url['detail'] = metadata.get('detail', 'auto')
                    if item.force_download:
                        image_content = await download_item(item, data_format='base64_uri', type_format='extension')
                        image_url['url'] = image_content['data']
                    content.append(ChatCompletionContentPartImageParam(image_url=image_url, type='image_url'))
                elif isinstance(item, BinaryContent):
                    if self._is_text_like_media_type(item.media_type):
                        # Inline text-like binary content as a text block
                        content.append(
                            self._inline_text_file_part(
                                item.data.decode('utf-8'),
                                media_type=item.media_type,
                                identifier=item.identifier,
                            )
                        )
                    elif item.is_image:
                        image_url = ImageURL(url=item.data_uri)
                        if metadata := item.vendor_metadata:
                            image_url['detail'] = metadata.get('detail', 'auto')
                        content.append(ChatCompletionContentPartImageParam(image_url=image_url, type='image_url'))
                    elif item.is_audio:
                        assert item.format in ('wav', 'mp3')
                        audio = InputAudio(data=base64.b64encode(item.data).decode('utf-8'), format=item.format)
                        content.append(ChatCompletionContentPartInputAudioParam(input_audio=audio, type='input_audio'))
                    elif item.is_document:
                        content.append(
                            File(
                                file=FileFile(
                                    file_data=item.data_uri,
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
                    if self._is_text_like_media_type(item.media_type):
                        downloaded_text = await download_item(item, data_format='text')
                        content.append(
                            self._inline_text_file_part(
                                downloaded_text['data'],
                                media_type=item.media_type,
                                identifier=item.identifier,
                            )
                        )
                    else:
                        downloaded_item = await download_item(item, data_format='base64_uri', type_format='extension')
                        content.append(
                            File(
                                file=FileFile(
                                    file_data=downloaded_item['data'],
                                    filename=f'filename.{downloaded_item["data_type"]}',
                                ),
                                type='file',
                            )
                        )
                elif isinstance(item, VideoUrl):  # pragma: no cover
                    raise NotImplementedError('VideoUrl is not supported for OpenAI')
                elif isinstance(item, CachePoint):
                    # OpenAI doesn't support prompt caching via CachePoint, so we filter it out
                    pass
                else:
                    assert_never(item)
        return chat.ChatCompletionUserMessageParam(role='user', content=content)

    @staticmethod
    def _is_text_like_media_type(media_type: str) -> bool:
        return (
            media_type.startswith('text/')
            or media_type == 'application/json'
            or media_type.endswith('+json')
            or media_type == 'application/xml'
            or media_type.endswith('+xml')
            or media_type in ('application/x-yaml', 'application/yaml')
        )

    @staticmethod
    def _inline_text_file_part(text: str, *, media_type: str, identifier: str) -> ChatCompletionContentPartTextParam:
        text = '\n'.join(
            [
                f'-----BEGIN FILE id="{identifier}" type="{media_type}"-----',
                text,
                f'-----END FILE id="{identifier}"-----',
            ]
        )
        return ChatCompletionContentPartTextParam(text=text, type='text')


@deprecated(
    '`OpenAIModel` was renamed to `OpenAIChatModel` to clearly distinguish it from `OpenAIResponsesModel` which '
    "uses OpenAI's newer Responses API. Use that unless you're using an OpenAI Chat Completions-compatible API, or "
    "require a feature that the Responses API doesn't support yet like audio."
)
@dataclass(init=False)
class OpenAIModel(OpenAIChatModel):
    """Deprecated alias for `OpenAIChatModel`."""


@dataclass(init=False)
class OpenAIResponsesModel(Model):
    """A model that uses the OpenAI Responses API.

    The [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) is the
    new API for OpenAI models.

    If you are interested in the differences between the Responses API and the Chat Completions API,
    see the [OpenAI API docs](https://platform.openai.com/docs/guides/responses-vs-chat-completions).
    """

    client: AsyncOpenAI = field(repr=False)

    _model_name: OpenAIModelName = field(repr=False)
    _provider: Provider[AsyncOpenAI] = field(repr=False)

    def __init__(
        self,
        model_name: OpenAIModelName,
        *,
        provider: Literal[
            'openai',
            'deepseek',
            'azure',
            'openrouter',
            'grok',
            'fireworks',
            'together',
            'nebius',
            'ovhcloud',
            'gateway',
        ]
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
            provider = infer_provider('gateway/openai' if provider == 'gateway' else provider)
        self._provider = provider
        self.client = provider.client

        super().__init__(settings=settings, profile=profile or provider.model_profile)

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def model_name(self) -> OpenAIModelName:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The model provider."""
        return self._provider.name

    async def request(
        self,
        messages: list[ModelRequest | ModelResponse],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        response = await self._responses_create(
            messages, False, cast(OpenAIResponsesModelSettings, model_settings or {}), model_request_parameters
        )
        return self._process_response(response, model_request_parameters)

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
        response = await self._responses_create(
            messages, True, cast(OpenAIResponsesModelSettings, model_settings or {}), model_request_parameters
        )
        async with response:
            yield await self._process_streamed_response(response, model_request_parameters)

    def _process_response(  # noqa: C901
        self, response: responses.Response, model_request_parameters: ModelRequestParameters
    ) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        timestamp = number_to_datetime(response.created_at)
        items: list[ModelResponsePart] = []
        for item in response.output:
            if isinstance(item, responses.ResponseReasoningItem):
                signature = item.encrypted_content
                if item.summary:
                    for summary in item.summary:
                        # We use the same id for all summaries so that we can merge them on the round trip.
                        items.append(
                            ThinkingPart(
                                content=summary.text,
                                id=item.id,
                                signature=signature,
                                provider_name=self.system if signature else None,
                            )
                        )
                        # We only need to store the signature once.
                        signature = None
                elif signature:
                    items.append(
                        ThinkingPart(
                            content='',
                            id=item.id,
                            signature=signature,
                            provider_name=self.system,
                        )
                    )
                # NOTE: We don't currently handle the raw CoT from gpt-oss `reasoning_text`: https://cookbook.openai.com/articles/gpt-oss/handle-raw-cot
                # If you need this, please file an issue.
            elif isinstance(item, responses.ResponseOutputMessage):
                for content in item.content:
                    if isinstance(content, responses.ResponseOutputText):  # pragma: no branch
                        items.append(TextPart(content.text, id=item.id))
            elif isinstance(item, responses.ResponseFunctionToolCall):
                items.append(
                    ToolCallPart(
                        item.name,
                        item.arguments,
                        tool_call_id=item.call_id,
                        id=item.id,
                    )
                )
            elif isinstance(item, responses.ResponseCodeInterpreterToolCall):
                call_part, return_part, file_parts = _map_code_interpreter_tool_call(item, self.system)
                items.append(call_part)
                if file_parts:
                    items.extend(file_parts)
                items.append(return_part)
            elif isinstance(item, responses.ResponseFunctionWebSearch):
                call_part, return_part = _map_web_search_tool_call(item, self.system)
                items.append(call_part)
                items.append(return_part)
            elif isinstance(item, responses.response_output_item.ImageGenerationCall):
                call_part, return_part, file_part = _map_image_generation_tool_call(item, self.system)
                items.append(call_part)
                if file_part:  # pragma: no branch
                    items.append(file_part)
                items.append(return_part)
            elif isinstance(item, responses.ResponseComputerToolCall):  # pragma: no cover
                # Pydantic AI doesn't yet support the ComputerUse built-in tool
                pass
            elif isinstance(item, responses.ResponseCustomToolCall):  # pragma: no cover
                # Support is being implemented in https://github.com/pydantic/pydantic-ai/pull/2572
                pass
            elif isinstance(item, responses.response_output_item.LocalShellCall):  # pragma: no cover
                # Pydantic AI doesn't yet support the `codex-mini-latest` LocalShell built-in tool
                pass
            elif isinstance(item, responses.ResponseFileSearchToolCall):  # pragma: no cover
                # Pydantic AI doesn't yet support the FileSearch built-in tool
                pass
            elif isinstance(item, responses.response_output_item.McpCall):
                call_part, return_part = _map_mcp_call(item, self.system)
                items.append(call_part)
                items.append(return_part)
            elif isinstance(item, responses.response_output_item.McpListTools):
                call_part, return_part = _map_mcp_list_tools(item, self.system)
                items.append(call_part)
                items.append(return_part)
            elif isinstance(item, responses.response_output_item.McpApprovalRequest):  # pragma: no cover
                # Pydantic AI doesn't yet support McpApprovalRequest (explicit tool usage approval)
                pass

        finish_reason: FinishReason | None = None
        provider_details: dict[str, Any] | None = None
        raw_finish_reason = details.reason if (details := response.incomplete_details) else response.status
        if raw_finish_reason:
            provider_details = {'finish_reason': raw_finish_reason}
            finish_reason = _RESPONSES_FINISH_REASON_MAP.get(raw_finish_reason)

        return ModelResponse(
            parts=items,
            usage=_map_usage(response, self._provider.name, self._provider.base_url, self._model_name),
            model_name=response.model,
            provider_response_id=response.id,
            timestamp=timestamp,
            provider_name=self._provider.name,
            finish_reason=finish_reason,
            provider_details=provider_details,
        )

    async def _process_streamed_response(
        self,
        response: AsyncStream[responses.ResponseStreamEvent],
        model_request_parameters: ModelRequestParameters,
    ) -> OpenAIResponsesStreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):  # pragma: no cover
            raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')

        assert isinstance(first_chunk, responses.ResponseCreatedEvent)
        return OpenAIResponsesStreamedResponse(
            model_request_parameters=model_request_parameters,
            _model_name=first_chunk.response.model,
            _response=peekable_response,
            _timestamp=number_to_datetime(first_chunk.response.created_at),
            _provider_name=self._provider.name,
            _provider_url=self._provider.base_url,
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
        tools = (
            self._get_builtin_tools(model_request_parameters)
            + list(model_settings.get('openai_builtin_tools', []))
            + self._get_tools(model_request_parameters)
        )
        profile = OpenAIModelProfile.from_profile(self.profile)
        if not tools:
            tool_choice: Literal['none', 'required', 'auto'] | None = None
        elif not model_request_parameters.allow_text_output and profile.openai_supports_tool_choice_required:
            tool_choice = 'required'
        else:
            tool_choice = 'auto'

        previous_response_id = model_settings.get('openai_previous_response_id')
        if previous_response_id == 'auto':
            previous_response_id, messages = self._get_previous_response_id_and_new_messages(messages)

        instructions, openai_messages = await self._map_messages(messages, model_settings, model_request_parameters)
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
            instructions = OMIT

        if verbosity := model_settings.get('openai_text_verbosity'):
            text = text or {}
            text['verbosity'] = verbosity

        unsupported_model_settings = profile.openai_unsupported_model_settings
        for setting in unsupported_model_settings:
            model_settings.pop(setting, None)

        include: list[responses.ResponseIncludable] = []
        if profile.openai_supports_encrypted_reasoning_content:
            include.append('reasoning.encrypted_content')
        if model_settings.get('openai_include_code_execution_outputs'):
            include.append('code_interpreter_call.outputs')
        if model_settings.get('openai_include_web_search_sources'):
            include.append('web_search_call.action.sources')

        try:
            extra_headers = model_settings.get('extra_headers', {})
            extra_headers.setdefault('User-Agent', get_user_agent())
            return await self.client.responses.create(
                input=openai_messages,
                model=self._model_name,
                instructions=instructions,
                parallel_tool_calls=model_settings.get('parallel_tool_calls', OMIT),
                tools=tools or OMIT,
                tool_choice=tool_choice or OMIT,
                max_output_tokens=model_settings.get('max_tokens', OMIT),
                stream=stream,
                temperature=model_settings.get('temperature', OMIT),
                top_p=model_settings.get('top_p', OMIT),
                truncation=model_settings.get('openai_truncation', OMIT),
                timeout=model_settings.get('timeout', NOT_GIVEN),
                service_tier=model_settings.get('openai_service_tier', OMIT),
                previous_response_id=previous_response_id or OMIT,
                reasoning=reasoning,
                user=model_settings.get('openai_user', OMIT),
                text=text or OMIT,
                include=include or OMIT,
                extra_headers=extra_headers,
                extra_body=model_settings.get('extra_body'),
            )
        except APIStatusError as e:
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise  # pragma: lax no cover

    def _get_reasoning(self, model_settings: OpenAIResponsesModelSettings) -> Reasoning | Omit:
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
            return OMIT
        return Reasoning(effort=reasoning_effort, summary=reasoning_summary)

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[responses.FunctionToolParam]:
        return [self._map_tool_definition(r) for r in model_request_parameters.tool_defs.values()]

    def _get_builtin_tools(self, model_request_parameters: ModelRequestParameters) -> list[responses.ToolParam]:
        tools: list[responses.ToolParam] = []
        has_image_generating_tool = False
        for tool in model_request_parameters.builtin_tools:
            if isinstance(tool, WebSearchTool):
                web_search_tool = responses.WebSearchToolParam(
                    type='web_search', search_context_size=tool.search_context_size
                )
                if tool.user_location:
                    web_search_tool['user_location'] = responses.web_search_tool_param.UserLocation(
                        type='approximate', **tool.user_location
                    )
                tools.append(web_search_tool)
            elif isinstance(tool, CodeExecutionTool):
                has_image_generating_tool = True
                tools.append({'type': 'code_interpreter', 'container': {'type': 'auto'}})
            elif isinstance(tool, MCPServerTool):
                mcp_tool = responses.tool_param.Mcp(
                    type='mcp',
                    server_label=tool.id,
                    require_approval='never',
                )

                if tool.authorization_token:  # pragma: no branch
                    mcp_tool['authorization'] = tool.authorization_token

                if tool.allowed_tools is not None:  # pragma: no branch
                    mcp_tool['allowed_tools'] = tool.allowed_tools

                if tool.description:  # pragma: no branch
                    mcp_tool['server_description'] = tool.description

                if tool.headers:  # pragma: no branch
                    mcp_tool['headers'] = tool.headers

                if tool.url.startswith(MCP_SERVER_TOOL_CONNECTOR_URI_SCHEME + ':'):
                    _, connector_id = tool.url.split(':', maxsplit=1)
                    mcp_tool['connector_id'] = connector_id  # pyright: ignore[reportGeneralTypeIssues]
                else:
                    mcp_tool['server_url'] = tool.url

                tools.append(mcp_tool)
            elif isinstance(tool, ImageGenerationTool):  # pragma: no branch
                has_image_generating_tool = True
                tools.append(
                    responses.tool_param.ImageGeneration(
                        type='image_generation',
                        background=tool.background,
                        input_fidelity=tool.input_fidelity,
                        moderation=tool.moderation,
                        output_compression=tool.output_compression,
                        output_format=tool.output_format or 'png',
                        partial_images=tool.partial_images,
                        quality=tool.quality,
                        size=tool.size,
                    )
                )
            else:
                raise UserError(  # pragma: no cover
                    f'`{tool.__class__.__name__}` is not supported by `OpenAIResponsesModel`. If it should be, please file an issue.'
                )

        if model_request_parameters.allow_image_output and not has_image_generating_tool:
            tools.append({'type': 'image_generation'})
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

    def _get_previous_response_id_and_new_messages(
        self, messages: list[ModelMessage]
    ) -> tuple[str | None, list[ModelMessage]]:
        # When `openai_previous_response_id` is set to 'auto', the most recent
        # `provider_response_id` from the message history is selected and all
        # earlier messages are omitted. This allows the OpenAI SDK to reuse
        # server-side history for efficiency. The returned tuple contains the
        # `previous_response_id` (if found) and the trimmed list of messages.
        previous_response_id = None
        trimmed_messages: list[ModelMessage] = []
        for m in reversed(messages):
            if isinstance(m, ModelResponse) and m.provider_name == self.system:
                previous_response_id = m.provider_response_id
                break
            else:
                trimmed_messages.append(m)

        if previous_response_id and trimmed_messages:
            return previous_response_id, list(reversed(trimmed_messages))
        else:
            return None, messages

    async def _map_messages(  # noqa: C901
        self,
        messages: list[ModelMessage],
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[str | Omit, list[responses.ResponseInputItemParam]]:
        """Just maps a `pydantic_ai.Message` to a `openai.types.responses.ResponseInputParam`."""
        profile = OpenAIModelProfile.from_profile(self.profile)
        send_item_ids = model_settings.get(
            'openai_send_reasoning_ids', profile.openai_supports_encrypted_reasoning_content
        )

        openai_messages: list[responses.ResponseInputItemParam] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, SystemPromptPart):
                        openai_messages.append(responses.EasyInputMessageParam(role='system', content=part.content))
                    elif isinstance(part, UserPromptPart):
                        openai_messages.append(await self._map_user_prompt(part))
                    elif isinstance(part, ToolReturnPart):
                        call_id = _guard_tool_call_id(t=part)
                        call_id, _ = _split_combined_tool_call_id(call_id)
                        item = FunctionCallOutput(
                            type='function_call_output',
                            call_id=call_id,
                            output=part.model_response_str(),
                        )
                        openai_messages.append(item)
                    elif isinstance(part, RetryPromptPart):
                        if part.tool_name is None:
                            openai_messages.append(
                                Message(role='user', content=[{'type': 'input_text', 'text': part.model_response()}])
                            )
                        else:
                            call_id = _guard_tool_call_id(t=part)
                            call_id, _ = _split_combined_tool_call_id(call_id)
                            item = FunctionCallOutput(
                                type='function_call_output',
                                call_id=call_id,
                                output=part.model_response(),
                            )
                            openai_messages.append(item)
                    else:
                        assert_never(part)
            elif isinstance(message, ModelResponse):
                send_item_ids = send_item_ids and message.provider_name == self.system

                message_item: responses.ResponseOutputMessageParam | None = None
                reasoning_item: responses.ResponseReasoningItemParam | None = None
                web_search_item: responses.ResponseFunctionWebSearchParam | None = None
                code_interpreter_item: responses.ResponseCodeInterpreterToolCallParam | None = None
                for item in message.parts:
                    if isinstance(item, TextPart):
                        if item.id and send_item_ids:
                            if message_item is None or message_item['id'] != item.id:  # pragma: no branch
                                message_item = responses.ResponseOutputMessageParam(
                                    role='assistant',
                                    id=item.id,
                                    content=[],
                                    type='message',
                                    status='completed',
                                )
                                openai_messages.append(message_item)

                            message_item['content'] = [
                                *message_item['content'],
                                responses.ResponseOutputTextParam(
                                    text=item.content, type='output_text', annotations=[]
                                ),
                            ]
                        else:
                            openai_messages.append(
                                responses.EasyInputMessageParam(role='assistant', content=item.content)
                            )
                    elif isinstance(item, ToolCallPart):
                        call_id = _guard_tool_call_id(t=item)
                        call_id, id = _split_combined_tool_call_id(call_id)
                        id = id or item.id

                        param = responses.ResponseFunctionToolCallParam(
                            name=item.tool_name,
                            arguments=item.args_as_json_str(),
                            call_id=call_id,
                            type='function_call',
                        )
                        if profile.openai_responses_requires_function_call_status_none:
                            param['status'] = None  # type: ignore[reportGeneralTypeIssues]
                        if id and send_item_ids:  # pragma: no branch
                            param['id'] = id
                        openai_messages.append(param)
                    elif isinstance(item, BuiltinToolCallPart):
                        if item.provider_name == self.system and send_item_ids:
                            if (
                                item.tool_name == CodeExecutionTool.kind
                                and item.tool_call_id
                                and (args := item.args_as_dict())
                                and (container_id := args.get('container_id'))
                            ):
                                code_interpreter_item = responses.ResponseCodeInterpreterToolCallParam(
                                    id=item.tool_call_id,
                                    code=args.get('code'),
                                    container_id=container_id,
                                    outputs=None,  # These can be read server-side
                                    status='completed',
                                    type='code_interpreter_call',
                                )
                                openai_messages.append(code_interpreter_item)
                            elif (
                                item.tool_name == WebSearchTool.kind
                                and item.tool_call_id
                                and (args := item.args_as_dict())
                            ):
                                web_search_item = responses.ResponseFunctionWebSearchParam(
                                    id=item.tool_call_id,
                                    action=cast(responses.response_function_web_search_param.Action, args),
                                    status='completed',
                                    type='web_search_call',
                                )
                                openai_messages.append(web_search_item)
                            elif item.tool_name == ImageGenerationTool.kind and item.tool_call_id:
                                # The cast is necessary because of https://github.com/openai/openai-python/issues/2648
                                image_generation_item = cast(
                                    responses.response_input_item_param.ImageGenerationCall,
                                    {
                                        'id': item.tool_call_id,
                                        'type': 'image_generation_call',
                                    },
                                )
                                openai_messages.append(image_generation_item)
                            elif (  # pragma: no branch
                                item.tool_name.startswith(MCPServerTool.kind)
                                and item.tool_call_id
                                and (server_id := item.tool_name.split(':', 1)[1])
                                and (args := item.args_as_dict())
                                and (action := args.get('action'))
                            ):
                                if action == 'list_tools':
                                    mcp_list_tools_item = responses.response_input_item_param.McpListTools(
                                        id=item.tool_call_id,
                                        type='mcp_list_tools',
                                        server_label=server_id,
                                        tools=[],  # These can be read server-side
                                    )
                                    openai_messages.append(mcp_list_tools_item)
                                elif (  # pragma: no branch
                                    action == 'call_tool'
                                    and (tool_name := args.get('tool_name'))
                                    and (tool_args := args.get('tool_args'))
                                ):
                                    mcp_call_item = responses.response_input_item_param.McpCall(
                                        id=item.tool_call_id,
                                        server_label=server_id,
                                        name=tool_name,
                                        arguments=to_json(tool_args).decode(),
                                        error=None,  # These can be read server-side
                                        output=None,  # These can be read server-side
                                        type='mcp_call',
                                    )
                                    openai_messages.append(mcp_call_item)

                    elif isinstance(item, BuiltinToolReturnPart):
                        if item.provider_name == self.system and send_item_ids:
                            if (
                                item.tool_name == CodeExecutionTool.kind
                                and code_interpreter_item is not None
                                and isinstance(item.content, dict)
                                and (content := cast(dict[str, Any], item.content))  # pyright: ignore[reportUnknownMemberType]
                                and (status := content.get('status'))
                            ):
                                code_interpreter_item['status'] = status
                            elif (
                                item.tool_name == WebSearchTool.kind
                                and web_search_item is not None
                                and isinstance(item.content, dict)  # pyright: ignore[reportUnknownMemberType]
                                and (content := cast(dict[str, Any], item.content))  # pyright: ignore[reportUnknownMemberType]
                                and (status := content.get('status'))
                            ):
                                web_search_item['status'] = status
                            elif item.tool_name == ImageGenerationTool.kind:
                                # Image generation result does not need to be sent back, just the `id` off of `BuiltinToolCallPart`.
                                pass
                            elif item.tool_name.startswith(MCPServerTool.kind):  # pragma: no branch
                                # MCP call result does not need to be sent back, just the fields off of `BuiltinToolCallPart`.
                                pass
                    elif isinstance(item, FilePart):
                        # This was generated by the `ImageGenerationTool` or `CodeExecutionTool`,
                        # and does not need to be sent back separately from the corresponding `BuiltinToolReturnPart`.
                        # If `send_item_ids` is false, we won't send the `BuiltinToolReturnPart`, but OpenAI does not have a type for files from the assistant.
                        pass
                    elif isinstance(item, ThinkingPart):
                        if item.id and send_item_ids:
                            signature: str | None = None
                            if (
                                item.signature
                                and item.provider_name == self.system
                                and profile.openai_supports_encrypted_reasoning_content
                            ):
                                signature = item.signature

                            if (reasoning_item is None or reasoning_item['id'] != item.id) and (
                                signature or item.content
                            ):  # pragma: no branch
                                reasoning_item = responses.ResponseReasoningItemParam(
                                    id=item.id,
                                    summary=[],
                                    encrypted_content=signature,
                                    type='reasoning',
                                )
                                openai_messages.append(reasoning_item)

                            if item.content:
                                # The check above guarantees that `reasoning_item` is not None
                                assert reasoning_item is not None
                                reasoning_item['summary'] = [
                                    *reasoning_item['summary'],
                                    Summary(text=item.content, type='summary_text'),
                                ]
                        else:
                            start_tag, end_tag = profile.thinking_tags
                            openai_messages.append(
                                responses.EasyInputMessageParam(
                                    role='assistant', content='\n'.join([start_tag, item.content, end_tag])
                                )
                            )
                    else:
                        assert_never(item)
            else:
                assert_never(message)
        instructions = self._get_instructions(messages, model_request_parameters) or OMIT
        return instructions, openai_messages

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
    async def _map_user_prompt(part: UserPromptPart) -> responses.EasyInputMessageParam:  # noqa: C901
        content: str | list[responses.ResponseInputContentParam]
        if isinstance(part.content, str):
            content = part.content
        else:
            content = []
            for item in part.content:
                if isinstance(item, str):
                    content.append(responses.ResponseInputTextParam(text=item, type='input_text'))
                elif isinstance(item, BinaryContent):
                    if item.is_image:
                        detail: Literal['auto', 'low', 'high'] = 'auto'
                        if metadata := item.vendor_metadata:
                            detail = cast(
                                Literal['auto', 'low', 'high'],
                                metadata.get('detail', 'auto'),
                            )
                        content.append(
                            responses.ResponseInputImageParam(
                                image_url=item.data_uri,
                                type='input_image',
                                detail=detail,
                            )
                        )
                    elif item.is_document:
                        content.append(
                            responses.ResponseInputFileParam(
                                type='input_file',
                                file_data=item.data_uri,
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
                    detail: Literal['auto', 'low', 'high'] = 'auto'
                    image_url = item.url
                    if metadata := item.vendor_metadata:
                        detail = cast(Literal['auto', 'low', 'high'], metadata.get('detail', 'auto'))
                    if item.force_download:
                        downloaded_item = await download_item(item, data_format='base64_uri', type_format='extension')
                        image_url = downloaded_item['data']

                    content.append(
                        responses.ResponseInputImageParam(
                            image_url=image_url,
                            type='input_image',
                            detail=detail,
                        )
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
                elif isinstance(item, CachePoint):
                    # OpenAI doesn't support prompt caching via CachePoint, so we filter it out
                    pass
                else:
                    assert_never(item)
        return responses.EasyInputMessageParam(role='user', content=content)


@dataclass
class OpenAIStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for OpenAI models."""

    _model_name: OpenAIModelName
    _model_profile: ModelProfile
    _response: AsyncIterable[ChatCompletionChunk]
    _timestamp: datetime
    _provider_name: str
    _provider_url: str

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        async for chunk in self._response:
            self._usage += _map_usage(chunk, self._provider_name, self._provider_url, self._model_name)

            if chunk.id:  # pragma: no branch
                self.provider_response_id = chunk.id

            if chunk.model:
                self._model_name = chunk.model

            try:
                choice = chunk.choices[0]
            except IndexError:
                continue

            # When using Azure OpenAI and an async content filter is enabled, the openai SDK can return None deltas.
            if choice.delta is None:  # pyright: ignore[reportUnnecessaryComparison]
                continue

            if raw_finish_reason := choice.finish_reason:
                self.provider_details = {'finish_reason': raw_finish_reason}
                self.finish_reason = _CHAT_FINISH_REASON_MAP.get(raw_finish_reason)

            # The `reasoning_content` field is only present in DeepSeek models.
            # https://api-docs.deepseek.com/guides/reasoning_model
            if reasoning_content := getattr(choice.delta, 'reasoning_content', None):
                yield self._parts_manager.handle_thinking_delta(
                    vendor_part_id='reasoning_content',
                    id='reasoning_content',
                    content=reasoning_content,
                    provider_name=self.provider_name,
                )

            # The `reasoning` field is only present in gpt-oss via Ollama and OpenRouter.
            # - https://cookbook.openai.com/articles/gpt-oss/handle-raw-cot#chat-completions-api
            # - https://openrouter.ai/docs/use-cases/reasoning-tokens#basic-usage-with-reasoning-tokens
            if reasoning := getattr(choice.delta, 'reasoning', None):  # pragma: no cover
                yield self._parts_manager.handle_thinking_delta(
                    vendor_part_id='reasoning',
                    id='reasoning',
                    content=reasoning,
                    provider_name=self.provider_name,
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
                    if isinstance(maybe_event, PartStartEvent) and isinstance(maybe_event.part, ThinkingPart):
                        maybe_event.part.id = 'content'
                        maybe_event.part.provider_name = self.provider_name
                    yield maybe_event

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
    def provider_name(self) -> str:
        """Get the provider name."""
        return self._provider_name

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
    _provider_name: str
    _provider_url: str

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:  # noqa: C901
        async for chunk in self._response:
            # NOTE: You can inspect the builtin tools used checking the `ResponseCompletedEvent`.
            if isinstance(chunk, responses.ResponseCompletedEvent):
                self._usage += self._map_usage(chunk.response)

                raw_finish_reason = (
                    details.reason if (details := chunk.response.incomplete_details) else chunk.response.status
                )
                if raw_finish_reason:  # pragma: no branch
                    self.provider_details = {'finish_reason': raw_finish_reason}
                    self.finish_reason = _RESPONSES_FINISH_REASON_MAP.get(raw_finish_reason)

            elif isinstance(chunk, responses.ResponseContentPartAddedEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseContentPartDoneEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseCreatedEvent):
                if chunk.response.id:  # pragma: no branch
                    self.provider_response_id = chunk.response.id

            elif isinstance(chunk, responses.ResponseFailedEvent):  # pragma: no cover
                self._usage += self._map_usage(chunk.response)

            elif isinstance(chunk, responses.ResponseFunctionCallArgumentsDeltaEvent):
                maybe_event = self._parts_manager.handle_tool_call_delta(
                    vendor_part_id=chunk.item_id,
                    args=chunk.delta,
                )
                if maybe_event is not None:  # pragma: no branch
                    yield maybe_event

            elif isinstance(chunk, responses.ResponseFunctionCallArgumentsDoneEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseIncompleteEvent):  # pragma: no cover
                self._usage += self._map_usage(chunk.response)

            elif isinstance(chunk, responses.ResponseInProgressEvent):
                self._usage += self._map_usage(chunk.response)

            elif isinstance(chunk, responses.ResponseOutputItemAddedEvent):
                if isinstance(chunk.item, responses.ResponseFunctionToolCall):
                    yield self._parts_manager.handle_tool_call_part(
                        vendor_part_id=chunk.item.id,
                        tool_name=chunk.item.name,
                        args=chunk.item.arguments,
                        tool_call_id=chunk.item.call_id,
                        id=chunk.item.id,
                    )
                elif isinstance(chunk.item, responses.ResponseReasoningItem):
                    pass
                elif isinstance(chunk.item, responses.ResponseOutputMessage):
                    pass
                elif isinstance(chunk.item, responses.ResponseFunctionWebSearch):
                    call_part, _ = _map_web_search_tool_call(chunk.item, self.provider_name)
                    yield self._parts_manager.handle_part(
                        vendor_part_id=f'{chunk.item.id}-call', part=replace(call_part, args=None)
                    )
                elif isinstance(chunk.item, responses.ResponseCodeInterpreterToolCall):
                    call_part, _, _ = _map_code_interpreter_tool_call(chunk.item, self.provider_name)

                    args_json = call_part.args_as_json_str()
                    # Drop the final `"}` so that we can add code deltas
                    args_json_delta = args_json[:-2]
                    assert args_json_delta.endswith('"code":"'), f'Expected {args_json_delta!r} to end in `"code":"`'

                    yield self._parts_manager.handle_part(
                        vendor_part_id=f'{chunk.item.id}-call', part=replace(call_part, args=None)
                    )
                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=f'{chunk.item.id}-call',
                        args=args_json_delta,
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event
                elif isinstance(chunk.item, responses.response_output_item.ImageGenerationCall):
                    call_part, _, _ = _map_image_generation_tool_call(chunk.item, self.provider_name)
                    yield self._parts_manager.handle_part(vendor_part_id=f'{chunk.item.id}-call', part=call_part)
                elif isinstance(chunk.item, responses.response_output_item.McpCall):
                    call_part, _ = _map_mcp_call(chunk.item, self.provider_name)

                    args_json = call_part.args_as_json_str()
                    # Drop the final `{}}` so that we can add tool args deltas
                    args_json_delta = args_json[:-3]
                    assert args_json_delta.endswith('"tool_args":'), (
                        f'Expected {args_json_delta!r} to end in `"tool_args":"`'
                    )

                    yield self._parts_manager.handle_part(
                        vendor_part_id=f'{chunk.item.id}-call', part=replace(call_part, args=None)
                    )
                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=f'{chunk.item.id}-call',
                        args=args_json_delta,
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event
                elif isinstance(chunk.item, responses.response_output_item.McpListTools):
                    call_part, _ = _map_mcp_list_tools(chunk.item, self.provider_name)
                    yield self._parts_manager.handle_part(vendor_part_id=f'{chunk.item.id}-call', part=call_part)
                else:
                    warnings.warn(  # pragma: no cover
                        f'Handling of this item type is not yet implemented. Please report on our GitHub: {chunk}',
                        UserWarning,
                    )

            elif isinstance(chunk, responses.ResponseOutputItemDoneEvent):
                if isinstance(chunk.item, responses.ResponseReasoningItem):
                    if signature := chunk.item.encrypted_content:  # pragma: no branch
                        # Add the signature to the part corresponding to the first summary item
                        yield self._parts_manager.handle_thinking_delta(
                            vendor_part_id=f'{chunk.item.id}-0',
                            id=chunk.item.id,
                            signature=signature,
                            provider_name=self.provider_name,
                        )
                elif isinstance(chunk.item, responses.ResponseCodeInterpreterToolCall):
                    _, return_part, file_parts = _map_code_interpreter_tool_call(chunk.item, self.provider_name)
                    for i, file_part in enumerate(file_parts):
                        yield self._parts_manager.handle_part(
                            vendor_part_id=f'{chunk.item.id}-file-{i}', part=file_part
                        )
                    yield self._parts_manager.handle_part(vendor_part_id=f'{chunk.item.id}-return', part=return_part)
                elif isinstance(chunk.item, responses.ResponseFunctionWebSearch):
                    call_part, return_part = _map_web_search_tool_call(chunk.item, self.provider_name)

                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=f'{chunk.item.id}-call',
                        args=call_part.args,
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event

                    yield self._parts_manager.handle_part(vendor_part_id=f'{chunk.item.id}-return', part=return_part)
                elif isinstance(chunk.item, responses.response_output_item.ImageGenerationCall):
                    _, return_part, file_part = _map_image_generation_tool_call(chunk.item, self.provider_name)
                    if file_part:  # pragma: no branch
                        yield self._parts_manager.handle_part(vendor_part_id=f'{chunk.item.id}-file', part=file_part)
                    yield self._parts_manager.handle_part(vendor_part_id=f'{chunk.item.id}-return', part=return_part)

                elif isinstance(chunk.item, responses.response_output_item.McpCall):
                    _, return_part = _map_mcp_call(chunk.item, self.provider_name)
                    yield self._parts_manager.handle_part(vendor_part_id=f'{chunk.item.id}-return', part=return_part)
                elif isinstance(chunk.item, responses.response_output_item.McpListTools):
                    _, return_part = _map_mcp_list_tools(chunk.item, self.provider_name)
                    yield self._parts_manager.handle_part(vendor_part_id=f'{chunk.item.id}-return', part=return_part)

            elif isinstance(chunk, responses.ResponseReasoningSummaryPartAddedEvent):
                yield self._parts_manager.handle_thinking_delta(
                    vendor_part_id=f'{chunk.item_id}-{chunk.summary_index}',
                    content=chunk.part.text,
                    id=chunk.item_id,
                )

            elif isinstance(chunk, responses.ResponseReasoningSummaryPartDoneEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseReasoningSummaryTextDoneEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseReasoningSummaryTextDeltaEvent):
                yield self._parts_manager.handle_thinking_delta(
                    vendor_part_id=f'{chunk.item_id}-{chunk.summary_index}',
                    content=chunk.delta,
                    id=chunk.item_id,
                )

            elif isinstance(chunk, responses.ResponseOutputTextAnnotationAddedEvent):
                # TODO(Marcelo): We should support annotations in the future.
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseTextDeltaEvent):
                maybe_event = self._parts_manager.handle_text_delta(
                    vendor_part_id=chunk.item_id, content=chunk.delta, id=chunk.item_id
                )
                if maybe_event is not None:  # pragma: no branch
                    yield maybe_event

            elif isinstance(chunk, responses.ResponseTextDoneEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseWebSearchCallInProgressEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseWebSearchCallSearchingEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseWebSearchCallCompletedEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseAudioDeltaEvent):  # pragma: lax no cover
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseCodeInterpreterCallCodeDeltaEvent):
                json_args_delta = to_json(chunk.delta).decode()[1:-1]  # Drop the surrounding `"`
                maybe_event = self._parts_manager.handle_tool_call_delta(
                    vendor_part_id=f'{chunk.item_id}-call',
                    args=json_args_delta,
                )
                if maybe_event is not None:  # pragma: no branch
                    yield maybe_event

            elif isinstance(chunk, responses.ResponseCodeInterpreterCallCodeDoneEvent):
                maybe_event = self._parts_manager.handle_tool_call_delta(
                    vendor_part_id=f'{chunk.item_id}-call',
                    args='"}',
                )
                if maybe_event is not None:  # pragma: no branch
                    yield maybe_event

            elif isinstance(chunk, responses.ResponseCodeInterpreterCallCompletedEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseCodeInterpreterCallInProgressEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseCodeInterpreterCallInterpretingEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseImageGenCallCompletedEvent):  # pragma: no cover
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseImageGenCallGeneratingEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseImageGenCallInProgressEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseImageGenCallPartialImageEvent):
                # Not present on the type, but present on the actual object.
                # See https://github.com/openai/openai-python/issues/2649
                output_format = getattr(chunk, 'output_format', 'png')
                file_part = FilePart(
                    content=BinaryImage(
                        data=base64.b64decode(chunk.partial_image_b64),
                        media_type=f'image/{output_format}',
                    ),
                    id=chunk.item_id,
                )
                yield self._parts_manager.handle_part(vendor_part_id=f'{chunk.item_id}-file', part=file_part)

            elif isinstance(chunk, responses.ResponseMcpCallArgumentsDoneEvent):
                maybe_event = self._parts_manager.handle_tool_call_delta(
                    vendor_part_id=f'{chunk.item_id}-call',
                    args='}',
                )
                if maybe_event is not None:  # pragma: no branch
                    yield maybe_event

            elif isinstance(chunk, responses.ResponseMcpCallArgumentsDeltaEvent):
                maybe_event = self._parts_manager.handle_tool_call_delta(
                    vendor_part_id=f'{chunk.item_id}-call',
                    args=chunk.delta,
                )
                if maybe_event is not None:  # pragma: no branch
                    yield maybe_event

            elif isinstance(chunk, responses.ResponseMcpListToolsInProgressEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseMcpListToolsCompletedEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseMcpListToolsFailedEvent):  # pragma: no cover
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseMcpCallInProgressEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseMcpCallFailedEvent):  # pragma: no cover
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseMcpCallCompletedEvent):
                pass  # there's nothing we need to do here

            else:  # pragma: no cover
                warnings.warn(
                    f'Handling of this event type is not yet implemented. Please report on our GitHub: {chunk}',
                    UserWarning,
                )

    def _map_usage(self, response: responses.Response):
        return _map_usage(response, self._provider_name, self._provider_url, self._model_name)

    @property
    def model_name(self) -> OpenAIModelName:
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


def _map_usage(
    response: chat.ChatCompletion | ChatCompletionChunk | responses.Response,
    provider: str,
    provider_url: str,
    model: str,
) -> usage.RequestUsage:
    response_usage = response.usage
    if response_usage is None:
        return usage.RequestUsage()

    usage_data = response_usage.model_dump(exclude_none=True)
    details = {
        k: v
        for k, v in usage_data.items()
        if k not in {'prompt_tokens', 'completion_tokens', 'input_tokens', 'output_tokens', 'total_tokens'}
        if isinstance(v, int)
    }
    response_data = dict(model=model, usage=usage_data)
    if isinstance(response_usage, responses.ResponseUsage):
        api_flavor = 'responses'

        if getattr(response_usage, 'output_tokens_details', None) is not None:
            details['reasoning_tokens'] = response_usage.output_tokens_details.reasoning_tokens
        else:
            details['reasoning_tokens'] = 0
    else:
        api_flavor = 'chat'

        if response_usage.completion_tokens_details is not None:
            details.update(response_usage.completion_tokens_details.model_dump(exclude_none=True))

    return usage.RequestUsage.extract(
        response_data,
        provider=provider,
        provider_url=provider_url,
        provider_fallback='openai',
        api_flavor=api_flavor,
        details=details,
    )


def _split_combined_tool_call_id(combined_id: str) -> tuple[str, str | None]:
    # When reasoning, the Responses API requires the `ResponseFunctionToolCall` to be returned with both the `call_id` and `id` fields.
    # Before our `ToolCallPart` gained the `id` field alongside `tool_call_id` field, we combined the two fields into a single string stored on `tool_call_id`.
    if '|' in combined_id:
        call_id, id = combined_id.split('|', 1)
        return call_id, id
    else:
        return combined_id, None


def _map_code_interpreter_tool_call(
    item: responses.ResponseCodeInterpreterToolCall, provider_name: str
) -> tuple[BuiltinToolCallPart, BuiltinToolReturnPart, list[FilePart]]:
    result: dict[str, Any] = {
        'status': item.status,
    }

    file_parts: list[FilePart] = []
    logs: list[str] = []
    if item.outputs:
        for output in item.outputs:
            if isinstance(output, responses.response_code_interpreter_tool_call.OutputImage):
                file_parts.append(
                    FilePart(
                        content=BinaryImage.from_data_uri(output.url),
                        id=item.id,
                    )
                )
            elif isinstance(output, responses.response_code_interpreter_tool_call.OutputLogs):
                logs.append(output.logs)
            else:
                assert_never(output)

    if logs:
        result['logs'] = logs

    return (
        BuiltinToolCallPart(
            tool_name=CodeExecutionTool.kind,
            tool_call_id=item.id,
            args={
                'container_id': item.container_id,
                'code': item.code or '',
            },
            provider_name=provider_name,
        ),
        BuiltinToolReturnPart(
            tool_name=CodeExecutionTool.kind,
            tool_call_id=item.id,
            content=result,
            provider_name=provider_name,
        ),
        file_parts,
    )


def _map_web_search_tool_call(
    item: responses.ResponseFunctionWebSearch, provider_name: str
) -> tuple[BuiltinToolCallPart, BuiltinToolReturnPart]:
    args: dict[str, Any] | None = None

    result = {
        'status': item.status,
    }

    if action := item.action:
        args = action.model_dump(mode='json')

        # To prevent `Unknown parameter: 'input[2].action.sources'` for `ActionSearch`
        if sources := args.pop('sources', None):
            result['sources'] = sources

    return (
        BuiltinToolCallPart(
            tool_name=WebSearchTool.kind,
            tool_call_id=item.id,
            args=args,
            provider_name=provider_name,
        ),
        BuiltinToolReturnPart(
            tool_name=WebSearchTool.kind,
            tool_call_id=item.id,
            content=result,
            provider_name=provider_name,
        ),
    )


def _map_image_generation_tool_call(
    item: responses.response_output_item.ImageGenerationCall, provider_name: str
) -> tuple[BuiltinToolCallPart, BuiltinToolReturnPart, FilePart | None]:
    result = {
        'status': item.status,
    }

    # Not present on the type, but present on the actual object.
    # See https://github.com/openai/openai-python/issues/2649
    if background := getattr(item, 'background', None):
        result['background'] = background
    if quality := getattr(item, 'quality', None):
        result['quality'] = quality
    if size := getattr(item, 'size', None):
        result['size'] = size
    if revised_prompt := getattr(item, 'revised_prompt', None):
        result['revised_prompt'] = revised_prompt
    output_format = getattr(item, 'output_format', 'png')

    file_part: FilePart | None = None
    if item.result:
        file_part = FilePart(
            content=BinaryImage(
                data=base64.b64decode(item.result),
                media_type=f'image/{output_format}',
            ),
            id=item.id,
        )

        # For some reason, the streaming API leaves `status` as `generating` even though generation has completed.
        result['status'] = 'completed'

    return (
        BuiltinToolCallPart(
            tool_name=ImageGenerationTool.kind,
            tool_call_id=item.id,
            provider_name=provider_name,
        ),
        BuiltinToolReturnPart(
            tool_name=ImageGenerationTool.kind,
            tool_call_id=item.id,
            content=result,
            provider_name=provider_name,
        ),
        file_part,
    )


def _map_mcp_list_tools(
    item: responses.response_output_item.McpListTools, provider_name: str
) -> tuple[BuiltinToolCallPart, BuiltinToolReturnPart]:
    tool_name = ':'.join([MCPServerTool.kind, item.server_label])
    return (
        BuiltinToolCallPart(
            tool_name=tool_name,
            tool_call_id=item.id,
            provider_name=provider_name,
            args={'action': 'list_tools'},
        ),
        BuiltinToolReturnPart(
            tool_name=tool_name,
            tool_call_id=item.id,
            content=item.model_dump(mode='json', include={'tools', 'error'}),
            provider_name=provider_name,
        ),
    )


def _map_mcp_call(
    item: responses.response_output_item.McpCall, provider_name: str
) -> tuple[BuiltinToolCallPart, BuiltinToolReturnPart]:
    tool_name = ':'.join([MCPServerTool.kind, item.server_label])
    return (
        BuiltinToolCallPart(
            tool_name=tool_name,
            tool_call_id=item.id,
            args={
                'action': 'call_tool',
                'tool_name': item.name,
                'tool_args': json.loads(item.arguments) if item.arguments else {},
            },
            provider_name=provider_name,
        ),
        BuiltinToolReturnPart(
            tool_name=tool_name,
            tool_call_id=item.id,
            content={
                'output': item.output,
                'error': item.error,
            },
            provider_name=provider_name,
        ),
    )
