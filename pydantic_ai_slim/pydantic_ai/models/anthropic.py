from __future__ import annotations as _annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, Union, cast, overload

from httpx import AsyncClient as AsyncHTTPClient
from typing_extensions import assert_never

from .. import result
from .._utils import guard_tool_call_id as _guard_tool_call_id
from ..messages import (
    ArgsDict,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
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
    AgentModel,
    EitherStreamedResponse,
    Model,
    cached_async_http_client,
    check_allow_model_requests,
)

try:
    from anthropic import NOT_GIVEN, AsyncAnthropic, AsyncStream
    from anthropic.types import (
        Message as AnthropicMessage,
        MessageParam,
        RawMessageDeltaEvent,
        RawMessageStartEvent,
        RawMessageStreamEvent,
        TextBlock,
        TextBlockParam,
        ToolChoiceParam,
        ToolParam,
        ToolResultBlockParam,
        ToolUseBlock,
        ToolUseBlockParam,
    )
except ImportError as _import_error:
    raise ImportError(
        'Please install `anthropic` to use the Anthropic model, '
        "you can use the `anthropic` optional group — `pip install 'pydantic-ai-slim[anthropic]'`"
    ) from _import_error

LatestAnthropicModelNames = Literal[
    'claude-3-5-haiku-latest',
    'claude-3-5-sonnet-latest',
    'claude-3-opus-latest',
]
"""Latest named Anthropic models."""

AnthropicModelName = Union[str, LatestAnthropicModelNames]
"""Possible Anthropic model names.

Since Anthropic supports a variety of date-stamped models, we explicitly list the latest models but
allow any name in the type hints.
Since [the Anthropic docs](https://docs.anthropic.com/en/docs/about-claude/models) for a full list.
"""


@dataclass(init=False)
class AnthropicModel(Model):
    """A model that uses the Anthropic API.

    Internally, this uses the [Anthropic Python client](https://github.com/anthropics/anthropic-sdk-python) to interact with the API.

    Apart from `__init__`, all methods are private or match those of the base class.

    !!! note
        The `AnthropicModel` class does not yet support streaming responses.
        We anticipate adding support for streaming responses in a near-term future release.
    """

    model_name: AnthropicModelName
    client: AsyncAnthropic = field(repr=False)

    def __init__(
        self,
        model_name: AnthropicModelName,
        *,
        api_key: str | None = None,
        anthropic_client: AsyncAnthropic | None = None,
        http_client: AsyncHTTPClient | None = None,
    ):
        """Initialize an Anthropic model.

        Args:
            model_name: The name of the Anthropic model to use. List of model names available
                [here](https://docs.anthropic.com/en/docs/about-claude/models).
            api_key: The API key to use for authentication, if not provided, the `ANTHROPIC_API_KEY` environment variable
                will be used if available.
            anthropic_client: An existing
                [`AsyncAnthropic`](https://github.com/anthropics/anthropic-sdk-python?tab=readme-ov-file#async-usage)
                client to use, if provided, `api_key` and `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        self.model_name = model_name
        if anthropic_client is not None:
            assert http_client is None, 'Cannot provide both `anthropic_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `anthropic_client` and `api_key`'
            self.client = anthropic_client
        elif http_client is not None:
            self.client = AsyncAnthropic(api_key=api_key, http_client=http_client)
        else:
            self.client = AsyncAnthropic(api_key=api_key, http_client=cached_async_http_client())

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        check_allow_model_requests()
        tools = [self._map_tool_definition(r) for r in function_tools]
        if result_tools:
            tools += [self._map_tool_definition(r) for r in result_tools]
        return AnthropicAgentModel(
            self.client,
            self.model_name,
            allow_text_result,
            tools,
        )

    def name(self) -> str:
        return self.model_name

    @staticmethod
    def _map_tool_definition(f: ToolDefinition) -> ToolParam:
        return {
            'name': f.name,
            'description': f.description,
            'input_schema': f.parameters_json_schema,
        }


@dataclass
class AnthropicAgentModel(AgentModel):
    """Implementation of `AgentModel` for Anthropic models."""

    client: AsyncAnthropic
    model_name: str
    allow_text_result: bool
    tools: list[ToolParam]

    async def request(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None
    ) -> tuple[ModelResponse, result.Usage]:
        response = await self._messages_create(messages, False, model_settings)
        return self._process_response(response), _map_usage(response)

    @asynccontextmanager
    async def request_stream(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None
    ) -> AsyncIterator[EitherStreamedResponse]:
        response = await self._messages_create(messages, True, model_settings)
        async with response:
            yield await self._process_streamed_response(response)

    @overload
    async def _messages_create(
        self, messages: list[ModelMessage], stream: Literal[True], model_settings: ModelSettings | None
    ) -> AsyncStream[RawMessageStreamEvent]:
        pass

    @overload
    async def _messages_create(
        self, messages: list[ModelMessage], stream: Literal[False], model_settings: ModelSettings | None
    ) -> AnthropicMessage:
        pass

    async def _messages_create(
        self, messages: list[ModelMessage], stream: bool, model_settings: ModelSettings | None
    ) -> AnthropicMessage | AsyncStream[RawMessageStreamEvent]:
        # standalone function to make it easier to override
        if not self.tools:
            tool_choice: ToolChoiceParam | None = None
        elif not self.allow_text_result:
            tool_choice = {'type': 'any'}
        else:
            tool_choice = {'type': 'auto'}

        system_prompt, anthropic_messages = self._map_message(messages)

        model_settings = model_settings or {}

        return await self.client.messages.create(
            max_tokens=model_settings.get('max_tokens', 1024),
            system=system_prompt or NOT_GIVEN,
            messages=anthropic_messages,
            model=self.model_name,
            tools=self.tools or NOT_GIVEN,
            tool_choice=tool_choice or NOT_GIVEN,
            stream=stream,
            temperature=model_settings.get('temperature', NOT_GIVEN),
            top_p=model_settings.get('top_p', NOT_GIVEN),
            timeout=model_settings.get('timeout', NOT_GIVEN),
        )

    @staticmethod
    def _process_response(response: AnthropicMessage) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        items: list[ModelResponsePart] = []
        for item in response.content:
            if isinstance(item, TextBlock):
                items.append(TextPart(item.text))
            else:
                assert isinstance(item, ToolUseBlock), 'unexpected item type'
                items.append(
                    ToolCallPart.from_raw_args(
                        item.name,
                        cast(dict[str, Any], item.input),
                        item.id,
                    )
                )

        return ModelResponse(items)

    @staticmethod
    async def _process_streamed_response(response: AsyncStream[RawMessageStreamEvent]) -> EitherStreamedResponse:
        """TODO: Process a streamed response, and prepare a streaming response to return."""
        # We don't yet support streamed responses from Anthropic, so we raise an error here for now.
        # Streamed responses will be supported in a future release.

        raise RuntimeError('Streamed responses are not yet supported for Anthropic models.')

        # Should be returning some sort of AnthropicStreamTextResponse or AnthropicStreamStructuredResponse
        # depending on the type of chunk we get, but we need to establish how we handle (and when we get) the following:
        # RawMessageStartEvent
        # RawMessageDeltaEvent
        # RawMessageStopEvent
        # RawContentBlockStartEvent
        # RawContentBlockDeltaEvent
        # RawContentBlockDeltaEvent
        #
        # We might refactor streaming internally before we implement this...

    @staticmethod
    def _map_message(messages: list[ModelMessage]) -> tuple[str, list[MessageParam]]:
        """Just maps a `pydantic_ai.Message` to a `anthropic.types.MessageParam`."""
        system_prompt: str = ''
        anthropic_messages: list[MessageParam] = []
        for m in messages:
            if isinstance(m, ModelRequest):
                for part in m.parts:
                    if isinstance(part, SystemPromptPart):
                        system_prompt += part.content
                    elif isinstance(part, UserPromptPart):
                        anthropic_messages.append(MessageParam(role='user', content=part.content))
                    elif isinstance(part, ToolReturnPart):
                        anthropic_messages.append(
                            MessageParam(
                                role='user',
                                content=[
                                    ToolResultBlockParam(
                                        tool_use_id=_guard_tool_call_id(t=part, model_source='Anthropic'),
                                        type='tool_result',
                                        content=part.model_response_str(),
                                        is_error=False,
                                    )
                                ],
                            )
                        )
                    elif isinstance(part, RetryPromptPart):
                        if part.tool_name is None:
                            anthropic_messages.append(MessageParam(role='user', content=part.model_response()))
                        else:
                            anthropic_messages.append(
                                MessageParam(
                                    role='user',
                                    content=[
                                        ToolResultBlockParam(
                                            tool_use_id=_guard_tool_call_id(t=part, model_source='Anthropic'),
                                            type='tool_result',
                                            content=part.model_response(),
                                            is_error=True,
                                        ),
                                    ],
                                )
                            )
            elif isinstance(m, ModelResponse):
                content: list[TextBlockParam | ToolUseBlockParam] = []
                for item in m.parts:
                    if isinstance(item, TextPart):
                        content.append(TextBlockParam(text=item.content, type='text'))
                    else:
                        assert isinstance(item, ToolCallPart)
                        content.append(_map_tool_call(item))
                anthropic_messages.append(MessageParam(role='assistant', content=content))
            else:
                assert_never(m)
        return system_prompt, anthropic_messages


def _map_tool_call(t: ToolCallPart) -> ToolUseBlockParam:
    assert isinstance(t.args, ArgsDict), f'Expected ArgsDict, got {t.args}'
    return ToolUseBlockParam(
        id=_guard_tool_call_id(t=t, model_source='Anthropic'),
        type='tool_use',
        name=t.tool_name,
        input=t.args_as_dict(),
    )


def _map_usage(message: AnthropicMessage | RawMessageStreamEvent) -> result.Usage:
    if isinstance(message, AnthropicMessage):
        usage = message.usage
    else:
        if isinstance(message, RawMessageStartEvent):
            usage = message.message.usage
        elif isinstance(message, RawMessageDeltaEvent):
            usage = message.usage
        else:
            # No usage information provided in:
            # - RawMessageStopEvent
            # - RawContentBlockStartEvent
            # - RawContentBlockDeltaEvent
            # - RawContentBlockStopEvent
            usage = None

    if usage is None:
        return result.Usage()

    request_tokens = getattr(usage, 'input_tokens', None)

    return result.Usage(
        # Usage coming from the RawMessageDeltaEvent doesn't have input token data, hence this getattr
        request_tokens=request_tokens,
        response_tokens=usage.output_tokens,
        total_tokens=(request_tokens or 0) + usage.output_tokens,
    )
