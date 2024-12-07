from __future__ import annotations as _annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Literal

from httpx import AsyncClient as AsyncHTTPClient
from mistralai import UsageInfo
from mistralai.types.basemodel import Unset
from typing_extensions import assert_never

from .. import UnexpectedModelBehavior, result
from ..messages import (
    Message,
    ModelAnyResponse,
    ModelStructuredResponse,
    ModelTextResponse,
    ToolCall,
)
from . import (
    AbstractToolDefinition,
    AgentModel,
    EitherStreamedResponse,
    Model,
    cached_async_http_client,
    check_allow_model_requests,
)

try:
    from mistralai import Mistral, models, utils
    from mistralai.models.assistantmessage import (
        AssistantMessage,
        AssistantMessageContent,
        AssistantMessageContentTypedDict,
        AssistantMessageRole,
        AssistantMessageTypedDict,
    )
    from mistralai.models.chatcompletionchoice import (
        ChatCompletionChoice,
        ChatCompletionChoiceTypedDict,
    )
    from mistralai.models.function import Function, FunctionTypedDict
    from mistralai.models.toolmessage import (
        ToolMessage,
        ToolMessageContent,
        ToolMessageContentTypedDict,
        ToolMessageRole,
        ToolMessageTypedDict,
    )
    from mistralai.models.usermessage import (
        UserMessage,
        UserMessageContent,
        UserMessageContentTypedDict,
        UserMessageRole,
        UserMessageTypedDict,
    )
    from mistralai.types import UNSET
except ImportError as e:
    raise ImportError(
        "Please install `mistral` to use the Mistral model, "
        "you can use the `mistral` optional group — `pip install 'pydantic-ai[mistral]'`"
    ) from e

MistralModelName = Literal["small-mistral",]


@dataclass(init=False)
class MistralModel(Model):
    model_name: MistralModelName
    client: Mistral = field(repr=False)

    def __init__(
        self,
        model_name: MistralModelName,
        *,
        api_key: str | None = None,
        client: Mistral | None = None,
        http_client: AsyncHTTPClient | None = None,
    ):
        self.model_name = model_name
        if client is not None:
            assert (
                http_client is None
            ), "Cannot provide both `mistral_client` and `http_client`"
            assert api_key is None, "Cannot provide both `mistral_client` and `api_key`"
            self.client = client
        elif http_client is not None:
            self.client = Mistral(api_key=api_key, async_client=http_client)
        else:
            self.client = Mistral(
                api_key=api_key, async_client=cached_async_http_client()
            )
            # self.client.chat.complete_async

    async def agent_model(
        self,
        function_tools: Mapping[str, AbstractToolDefinition],
        allow_text_result: bool,
        result_tools: Sequence[AbstractToolDefinition] | None,
    ) -> AgentModel:
        check_allow_model_requests()
        tools = [self._map_tool_definition(r) for r in function_tools.values()]
        if result_tools is not None:
            tools += [self._map_tool_definition(r) for r in result_tools]
        return MistralAgentModel(
            self.client,
            self.model_name,
            allow_text_result,
            tools,
        )

    def name(self) -> str:
        return f"mistral:{self.model_name}"

    @staticmethod
    def _map_tool_definition(
        f: AbstractToolDefinition,
    ) -> models.Tool | models.ToolTypedDict:
        """Convert an `AbstractToolDefinition` to a `Tool` or `ToolTypedDict`.

        This is a utility function used to convert our internal representation of a tool to the
        representation expected by the Mistral API.
        """
        function = Function(
            name=f.name, parameters=f.json_schema, description=f.description
        )
        return models.Tool(function=function)


@dataclass
class MistralAgentModel(AgentModel):
    """Implementation of `AgentModel` for Mistral models."""

    client: Mistral
    model_name: str
    allow_text_result: bool
    tools: list[models.Tool | models.ToolTypedDict] = field(default_factory=list)

    async def request(
        self, messages: list[Message]
    ) -> tuple[ModelAnyResponse, result.Cost]:
        response = await self._completions_create(messages, False)
        return self._process_response(response), _map_cost(response)

    @asynccontextmanager
    async def request_stream(
        self, messages: list[Message]
    ) -> AsyncIterator[EitherStreamedResponse]:
        response = await self._completions_create(messages, True)
        async with response:
            yield await self._process_streamed_response(response)

    # @overload
    # async def _completions_create(
    #     self, messages: list[Message], stream: Literal[True]
    # ) -> AsyncStream[ChatCompletionChunk]:
    #     pass

    # @overload
    # async def _completions_create(
    #     self, messages: list[Message], stream: Literal[False]
    # ) -> chat.ChatCompletion:
    #     pass

    async def _completions_create(
        self, messages: list[Message], stream: bool
    ) -> models.ChatCompletionResponse:
        # standalone function to make it easier to override
        if not self.tools:
            tool_choice: Literal["none", "required", "auto"] | None = None
        elif not self.allow_text_result:
            tool_choice = "required"
        else:
            tool_choice = "auto"

        mistral_messages = [self._map_message(m) for m in messages]
        return await self.client.chat.complete_async(
            model=str(self.model_name),
            messages=mistral_messages,
            temperature=0.0,
            n=1,
            # parallel_tool_calls=True if self.tools else NOT_GIVEN,
            tools=self.tools or UNSET,
            tool_choice=tool_choice or None,
            stream=stream,
        ) or models.ChatCompletionResponse(  # TODO: See when return None
            id="",
            object="",
            model="",
            usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

    @staticmethod
    def _process_response(response: models.ChatCompletionResponse) -> ModelAnyResponse:
        """Process a non-streamed response, and prepare a message to return."""
        timestamp: datetime | None
        if response.created:
            timestamp = datetime.fromtimestamp(response.created, tz=timezone.utc)
        else:
            timestamp = datetime.now(tz=timezone.utc)  # TODO: see when datatime

        choices = response.choices  # TODO: Adjust this part.
        assert choices
        assert choices[0]

        choice = choices[0]

        if choice.message.tool_calls is not None and not isinstance(
            choice.message.tool_calls, Unset
        ):
            tools_calls = choice.message.tool_calls
            tools: List[ToolCall] = [
                ToolCall.from_json(
                    tool_name=c.function.name,
                    args_json=c.function.arguments,
                    tool_id=c.id,
                )
                for c in tools_calls
            ]
            return ModelStructuredResponse(
                tools,
                timestamp=timestamp or None,
            )
        else:
            assert choice.message.content is not None, choice
            return ModelTextResponse(
                choice.message.content, timestamp=timestamp or None
            )

    @staticmethod
    async def _process_streamed_response(
        response: models.ChatCompletionResponse,
    ) -> EitherStreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        assert False
        try:
            first_chunk = await response.__anext__()
        except StopAsyncIteration as e:  # pragma: no cover
            raise UnexpectedModelBehavior(
                "Streamed response ended without content or tool calls"
            ) from e
        timestamp = datetime.fromtimestamp(first_chunk.created, tz=timezone.utc)
        delta = first_chunk.choices[0].delta
        start_cost = _map_cost(first_chunk)

        # the first chunk may only contain `role`, so we iterate until we get either `tool_calls` or `content`
        while delta.tool_calls is None and delta.content is None:
            try:
                next_chunk = await response.__anext__()
            except StopAsyncIteration as e:
                raise UnexpectedModelBehavior(
                    "Streamed response ended without content or tool calls"
                ) from e
            delta = next_chunk.choices[0].delta
            start_cost += _map_cost(next_chunk)

        if delta.content is not None:
            return MistralStreamTextResponse(
                delta.content, response, timestamp, start_cost
            )
        else:
            assert (
                delta.tool_calls is not None
            ), f"Expected delta with tool_calls, got {delta}"
            return MistralStreamStructuredResponse(
                response,
                {c.index: c for c in delta.tool_calls},
                timestamp,
                start_cost,
            )

    @staticmethod
    def _map_message(message: Message) -> models.Messages:
        """Just maps a `pydantic_ai.Message` to a `Mistral.types.ChatCompletionMessageParam`."""
        if message.role == "system":
            # SystemPrompt ->
            return AssistantMessage(content=message.content)
        elif message.role == "user":
            # UserPrompt ->
            return UserMessage(content=message.content)
        elif message.role == "tool-return":
            # ToolReturn ->
            return ToolMessage(
                tool_call_id=_guard_tool_id(message),
                content=message.model_response_str(),
            )
        elif message.role == "retry-prompt":
            # RetryPrompt ->
            if message.tool_name is None:
                return UserMessage(content=message.model_response())
            else:
                return ToolMessage(
                    tool_call_id=_guard_tool_id(message),
                    content=message.model_response(),
                )
        elif message.role == "model-text-response":
            # ModelTextResponse ->
            return AssistantMessage(content=message.content)
        elif message.role == "model-structured-response":
            assert (
                message.role == "model-structured-response"
            ), f'Expected role to be "llm-tool-calls", got {message.role}'
            # ModelStructuredResponse ->
            return AssistantMessage(
                tool_calls=[_map_tool_call(t) for t in message.calls],
            )
        else:
            assert_never(message)


def _guard_tool_id(t: ToolCall) -> str:
    """Type guard that checks a `tool_id` is not None both for static typing and runtime."""
    assert t.tool_id is not None, f"Mistral requires `tool_id` to be set: {t}"
    return t.tool_id


def _map_tool_call(t: ToolCall) -> models.ToolCall:
    # assert isinstance(t.args, ArgsJson), f"Expected ArgsJson, got {t.args}"
    return models.ToolCall(
        id=_guard_tool_id(t),
        type="function",
        function={"name": t.tool_name, "arguments": t.args.args_json},
    )


def _map_cost(completion: models.ChatCompletionResponse) -> result.Cost:

    usage = completion.usage

    return result.Cost(
        request_tokens=usage.prompt_tokens,
        response_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
    )
