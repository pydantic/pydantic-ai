from __future__ import annotations as _annotations

# Python imports
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

# Third-party imports
from typing_extensions import assert_never

# Local imports
from .. import result
from ..messages import (
    ArgsJson,
    Message,
    ModelAnyResponse,
    ModelStructuredResponse,
    ModelTextResponse,
    RetryPrompt,
    ToolCall,
    ToolReturn,
)
from ..tools import ToolDefinition
from . import (
    AgentModel,
    EitherStreamedResponse,
    Model,
    check_allow_model_requests,
)

try:
    from cohere import (
        AssistantChatMessageV2,
        AsyncClientV2,
        ChatMessageV2,
        ChatResponse,
        SystemChatMessageV2,
        ToolCallV2,
        ToolCallV2Function,
        ToolChatMessageV2,
        ToolV2,
        ToolV2Function,
        UserChatMessageV2,
    )
    from cohere.v2.client import OMIT
except ImportError as e:
    raise ImportError(
        'Please install `cohere` to use the Cohere model, '
        "you can use the `cohere` optional group — `pip install 'pydantic-ai[cohere]'`"
    ) from e


# TODO Create a union of all the models available in the Cohere API
type ChatModel = str


@dataclass(init=False)
class CohereModel(Model):
    """A model that uses the Cohere API.

    Internally, this uses the [Cohere Python client](https://github.com/cohere-ai/cohere-python)
    to interact with the API.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    model_name: ChatModel
    client: AsyncClientV2 = field(repr=False)

    def __init__(
        self,
        model_name: ChatModel,
        *,
        api_key: str | None = None,
        cohere_client: AsyncClientV2 | None = None,
    ):
        """Initialize a Cohere model.

        Args:
            model_name: The name of the Cohere model to use. List of model names available
                [here](https://github.com/openai/openai-python/blob/v1.54.3/src/openai/types/chat_model.py#L7)
                (Unfortunately, despite being ask to do so, OpenAI do not provide `.inv` files for their API).
            api_key: The API key to use for authentication, if not provided, the `OPENAI_API_KEY` environment variable
                will be used if available.
            cohere_client: An existing
                [`AsyncClientV2`](https://github.com/openai/openai-python?tab=readme-ov-file#async-usage)
                client to use, if provided, `api_key` and `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        self.model_name: ChatModel = model_name
        if cohere_client is not None:
            assert api_key is None, 'Cannot provide both `cohere_client` and `api_key`'
            self.client = cohere_client
        else:
            self.client = AsyncClientV2(api_key=api_key)  # type: ignore

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
        return CohereAgentModel(
            self.client,
            self.model_name,
            allow_text_result,
            tools,
        )

    def name(self) -> str:
        return f'openai:{self.model_name}'

    @staticmethod
    def _map_tool_definition(f: ToolDefinition) -> ToolV2:
        return ToolV2(
            type='function',
            function=ToolV2Function(
                name=f.name,
                description=f.description,
                parameters=f.parameters_json_schema,
            ),
        )


@dataclass
class CohereAgentModel(AgentModel):
    """Implementation of `AgentModel` for OpenAI models."""

    client: AsyncClientV2
    model_name: ChatModel
    allow_text_result: bool
    tools: list[ToolV2]

    async def request(self, messages: list[Message]) -> tuple[ModelAnyResponse, result.Cost]:
        response = await self._chat(messages)
        return self._process_response(response), _map_cost(response)

    # TODO Add streaming support.
    @asynccontextmanager
    async def request_stream(self, messages: list[Message]) -> AsyncIterator[EitherStreamedResponse]:
        """Make a request to the model and return a streaming response."""
        raise NotImplementedError(f'Streamed requests not supported by this {self.__class__.__name__}')
        # yield is required to make this a generator for type checking
        # noinspection PyUnreachableCode
        yield  # pragma: no cover

    async def _chat(self, messages: list[Message]):
        openai_messages = [self._map_message(m) for m in messages]
        return await self.client.chat(
            model=self.model_name,
            messages=openai_messages,
            tools=self.tools or OMIT,
        )

    @staticmethod
    def _process_response(response: ChatResponse) -> ModelAnyResponse:
        """Process a non-streamed response, and prepare a message to return."""
        if response.message.tool_calls is not None:
            return ModelStructuredResponse(
                [
                    ToolCall.from_json(c.function.name, c.function.arguments, c.id)
                    for c in response.message.tool_calls
                    if c.function and c.function.name and c.function.arguments
                ],
            )
        else:
            assert response.message is not None
            assert response.message.content is not None
            assert len(response.message.content) > 0
            return ModelTextResponse(response.message.content[0].text)

    @staticmethod
    def _map_message(message: Message) -> ChatMessageV2:
        """Just maps a `pydantic_ai.Message` to a `openai.types.ChatCompletionMessageParam`."""
        if message.role == 'system':
            # SystemPrompt ->
            return SystemChatMessageV2(role='system', content=message.content)
        elif message.role == 'user':
            # UserPrompt ->
            return UserChatMessageV2(role='user', content=message.content)
        elif message.role == 'tool-return':
            # ToolReturn ->
            return ToolChatMessageV2(
                role='tool',
                tool_call_id=_guard_tool_id(message),
                content=message.model_response_str(),
            )
        elif message.role == 'retry-prompt':
            # RetryPrompt ->
            if message.tool_name is None:
                return UserChatMessageV2(role='user', content=message.model_response())
            else:
                return ToolChatMessageV2(
                    role='tool',
                    tool_call_id=_guard_tool_id(message),
                    content=message.model_response(),
                )
        elif message.role == 'model-text-response':
            # ModelTextResponse ->
            return AssistantChatMessageV2(role='assistant', content=message.content)
        elif message.role == 'model-structured-response':
            assert (
                message.role == 'model-structured-response'
            ), f'Expected role to be "llm-tool-calls", got {message.role}'
            # ModelStructuredResponse ->
            return AssistantChatMessageV2(
                role='assistant',
                tool_calls=[_map_tool_call(t) for t in message.calls],
            )
        else:
            assert_never(message)


def _guard_tool_id(t: ToolCall | ToolReturn | RetryPrompt) -> str:
    """Type guard that checks a `tool_id` is not None both for static typing and runtime."""
    assert t.tool_id is not None, f'Cohere requires `tool_id` to be set: {t}'
    return t.tool_id


def _map_tool_call(t: ToolCall) -> ToolCallV2:
    assert isinstance(t.args, ArgsJson), f'Expected ArgsJson, got {t.args}'
    return ToolCallV2(
        id=_guard_tool_id(t),
        type='function',
        function=ToolCallV2Function(
            name=t.tool_name,
            arguments=t.args.args_json,
        ),
    )


def _map_cost(response: ChatResponse) -> result.Cost:
    usage = response.usage
    if usage is None:
        return result.Cost()
    else:
        return result.Cost(
            # todo fix typing
            request_tokens=int(usage.tokens.input_tokens),  # type: ignore
            response_tokens=int(usage.tokens.output_tokens),  # type: ignore
            total_tokens=int(usage.tokens.input_tokens + usage.tokens.output_tokens),  # type: ignore
            # todo add details
            details=None,
        )
