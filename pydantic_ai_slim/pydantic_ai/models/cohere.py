from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from typing import Literal

from typing_extensions import assert_never

from pydantic_ai.settings import ModelSettings

from .._utils import guard_tool_call_id as _guard_tool_call_id
from ..messages import (
    ArgsJson,
    Message,
    ModelResponse,
    ModelResponsePart,
    RetryPrompt,
    SystemPrompt,
    TextPart,
    ToolCallPart,
    ToolReturn,
    UserPrompt,
)
from ..result import Cost
from ..tools import ToolDefinition
from . import AgentModel, Model, check_allow_model_requests

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


type ChatModel = Literal[
    'command-r',
    'command-r-plus',
]


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
            model_name: The name of the Cohere model to use.
            api_key: The API key to use for authentication, if not provided, the
                `COHERE_API_KEY` environment variable will be used if available.
            cohere_client: An optional existing AsyncClientV2 client to use. If
                provided, `api_key`  must be `None`.
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
        return f'cohere:{self.model_name}'

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
    """Implementation of `AgentModel` for Cohere models."""

    client: AsyncClientV2
    model_name: ChatModel
    allow_text_result: bool
    tools: list[ToolV2]

    async def request(
        self, messages: list[Message], model_settings: ModelSettings | None
    ) -> tuple[ModelResponse, Cost]:
        response = await self._chat(messages)
        return self._process_response(response), _map_cost(response)

    async def _chat(self, messages: list[Message]):
        cohere_messages = [self._map_message(m) for m in messages]
        return await self.client.chat(
            model=self.model_name,
            messages=cohere_messages,
            tools=self.tools or OMIT,
        )

    @staticmethod
    def _process_response(response: ChatResponse) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        parts: list[ModelResponsePart] = []
        if response.message.content is not None and len(response.message.content) > 0:
            choice = response.message.content[0]
            parts.append(TextPart(choice.text))
        for c in response.message.tool_calls or []:
            if c.function and c.function.name and c.function.arguments:
                parts.append(
                    ToolCallPart.from_json(
                        tool_name=c.function.name,
                        args_json=c.function.arguments,
                        tool_call_id=c.id,
                    )
                )
        return ModelResponse(parts=parts)

    @staticmethod
    def _map_message(message: Message) -> ChatMessageV2:
        """Just maps a `pydantic_ai.Message` to a `cohere.ChatMessageV2`."""
        if isinstance(message, SystemPrompt):
            return SystemChatMessageV2(role='system', content=message.content)
        elif isinstance(message, UserPrompt):
            return UserChatMessageV2(role='user', content=message.content)
        elif isinstance(message, ToolReturn):
            return ToolChatMessageV2(
                role='tool',
                tool_call_id=_guard_tool_call_id(message, model_source='Cohere'),
                content=message.model_response_str(),
            )
        elif isinstance(message, RetryPrompt):
            # RetryPrompt ->
            if message.tool_name is None:
                return UserChatMessageV2(role='user', content=message.model_response())
            else:
                return ToolChatMessageV2(
                    role='tool',
                    tool_call_id=_guard_tool_call_id(message, model_source='Cohere'),
                    content=message.model_response(),
                )
        elif isinstance(message, ModelResponse):
            texts: list[str] = []
            tool_calls: list[ToolCallV2] = []
            for item in message.parts:
                if isinstance(item, TextPart):
                    texts.append(item.content)
                elif isinstance(item, ToolCallPart):
                    tool_calls.append(_map_tool_call(item))
                else:
                    assert_never(item)
            return AssistantChatMessageV2(
                content='\n\n'.join(texts),
                tool_calls=tool_calls,
            )
        else:
            assert_never(message)


def _map_tool_call(t: ToolCallPart) -> ToolCallV2:
    assert isinstance(t.args, ArgsJson), f'Expected ArgsJson, got {t.args}'
    return ToolCallV2(
        id=_guard_tool_call_id(t, model_source='Cohere'),
        type='function',
        function=ToolCallV2Function(
            name=t.tool_name,
            arguments=t.args.args_json,
        ),
    )


def _map_cost(response: ChatResponse) -> Cost:
    usage = response.usage
    if usage is None:
        return Cost()
    else:
        return Cost(
            # todo fix typing
            request_tokens=int(usage.tokens.input_tokens),  # type: ignore
            response_tokens=int(usage.tokens.output_tokens),  # type: ignore
            total_tokens=int(usage.tokens.input_tokens + usage.tokens.output_tokens),  # type: ignore
            # todo add details
            details=None,
        )
