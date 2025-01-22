from __future__ import annotations as _annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import chain
from typing import Literal, TypeAlias, Union

from cohere import TextAssistantMessageContentItem
from typing_extensions import assert_never

from .. import result
from .._utils import guard_tool_call_id as _guard_tool_call_id
from ..messages import (
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
except ImportError as _import_error:
    raise ImportError(
        'Please install `cohere` to use the Cohere model, '
        "you can use the `cohere` optional group — `pip install 'pydantic-ai-slim[cohere]'`"
    ) from _import_error

CohereModelName: TypeAlias = Union[
    str,
    Literal[
        'c4ai-aya-expanse-32b',
        'c4ai-aya-expanse-8b',
        'command',
        'command-light',
        'command-light-nightly',
        'command-nightly',
        'command-r',
        'command-r-03-2024',
        'command-r-08-2024',
        'command-r-plus',
        'command-r-plus-04-2024',
        'command-r-plus-08-2024',
        'command-r7b-12-2024',
    ],
]


@dataclass(init=False)
class CohereModel(Model):
    """A model that uses the Cohere API.

    Internally, this uses the [Cohere Python client](
    https://github.com/cohere-ai/cohere-python) to interact with the API.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    model_name: CohereModelName
    client: AsyncClientV2 = field(repr=False)

    def __init__(
        self,
        model_name: CohereModelName,
        *,
        api_key: str | None = None,
        cohere_client: AsyncClientV2 | None = None,
    ):
        """Initialize an Cohere model.

        Args:
            model_name: The name of the Cohere model to use. List of model names
                available [here](https://docs.cohere.com/docs/models#command).
            api_key: The API key to use for authentication, if not provided, the
                `COHERE_API_KEY` environment variable will be used if available.
            cohere_client: An existing Cohere async client to use. If provided,
                `api_key` must be `None`.
        """
        self.model_name: CohereModelName = model_name
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
    model_name: CohereModelName
    allow_text_result: bool
    tools: list[ToolV2]

    async def request(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None
    ) -> tuple[ModelResponse, result.Usage]:
        response = await self._chat(messages, model_settings)
        return self._process_response(response), _map_usage(response)

    async def _chat(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
    ) -> ChatResponse:
        cohere_messages = list(chain(*(self._map_message(m) for m in messages)))
        model_settings = model_settings or {}
        return await self.client.chat(
            model=self.model_name,
            messages=cohere_messages,
            tools=self.tools or OMIT,
            max_tokens=model_settings.get('max_tokens', OMIT),
            temperature=model_settings.get('temperature', OMIT),
            p=model_settings.get('top_p', OMIT),
        )

    def _process_response(self, response: ChatResponse) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        parts: list[ModelResponsePart] = []
        if response.message.content is not None and len(response.message.content) > 0:
            # While Cohere's API returns a list, it only does that for future proofing
            # and currently only one item is being returned.
            choice = response.message.content[0]
            parts.append(TextPart(choice.text))
        for c in response.message.tool_calls or []:
            if c.function and c.function.name and c.function.arguments:
                parts.append(
                    ToolCallPart.from_raw_args(
                        tool_name=c.function.name,
                        args=c.function.arguments,
                        tool_call_id=c.id,
                    )
                )
        return ModelResponse(parts=parts, model_name=self.model_name)

    @classmethod
    def _map_message(cls, message: ModelMessage) -> Iterable[ChatMessageV2]:
        """Just maps a `pydantic_ai.Message` to a `cohere.ChatMessageV2`."""
        if isinstance(message, ModelRequest):
            yield from cls._map_user_message(message)
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
            message_param = AssistantChatMessageV2(role='assistant')
            if texts:
                message_param.content = [TextAssistantMessageContentItem(text='\n\n'.join(texts))]
            if tool_calls:
                message_param.tool_calls = tool_calls
            yield message_param
        else:
            assert_never(message)

    @classmethod
    def _map_user_message(cls, message: ModelRequest) -> Iterable[ChatMessageV2]:
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                yield SystemChatMessageV2(role='system', content=part.content)
            elif isinstance(part, UserPromptPart):
                yield UserChatMessageV2(role='user', content=part.content)
            elif isinstance(part, ToolReturnPart):
                yield ToolChatMessageV2(
                    role='tool',
                    tool_call_id=_guard_tool_call_id(t=part, model_source='Cohere'),
                    content=part.model_response_str(),
                )
            elif isinstance(part, RetryPromptPart):
                if part.tool_name is None:
                    yield UserChatMessageV2(role='user', content=part.model_response())
                else:
                    yield ToolChatMessageV2(
                        role='tool',
                        tool_call_id=_guard_tool_call_id(t=part, model_source='Cohere'),
                        content=part.model_response(),
                    )
            else:
                assert_never(part)


def _map_tool_call(t: ToolCallPart) -> ToolCallV2:
    return ToolCallV2(
        id=_guard_tool_call_id(t=t, model_source='Cohere'),
        type='function',
        function=ToolCallV2Function(
            name=t.tool_name,
            arguments=t.args_as_json_str(),
        ),
    )


def _map_usage(response: ChatResponse) -> result.Usage:
    usage = response.usage
    if usage is None:
        return result.Usage()
    else:
        details: dict[str, int] = {}
        if usage.billed_units is not None:
            if usage.billed_units.input_tokens:
                details['input_tokens'] = int(usage.billed_units.input_tokens)
            if usage.billed_units.output_tokens:
                details['output_tokens'] = int(usage.billed_units.output_tokens)
            if usage.billed_units.search_units:
                details['search_units'] = int(usage.billed_units.search_units)
            if usage.billed_units.classifications:
                details['classifications'] = int(usage.billed_units.classifications)

        request_tokens = int(usage.tokens.input_tokens) if usage.tokens and usage.tokens.input_tokens else None
        response_tokens = int(usage.tokens.output_tokens) if usage.tokens and usage.tokens.output_tokens else None
        return result.Usage(
            request_tokens=request_tokens,
            response_tokens=response_tokens,
            total_tokens=(request_tokens or 0) + (response_tokens or 0),
            details=details,
        )
