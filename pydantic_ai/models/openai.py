from __future__ import annotations as _annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from httpx import AsyncClient as AsyncHTTPClient
from openai import AsyncOpenAI
from openai.types import ChatModel, chat
from typing_extensions import assert_never

from .. import shared
from ..messages import (
    ArgsJson,
    LLMMessage,
    LLMResponse,
    LLMToolCalls,
    Message,
    RetryPrompt,
    ToolCall,
    ToolReturn,
)
from . import AbstractToolDefinition, AgentModel, Model, cached_async_http_client


@dataclass(init=False)
class OpenAIModel(Model):
    model_name: ChatModel
    client: AsyncOpenAI = field(repr=False)

    def __init__(
        self,
        model_name: ChatModel,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: AsyncHTTPClient | None = None,
    ):
        self.model_name: ChatModel = model_name
        if openai_client is not None:
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            self.client = openai_client
        elif http_client is not None:
            self.client = AsyncOpenAI(api_key=api_key, http_client=http_client)
        else:
            self.client = AsyncOpenAI(api_key=api_key, http_client=cached_async_http_client())

    def agent_model(
        self,
        retrievers: Mapping[str, AbstractToolDefinition],
        allow_text_result: bool,
        result_tools: Sequence[AbstractToolDefinition] | None,
    ) -> AgentModel:
        tools = [self.map_tool_definition(r) for r in retrievers.values()]
        if result_tools is not None:
            tools += [self.map_tool_definition(r) for r in result_tools]
        return OpenAIAgentModel(
            self.client,
            self.model_name,
            allow_text_result,
            tools,
        )

    def name(self) -> str:
        return f'openai:{self.model_name}'

    @staticmethod
    def map_tool_definition(f: AbstractToolDefinition) -> chat.ChatCompletionToolParam:
        return {
            'type': 'function',
            'function': {
                'name': f.name,
                'description': f.description,
                'parameters': f.json_schema,
            },
        }


@dataclass
class OpenAIAgentModel(AgentModel):
    client: AsyncOpenAI
    model_name: ChatModel
    allow_text_result: bool
    tools: list[chat.ChatCompletionToolParam]

    async def request(self, messages: list[Message]) -> tuple[LLMMessage, shared.Cost]:
        response = await self.completions_create(messages)
        return self.process_response(response), _map_cost(response)

    @staticmethod
    def process_response(response: chat.ChatCompletion) -> LLMMessage:
        choice = response.choices[0]
        timestamp = datetime.fromtimestamp(response.created)
        if choice.message.tool_calls is not None:
            return LLMToolCalls(
                [ToolCall.from_json(c.function.name, c.function.arguments, c.id) for c in choice.message.tool_calls],
                timestamp=timestamp,
            )
        else:
            assert choice.message.content is not None, choice
            return LLMResponse(choice.message.content, timestamp=timestamp)

    async def completions_create(self, messages: list[Message]) -> chat.ChatCompletion:
        # standalone function to make it easier to override
        if not self.tools:
            tool_choice: Literal['none', 'required', 'auto'] = 'none'
        elif not self.allow_text_result:
            tool_choice = 'required'
        else:
            tool_choice = 'auto'

        openai_messages = [self.map_message(m) for m in messages]
        return await self.client.chat.completions.create(
            model=self.model_name,
            messages=openai_messages,
            n=1,
            parallel_tool_calls=True,
            tools=self.tools,
            tool_choice=tool_choice,
        )

    @staticmethod
    def map_message(message: Message) -> chat.ChatCompletionMessageParam:
        """Just maps a `pydantic_ai.Message` to a `openai.types.ChatCompletionMessageParam`."""
        if message.role == 'system':
            # SystemPrompt ->
            return chat.ChatCompletionSystemMessageParam(role='system', content=message.content)
        elif message.role == 'user':
            # UserPrompt ->
            return chat.ChatCompletionUserMessageParam(role='user', content=message.content)
        elif message.role == 'tool-return':
            # ToolReturn ->
            return chat.ChatCompletionToolMessageParam(
                role='tool',
                tool_call_id=_guard_tool_id(message),
                content=message.model_response_str(),
            )
        elif message.role == 'retry-prompt':
            # RetryPrompt ->
            if message.tool_name is None:
                return chat.ChatCompletionUserMessageParam(role='user', content=message.model_response())
            else:
                return chat.ChatCompletionToolMessageParam(
                    role='tool',
                    tool_call_id=_guard_tool_id(message),
                    content=message.model_response(),
                )
        elif message.role == 'llm-response':
            # LLMResponse ->
            return chat.ChatCompletionAssistantMessageParam(role='assistant', content=message.content)
        elif message.role == 'llm-tool-calls':
            # LLMToolCalls ->
            return chat.ChatCompletionAssistantMessageParam(
                role='assistant',
                tool_calls=[_map_tool_call(t) for t in message.calls],
            )
        else:
            assert_never(message)


def _guard_tool_id(t: ToolCall | ToolReturn | RetryPrompt) -> str:
    """Type guard that checks a `tool_id` is not None both for static typing and runtime."""
    assert t.tool_id is not None, f'OpenAI requires `tool_id` to be set: {t}'
    return t.tool_id


def _map_tool_call(t: ToolCall) -> chat.ChatCompletionMessageToolCallParam:
    assert isinstance(t.args, ArgsJson), f'Expected ArgsJson, got {t.args}'
    return chat.ChatCompletionMessageToolCallParam(
        id=_guard_tool_id(t),
        type='function',
        function={'name': t.tool_name, 'arguments': t.args.args_json},
    )


def _map_cost(response: chat.ChatCompletion) -> shared.Cost:
    usage = response.usage
    if usage is None:
        return shared.Cost()
    else:
        details: dict[str, int] = {}
        if usage.completion_tokens_details is not None:
            details.update(usage.completion_tokens_details.model_dump(exclude_none=True))
        if usage.prompt_tokens_details is not None:
            details.update(usage.prompt_tokens_details.model_dump(exclude_none=True))
        return shared.Cost(
            request_tokens=usage.prompt_tokens,
            response_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            details=details,
        )
