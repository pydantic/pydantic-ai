from __future__ import annotations as _annotations

from dataclasses import dataclass
from datetime import datetime
from functools import cache
from typing import Literal, assert_never

from openai import AsyncClient
from openai.types import ChatModel
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam, ChatCompletionToolParam

from ..messages import (
    LLMMessage,
    LLMResponse,
    LLMToolCalls,
    Message,
    ToolCall,
    ToolRetry,
    ToolReturn,
)
from . import AbstractToolDefinition, AgentModel, Model


class OpenAIModel(Model):
    def __init__(self, model_name: ChatModel, *, api_key: str | None = None, client: AsyncClient | None = None):
        if model_name not in ChatModel.__args__:
            raise ValueError(f'Invalid model name: {model_name}')
        self.model_name: ChatModel = model_name
        self.client = client or cached_async_client(api_key)

    def agent_model(
        self, allow_text_result: bool, tools: list[AbstractToolDefinition], result_tool_name: str | None
    ) -> AgentModel:
        return OpenAIAgentModel(
            self.client,
            self.model_name,
            allow_text_result,
            [map_tool_definition(t) for t in tools],
        )


@dataclass
class OpenAIAgentModel(AgentModel):
    client: AsyncClient
    model_name: ChatModel
    allow_text_result: bool
    tools: list[ChatCompletionToolParam]

    async def request(self, messages: list[Message]) -> LLMMessage:
        response = await self.completions_create(messages)
        choice = response.choices[0]
        timestamp = datetime.fromtimestamp(response.created)
        if choice.message.tool_calls is not None:
            return LLMToolCalls(
                [
                    ToolCall(
                        tool_name=c.function.name,
                        arguments=c.function.arguments,
                        tool_id=c.id,
                    )
                    for c in choice.message.tool_calls
                ],
                timestamp=timestamp,
            )
        else:
            assert choice.message.content is not None, choice
            return LLMResponse(choice.message.content, timestamp=timestamp)

    async def completions_create(self, messages: list[Message]) -> ChatCompletion:
        # standalone function to make it easier to override
        if not self.tools:
            tool_choice: Literal['none', 'required', 'auto'] = 'none'
        elif not self.allow_text_result:
            tool_choice = 'required'
        else:
            tool_choice = 'auto'

        openai_messages = [map_message(m) for m in messages]
        return await self.client.chat.completions.create(
            model=self.model_name,
            messages=openai_messages,
            n=1,
            parallel_tool_calls=True,
            tools=self.tools,
            tool_choice=tool_choice,
        )


@cache
def cached_async_client(api_key: str) -> AsyncClient:
    return AsyncClient(api_key=api_key)


def map_tool_definition(f: AbstractToolDefinition) -> ChatCompletionToolParam:
    return {
        'type': 'function',
        'function': {
            'name': f.name,
            'description': f.description,
            'parameters': f.json_schema,  # type: ignore
        },
    }


def map_message(message: Message) -> ChatCompletionMessageParam:
    """Just maps a `pydantic_ai.Message` to a `openai.types.ChatCompletionMessageParam`."""
    if message.role == 'system':
        # SystemPrompt -> ChatCompletionSystemMessageParam
        return {'role': 'system', 'content': message.content}
    elif message.role == 'user':
        # UserPrompt -> ChatCompletionUserMessageParam
        return {'role': 'user', 'content': message.content}
    elif message.role == 'tool-return' or message.role == 'tool-retry':
        # ToolReturn or ToolRetry -> ChatCompletionToolMessageParam
        return {
            'role': 'tool',
            'tool_call_id': guard_tool_id(message),
            'content': message.llm_response(),
        }
    elif message.role == 'llm-response':
        # LLMResponse -> ChatCompletionAssistantMessageParam
        return {'role': 'assistant', 'content': message.content}
    elif message.role == 'llm-tool-calls':
        # LLMToolCalls -> ChatCompletionAssistantMessageParam
        return {
            'role': 'assistant',
            'tool_calls': [
                {
                    'id': guard_tool_id(t),
                    'type': 'function',
                    'function': {'name': t.tool_name, 'arguments': t.arguments},
                }
                for t in message.calls
            ],
        }
    elif message.role == 'plain-response-forbidden':
        # PlainResponseForbidden -> ChatCompletionUserMessageParam
        return {
            'role': 'user',
            'content': 'Plain text responses are not allowed, please call one of the functions instead.',
        }
    else:
        assert_never(message)


def guard_tool_id(t: ToolCall | ToolReturn | ToolRetry) -> str:
    """Type guard that checks a `tool_id` is not None both for static typing and runtime."""
    assert t.tool_id is not None, f'OpenAI requires `tool_id` to be set: {t}'
    return t.tool_id
