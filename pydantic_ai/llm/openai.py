from dataclasses import dataclass
from datetime import datetime
from functools import cache
from typing import assert_never

from openai import AsyncClient
from openai.types import ChatModel
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam, ChatCompletionToolParam

from ..messages import (
    FunctionCall,
    LLMFunctionCalls,
    LLMMessage,
    LLMResponse,
    Message,
)
from . import AgentModel, FunctionDefinition, Model
from .utils import function_response_content, function_validation_error_content


class OpenAIModel(Model):
    def __init__(self, model_name: ChatModel, *, api_key: str | None = None, client: AsyncClient | None = None):
        if model_name not in ChatModel.__args__:
            raise ValueError(f'Invalid model name: {model_name}')
        self.model_name: ChatModel = model_name
        self.client = client or cached_async_client(api_key)

    def agent_model(self, allow_plain_message: bool, functions: list[FunctionDefinition]) -> AgentModel:
        return OpenAIAgentModel(
            self.client,
            self.model_name,
            allow_plain_message,
            [map_function_definition(f) for f in functions],
        )


@dataclass
class OpenAIAgentModel(AgentModel):
    client: AsyncClient
    model_name: ChatModel
    allow_plain_message: bool
    tools: list[ChatCompletionToolParam]

    async def request(self, messages: list[Message]) -> LLMMessage:
        response = await self.completions_create(messages)
        choice = response.choices[0]
        timestamp = datetime.fromtimestamp(response.created)
        if choice.finish_reason == 'tool_calls':
            assert choice.message.tool_calls is not None, choice
            return LLMFunctionCalls(
                role='llm-function-calls',
                timestamp=timestamp,
                calls=[
                    FunctionCall(
                        function_id=c.id,
                        function_name=c.function.name,
                        arguments=c.function.arguments,
                    )
                    for c in choice.message.tool_calls
                ],
            )
        else:
            assert choice.message.content is not None, choice
            return LLMResponse(role='llm-response', timestamp=timestamp, content=choice.message.content)

    async def completions_create(self, messages: list[Message]) -> ChatCompletion:
        # standalone function to make it easier to override
        openai_messages = [map_message(m) for m in messages]
        return await self.client.chat.completions.create(
            model=self.model_name,
            messages=openai_messages,
            n=1,
            parallel_tool_calls=True,
            tools=self.tools,
            tool_choice='auto' if self.allow_plain_message else 'required',
        )


@cache
def cached_async_client(api_key: str) -> AsyncClient:
    return AsyncClient(api_key=api_key)


def map_function_definition(f: FunctionDefinition) -> ChatCompletionToolParam:
    return {
        'type': 'function',
        'function': {
            'name': f.name,
            'description': f.description,
            'parameters': f.json_schema,
        },
    }


def map_message(message: Message) -> ChatCompletionMessageParam:
    """Just maps a `pydantic_ai.Message` to a `openai.types.ChatCompletionMessageParam`."""
    if message['role'] == 'system':
        # SystemPrompt -> ChatCompletionSystemMessageParam
        return {'role': 'system', 'content': message['content']}
    elif message['role'] == 'user':
        # UserPrompt -> ChatCompletionUserMessageParam
        return {'role': 'user', 'content': message['content']}
    elif message['role'] == 'function-response':
        # FunctionResponse -> ChatCompletionToolMessageParam
        return {
            'role': 'tool',
            'tool_call_id': message['function_id'],
            'content': function_response_content(message),
        }
    elif message['role'] == 'function-validation-error':
        # FunctionValidationError -> ChatCompletionUserMessageParam
        return {
            'role': 'tool',
            'tool_call_id': message['function_id'],
            'content': function_validation_error_content(message),
        }
    elif message['role'] == 'llm-response':
        # LLMResponse -> ChatCompletionAssistantMessageParam
        return {'role': 'assistant', 'content': message['content']}
    elif message['role'] == 'llm-function-calls':
        # LLMFunctionCalls -> ChatCompletionAssistantMessageParam
        return {
            'role': 'assistant',
            'tool_calls': [
                {
                    'id': f['function_id'],
                    'type': 'function',
                    'function': {
                        'name': f['function_name'],
                        'arguments': f['arguments'],
                    },
                }
                for f in message['calls']
            ],
        }
    elif message['role'] == 'plain-response-forbidden':
        # PlainResponseForbidden -> ChatCompletionUserMessageParam
        return {
            'role': 'user',
            'content': 'Plain text responses are not allowed, please call one of the functions instead.',
        }
    else:
        assert_never(message)
