import json
from datetime import datetime

from openai import AsyncClient
from openai.types import ChatModel
from openai.types.chat import ChatCompletionMessageParam

from ..result import (
    FunctionCall,
    FunctionResponse,
    FunctionValidationError,
    LLMFunctionCalls,
    LLMMessage,
    LLMResponse,
    Message,
)
from . import Model


class OpenAIModel(Model):
    def __init__(self, model_name: ChatModel, *, api_key: str | None = None, client: AsyncClient | None = None):
        if model_name not in ChatModel.__args__:
            raise ValueError(f'Invalid model name: {model_name}')
        self.model_name = model_name
        self.client = client or AsyncClient(api_key=api_key)

    async def request(self, messages: list[Message]) -> LLMMessage:
        openai_messages = [map_message(m) for m in messages]
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=openai_messages,
            n=1,
            parallel_tool_calls=True,
        )
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
    else:
        assert message['role'] == 'llm-function-calls', message
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


def function_response_content(m: FunctionResponse) -> str:
    return f'Response from calling {m["function_name"]}: {m["response"]}'


def function_validation_error_content(m: FunctionValidationError) -> str:
    errors_json = json.dumps(m['errors'], indent=2)
    return f'Validation error calling {m["function_name"]}:\n{errors_json}\n\nPlease fix the errors and try again.'
