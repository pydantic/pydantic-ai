from __future__ import annotations as _annotations

import base64
from typing import TYPE_CHECKING, Annotated, Any, NotRequired

from fastapi import FastAPI, HTTPException
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAudioParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionReasoningEffort,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage
from pydantic import Field, TypeAdapter
from typing_extensions import TypedDict

from pydantic_ai import UserError, _utils
from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
)
from pydantic_ai.settings import ModelSettings

if TYPE_CHECKING:
    from pydantic_ai.agent import Agent, AgentRunResult

Number = Annotated[float, Field(ge=-2.0, le=2.0)]
PositiveNumber = Annotated[float, Field(ge=0.0, le=2.0)]


class ChatCompletionInput(TypedDict):
    messages: list[ChatCompletionMessageParam]
    model: str
    audio: NotRequired[ChatCompletionAudioParam]
    frequency_penalty: NotRequired[float | None]
    logit_bias: NotRequired[dict[str, int] | None]
    logprobs: NotRequired[bool | None]
    max_completion_tokens: NotRequired[int | None]
    parallel_tool_calls: NotRequired[bool | None]
    presence_penalty: NotRequired[Number | None]
    reasoning_effort: NotRequired[ChatCompletionReasoningEffort | None]
    seed: NotRequired[int | None]
    stop: NotRequired[str | None]
    stream: NotRequired[bool | None]
    temperature: NotRequired[PositiveNumber | None]
    # TODO(Marcelo): We need to support tools.
    tools: NotRequired[list[ChatCompletionToolParam]]
    top_logprobs: NotRequired[Annotated[int, Field(ge=0, le=20)] | None]
    top_p: NotRequired[PositiveNumber | None]
    user: NotRequired[str]
    extra_headers: NotRequired[dict[str, str]]
    extra_body: NotRequired[object]


model_settings_ta = TypeAdapter(ModelSettings)


def to_chat_completions(agent: Agent[None], path: str = '/chat/completions') -> FastAPI:
    # TODO(Marcelo): PydanticAI deps should be created in the lifespan.
    app = FastAPI()

    async def chat_completions(body: ChatCompletionInput) -> ChatCompletion:
        message_history = openai2pai(body['messages'])

        if body.get('tools'):
            raise HTTPException(status_code=400, detail='Tools are not supported yet')

        try:
            result = await agent.run(
                model=body['model'],
                message_history=message_history,
                model_settings=model_settings_ta.validate_python(body),
            )
        except UserError as e:
            raise HTTPException(status_code=400, detail=e.message)
        return pai_result_2_openai(result=result, model=body['model'])

    app.add_api_route(path, chat_completions, methods=['POST'])

    return app


def openai2pai(messages: list[ChatCompletionMessageParam]) -> list[ModelMessage]:
    """Convert OpenAI ChatCompletionMessageParam list to pydantic-ai ModelMessage format."""
    result: list[ModelMessage] = []
    current_model_request = ModelRequest(parts=[])
    current_model_response = ModelResponse(parts=[])

    for message in messages:
        if message['role'] == 'system' or message['role'] == 'developer':
            content = message['content']
            if not isinstance(content, str):
                content = '\n'.join(part['text'] for part in content)
            current_model_request.parts.append(SystemPromptPart(content=content))

        elif message['role'] == 'user':
            content = message['content']
            user_content: str | list[UserContent]
            if isinstance(content, str):
                user_content = content
            else:
                user_content = []
                for part in content:
                    if part['type'] == 'text':
                        user_content.append(part['text'])
                    elif part['type'] == 'image_url':
                        user_content.append(ImageUrl(url=part['image_url']['url']))
                    elif part['type'] == 'input_audio':
                        user_content.append(
                            BinaryContent(
                                data=base64.b64decode(part['input_audio']['data']),
                                media_type=part['input_audio']['format'],
                            )
                        )
                    elif part['type'] == 'file':
                        assert 'file' in part['file']
                        user_content.append(
                            BinaryContent(
                                data=base64.b64decode(part['file']['file_data']),
                                media_type=part['file']['file']['type'],
                            )
                        )
                    else:
                        raise ValueError(f'Unknown content type: {part["type"]}')
            current_model_request.parts.append(UserPromptPart(content=user_content))

        elif message['role'] == 'assistant':
            if current_model_request.parts:
                result.append(current_model_request)
                current_model_request = ModelRequest(parts=[])

            response_parts: list[ModelResponsePart] = []
            content = message.get('content')
            tool_calls = message.get('tool_calls')

            if content:
                if isinstance(content, str):
                    response_parts.append(TextPart(content=content))
                else:
                    content_text = '\n'.join(part['text'] for part in content if part['type'] == 'text')
                    if content_text:
                        response_parts.append(TextPart(content=content_text))

            if tool_calls:
                for tool_call in tool_calls:
                    if tool_call['type'] == 'function':
                        response_parts.append(
                            ToolCallPart(
                                tool_name=tool_call['function']['name'],
                                args=tool_call['function']['arguments'],
                                tool_call_id=tool_call['id'],
                            )
                        )

            if response_parts:
                current_model_response = ModelResponse(parts=response_parts)
                result.append(current_model_response)
                current_model_response = ModelResponse(parts=[])

        elif message['role'] == 'tool':
            tool_call_id = message['tool_call_id']
            content = message['content']
            tool_name = message.get('name', 'unknown')

            current_model_request.parts.append(
                ToolReturnPart(
                    tool_name=tool_name,
                    content=content,
                    tool_call_id=tool_call_id,
                )
            )

        elif message['role'] == 'function':
            name = message['name']
            content = message['content']

            current_model_request.parts.append(
                ToolReturnPart(
                    tool_name=name,
                    content=content,
                    tool_call_id=f'call_{name}',
                )
            )

        else:
            raise ValueError(f'Unknown role: {message["role"]}')

    if current_model_request.parts:
        result.append(current_model_request)
    if current_model_response.parts:
        result.append(current_model_response)

    return result


def pai_result_2_openai(result: AgentRunResult[Any], model: str) -> ChatCompletion:
    """Convert a PydanticAI AgentRunResult to OpenAI ChatCompletion format."""
    content = ''
    if result.output:
        content = str(result.output)
    elif result.all_messages():
        # Get the last message content
        last_message = result.all_messages()[-1]
        if isinstance(last_message, ModelResponse):
            for part in last_message.parts:
                if isinstance(part, TextPart):
                    content += part.content

    return ChatCompletion(
        id=f'chatcmpl-{_utils.now_utc().isoformat()}',
        object='chat.completion',
        created=int(_utils.now_utc().timestamp()),
        model=model,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role='assistant',
                    content=content,
                ),
                finish_reason='stop',
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        ),
    )
