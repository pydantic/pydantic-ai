import base64
import json
from collections.abc import Iterable
from typing import Any, cast

try:
    from openai.types.chat import (
        ChatCompletion,
        ChatCompletionMessage,
        ChatCompletionMessageParam,
    )
    from openai.types.chat.chat_completion import Choice
    from openai.types.completion_usage import CompletionUsage
    from openai.types.responses import (
        Response,
        ResponseInputParam,
        ResponseOutputMessage,
        ResponseOutputText,
        ResponseUsage,
    )
    from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to enable the fastapi openai compatible endpoint, '
        'you can use the `openai` and `fastapi` optional group — `pip install "pydantic-ai-slim[openai,fastapi]"`'
    ) from _import_error

from pydantic import TypeAdapter

from pydantic_ai import _utils
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    DocumentUrl,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.settings import ModelSettings

model_settings_ta = TypeAdapter(ModelSettings)


def _flush_request_if_any(
    result: list[ModelMessage],
    current_request_parts: list[ModelRequestPart],
) -> None:
    """Append a ModelRequest to result if there are collected request parts and clear the collector."""
    if current_request_parts:
        result.append(ModelRequest(parts=current_request_parts.copy()))
        current_request_parts.clear()


def _flush_response_if_any(
    result: list[ModelMessage],
    current_response_parts: list[ModelResponsePart],
) -> None:
    """Append a ModelResponse to result if there are collected response parts and clear the collector."""
    if current_response_parts:
        result.append(ModelResponse(parts=current_response_parts.copy()))
        current_response_parts.clear()


def _extract_text_from_content(
    content: str | list[dict[str, Any]] | Iterable[Any],
) -> str:
    """Extract plain text from content (for system/developer messages).

    The Responses SDK accepts a plain string, a list of dicts, or any iterable
    of content items. Accept strings, iterables of dict-like
    objects, or iterables of primitive/text items. Non-dict items will be coerced
    to string where reasonable.
    """
    if isinstance(content, str):
        return content

    text_parts: list[str] = []
    # `content` may be any iterable (list, generator, etc.)
    for item in content:
        # strings are direct text pieces
        if isinstance(item, str):
            text_parts.append(item)
            continue

        # If item is a dict-like object with a type/text schema, prefer those fields
        if isinstance(item, dict):
            item: dict[str, Any]
            itype = item.get('type')
            if itype in ('input_text', 'output_text'):
                text = item.get('text', '')
                if text:
                    text_parts.append(text)
                continue
            # fallback to common 'text' key if present
            if 'text' in item:
                maybe_text = item.get('text')
                if isinstance(maybe_text, str):
                    text_parts.append(maybe_text)
                    continue
            # some items nest content under 'content'
            nested = item.get('content')
            if isinstance(nested, str):
                text_parts.append(nested)
                continue
            # if nothing else, try to stringify the dict (best-effort)
            try:
                text_parts.append(str(item))
            except Exception:
                continue

    # Filter out empty strings and join
    return '\n'.join(p for p in text_parts if p)


def _convert_content_to_user_content(  # noqa: C901
    content: str | Iterable[Any],
) -> str | list[Any]:  # UserContent types
    """Convert ResponseInputMessageContentListParam to pydantic-ai UserContent.

    Be permissive about the incoming shape: the Responses SDK may provide a
    plain string or an iterable (list, generator, or SDK iterable) of content
    items. Handle strings, dict-like items, and fall back to best-effort
    stringification for unknown types.
    """
    if isinstance(content, str):
        return content

    user_content: list[Any] = []
    for item in content:
        if isinstance(item, str):
            user_content.append(item)
            continue

        if not isinstance(item, dict):
            try:
                user_content.append(str(item))
            except Exception:
                continue
            continue

        item: dict[str, Any]
        item_type = item.get('type')

        if item_type == 'input_text':
            text = item.get('text')
            if isinstance(text, str):
                user_content.append(text)

        elif item_type == 'input_image':
            if item.get('image_url'):
                user_content.append(ImageUrl(url=item['image_url']))

        elif item_type == 'input_file':
            if item.get('file_data'):
                file_data = base64.b64decode(item['file_data'])
                media_type = _guess_media_type_from_filename(item.get('filename', ''))
                user_content.append(
                    BinaryContent(data=file_data, media_type=media_type),
                )
            elif item.get('file_url'):
                media_type = _guess_media_type_from_filename(item.get('filename', ''))
                if media_type.startswith('image/'):
                    user_content.append(ImageUrl(url=item['file_url']))
                elif media_type.startswith('audio/'):
                    user_content.append(AudioUrl(url=item['file_url']))
                elif media_type.startswith('video/'):
                    user_content.append(VideoUrl(url=item['file_url']))
                else:
                    user_content.append(DocumentUrl(url=item['file_url']))

        elif item_type == 'input_audio':
            input_audio = item.get('input_audio', {})
            if isinstance(input_audio, dict) and 'data' in input_audio:
                input_audio: dict[str, Any]
                audio_data = base64.b64decode(input_audio['data'])
                media_type = f'audio/{input_audio.get("format", "wav")}'
                user_content.append(
                    BinaryContent(data=audio_data, media_type=media_type),
                )

        else:
            if 'text' in item and isinstance(item.get('text'), str):
                user_content.append(item.get('text'))
            elif 'content' in item and isinstance(item.get('content'), str):
                user_content.append(item.get('content'))
            else:
                try:
                    user_content.append(str(item))
                except Exception:
                    continue

    return user_content if user_content else ''


def _convert_function_output_to_content(
    output_list: list[dict[str, Any]],
) -> Any:
    """Convert ResponseFunctionCallOutputItemListParam to content."""
    # For simplicity, extract text content or serialize as JSON
    text_parts: list[Any] = []
    for item in output_list:
        item: dict[str, Any]
        if item.get('type') == 'output_text':
            text_parts.append(item['text'])
        else:
            # For non-text items, serialize as JSON
            text_parts.append(json.dumps(item))

    return '\n'.join(text_parts) if text_parts else json.dumps(output_list)


def _guess_media_type_from_filename(filename: str) -> str:
    """Guess media type from filename extension."""
    ext = filename.lower().split('.')[-1] if '.' in filename else ''

    # Image formats
    if ext in ('jpg', 'jpeg'):
        return 'image/jpeg'
    elif ext == 'png':
        return 'image/png'
    elif ext == 'gif':
        return 'image/gif'
    elif ext == 'webp':
        return 'image/webp'

    # Audio formats
    elif ext in ('mp3', 'mpeg'):
        return 'audio/mpeg'
    elif ext == 'wav':
        return 'audio/wav'
    elif ext == 'ogg':
        return 'audio/ogg'

    # Video formats
    elif ext == 'mp4':
        return 'video/mp4'
    elif ext == 'webm':
        return 'video/webm'

    # Document formats
    elif ext == 'pdf':
        return 'application/pdf'
    elif ext == 'txt':
        return 'text/plain'

    # Default
    return 'application/octet-stream'


def openai_responses_input_to_pai(  # noqa: C901
    items: ResponseInputParam | str,
) -> list[ModelMessage]:
    """Convert OpenAI Responses API ResponseInputParam to pydantic-ai ModelMessage format."""
    result: list[ModelMessage] = []
    current_request_parts: list[ModelRequestPart] = []
    current_response_parts: list[ModelResponsePart] = []

    # Track tool call IDs to tool names for matching outputs to calls
    tool_call_map: dict[str, str] = {}

    if isinstance(items, str):
        current_request_parts = [UserPromptPart(content=items)]
    else:
        for item in items:
            item_type = item.get(
                'type',
                'message',
            )  # Get item type - default to "message" if not specified (for EasyInputMessageParam)

            if item_type == 'message':
                if 'role' not in item:
                    continue

                role = item['role']
                content = item.get('content')

                if role in ('system', 'developer'):
                    text_content = _extract_text_from_content(content)
                    current_request_parts.append(SystemPromptPart(content=text_content))

                elif role == 'user':
                    user_content = _convert_content_to_user_content(content)
                    current_request_parts.append(UserPromptPart(content=user_content))

                elif role == 'assistant':
                    _flush_request_if_any(result, current_request_parts)
                    if isinstance(content, str):
                        current_response_parts.append(TextPart(content=content))
                    elif isinstance(content, list):
                        for content_item in content:
                            if content_item.get('type') == 'output_text':
                                current_response_parts.append(
                                    TextPart(content=content_item['text']),  # type:ignore
                                )
                            elif content_item.get('type') == 'refusal':
                                current_response_parts.append(
                                    TextPart(content=f'[REFUSAL] {content_item["refusal"]}'),  # type:ignore
                                )
                    # Flush the response immediately so the request/response ordering
                    _flush_response_if_any(result, current_response_parts)

            elif item_type == 'function_call':
                if 'name' not in item or 'call_id' not in item or 'arguments' not in item:
                    continue

                _flush_request_if_any(result, current_request_parts)

                tool_name = item['name']
                call_id = item['call_id']
                tool_call_map[call_id] = tool_name

                current_response_parts.append(
                    ToolCallPart(
                        tool_name=tool_name,
                        args=item['arguments'],
                        tool_call_id=call_id,
                    ),
                )
                # Flush response immediately so the ToolCall is emitted as a response
                # before any subsequent ToolReturnParts are added (preserves ordering).
                _flush_response_if_any(result, current_response_parts)

            elif item_type == 'function_call_output':
                if 'call_id' not in item or 'output' not in item:
                    continue

                call_id = item['call_id']
                tool_name = tool_call_map.get(call_id, 'unknown_function')
                output = item['output']

                if isinstance(output, str):
                    content = output
                else:
                    content = _convert_function_output_to_content(output)  # type: ignore

                current_request_parts.append(
                    ToolReturnPart(
                        tool_name=tool_name,
                        content=content,
                        tool_call_id=call_id,
                    ),
                )

            elif item_type == 'file_search_call':
                if 'id' not in item or 'queries' not in item:
                    continue

                _flush_request_if_any(result, current_request_parts)

                current_response_parts.append(
                    BuiltinToolCallPart(
                        tool_name='file_search',
                        args=json.dumps({'queries': item['queries']}),
                        tool_call_id=item['id'],
                        provider_name='openai',
                    ),
                )

                # If results are present, add as tool return
                if item.get('results'):
                    _flush_response_if_any(result, current_response_parts)

                    current_response_parts.append(
                        BuiltinToolReturnPart(
                            tool_name='file_search',
                            content=item.get('results'),
                            tool_call_id=item['id'],
                            provider_name='openai',
                        ),
                    )

            elif item_type == 'computer_call':
                if 'call_id' not in item or 'action' not in item or 'id' not in item:
                    continue

                _flush_request_if_any(result, current_request_parts)

                call_id = item['call_id']
                tool_call_map[call_id] = 'computer_use'

                current_response_parts.append(
                    BuiltinToolCallPart(
                        tool_name='computer_use',
                        args=json.dumps(item['action']),
                        tool_call_id=call_id,
                        provider_name='openai',
                    ),
                )

            elif item_type == 'computer_call_output':
                if 'call_id' not in item or 'output' not in item:
                    continue

                call_id = item['call_id']

                current_response_parts.append(
                    BuiltinToolReturnPart(
                        tool_name='computer_use',
                        content=item['output'],
                        tool_call_id=call_id,
                        provider_name='openai',
                    ),
                )

            elif item_type == 'web_search_call':
                if 'id' not in item or 'action' not in item:
                    continue

                _flush_request_if_any(result, current_request_parts)

                current_response_parts.append(
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args=json.dumps(item['action']),
                        tool_call_id=item['id'],
                        provider_name='openai',
                    ),
                )

            elif item_type == 'reasoning':
                if 'id' not in item or 'summary' not in item:
                    continue

                _flush_request_if_any(result, current_request_parts)

                summary = item.get('summary') or []
                content = item.get('content') or []
                summary_texts = [s['text'] for s in summary if isinstance(s, dict) and 'text' in s]
                content_texts = [c['text'] for c in content if isinstance(c, dict) and 'text' in c]
                thinking_text = '\n'.join(summary_texts)
                if content_texts:
                    thinking_text = (thinking_text + '\n\n' if thinking_text else '') + '\n'.join(
                        content_texts,
                    )

                current_response_parts.append(
                    ThinkingPart(
                        content=thinking_text,
                        signature=item.get('encrypted_content'),
                        provider_name='openai',
                    ),
                )

            elif item_type == 'image_generation_call':
                if 'id' not in item:
                    continue

                _flush_request_if_any(result, current_request_parts)

                current_response_parts.append(
                    BuiltinToolCallPart(
                        tool_name='image_generation',
                        args=None,
                        tool_call_id=cast(str, item['id']),
                        provider_name='openai',
                    ),
                )

                if item.get('result'):
                    _flush_response_if_any(result, current_response_parts)

                    current_response_parts.append(
                        BuiltinToolReturnPart(
                            tool_name='image_generation',
                            content=item['result'],  # type: ignore
                            tool_call_id=cast(str, item['id']),
                            provider_name='openai',
                        ),
                    )

            elif item_type == 'code_interpreter_call':
                if 'id' not in item or 'container_id' not in item:
                    continue

                _flush_request_if_any(result, current_request_parts)

                args_dict = {'code': item.get('code'), 'container_id': item['container_id']}
                current_response_parts.append(
                    BuiltinToolCallPart(
                        tool_name='code_interpreter',
                        args=json.dumps(args_dict),
                        tool_call_id=item['id'],
                        provider_name='openai',
                    ),
                )

                if item.get('outputs'):
                    _flush_response_if_any(result, current_response_parts)

                    current_response_parts.append(
                        BuiltinToolReturnPart(
                            tool_name='code_interpreter',
                            content=item['outputs'],
                            tool_call_id=item['id'],
                            provider_name='openai',
                        ),
                    )

            elif item_type == 'local_shell_call':
                if 'call_id' not in item or 'action' not in item or 'id' not in item:
                    continue

                _flush_request_if_any(result, current_request_parts)

                call_id = item['call_id']
                tool_call_map[call_id] = 'local_shell'

                current_response_parts.append(
                    BuiltinToolCallPart(
                        tool_name='local_shell',
                        args=json.dumps(item['action']),
                        tool_call_id=call_id,
                        provider_name='local',
                    ),
                )

            elif item_type == 'local_shell_call_output':
                if 'id' not in item or 'output' not in item:
                    continue

                current_response_parts.append(
                    BuiltinToolReturnPart(
                        tool_name='local_shell',
                        content=item['output'],
                        tool_call_id=cast(str, item['id']),
                        provider_name='local',
                    ),
                )

            elif item_type == 'mcp_list_tools':
                if 'id' not in item or 'tools' not in item:
                    continue

                current_request_parts.append(
                    ToolReturnPart(
                        tool_name='mcp_list_tools',
                        content=item['tools'],
                        tool_call_id=item['id'],
                    ),
                )

            elif item_type == 'mcp_approval_request':
                if 'id' not in item or 'name' not in item or 'arguments' not in item:
                    continue

                _flush_request_if_any(result, current_request_parts)

                current_response_parts.append(
                    ToolCallPart(
                        tool_name=item['name'],
                        args=item['arguments'],
                        tool_call_id=item['id'],
                    ),
                )

            elif item_type == 'mcp_approval_response':
                if 'approval_request_id' not in item or 'approve' not in item:
                    continue

                approval_data = {
                    'approve': item['approve'],
                    'reason': item.get('reason'),
                }
                current_request_parts.append(
                    ToolReturnPart(
                        tool_name='mcp_approval',
                        content=json.dumps(approval_data),
                        tool_call_id=item['approval_request_id'],
                    ),
                )

            elif item_type == 'mcp_call':
                if 'id' not in item or 'name' not in item or 'arguments' not in item:
                    continue

                _flush_request_if_any(result, current_request_parts)

                call_id = item['id']
                tool_call_map[call_id] = item['name']

                current_response_parts.append(
                    ToolCallPart(
                        tool_name=item['name'],
                        args=item['arguments'],
                        tool_call_id=call_id,
                    ),
                )

                if item.get('output') or item.get('error'):
                    _flush_response_if_any(result, current_response_parts)

                    content = item.get('output') or f'Error: {item.get("error")}'
                    current_request_parts.append(
                        ToolReturnPart(
                            tool_name=item['name'],
                            content=content,
                            tool_call_id=call_id,
                        ),
                    )

            elif item_type == 'custom_tool_call':
                if 'call_id' not in item or 'name' not in item or 'input' not in item:
                    continue

                _flush_request_if_any(result, current_request_parts)

                call_id = item['call_id']
                tool_call_map[call_id] = item['name']

                current_response_parts.append(
                    ToolCallPart(
                        tool_name=item['name'],
                        args=item['input'],
                        tool_call_id=call_id,
                    ),
                )

            elif item_type == 'custom_tool_call_output':
                if 'call_id' not in item or 'output' not in item:
                    continue

                call_id = item['call_id']
                tool_name = tool_call_map.get(call_id, 'unknown_custom_tool')
                output = item['output']

                if isinstance(output, str):
                    content = output
                else:
                    content = _convert_function_output_to_content(output)  # type:ignore

                current_request_parts.append(
                    ToolReturnPart(
                        tool_name=tool_name,
                        content=content,
                        tool_call_id=call_id,
                    ),
                )

            elif item_type == 'item_reference':
                continue

    # Flush remaining messages
    _flush_request_if_any(result, current_request_parts)
    _flush_response_if_any(result, current_response_parts)

    return result


def openai_chat_completions_2pai(  # noqa: C901
    messages: str | list[ChatCompletionMessageParam],
) -> list[ModelMessage]:
    """Convert OpenAI ChatCompletionMessageParam list to pydantic-ai ModelMessage format."""
    result: list[ModelMessage] = []
    current_request_parts: list[ModelRequestPart] = []
    current_response_parts: list[ModelResponsePart] = []

    if isinstance(messages, str):
        current_request_parts = [UserPromptPart(content=messages)]
    else:
        for message in messages:
            if 'role' in message:
                if message.get('role') == 'system' or message.get('role') == 'developer':
                    content = message['content']  # type: ignore
                    if not isinstance(content, str):
                        content = '\n'.join(part['text'] for part in content)  # type: ignore
                    current_request_parts.append(SystemPromptPart(content=content))

                elif message.get('role') == 'user':
                    content = message['content']  # type: ignore
                    user_content: str | list[UserContent]
                    if isinstance(content, str):
                        user_content = content
                    else:
                        user_content = []
                        if content is not None:
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
                                        ),
                                    )
                                elif part['type'] == 'file':
                                    assert 'file' in part['file']
                                    user_content.append(
                                        BinaryContent(
                                            data=base64.b64decode(part['file']['file_data']),
                                            media_type=part['file']['file']['type'],
                                        ),
                                    )
                                else:
                                    raise ValueError(f'Unknown content type: {part["type"]}')
                    current_request_parts.append(UserPromptPart(content=user_content))

                elif message['role'] == 'assistant':
                    if current_request_parts:
                        result.append(ModelRequest(parts=current_request_parts))
                        current_request_parts = []

                    current_response_parts = []
                    content = message.get('content')
                    tool_calls = message.get('tool_calls')

                    if content:
                        if isinstance(content, str):
                            current_response_parts.append(TextPart(content=content))
                        else:
                            content_text = '\n'.join(part['text'] for part in content if part['type'] == 'text')
                            if content_text:
                                current_response_parts.append(TextPart(content=content_text))

                    if tool_calls:
                        for tool_call in tool_calls:
                            if tool_call['type'] == 'function' and 'function' in tool_call:
                                current_response_parts.append(
                                    ToolCallPart(
                                        tool_name=tool_call['function']['name'],
                                        args=tool_call['function']['arguments'],
                                        tool_call_id=tool_call['id'],
                                    ),
                                )
                            else:
                                raise NotImplementedError(
                                    'ChatCompletionMessageCustomToolCallParam translator not implemented',
                                )
                    if current_response_parts:
                        result.append(ModelResponse(parts=current_response_parts))
                        current_response_parts = []

                elif message['role'] == 'tool':
                    tool_call_id = message['tool_call_id']
                    content = message['content']
                    tool_name = message.get('name', 'unknown')

                    current_request_parts.append(
                        ToolReturnPart(
                            tool_name=tool_name,
                            content=content,
                            tool_call_id=tool_call_id,
                        ),
                    )

                elif message['role'] == 'function':
                    name = message['name']
                    content = message['content']

                    current_request_parts.append(
                        ToolReturnPart(
                            tool_name=name,
                            content=content,
                            tool_call_id=f'call_{name}',
                        ),
                    )

                else:
                    raise ValueError(f'Unknown role: {message["role"]}')
            else:
                raise NotImplementedError('ComputerCallOutput translator not implemented.')

    if current_request_parts:
        result.append(ModelRequest(parts=current_request_parts))
    if current_response_parts:
        result.append(ModelResponse(parts=current_response_parts))

    return result


def pai_result_to_openai_completions(result: AgentRunResult[Any], model: str) -> ChatCompletion:
    """Convert a PydanticAI AgentRunResult to OpenAI ChatCompletion format."""
    content = str(result.output)
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
            ),
        ],
        usage=CompletionUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        ),
    )


def pai_result_to_openai_responses(result: AgentRunResult[Any], model: str) -> Response:
    """Convert a PydanticAI AgentRunResult to OpenAI Responses format."""
    content = str(result.output)

    all_msgs = result.all_messages()
    message = all_msgs[-1]
    prov_id = getattr(message, 'provider_response_id', None) or f'resp_{_utils.now_utc().isoformat()}'
    # message.timestamp might be None; fall back to now if needed
    timestamp_obj = getattr(message, 'timestamp', None) or _utils.now_utc()
    created_at = timestamp_obj.timestamp()

    msg_id = prov_id.replace('resp_', 'msg_') if isinstance(prov_id, str) else f'msg_{_utils.now_utc().isoformat()}'

    return Response(
        id=prov_id,
        object='response',
        created_at=created_at,
        model=model,
        parallel_tool_calls=False,
        tools=[],
        tool_choice='auto',
        output=[
            ResponseOutputMessage(
                id=msg_id,
                status='completed',
                role='assistant',
                type='message',
                content=[
                    ResponseOutputText(text=content, annotations=[], type='output_text'),
                ],
            ),
        ],
        usage=ResponseUsage(
            input_tokens=0,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens=0,
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
            total_tokens=0,
        ),
    )
