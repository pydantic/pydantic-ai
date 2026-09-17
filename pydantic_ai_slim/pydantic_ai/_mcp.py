from __future__ import annotations

import base64
from collections.abc import Iterator, Sequence
from typing import Literal

from typing_extensions import assert_never

from . import exceptions, messages
from ._mcp_compat import mcp_field, mcp_field_value, mcp_optional_field

try:
    # `mcp.types` serves either SDK generation: v2 keeps it as an exact re-export of `mcp_types`.
    from mcp import types as mcp_types
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `mcp` package to use the MCP integrations, '
        'you can use the `mcp` optional group — `pip install "pydantic-ai-slim[mcp]"`'
    ) from _import_error


def map_from_mcp_params(params: mcp_types.CreateMessageRequestParams) -> list[messages.ModelMessage]:
    """Convert from MCP create message request parameters to pydantic-ai messages."""
    pai_messages: list[messages.ModelMessage] = []
    request_parts: list[messages.ModelRequestPart] = []
    if system_prompt := mcp_optional_field(params, 'system_prompt', str):
        request_parts.append(messages.SystemPromptPart(content=system_prompt))
    response_parts: list[messages.ModelResponsePart] = []
    tool_names: dict[str, str] = {}
    for msg in params.messages:
        contents = msg.content if isinstance(msg.content, list) else [msg.content]
        if msg.role == 'user':
            # if there are any response parts, add a response message wrapping them
            if response_parts:
                pai_messages.append(messages.ModelResponse(parts=response_parts))
                response_parts = []

            for content in contents:
                if isinstance(content, mcp_types.TextContent):
                    user_part_content: str | Sequence[messages.UserContent] = content.text
                elif isinstance(content, (mcp_types.ImageContent, mcp_types.AudioContent)):
                    user_part_content = [
                        messages.BinaryContent(
                            data=base64.b64decode(content.data),
                            media_type=mcp_field(content, 'mime_type', str),
                        )
                    ]
                elif isinstance(content, mcp_types.ToolResultContent):
                    request_parts.append(_map_sampling_tool_result(content, tool_names))
                    continue
                elif isinstance(content, mcp_types.ToolUseContent):
                    raise NotImplementedError(f'{type(content).__name__} cannot be used as user content')
                else:
                    assert_never(content)

                request_parts.append(messages.UserPromptPart(content=user_part_content))
        else:
            # role is assistant
            # if there are any request parts, add a request message wrapping them
            if request_parts:
                pai_messages.append(messages.ModelRequest(parts=request_parts))
                request_parts = []

            for content in contents:
                response_parts.append(_map_sampling_response_part(content, tool_names))

    if response_parts:
        pai_messages.append(messages.ModelResponse(parts=response_parts))
    if request_parts:
        pai_messages.append(messages.ModelRequest(parts=request_parts))
    return pai_messages


def _map_sampling_response_part(
    content: mcp_types.SamplingMessageContentBlock, tool_names: dict[str, str]
) -> messages.TextPart | messages.ToolCallPart:
    if isinstance(content, (mcp_types.TextContent, mcp_types.ImageContent, mcp_types.AudioContent)):
        return map_from_sampling_content(content)
    elif isinstance(content, mcp_types.ToolUseContent):
        tool_names[content.id] = content.name
        return messages.ToolCallPart(tool_name=content.name, args=content.input, tool_call_id=content.id)
    elif isinstance(content, mcp_types.ToolResultContent):
        raise NotImplementedError(f'Unsupported assistant content type: {type(content).__name__}')
    else:
        assert_never(content)


def _map_sampling_tool_result(
    content: mcp_types.ToolResultContent, tool_names: dict[str, str]
) -> messages.ToolReturnPart:
    # Share tool-result conversion without importing the optional FastMCP client for outbound sampling.
    from .mcp import _map_mcp_tool_results  # pyright: ignore[reportPrivateUsage]

    tool_call_id = mcp_field(content, 'tool_use_id', str)
    if tool_call_id not in tool_names:
        raise exceptions.UnexpectedModelBehavior(f'MCP tool result has no matching tool call: {tool_call_id!r}')
    structured_content = mcp_field_value(content, 'structured_content')
    return messages.ToolReturnPart(
        tool_name=tool_names[tool_call_id],
        tool_call_id=tool_call_id,
        content=structured_content if structured_content is not None else _map_mcp_tool_results(content.content),
        outcome='failed' if mcp_optional_field(content, 'is_error', bool) else 'success',
    )


def map_from_pai_messages(pai_messages: list[messages.ModelMessage]) -> tuple[str, list[mcp_types.SamplingMessage]]:
    """Convert from pydantic-ai messages to MCP sampling messages.

    Returns:
        A tuple containing the system prompt and a list of sampling messages.
    """
    sampling_msgs: list[mcp_types.SamplingMessage] = []

    def add_msg(
        role: Literal['user', 'assistant'],
        content: mcp_types.SamplingMessageContentBlock | list[mcp_types.SamplingMessageContentBlock],
    ):
        sampling_msgs.append(mcp_types.SamplingMessage(role=role, content=content))

    system_prompt: list[str] = []
    for pai_message in pai_messages:
        if isinstance(pai_message, messages.ModelRequest):
            request_start = len(sampling_msgs)
            tool_results: list[mcp_types.SamplingMessageContentBlock] = []
            if pai_message.instructions is not None:
                system_prompt.append(pai_message.instructions)

            for part in pai_message.parts:
                if isinstance(part, messages.SystemPromptPart):
                    system_prompt.append(part.content)
                elif isinstance(part, messages.UserPromptPart):
                    for content in _map_user_prompt(part):
                        add_msg('user', content)
                elif isinstance(part, messages.ToolReturnPart):
                    if part.files:
                        raise NotImplementedError('Multimodal tool results in MCP sampling are not yet supported')
                    tool_results.append(
                        mcp_types.ToolResultContent(
                            type='tool_result',
                            toolUseId=part.tool_call_id,
                            content=[
                                mcp_types.TextContent(type='text', text=part.model_response_str(wrap_if_error=False))
                            ],
                            isError=part.outcome == 'failed',
                        )
                    )
                elif isinstance(part, messages.RetryPromptPart):
                    content = mcp_types.TextContent(type='text', text=part.model_response())
                    if part.tool_name is None:
                        add_msg('user', content)
                    else:
                        tool_results.append(
                            mcp_types.ToolResultContent(
                                type='tool_result', toolUseId=part.tool_call_id, content=[content], isError=True
                            )
                        )
                elif isinstance(part, (messages.SpeechPart, messages.ToolAvailabilityDeltaPart)):
                    # These parts are not currently sent to MCP sampling.
                    continue
                else:
                    assert_never(part)
            if tool_results:
                # MCP requires all results from a turn in one message, before ordinary user content.
                sampling_msgs.insert(request_start, mcp_types.SamplingMessage(role='user', content=tool_results))
        else:
            add_msg('assistant', _map_response_history(pai_message))
    return ''.join(system_prompt), sampling_msgs


def _map_user_prompt(part: messages.UserPromptPart) -> Iterator[mcp_types.TextContent | mcp_types.ImageContent]:
    chunks = [part.content] if isinstance(part.content, str) else part.content
    for chunk in chunks:
        if isinstance(chunk, str):
            yield mcp_types.TextContent(type='text', text=chunk)
        elif isinstance(chunk, messages.BinaryContent) and chunk.is_image:
            yield mcp_types.ImageContent(type='image', data=chunk.base64, mimeType=chunk.media_type)
        # TODO(Marcelo): Add support for audio content.
        else:
            raise NotImplementedError(f'Unsupported content type: {type(chunk)}')


def _map_response_history(
    response: messages.ModelResponse,
) -> mcp_types.TextContent | list[mcp_types.SamplingMessageContentBlock]:
    if not any(isinstance(part, messages.ToolCallPart) for part in response.parts):
        return map_from_model_response(response)

    content: list[mcp_types.SamplingMessageContentBlock] = []
    for part in response.parts:
        if isinstance(part, messages.ToolCallPart):
            content.append(
                mcp_types.ToolUseContent(
                    type='tool_use', id=part.tool_call_id, name=part.tool_name, input=part.args_as_dict()
                )
            )
        elif isinstance(part, messages.ThinkingPart):
            continue
        elif isinstance(part, messages.TextPart):
            content.append(mcp_types.TextContent(type='text', text=part.content))
        elif isinstance(
            part,
            (
                messages.NativeToolCallPart,
                messages.NativeToolReturnPart,
                messages.FilePart,
                messages.SpeechPart,
                messages.CompactionPart,
            ),
        ):
            raise exceptions.UnexpectedModelBehavior(
                f'Unexpected part type: {type(part).__name__}, expected TextPart or ToolCallPart'
            )
        else:
            assert_never(part)
    return content


def map_from_model_response(model_response: messages.ModelResponse) -> mcp_types.TextContent:
    """Convert from a model response to MCP text content."""
    text_parts: list[str] = []
    for part in model_response.parts:
        if isinstance(part, messages.TextPart):
            text_parts.append(part.content)
        elif isinstance(part, messages.ThinkingPart):
            continue
        else:
            raise exceptions.UnexpectedModelBehavior(f'Unexpected part type: {type(part).__name__}, expected TextPart')
    return mcp_types.TextContent(type='text', text=''.join(text_parts))


def map_from_sampling_content(
    content: mcp_types.TextContent | mcp_types.ImageContent | mcp_types.AudioContent,
) -> messages.TextPart:
    """Convert from sampling content to a pydantic-ai text part."""
    if isinstance(content, mcp_types.TextContent):  # pragma: no branch
        return messages.TextPart(content=content.text)
    else:
        # TODO: Add support for Image/Audio using FilePart.
        raise NotImplementedError('Image and Audio responses in sampling are not yet supported')
