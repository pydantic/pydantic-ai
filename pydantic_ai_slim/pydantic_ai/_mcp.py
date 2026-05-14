from __future__ import annotations

import base64
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Union

import logfire
from pydantic import TypeAdapter
from pydantic.alias_generators import to_snake
from typing_extensions import TypedDict, assert_never

from . import exceptions, messages
from ._output import types_from_output_spec
from ._run_context import AgentDepsT
from .output import OutputDataT

try:
    from mcp import types as mcp_types
    from mcp.server.lowlevel.server import Server, StructuredContent
    from mcp.types import Tool
except ImportError as _import_error:
    raise ImportError(
        'Please install the `mcp` package to use the MCP server, '
        'you can use the `mcp` optional group — `pip install "pydantic-ai-slim[mcp]"`'
    ) from _import_error

if TYPE_CHECKING:
    from .agent.abstract import AbstractAgent


def map_from_mcp_params(params: mcp_types.CreateMessageRequestParams) -> list[messages.ModelMessage]:
    """Convert from MCP create message request parameters to pydantic-ai messages."""
    pai_messages: list[messages.ModelMessage] = []
    request_parts: list[messages.ModelRequestPart] = []
    if params.systemPrompt:
        request_parts.append(messages.SystemPromptPart(content=params.systemPrompt))
    response_parts: list[messages.ModelResponsePart] = []
    for msg in params.messages:
        content = msg.content
        if msg.role == 'user':
            # if there are any response parts, add a response message wrapping them
            if response_parts:
                pai_messages.append(messages.ModelResponse(parts=response_parts))
                response_parts = []

            # TODO(Marcelo): We can reuse the `_map_tool_result_part` from the mcp module here.
            if isinstance(content, mcp_types.TextContent):
                user_part_content: str | Sequence[messages.UserContent] = content.text
            elif isinstance(content, (mcp_types.ImageContent, mcp_types.AudioContent)):
                user_part_content = [
                    messages.BinaryContent(data=base64.b64decode(content.data), media_type=content.mimeType)
                ]
            elif isinstance(content, list):
                raise NotImplementedError('list content type is not yet supported')
            elif isinstance(content, (mcp_types.ToolUseContent, mcp_types.ToolResultContent)):
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

            if isinstance(content, (mcp_types.TextContent, mcp_types.ImageContent, mcp_types.AudioContent)):
                response_parts.append(map_from_sampling_content(content))
            else:
                raise NotImplementedError(f'Unsupported assistant content type: {type(content).__name__}')

    if response_parts:
        pai_messages.append(messages.ModelResponse(parts=response_parts))
    if request_parts:
        pai_messages.append(messages.ModelRequest(parts=request_parts))
    return pai_messages


def map_from_pai_messages(pai_messages: list[messages.ModelMessage]) -> tuple[str, list[mcp_types.SamplingMessage]]:
    """Convert from pydantic-ai messages to MCP sampling messages.

    Returns:
        A tuple containing the system prompt and a list of sampling messages.
    """
    sampling_msgs: list[mcp_types.SamplingMessage] = []

    def add_msg(
        role: Literal['user', 'assistant'],
        content: mcp_types.TextContent | mcp_types.ImageContent | mcp_types.AudioContent,
    ):
        sampling_msgs.append(mcp_types.SamplingMessage(role=role, content=content))

    system_prompt: list[str] = []
    for pai_message in pai_messages:
        if isinstance(pai_message, messages.ModelRequest):
            if pai_message.instructions is not None:
                system_prompt.append(pai_message.instructions)

            for part in pai_message.parts:
                if isinstance(part, messages.SystemPromptPart):
                    system_prompt.append(part.content)
                if isinstance(part, messages.UserPromptPart):
                    if isinstance(part.content, str):
                        add_msg('user', mcp_types.TextContent(type='text', text=part.content))
                    else:
                        for chunk in part.content:
                            if isinstance(chunk, str):
                                add_msg('user', mcp_types.TextContent(type='text', text=chunk))
                            elif isinstance(chunk, messages.BinaryContent) and chunk.is_image:
                                add_msg(
                                    'user',
                                    mcp_types.ImageContent(
                                        type='image',
                                        data=chunk.base64,
                                        mimeType=chunk.media_type,
                                    ),
                                )
                            # TODO(Marcelo): Add support for audio content.
                            else:
                                raise NotImplementedError(f'Unsupported content type: {type(chunk)}')
        else:
            add_msg('assistant', map_from_model_response(pai_message))
    return ''.join(system_prompt), sampling_msgs


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


class _AgentToolArgs(TypedDict):
    prompt: str


def agent_to_mcp(
    agent: AbstractAgent[AgentDepsT, OutputDataT],
    *,
    server_name: str | None = None,
    tool_name: str | None = None,
    tool_description: str | None = None,
    deps: AgentDepsT = None,
) -> Server:
    default_name = to_snake((agent.name or 'PydanticAI Agent').replace(' ', '_'))
    server_name = server_name or default_name
    tool_name = tool_name or default_name

    return_types = types_from_output_spec(agent.output_type)
    if len(return_types) == 1:
        output_adapter: TypeAdapter[Any] = TypeAdapter(return_types[0])
    else:
        output_adapter = TypeAdapter(Union[tuple(return_types)])  # noqa: UP007

    app: Server[Any, Any] = Server(name=server_name)

    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=tool_name,
                description=tool_description,
                inputSchema={
                    'type': 'object',
                    'properties': {'prompt': {'type': 'string'}},
                    'required': ['prompt'],
                },
                outputSchema={
                    'type': 'object',
                    'properties': {'result': agent.output_json_schema()},
                    'required': ['result'],
                },
            )
        ]

    async def call_tool(name: str, args: _AgentToolArgs) -> StructuredContent:
        if name != tool_name:
            raise ValueError(f'Unknown tool: {name!r}')

        logfire.info('Calling tool {name}', name=name, args=args)

        result = await agent.run(user_prompt=args['prompt'], deps=deps)

        return {'result': output_adapter.dump_python(result.output, mode='json')}

    app.list_tools()(list_tools)
    app.call_tool()(call_tool)

    return app
