"""Native tools: their request params and the output items they return."""

from __future__ import annotations as _annotations

import base64
import json
from collections.abc import Mapping
from typing import Any, Literal, assert_never, cast, get_args

from openai.types import responses

from ...exceptions import UserError
from ...messages import BinaryImage, FilePart, NativeToolCallPart, NativeToolReturnPart
from ...native_tools import (
    CodeExecutionTool,
    FileSearchTool,
    ImageAspectRatio,
    ImageGenerationTool,
    MCPServerTool,
    WebSearchTool,
)

MCP_SERVER_TOOL_CONNECTOR_URI_SCHEME: Literal['x-openai-connector'] = 'x-openai-connector'
"""
Prefix for OpenAI connector IDs. OpenAI supports either a URL or a connector ID when passing MCP configuration to a model,
by using that prefix like `x-openai-connector:<connector-id>` in a URL, you can pass a connector ID to a model.
"""


_OPENAI_ASPECT_RATIO_TO_SIZE: dict[ImageAspectRatio, Literal['1024x1024', '1024x1536', '1536x1024']] = {
    '1:1': '1024x1024',
    '2:3': '1024x1536',
    '3:2': '1536x1024',
}


_OPENAI_IMAGE_SIZE = Literal['auto', '1024x1024', '1024x1536', '1536x1024']


_OPENAI_IMAGE_SIZES: tuple[_OPENAI_IMAGE_SIZE, ...] = get_args(_OPENAI_IMAGE_SIZE)


def resolve_openai_image_generation_size(
    tool: ImageGenerationTool,
) -> _OPENAI_IMAGE_SIZE:
    """Map `ImageGenerationTool.aspect_ratio` to an OpenAI size string when provided."""
    aspect_ratio = tool.aspect_ratio
    if aspect_ratio is None:
        if tool.size is None:
            return 'auto'  # default
        if tool.size not in _OPENAI_IMAGE_SIZES:
            raise UserError(
                f'OpenAI image generation only supports `size` values: {_OPENAI_IMAGE_SIZES}. '
                f'Got: {tool.size}. Omit `size` to use the default (auto).'
            )
        return tool.size

    mapped_size = _OPENAI_ASPECT_RATIO_TO_SIZE.get(aspect_ratio)
    if mapped_size is None:
        supported = ', '.join(_OPENAI_ASPECT_RATIO_TO_SIZE)
        raise UserError(
            f'OpenAI image generation only supports `aspect_ratio` values: {supported}. Specify one of those values or omit `aspect_ratio`.'
        )
    # When aspect_ratio is set, size must be None, 'auto', or match the mapped size
    if tool.size not in (None, 'auto', mapped_size):
        raise UserError(
            '`ImageGenerationTool` cannot combine `aspect_ratio` with a conflicting `size` when using OpenAI.'
        )

    return mapped_size


def map_openai_image_generation_tool(tool: ImageGenerationTool) -> responses.tool_param.ImageGeneration:
    provider_settings = (tool.provider_settings or {}).get('openai', {})
    size = resolve_openai_image_generation_size(tool)
    output_compression = tool.output_compression if tool.output_compression is not None else 100
    image_generation_tool = responses.tool_param.ImageGeneration(
        type='image_generation',
        action=provider_settings.get('action', tool.action),
        background=provider_settings.get('background', tool.background),
        moderation=provider_settings.get('moderation', tool.moderation),
        output_compression=output_compression,
        output_format=tool.output_format or 'png',
        partial_images=provider_settings.get('partial_images', tool.partial_images),
        quality=provider_settings.get('quality', tool.quality),
        size=size,
    )
    if (model := provider_settings.get('model', tool.model)) is not None:
        image_generation_tool['model'] = model
    if (input_fidelity := provider_settings.get('input_fidelity', tool.input_fidelity)) is not None:
        image_generation_tool['input_fidelity'] = input_fidelity
    return image_generation_tool


def map_code_interpreter_tool_call(
    item: responses.ResponseCodeInterpreterToolCall, provider_name: str
) -> tuple[NativeToolCallPart, NativeToolReturnPart, list[FilePart]]:
    result: dict[str, Any] = {
        'status': item.status,
    }

    file_parts: list[FilePart] = []
    logs: list[str] = []
    if item.outputs:
        for output in item.outputs:
            if isinstance(output, responses.response_code_interpreter_tool_call.OutputImage):
                file_parts.append(
                    FilePart(
                        content=BinaryImage.from_data_uri(output.url),
                        id=item.id,
                    )
                )
            elif isinstance(output, responses.response_code_interpreter_tool_call.OutputLogs):
                logs.append(output.logs)
            else:
                assert_never(output)

    if logs:
        result['logs'] = logs

    call_part = NativeToolCallPart(
        tool_name=CodeExecutionTool.kind,
        tool_call_id=item.id,
        args={
            'container_id': item.container_id,
            'code': item.code or '',
        },
        provider_name=provider_name,
    )
    call_part.otel_metadata = {'code_arg_name': 'code', 'code_arg_language': 'python'}

    return (
        call_part,
        NativeToolReturnPart(
            tool_name=CodeExecutionTool.kind,
            tool_call_id=item.id,
            content=result,
            provider_name=provider_name,
        ),
        file_parts,
    )


def map_file_search_tool_param(tool: FileSearchTool) -> responses.FileSearchToolParam:
    file_search_tool = responses.FileSearchToolParam(type='file_search', vector_store_ids=list(tool.file_store_ids))
    if tool.max_num_results is not None:
        file_search_tool['max_num_results'] = tool.max_num_results
    return file_search_tool


def map_web_search_tool_param(tool: WebSearchTool) -> responses.WebSearchToolParam:
    """Map a [`WebSearchTool`][pydantic_ai.native_tools.WebSearchTool] to a Responses `web_search` tool."""
    web_search_tool = responses.WebSearchToolParam(type='web_search', search_context_size=tool.search_context_size)
    if tool.user_location:
        web_search_tool['user_location'] = responses.web_search_tool_param.UserLocation(
            type='approximate', **tool.user_location
        )
    filters = responses.web_search_tool_param.Filters()
    if tool.allowed_domains:
        filters['allowed_domains'] = tool.allowed_domains
    if tool.blocked_domains:
        # The OpenAI API supports this field, but the SDK's `Filters` does not include it yet.
        cast(dict[str, object], filters)['blocked_domains'] = tool.blocked_domains
    if filters:
        web_search_tool['filters'] = filters
    if tool.external_web_access is not None:
        # The OpenAI API supports this field, but the SDK's `WebSearchToolParam` does not include it yet.
        cast(dict[str, object], web_search_tool)['external_web_access'] = tool.external_web_access
    return web_search_tool


def map_code_interpreter_tool_param(
    tool: CodeExecutionTool, provider_name: str
) -> responses.tool_param.CodeInterpreter:
    """Map a [`CodeExecutionTool`][pydantic_ai.native_tools.CodeExecutionTool] to a Responses `code_interpreter` tool."""
    container: responses.tool_param.CodeInterpreterContainerCodeInterpreterToolAuto = {'type': 'auto'}
    if tool.files:
        # Cross-provider files are dropped silently here, not raised via
        # `_validate_uploaded_file_provider`; intentional per https://github.com/pydantic/pydantic-ai/issues/4338 (ignore over raise).
        provider_file_ids = [file.file_id for file in tool.files if file.provider_name == provider_name]
        if provider_file_ids:
            container['file_ids'] = provider_file_ids
    return {'type': 'code_interpreter', 'container': container}


def map_mcp_server_tool_param(tool: MCPServerTool) -> responses.tool_param.Mcp:
    """Map an [`MCPServerTool`][pydantic_ai.native_tools.MCPServerTool] to a Responses `mcp` tool."""
    mcp_tool = responses.tool_param.Mcp(
        type='mcp',
        server_label=tool.id,
        require_approval='never',
    )

    if tool.authorization_token:  # pragma: no branch
        mcp_tool['authorization'] = tool.authorization_token

    if tool.allowed_tools is not None:  # pragma: no branch
        mcp_tool['allowed_tools'] = tool.allowed_tools

    if tool.description:  # pragma: no branch
        mcp_tool['server_description'] = tool.description

    if tool.headers:  # pragma: no branch
        mcp_tool['headers'] = tool.headers

    if tool.url.startswith(MCP_SERVER_TOOL_CONNECTOR_URI_SCHEME + ':'):
        _, connector_id = tool.url.split(':', maxsplit=1)
        mcp_tool['connector_id'] = connector_id  # pyright: ignore[reportGeneralTypeIssues]
    else:
        mcp_tool['server_url'] = tool.url

    return mcp_tool


def map_web_search_call_for_replay(
    item: NativeToolCallPart, args: Mapping[str, object]
) -> responses.ResponseFunctionWebSearchParam:
    """Map a `WebSearchTool` call to the `web_search_call` item that replays it by its ID."""
    # We need to exclude None values because of https://github.com/pydantic/pydantic-ai/issues/3653
    action = {k: v for k, v in args.items() if v is not None}
    return responses.ResponseFunctionWebSearchParam(
        id=item.id or item.tool_call_id,
        action=cast(responses.response_function_web_search_param.Action, action),
        status='completed',
        type='web_search_call',
    )


def map_web_search_tool_call(
    item: responses.ResponseFunctionWebSearch, provider_name: str
) -> tuple[NativeToolCallPart, NativeToolReturnPart]:
    args: dict[str, Any] | None = None

    result = {
        'status': item.status,
    }

    if action := item.action:
        # We need to exclude None values because of https://github.com/pydantic/pydantic-ai/issues/3653
        args = action.model_dump(mode='json', exclude_none=True)

        # To prevent `Unknown parameter: 'input[2].action.sources'` for `ActionSearch`
        if sources := args.pop('sources', None):
            result['sources'] = sources

    return (
        NativeToolCallPart(
            tool_name=WebSearchTool.kind,
            tool_call_id=item.id,
            args=args,
            provider_name=provider_name,
            id=item.id,
        ),
        NativeToolReturnPart(
            tool_name=WebSearchTool.kind,
            tool_call_id=item.id,
            content=result,
            provider_name=provider_name,
        ),
    )


def map_file_search_tool_call(
    item: responses.ResponseFileSearchToolCall,
    provider_name: str,
) -> tuple[NativeToolCallPart, NativeToolReturnPart]:
    args = {'queries': item.queries}

    result: dict[str, Any] = {
        'status': item.status,
    }
    if item.results is not None:
        result['results'] = [r.model_dump(mode='json') for r in item.results]

    return (
        NativeToolCallPart(
            tool_name=FileSearchTool.kind,
            tool_call_id=item.id,
            args=args,
            provider_name=provider_name,
            id=item.id,
        ),
        NativeToolReturnPart(
            tool_name=FileSearchTool.kind,
            tool_call_id=item.id,
            content=result,
            provider_name=provider_name,
        ),
    )


def map_image_generation_tool_call(
    item: responses.response_output_item.ImageGenerationCall, provider_name: str
) -> tuple[NativeToolCallPart, NativeToolReturnPart, FilePart | None]:
    result = {
        'status': item.status,
    }

    # Not present on the type, but present on the actual object.
    # See https://github.com/openai/openai-python/issues/2649
    if background := getattr(item, 'background', None):
        result['background'] = background
    if quality := getattr(item, 'quality', None):
        result['quality'] = quality
    if size := getattr(item, 'size', None):
        result['size'] = size
    if revised_prompt := getattr(item, 'revised_prompt', None):
        result['revised_prompt'] = revised_prompt
    output_format = getattr(item, 'output_format', 'png')

    file_part: FilePart | None = None
    if item.result:
        file_part = FilePart(
            content=BinaryImage(
                data=base64.b64decode(item.result),
                media_type=f'image/{output_format}',
            ),
            id=item.id,
        )

        # For some reason, the streaming API leaves `status` as `generating` even though generation has completed.
        result['status'] = 'completed'

    return (
        NativeToolCallPart(
            tool_name=ImageGenerationTool.kind,
            tool_call_id=item.id,
            provider_name=provider_name,
        ),
        NativeToolReturnPart(
            tool_name=ImageGenerationTool.kind,
            tool_call_id=item.id,
            content=result,
            provider_name=provider_name,
        ),
        file_part,
    )


def map_mcp_list_tools(
    item: responses.response_output_item.McpListTools, provider_name: str
) -> tuple[NativeToolCallPart, NativeToolReturnPart]:
    tool_name = ':'.join([MCPServerTool.kind, item.server_label])
    return (
        NativeToolCallPart(
            tool_name=tool_name,
            tool_call_id=item.id,
            provider_name=provider_name,
            args={'action': 'list_tools'},
        ),
        NativeToolReturnPart(
            tool_name=tool_name,
            tool_call_id=item.id,
            content=item.model_dump(mode='json', include={'tools', 'error'}),
            provider_name=provider_name,
        ),
    )


def map_mcp_call(
    item: responses.response_output_item.McpCall, provider_name: str
) -> tuple[NativeToolCallPart, NativeToolReturnPart]:
    tool_name = ':'.join([MCPServerTool.kind, item.server_label])
    return (
        NativeToolCallPart(
            tool_name=tool_name,
            tool_call_id=item.id,
            args={
                'action': 'call_tool',
                'tool_name': item.name,
                'tool_args': json.loads(item.arguments) if item.arguments else {},
            },
            provider_name=provider_name,
        ),
        NativeToolReturnPart(
            tool_name=tool_name,
            tool_call_id=item.id,
            # Dumped rather than read off the item like `output` alone would allow: `McpCall.error` is an
            # SDK model union, so reading the attribute puts an SDK model into a message part that then
            # can't be serialized with the message history. `warnings=False` because an error `type` the
            # SDK doesn't know yet is stuffed into the first union member, whose `type` literal then warns.
            content=item.model_dump(mode='json', include={'output', 'error'}, warnings=False),
            provider_name=provider_name,
        ),
    )
