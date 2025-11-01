import base64
from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import Field

from . import _utils, exceptions, messages

try:
    from mcp import types as mcp_types
except ImportError as _import_error:
    raise ImportError(
        'Please install the `mcp` package to use the MCP server, '
        'you can use the `mcp` optional group — `pip install "pydantic-ai-slim[mcp]"`'
    ) from _import_error


@dataclass(repr=False, kw_only=True)
class ResourceAnnotations:
    """Additional properties describing MCP entities."""

    audience: list[mcp_types.Role] | None = None
    """Intended audience for this entity."""

    priority: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    """Priority level for this entity, ranging from 0.0 to 1.0."""

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(repr=False, kw_only=True)
class BaseResource(ABC):
    """Base class for MCP resources."""

    name: str
    """The programmatic name of the resource."""

    title: str | None = None
    """Human-readable title for UI contexts."""

    description: str | None = None
    """A description of what this resource represents."""

    mime_type: str | None = None
    """The MIME type of the resource, if known."""

    annotations: ResourceAnnotations | None = None
    """Optional annotations for the resource."""

    meta: dict[str, Any] | None = None
    """Optional metadata for the resource."""

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(repr=False, kw_only=True)
class Resource(BaseResource):
    """A resource that can be read from an MCP server."""

    uri: str
    """The URI of the resource."""

    size: int | None = None
    """The size of the raw resource content in bytes (before base64 encoding), if known."""


@dataclass(repr=False, kw_only=True)
class ResourceTemplate(BaseResource):
    """A template for parameterized resources on an MCP server."""

    uri_template: str
    """URI template (RFC 6570) for constructing resource URIs."""


@dataclass(repr=False, kw_only=True)
class ServerCapabilities:
    """Capabilities that an MCP server supports."""

    experimental: list[str] | None = None
    """Experimental, non-standard capabilities that the server supports."""

    logging: bool = False
    """Whether the server supports sending log messages to the client."""

    prompts: bool = False
    """Whether the server offers any prompt templates."""

    resources: bool = False
    """Whether the server offers any resources to read."""

    tools: bool = False
    """Whether the server offers any tools to call."""

    completions: bool = False
    """Whether the server offers autocompletion suggestions for prompts and resources."""

    __repr__ = _utils.dataclasses_no_defaults_repr


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
            else:
                # image content
                user_part_content = [
                    messages.BinaryContent(data=base64.b64decode(content.data), media_type=content.mimeType)
                ]

            request_parts.append(messages.UserPromptPart(content=user_part_content))
        else:
            # role is assistant
            # if there are any request parts, add a request message wrapping them
            if request_parts:
                pai_messages.append(messages.ModelRequest(parts=request_parts))
                request_parts = []

            response_parts.append(map_from_sampling_content(content))

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
                                        data=base64.b64decode(chunk.data).decode(),
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
        # TODO(Marcelo): We should ignore ThinkingPart here.
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
        raise NotImplementedError('Image and Audio responses in sampling are not yet supported')


def map_from_mcp_resource(mcp_resource: mcp_types.Resource) -> Resource:
    """Convert from MCP Resource to native Pydantic AI Resource."""
    return Resource(
        uri=str(mcp_resource.uri),
        name=mcp_resource.name,
        title=mcp_resource.title,
        description=mcp_resource.description,
        mime_type=mcp_resource.mimeType,
        size=mcp_resource.size,
        annotations=(
            ResourceAnnotations(audience=mcp_resource.annotations.audience, priority=mcp_resource.annotations.priority)
            if mcp_resource.annotations
            else None
        ),
        meta=mcp_resource.meta,
    )


def map_from_mcp_resource_template(mcp_template: mcp_types.ResourceTemplate) -> ResourceTemplate:
    """Convert from MCP ResourceTemplate to native Pydantic AI ResourceTemplate."""
    return ResourceTemplate(
        uri_template=mcp_template.uriTemplate,
        name=mcp_template.name,
        title=mcp_template.title,
        description=mcp_template.description,
        mime_type=mcp_template.mimeType,
        annotations=(
            ResourceAnnotations(audience=mcp_template.annotations.audience, priority=mcp_template.annotations.priority)
            if mcp_template.annotations
            else None
        ),
        meta=mcp_template.meta,
    )


def map_from_mcp_server_capabilities(mcp_capabilities: mcp_types.ServerCapabilities) -> ServerCapabilities:
    """Convert from MCP ServerCapabilities to native Pydantic AI ServerCapabilities."""
    return ServerCapabilities(
        experimental=list(mcp_capabilities.experimental.keys()) if mcp_capabilities.experimental else None,
        logging=mcp_capabilities.logging is not None,
        prompts=mcp_capabilities.prompts is not None,
        resources=mcp_capabilities.resources is not None,
        tools=mcp_capabilities.tools is not None,
        completions=mcp_capabilities.completions is not None,
    )
