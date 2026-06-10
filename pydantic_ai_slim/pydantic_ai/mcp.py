from __future__ import annotations

import asyncio
import base64
import functools
import os
import re
import ssl
import warnings
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol, TypeAlias, overload

import anyio
import httpx
import pydantic_core
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from pydantic import AnyUrl, BaseModel, Discriminator, Field, Tag
from pydantic_core import CoreSchema, core_schema
from typing_extensions import Self, assert_never, deprecated

from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition

from .direct import model_request
from .toolsets.abstract import AbstractToolset, ToolsetTool

try:
    from mcp import types as mcp_types
    from mcp.client.session import ClientSession, ElicitationFnT, LoggingFnT
    from mcp.client.sse import sse_client
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.client.streamable_http import streamable_http_client
    from mcp.shared import exceptions as mcp_exceptions
    from mcp.shared.context import RequestContext
    from mcp.shared.message import SessionMessage
    from mcp.shared.session import RequestResponder
except ImportError as _import_error:
    raise ImportError(
        'Please install the `mcp` package to use the MCP server, '
        'you can use the `mcp` optional group — `pip install "pydantic-ai-slim[mcp]"`'
    ) from _import_error

if TYPE_CHECKING:
    from fastmcp.client import Client as FastMCPClient
    from fastmcp.client.client import CallToolResult
    from fastmcp.client.elicitation import ElicitationHandler
    from fastmcp.client.logging import LogHandler
    from fastmcp.client.messages import MessageHandlerT
    from fastmcp.client.progress import ProgressHandler
    from fastmcp.client.roots import RootsHandler, RootsList
    from fastmcp.client.sampling import SamplingHandler
    from fastmcp.client.tasks import ToolTask
    from fastmcp.client.transports import (
        ClientTransport,
        SSETransport,
        StdioTransport,
        StreamableHttpTransport,
    )
    from fastmcp.exceptions import ToolError
    from fastmcp.mcp_config import infer_transport_type_from_url
    from fastmcp.server import FastMCP
    from mcp.server.fastmcp import FastMCP as FastMCP1Server


# `fastmcp` is optional at runtime: the `[mcp]` extra pulls `fastmcp-slim[client]` so `MCPToolset`
# works out of the box, but the legacy `MCPServer*` classes only need the bare `mcp` SDK. Defer the
# import error so users with a hand-installed `mcp` (no fastmcp) can still import the legacy classes
# from `pydantic_ai.mcp`; only when they try to construct an `MCPToolset` (or call a helper that
# needs fastmcp) do we raise. The `[fastmcp]` extra is deprecated; it's a v1-only alias for pulling
# the full `fastmcp` package, and will be removed in v2 — `[mcp]` will be the only MCP extra.
_fastmcp_import_error: ImportError | None
try:
    from fastmcp.client import Client as FastMCPClient
    from fastmcp.client.elicitation import ElicitationHandler
    from fastmcp.client.logging import LogHandler
    from fastmcp.client.messages import MessageHandlerT
    from fastmcp.client.progress import ProgressHandler
    from fastmcp.client.roots import RootsHandler, RootsList
    from fastmcp.client.sampling import SamplingHandler
    from fastmcp.client.transports import (
        ClientTransport,
        SSETransport,
        StdioTransport,
        StreamableHttpTransport,
    )
    from fastmcp.exceptions import ToolError
    from fastmcp.mcp_config import infer_transport_type_from_url
    from mcp.server.fastmcp import FastMCP as FastMCP1Server
except ImportError as _err:  # pragma: no cover
    _fastmcp_import_error = _err
else:
    _fastmcp_import_error = None


def _require_fastmcp() -> None:
    """Raise [`ImportError`][ImportError] if the fastmcp client isn't installed."""
    if _fastmcp_import_error is not None:  # pragma: no cover
        raise ImportError(
            'Please install the fastmcp client to use `MCPToolset` — '
            '`pip install "pydantic-ai-slim[mcp]"` pulls `fastmcp-slim[client]`, '
            'or install the full `fastmcp` package directly.'
        ) from _fastmcp_import_error


# after mcp imports so any import error maps to this file, not _mcp.py
from . import _mcp, _utils, exceptions, messages, models  # noqa: E402
from .settings import ModelSettings  # noqa: E402

__all__ = (
    'MCPToolset',
    'MCPToolsetClient',
    'load_mcp_toolsets',
    'MCPServer',
    'MCPServerStdio',
    'MCPServerHTTP',
    'MCPServerSSE',
    'MCPServerStreamableHTTP',
    'load_mcp_servers',
    'MCPError',
    'Resource',
    'ResourceAnnotations',
    'ResourceTemplate',
    'ServerCapabilities',
    'ProcessToolCallback',
    'CallToolFunc',
    'ToolResult',
    'Prompt',
    'PromptArgument',
    'PromptMessage',
    'PromptResult',
    'Icon',
    'ResourceLink',
    'EmbeddedResource',
    'ContentBlock',
    'PromptRole',
)


class MCPError(RuntimeError):
    """Raised when an MCP server returns an error response.

    This exception wraps error responses from MCP servers, following the ErrorData schema
    from the MCP specification.
    """

    message: str
    """The error message."""

    code: int
    """The error code returned by the server."""

    data: dict[str, Any] | None
    """Additional information about the error, if provided by the server."""

    def __init__(self, message: str, code: int, data: dict[str, Any] | None = None):
        self.message = message
        self.code = code
        self.data = data
        super().__init__(message)

    @classmethod
    def from_mcp_sdk(cls, error: mcp_exceptions.McpError) -> MCPError:
        """Create an MCPError from an MCP SDK McpError.

        Args:
            error: An McpError from the MCP SDK.
        """
        # Extract error data from the McpError.error attribute
        error_data = error.error
        return cls(message=error_data.message, code=error_data.code, data=error_data.data)

    def __str__(self) -> str:
        if self.data:
            return f'{self.message} (code: {self.code}, data: {self.data})'
        return f'{self.message} (code: {self.code})'


@dataclass(repr=False, kw_only=True)
class ResourceAnnotations:
    """Additional properties describing MCP entities.

    See the [resource annotations in the MCP specification](https://modelcontextprotocol.io/specification/2025-11-25/server/resources#annotations).
    """

    audience: list[mcp_types.Role] | None = None
    """Intended audience for this entity."""

    priority: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    """Priority level for this entity, ranging from 0.0 to 1.0."""

    last_modified: str | None = None
    """ISO 8601 timestamp of the last modification."""

    __repr__ = _utils.dataclasses_no_defaults_repr

    @classmethod
    def from_mcp_sdk(cls, mcp_annotations: mcp_types.Annotations) -> ResourceAnnotations:
        """Convert from MCP SDK Annotations to ResourceAnnotations.

        Args:
            mcp_annotations: The MCP SDK annotations object.
        """
        return cls(
            audience=mcp_annotations.audience,
            priority=mcp_annotations.priority,
            # `lastModified` is in the 2025-11-25 spec on `Annotations` but absent from `mcp` v1.25.0;
            # read defensively so we pick it up as soon as the SDK catches up.
            last_modified=getattr(mcp_annotations, 'lastModified', None),
        )


@dataclass(repr=False, kw_only=True)
class Icon:
    """An icon for display in user interfaces."""

    src: str
    """URL or data URI for the icon."""

    mime_type: str | None = None
    """Optional MIME type for the icon."""

    sizes: list[str] | None = None
    """Optional list of strings specifying icon dimensions (e.g., ["48x48", "96x96"])."""

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

    icons: list[Icon] | None = None
    """Optional icons for the resource."""

    metadata: dict[str, Any] | None = None
    """Optional metadata for the resource."""

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(repr=False, kw_only=True)
class Resource(BaseResource):
    """A resource that can be read from an MCP server.

    See the [resources in the MCP specification](https://modelcontextprotocol.io/specification/2025-11-25/server/resources).
    """

    uri: str
    """The URI of the resource."""

    size: int | None = None
    """The size of the raw resource content in bytes (before base64 encoding), if known."""

    @classmethod
    def from_mcp_sdk(cls, mcp_resource: mcp_types.Resource) -> Resource:
        """Convert from MCP SDK Resource to PydanticAI Resource.

        Args:
            mcp_resource: The MCP SDK Resource object.
        """
        return cls(
            uri=str(mcp_resource.uri),
            name=mcp_resource.name,
            title=mcp_resource.title,
            description=mcp_resource.description,
            mime_type=mcp_resource.mimeType,
            size=mcp_resource.size,
            annotations=ResourceAnnotations.from_mcp_sdk(mcp_resource.annotations)
            if mcp_resource.annotations
            else None,
            icons=[Icon(src=icon.src, mime_type=icon.mimeType, sizes=icon.sizes) for icon in mcp_resource.icons]
            if mcp_resource.icons
            else None,
            metadata=mcp_resource.meta,
        )


@dataclass(repr=False, kw_only=True)
class ResourceTemplate(BaseResource):
    """A template for parameterized resources on an MCP server.

    See the [resource templates in the MCP specification](https://modelcontextprotocol.io/specification/2025-11-25/server/resources#resource-templates).
    """

    uri_template: str
    """URI template (RFC 6570) for constructing resource URIs."""

    @classmethod
    def from_mcp_sdk(cls, mcp_template: mcp_types.ResourceTemplate) -> ResourceTemplate:
        """Convert from MCP SDK ResourceTemplate to PydanticAI ResourceTemplate.

        Args:
            mcp_template: The MCP SDK ResourceTemplate object.
        """
        return cls(
            uri_template=mcp_template.uriTemplate,
            name=mcp_template.name,
            title=mcp_template.title,
            description=mcp_template.description,
            mime_type=mcp_template.mimeType,
            annotations=ResourceAnnotations.from_mcp_sdk(mcp_template.annotations)
            if mcp_template.annotations
            else None,
            icons=[Icon(src=icon.src, mime_type=icon.mimeType, sizes=icon.sizes) for icon in mcp_template.icons]
            if mcp_template.icons
            else None,
            metadata=mcp_template.meta,
        )


@dataclass(repr=False, kw_only=True)
class ResourceLink:
    """A resource link referenced in a prompt or tool call result.

    Unlike [`EmbeddedResource`][pydantic_ai.mcp.EmbeddedResource], this does not include the resource
    content directly — it is a reference to a resource that the server can read.

    Note: resource links returned by tools are not guaranteed to appear in the results of
    `resources/list` requests.

    See the [MCP specification](https://modelcontextprotocol.io/specification/2025-11-25/server/resources).
    """

    uri: str
    """The URI of the linked resource."""

    name: str
    """The programmatic name of the linked resource."""

    title: str | None = None
    """Human-readable title for UI contexts."""

    description: str | None = None
    """A description of what this linked resource represents."""

    mime_type: str | None = None
    """The MIME type of the linked resource, if known."""

    size: int | None = None
    """The size of the raw resource content in bytes (before base64 encoding), if known."""

    annotations: ResourceAnnotations | None = None
    """Optional annotations for the linked resource."""

    icons: list[Icon] | None = None
    """Optional icons for the linked resource."""

    metadata: dict[str, Any] | None = None
    """Optional metadata for the linked resource."""

    type: Literal['resource_link'] = 'resource_link'
    """Discriminator for resource link content."""

    __repr__ = _utils.dataclasses_no_defaults_repr

    @classmethod
    def from_mcp_sdk(cls, mcp_resource_link: mcp_types.ResourceLink) -> ResourceLink:
        """Convert from MCP SDK ResourceLink to PydanticAI ResourceLink."""
        return cls(
            type='resource_link',
            uri=str(mcp_resource_link.uri),
            name=mcp_resource_link.name,
            title=mcp_resource_link.title,
            description=mcp_resource_link.description,
            mime_type=mcp_resource_link.mimeType,
            size=mcp_resource_link.size,
            annotations=ResourceAnnotations.from_mcp_sdk(mcp_resource_link.annotations)
            if mcp_resource_link.annotations
            else None,
            icons=[Icon(src=icon.src, mime_type=icon.mimeType, sizes=icon.sizes) for icon in mcp_resource_link.icons]
            if mcp_resource_link.icons
            else None,
            metadata=mcp_resource_link.meta,
        )


@dataclass(repr=False, kw_only=True)
class PromptArgument:
    """An argument for a prompt template."""

    name: str
    """The name of the argument."""

    title: str | None = None
    """Human-readable title for the argument."""

    description: str | None = None
    """A human-readable description of the argument."""

    required: bool | None = None
    """Whether the argument is required or optional. If not specified, the server may determine this based on context."""

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(repr=False, kw_only=True)
class Prompt:
    """A prompt or prompt template that the server offers."""

    name: str
    """The programmatic name of the prompt."""

    title: str | None = None
    """Human-readable title for prompt."""

    description: str | None = None
    """An optional description of what this prompt provides."""

    arguments: list[PromptArgument] | None = None
    """A list of arguments to use for templating the prompt."""

    icons: list[Icon] | None = None
    """An optional list of icons for this prompt."""

    metadata: dict[str, Any] | None = None
    """
    See [MCP specification](https://modelcontextprotocol.io/specification/2025-11-25/basic#_meta)
    for notes on _meta usage.
    """

    __repr__ = _utils.dataclasses_no_defaults_repr

    @classmethod
    def from_mcp_sdk(cls, mcp_prompt: mcp_types.Prompt) -> Prompt:
        """Convert from MCP SDK Prompt to PydanticAI Prompt.

        Args:
            mcp_prompt: The MCP SDK Prompt object.
        """
        return cls(
            name=mcp_prompt.name,
            title=mcp_prompt.title,
            description=mcp_prompt.description,
            arguments=[
                PromptArgument(
                    name=arg.name,
                    # `title` is in the 2025-11-25 spec on `PromptArgument` (via `BaseMetadata`)
                    # but absent from `mcp` v1.25.0; read defensively until the SDK catches up.
                    title=getattr(arg, 'title', None),
                    description=arg.description,
                    required=arg.required,
                )
                for arg in mcp_prompt.arguments
            ]
            if mcp_prompt.arguments
            else None,
            icons=[
                Icon(
                    src=icon.src,
                    mime_type=icon.mimeType,
                    sizes=icon.sizes,
                )
                for icon in mcp_prompt.icons
            ]
            if mcp_prompt.icons
            else None,
            metadata=mcp_prompt.meta,
        )


PromptRole = Literal['user', 'assistant']


@dataclass(repr=False, kw_only=True)
class EmbeddedResource:
    """A resource embedded into a prompt or tool call result.

    Contains the actual resource content alongside its metadata, unlike
    [`ResourceLink`][pydantic_ai.mcp.ResourceLink] which is only a reference.

    See the [MCP specification](https://modelcontextprotocol.io/specification/2025-11-25/server/resources).
    """

    uri: str
    """The URI of the embedded resource."""

    content: str | messages.BinaryContent
    """The content of the embedded resource."""

    type: Literal['resource'] = 'resource'
    """Discriminator for embedded resource content."""

    mime_type: str | None = None
    """The MIME type of the resource, if known."""

    annotations: ResourceAnnotations | None = None
    """Optional annotations for the resource."""

    metadata: dict[str, Any] | None = None
    """
    See [MCP specification](https://modelcontextprotocol.io/specification/2025-11-25/basic#_meta)
    for notes on _meta usage.
    """

    resource_metadata: dict[str, Any] | None = None
    """`_meta` carried on the nested resource contents (separate from the embedding's own `_meta`)."""

    __repr__ = _utils.dataclasses_no_defaults_repr

    @classmethod
    def from_mcp_sdk(cls, part: mcp_types.EmbeddedResource, content: str | messages.BinaryContent) -> EmbeddedResource:
        """Convert from MCP SDK EmbeddedResource to PydanticAI EmbeddedResource."""
        return cls(
            uri=str(part.resource.uri),
            content=content,
            mime_type=part.resource.mimeType,
            annotations=ResourceAnnotations.from_mcp_sdk(part.annotations) if part.annotations else None,
            metadata=part.meta,
            resource_metadata=part.resource.meta,
        )


ContentBlock = messages.TextContent | messages.BinaryContent | ResourceLink | EmbeddedResource
"""A content block that can be used in prompts and tool results."""


@dataclass(repr=False, kw_only=True)
class PromptMessage:
    """A message returned as part of a prompt result."""

    role: PromptRole
    """The role of the message sender."""

    content: ContentBlock
    """The content of the message."""

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(repr=False, kw_only=True)
class PromptResult:
    """The result of a [`get_prompt`][pydantic_ai.mcp.MCPServer.get_prompt] request."""

    messages: list[PromptMessage]
    """The prompt messages."""

    description: str | None = None
    """An optional description for the prompt."""

    metadata: dict[str, Any] | None = None
    """
    See [MCP specification](https://modelcontextprotocol.io/specification/2025-11-25/basic#_meta)
    for notes on _meta usage.
    """

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(repr=False, kw_only=True)
class ServerCapabilities:
    """Capabilities that an MCP server supports."""

    experimental: list[str] | None = None
    """Experimental, non-standard capabilities that the server supports."""

    logging: bool = False
    """Whether the server supports sending log messages to the client."""

    prompts: bool = False
    """Whether the server offers any prompt templates."""

    prompts_list_changed: bool = False
    """Whether the server will emit notifications when the list of prompts changes."""

    resources: bool = False
    """Whether the server offers any resources to read."""

    resources_list_changed: bool = False
    """Whether the server will emit notifications when the list of resources changes."""

    tools: bool = False
    """Whether the server offers any tools to call."""

    tools_list_changed: bool = False
    """Whether the server will emit notifications when the list of tools changes."""

    completions: bool = False
    """Whether the server offers autocompletion suggestions for prompts and resources."""

    __repr__ = _utils.dataclasses_no_defaults_repr

    @classmethod
    def from_mcp_sdk(cls, mcp_capabilities: mcp_types.ServerCapabilities) -> ServerCapabilities:
        """Convert from MCP SDK ServerCapabilities to PydanticAI ServerCapabilities.

        Args:
            mcp_capabilities: The MCP SDK ServerCapabilities object.
        """
        prompts_cap = mcp_capabilities.prompts
        resources_cap = mcp_capabilities.resources
        tools_cap = mcp_capabilities.tools
        return cls(
            experimental=list(mcp_capabilities.experimental.keys()) if mcp_capabilities.experimental else None,
            logging=mcp_capabilities.logging is not None,
            prompts=prompts_cap is not None,
            prompts_list_changed=bool(prompts_cap.listChanged) if prompts_cap else False,
            resources=resources_cap is not None,
            resources_list_changed=bool(resources_cap.listChanged) if resources_cap else False,
            tools=tools_cap is not None,
            tools_list_changed=bool(tools_cap.listChanged) if tools_cap else False,
            completions=mcp_capabilities.completions is not None,
        )


TOOL_SCHEMA_VALIDATOR = pydantic_core.SchemaValidator(
    schema=pydantic_core.core_schema.dict_schema(
        pydantic_core.core_schema.str_schema(), pydantic_core.core_schema.any_schema()
    )
)

# Environment variable expansion pattern
# Supports both ${VAR_NAME} and ${VAR_NAME:-default} syntax
# Group 1: variable name
# Group 2: the ':-' separator (to detect if default syntax is used)
# Group 3: the default value (can be empty)
_ENV_VAR_PATTERN = re.compile(r'\$\{([^}:]+)(:-([^}]*))?\}')


_SHUTDOWN_GRACE_SECONDS = 3
"""How long to wait for the session task to wind down at each shutdown phase
(graceful stop in `__aexit__`, force-cancel in either `__aenter__` cancel cleanup
or `__aexit__` escalation). Bounds worst-case cleanup time when the underlying
transport is unresponsive (e.g. a hung subprocess); past this we move on without
awaiting it."""


@dataclass
class _MCPSessionState:
    """State for the single background session task that owns an MCPServer's connection.

    The session task is spawned on first `__aenter__`, runs in its own asyncio.Task
    (escaping structured concurrency so it can outlive nested `async with` scopes),
    and is torn down when the last `__aexit__` decrements the ref count to zero.

    Because the task enters and exits its cancel scope in the same task, the
    `RuntimeError: Attempted to exit cancel scope in a different task` error from
    the underlying anyio transports cannot occur — regardless of which task
    originally called `__aenter__` / `__aexit__`.
    """

    session_task: asyncio.Task[None] | None = None
    ready_event: anyio.Event | None = None
    stop_event: anyio.Event | None = None
    nesting_counter: int = 0
    client: ClientSession | None = None
    connect_error: BaseException | None = None

    async def force_close(self, task: asyncio.Task[None]) -> None:
        """Cancel `task` and wait up to `_SHUTDOWN_GRACE_SECONDS` for it to unwind.

        Shielded against external cancellation so cleanup completes regardless of
        the caller's cancel state; the timeout bounds worst-case wait when the
        underlying transport's `__aexit__` can't unwind cleanly (e.g. hung
        subprocess, server that never closes the connection).
        """
        task.cancel()
        with anyio.CancelScope(shield=True):
            with anyio.move_on_after(_SHUTDOWN_GRACE_SECONDS):
                try:
                    await task
                except BaseException:
                    pass


class MCPServer(AbstractToolset[Any], ABC):
    """Base class for attaching agents to MCP servers.

    See <https://modelcontextprotocol.io> for more information.

    !!! warning "Deprecated"
        This class hierarchy (`MCPServer`, `MCPServerStdio`, `MCPServerSSE`,
        `MCPServerStreamableHTTP`, `MCPServerHTTP`) is deprecated in favor of
        [`MCPToolset`][pydantic_ai.mcp.MCPToolset], which is built on the more capable FastMCP
        client and supports the full MCP protocol. The concrete subclasses will be removed in v2.
    """

    tool_prefix: str | None
    """A prefix to add to all tools that are registered with the server.

    If not empty, will include a trailing underscore(`_`).

    e.g. if `tool_prefix='foo'`, then a tool named `bar` will be registered as `foo_bar`
    """

    log_level: mcp_types.LoggingLevel | None
    """The log level to set when connecting to the server, if any.

    See <https://modelcontextprotocol.io/specification/2025-03-26/server/utilities/logging#logging> for more details.

    If `None`, no log level will be set.
    """

    log_handler: LoggingFnT | None
    """A handler for logging messages from the server."""

    timeout: float
    """The timeout in seconds to wait for the client to initialize."""

    read_timeout: float
    """Maximum time in seconds to wait for new messages before timing out.

    This timeout applies to the long-lived connection after it's established.
    If no new messages are received within this time, the connection will be considered stale
    and may be closed. Defaults to 5 minutes (300 seconds).
    """

    process_tool_call: ProcessToolCallback | None
    """Hook to customize tool calling and optionally pass extra metadata."""

    allow_sampling: bool
    """Whether to allow MCP sampling through this client."""

    sampling_model: models.Model | None
    """The model to use for sampling."""

    max_retries: int
    """The maximum number of times to retry a tool call."""

    elicitation_callback: ElicitationFnT | None = None
    """Callback function to handle elicitation requests from the server."""

    cache_prompts: bool
    """Whether to cache the list of prompts.

    When enabled (default), prompts are fetched once and cached until either:
    - The server sends a `notifications/prompts/list_changed` notification
    - The connection is closed
    """

    cache_tools: bool
    """Whether to cache the list of tools.

    When enabled (default), tools are fetched once and cached until either:
    - The server sends a `notifications/tools/list_changed` notification
    - [`MCPServer.__aexit__`][pydantic_ai.mcp.MCPServer.__aexit__] is called (when the last context exits)

    Set to `False` for servers that change tools dynamically without sending notifications.

    Note: When using durable execution (Temporal, DBOS), tool definitions are additionally cached
    at the wrapper level across activities/steps, to avoid redundant MCP connections. This
    wrapper-level cache is not invalidated by `tools/list_changed` notifications.
    Set to `False` to disable all caching if tools may change during a workflow.
    """

    cache_resources: bool
    """Whether to cache the list of resources.

    When enabled (default), resources are fetched once and cached until either:
    - The server sends a `notifications/resources/list_changed` notification
    - [`MCPServer.__aexit__`][pydantic_ai.mcp.MCPServer.__aexit__] is called (when the last context exits)

    Set to `False` for servers that change resources dynamically without sending notifications.
    """

    include_instructions: bool
    """Whether to include the server's instructions in the agent's instructions.

    Defaults to `False` for backward compatibility.
    """

    include_return_schema: bool | None
    """Whether to include return schemas in tool definitions sent to the model.

    When `None` (default), defaults to `False` unless the
    [`IncludeToolReturnSchemas`][pydantic_ai.capabilities.IncludeToolReturnSchemas] capability is used.
    """

    _id: str | None

    _session_state: _MCPSessionState = field(compare=False)

    _server_info: mcp_types.Implementation
    _server_capabilities: ServerCapabilities
    _instructions: str | None

    _cached_prompts: list[Prompt] | None
    _cached_tools: list[mcp_types.Tool] | None
    _cached_resources: list[Resource] | None

    @functools.cached_property
    def _enter_lock(self) -> anyio.Lock:
        return anyio.Lock()

    # TODO (v2): enforce the arguments to be passed as keyword arguments only
    def __init__(
        self,
        tool_prefix: str | None = None,
        log_level: mcp_types.LoggingLevel | None = None,
        log_handler: LoggingFnT | None = None,
        timeout: float = 5,
        read_timeout: float = 5 * 60,
        process_tool_call: ProcessToolCallback | None = None,
        allow_sampling: bool = True,
        sampling_model: models.Model | None = None,
        max_retries: int = 1,
        elicitation_callback: ElicitationFnT | None = None,
        *,
        cache_prompts: bool = True,
        cache_tools: bool = True,
        cache_resources: bool = True,
        include_instructions: bool = False,
        include_return_schema: bool | None = None,
        id: str | None = None,
        client_info: mcp_types.Implementation | None = None,
    ):
        self.tool_prefix = tool_prefix
        self.log_level = log_level
        self.log_handler = log_handler
        self.timeout = timeout
        self.read_timeout = read_timeout
        self.process_tool_call = process_tool_call
        self.allow_sampling = allow_sampling
        self.sampling_model = sampling_model
        self.max_retries = max_retries
        self.elicitation_callback = elicitation_callback
        self.cache_prompts = cache_prompts
        self.cache_tools = cache_tools
        self.cache_resources = cache_resources
        self.include_instructions = include_instructions
        self.include_return_schema = include_return_schema
        self.client_info = client_info

        self._id = id or tool_prefix

        self.__post_init__()

    def __post_init__(self):
        self._session_state = _MCPSessionState()
        self._cached_prompts = None
        self._cached_tools = None
        self._cached_resources = None

    @abstractmethod
    @asynccontextmanager
    async def client_streams(
        self,
    ) -> AsyncGenerator[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
        ]
    ]:
        """Create the streams for the MCP server."""
        raise NotImplementedError('MCP Server subclasses must implement this method.')
        yield

    @property
    def id(self) -> str | None:
        return self._id

    @id.setter
    def id(self, value: str | None):
        self._id = value

    @property
    def label(self) -> str:
        if self.id:
            return super().label  # pragma: no cover
        else:
            return repr(self)

    @property
    def tool_name_conflict_hint(self) -> str:
        return 'Set the `tool_prefix` attribute to avoid name conflicts.'

    @property
    def server_info(self) -> mcp_types.Implementation:
        """Access the information send by the MCP server during initialization."""
        if getattr(self, '_server_info', None) is None:
            raise AttributeError(
                f'The `{self.__class__.__name__}.server_info` is only instantiated after initialization.'
            )
        return self._server_info

    @property
    def capabilities(self) -> ServerCapabilities:
        """Access the capabilities advertised by the MCP server during initialization."""
        if getattr(self, '_server_capabilities', None) is None:
            raise AttributeError(
                f'The `{self.__class__.__name__}.capabilities` is only instantiated after initialization.'
            )
        return self._server_capabilities

    @property
    def instructions(self) -> str | None:
        """Access the instructions sent by the MCP server during initialization."""
        if not hasattr(self, '_instructions'):
            raise AttributeError(
                f'The `{self.__class__.__name__}.instructions` is only available after initialization.'
            )
        return self._instructions

    async def get_instructions(self, ctx: RunContext[Any]) -> messages.InstructionPart | None:
        """Return the MCP server's instructions for how to use its tools.

        If [`include_instructions`][pydantic_ai.mcp.MCPServer.include_instructions] is `True`, returns
        the [`instructions`][pydantic_ai.mcp.MCPServer.instructions] sent by the MCP server during
        initialization. Otherwise, returns `None`.

        Instructions from external servers are marked as dynamic since they may change between connections.

        Args:
            ctx: The run context for this agent run.

        Returns:
            An `InstructionPart` with the server's instructions if `include_instructions` is enabled, otherwise `None`.
        """
        if not self.include_instructions:
            return None
        try:
            instr = self.instructions
        except AttributeError:
            # Server not yet initialized — return None rather than propagating.
            # Durable execution wrappers detect this and fetch via activity/step.
            return None
        return messages.InstructionPart(content=instr, dynamic=True) if instr is not None else None

    async def list_prompts(self) -> list[Prompt]:
        """Retrieve prompts that are currently active on the server.

        Prompts are cached by default, with cache invalidation on:
        - `notifications/prompts/list_changed` notifications from the server
        - Connection close (cache is cleared in `__aexit__`)

        Set `cache_prompts=False` for servers that change prompts without sending notifications.

        Raises:
            MCPError: If the server returns an error.
        """
        if self.cache_prompts and self._cached_prompts is not None:
            return self._cached_prompts

        async with self:
            if not self.capabilities.prompts:
                return []
            try:
                # Follow `nextCursor` until the server stops handing one back. The pre-existing
                # `list_tools` / `list_resources` on this class do not paginate yet; tracked as a
                # follow-up so all three list methods behave the same.
                # See https://modelcontextprotocol.io/specification/2025-11-25/server/utilities/pagination
                client = self._get_client()
                prompts: list[Prompt] = []
                cursor: str | None = None
                while True:
                    result = await client.list_prompts(params=mcp_types.PaginatedRequestParams(cursor=cursor))
                    prompts.extend(Prompt.from_mcp_sdk(p) for p in result.prompts)
                    # Treat falsy `nextCursor` (`None` or `""`) as end-of-results — guards against a
                    # misbehaving server that would otherwise loop the client on `cursor=""` forever.
                    if not result.nextCursor:
                        break
                    cursor = result.nextCursor
                if self.cache_prompts:
                    self._cached_prompts = prompts
                return prompts
            except mcp_exceptions.McpError as e:
                raise MCPError.from_mcp_sdk(e) from e

    async def get_prompt(self, name: str, arguments: dict[str, str] | None = None) -> PromptResult:
        """Retrieve a specific prompt by name.

        Args:
            name: The name of the prompt to retrieve.
            arguments: Arguments to parameterize the prompt, if applicable.

        Returns:
            The prompt result with description and messages.

        Raises:
            MCPError: If the server doesn't advertise the `prompts` capability, or if it returns
                an error response.
        """
        async with self:
            if not self.capabilities.prompts:
                raise MCPError(
                    message=f'Server does not advertise the `prompts` capability; cannot get prompt {name!r}.',
                    code=-32601,
                )
            try:
                result = await self._get_client().get_prompt(name, arguments)
            except mcp_exceptions.McpError as e:
                raise MCPError.from_mcp_sdk(e) from e

            return PromptResult(
                description=result.description,
                metadata=result.meta,
                messages=[
                    PromptMessage(role=msg.role, content=_map_mcp_prompt_part(msg.content)) for msg in result.messages
                ],
            )

    async def list_tools(self) -> list[mcp_types.Tool]:
        """Retrieve tools that are currently active on the server.

        Tools are cached by default, with cache invalidation on:
        - `notifications/tools/list_changed` notifications from the server
        - `__aexit__` when the last context exits

        Set `cache_tools=False` for servers that change tools without sending notifications.
        """
        if self.cache_tools and self._cached_tools is not None:
            return self._cached_tools

        async with self:
            result = await self._get_client().list_tools()
            if self.cache_tools:
                self._cached_tools = result.tools
            return result.tools

    async def direct_call_tool(
        self,
        name: str,
        args: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Call a tool on the server.

        Args:
            name: The name of the tool to call.
            args: The arguments to pass to the tool.
            metadata: Request-level metadata (optional)

        Returns:
            The result of the tool call.

        Raises:
            ModelRetry: If the tool call fails.
        """
        async with self:  # Ensure server is running
            try:
                result = await self._get_client().send_request(
                    mcp_types.ClientRequest(
                        mcp_types.CallToolRequest(
                            method='tools/call',
                            params=mcp_types.CallToolRequestParams(
                                name=name,
                                arguments=args,
                                _meta=mcp_types.RequestParams.Meta(**metadata) if metadata else None,
                            ),
                        )
                    ),
                    mcp_types.CallToolResult,
                )
            except mcp_exceptions.McpError as e:
                raise exceptions.ModelRetry(e.error.message)

        if result.isError:
            message: str | None = None
            if result.content:  # pragma: no branch
                text_parts = [part.text for part in result.content if isinstance(part, mcp_types.TextContent)]
                message = '\n'.join(text_parts)

            raise exceptions.ModelRetry(message or 'MCP tool call failed')

        # Prefer structured content if there are only text parts, which per the docs would contain the JSON-encoded structured content for backward compatibility.
        # See https://github.com/modelcontextprotocol/python-sdk#structured-output
        if (structured := result.structuredContent) and not any(
            not isinstance(part, mcp_types.TextContent) for part in result.content
        ):
            # The MCP SDK wraps primitives and generic types like list in a `result` key, but we want to use the raw value returned by the tool function.
            # See https://github.com/modelcontextprotocol/python-sdk#structured-output
            if isinstance(structured, dict) and len(structured) == 1 and 'result' in structured:
                return structured['result']
            return structured

        mapped = [await self._map_tool_result_part(part) for part in result.content]
        return mapped[0] if len(mapped) == 1 else mapped

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
    ) -> ToolResult:
        if self.tool_prefix:
            name = name.removeprefix(f'{self.tool_prefix}_')
            ctx = replace(ctx, tool_name=name)

        if self.process_tool_call is not None:
            return await self.process_tool_call(ctx, self.direct_call_tool, name, tool_args)
        else:
            return await self.direct_call_tool(name, tool_args)

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        return {
            name: self.tool_for_tool_def(
                ToolDefinition(
                    name=name,
                    description=mcp_tool.description,
                    parameters_json_schema=mcp_tool.inputSchema,
                    metadata={
                        'meta': mcp_tool.meta,
                        'annotations': mcp_tool.annotations.model_dump() if mcp_tool.annotations else None,
                        'output_schema': mcp_tool.outputSchema or None,
                    },
                    return_schema=mcp_tool.outputSchema or None,
                    include_return_schema=self.include_return_schema,
                ),
            )
            for mcp_tool in await self.list_tools()
            if (name := f'{self.tool_prefix}_{mcp_tool.name}' if self.tool_prefix else mcp_tool.name)
        }

    def tool_for_tool_def(self, tool_def: ToolDefinition) -> ToolsetTool[Any]:
        return ToolsetTool(
            toolset=self,
            tool_def=tool_def,
            max_retries=self.max_retries,
            args_validator=TOOL_SCHEMA_VALIDATOR,
        )

    async def list_resources(self) -> list[Resource]:
        """Retrieve resources that are currently present on the server.

        Resources are cached by default, with cache invalidation on:
        - `notifications/resources/list_changed` notifications from the server
        - `__aexit__` when the last context exits

        Set `cache_resources=False` for servers that change resources without sending notifications.

        Raises:
            MCPError: If the server returns an error.
        """
        if self.cache_resources and self._cached_resources is not None:
            return self._cached_resources

        async with self:
            if not self.capabilities.resources:
                return []
            try:
                result = await self._get_client().list_resources()
                resources = [Resource.from_mcp_sdk(r) for r in result.resources]
                if self.cache_resources:
                    self._cached_resources = resources
                return resources
            except mcp_exceptions.McpError as e:
                raise MCPError.from_mcp_sdk(e) from e

    async def list_resource_templates(self) -> list[ResourceTemplate]:
        """Retrieve resource templates that are currently present on the server.

        Raises:
            MCPError: If the server returns an error.
        """
        async with self:  # Ensure server is running
            if not self.capabilities.resources:
                return []
            try:
                result = await self._get_client().list_resource_templates()
            except mcp_exceptions.McpError as e:
                raise MCPError.from_mcp_sdk(e) from e
        return [ResourceTemplate.from_mcp_sdk(t) for t in result.resourceTemplates]

    @overload
    async def read_resource(self, uri: str) -> str | messages.BinaryContent | list[str | messages.BinaryContent]: ...

    @overload
    async def read_resource(
        self, uri: Resource
    ) -> str | messages.BinaryContent | list[str | messages.BinaryContent]: ...

    async def read_resource(
        self, uri: str | Resource
    ) -> str | messages.BinaryContent | list[str | messages.BinaryContent]:
        """Read the contents of a specific resource by URI.

        Args:
            uri: The URI of the resource to read, or a Resource object.

        Returns:
            The resource contents. If the resource has a single content item, returns that item directly.
            If the resource has multiple content items, returns a list of items.

        Raises:
            MCPError: If the server returns an error.
        """
        resource_uri = uri if isinstance(uri, str) else uri.uri
        async with self:  # Ensure server is running
            try:
                result = await self._get_client().read_resource(AnyUrl(resource_uri))
            except mcp_exceptions.McpError as e:
                raise MCPError.from_mcp_sdk(e) from e

        return (
            _resource_content_to_pai(result.contents[0])
            if len(result.contents) == 1
            else [_resource_content_to_pai(resource) for resource in result.contents]
        )

    def _get_client(self) -> ClientSession:
        client = self._session_state.client
        if client is None:
            raise RuntimeError(  # pragma: no cover
                f'{self.__class__.__name__} is not connected. Use `async with server:` to open a connection first.'
            )
        return client

    async def _session_runner(self) -> None:
        """Own the MCP session's lifecycle for this server.

        Entered AND exited inside this single dedicated asyncio.Task, so the underlying
        anyio cancel scopes (from stdio_client / streamable_http_client / etc.) are
        always exited in the same task they were entered in.
        """
        state = self._session_state
        # Capture local references so a recycled session (new __aenter__ replacing
        # state.ready_event/state.stop_event before this runner's `finally` runs)
        # cannot corrupt the next session's events.
        ready_event = state.ready_event
        stop_event = state.stop_event
        assert ready_event is not None
        assert stop_event is not None
        client: ClientSession | None = None
        try:
            async with AsyncExitStack() as stack:
                read_stream, write_stream = await stack.enter_async_context(self.client_streams())
                session = ClientSession(
                    read_stream=read_stream,
                    write_stream=write_stream,
                    sampling_callback=self._sampling_callback if self.allow_sampling else None,
                    elicitation_callback=self.elicitation_callback,
                    logging_callback=self.log_handler,
                    read_timeout_seconds=timedelta(seconds=self.read_timeout),
                    message_handler=self._handle_notification,
                    client_info=self.client_info,
                )
                client = await stack.enter_async_context(session)

                with anyio.fail_after(self.timeout):
                    result = await client.initialize()
                    self._server_info = result.serverInfo
                    self._server_capabilities = ServerCapabilities.from_mcp_sdk(result.capabilities)
                    self._instructions = result.instructions
                    if log_level := self.log_level:
                        await client.set_logging_level(log_level)

                state.client = client
                ready_event.set()
                await stop_event.wait()
        except BaseException as e:
            # Only record the error if we are still the active session — otherwise
            # __aenter__ has already moved on with a fresh session_task.
            if state.session_task is asyncio.current_task():
                state.connect_error = e
        finally:
            # Only clear state.client if it still references *our* client; a
            # recycled session may have already installed a new one.
            if state.client is client:
                state.client = None
            ready_event.set()

    async def __aenter__(self) -> Self:
        """Enter the MCP server context.

        The first call starts the connection (spawning a subprocess for stdio servers,
        opening an HTTP connection for HTTP servers). Subsequent calls — from any task
        — share the same connection via reference counting. The connection is torn
        down when the last `async with` scope exits.

        Because the session runs in a dedicated background task, entering and exiting
        from different tasks (e.g. `asyncio.gather` children, fasta2a workers, or
        graph node tasks) is safe: the underlying transport's cancel scopes never
        cross task boundaries.
        """
        async with self._enter_lock:
            state = self._session_state
            need_to_start = state.session_task is None or state.session_task.done()
            if need_to_start:
                state.stop_event = anyio.Event()
                state.ready_event = anyio.Event()
                state.connect_error = None
                state.client = None
                state.session_task = asyncio.create_task(self._session_runner())
                try:
                    await state.ready_event.wait()
                except BaseException:
                    # Cancelled while waiting for startup: tear down the session task
                    # without impacting anyone else (we hold the lock and just started it)
                    task = state.session_task
                    state.stop_event.set()
                    await state.force_close(task)
                    state.session_task = None
                    state.client = None
                    raise
                if state.connect_error is not None:
                    # Connection failed during startup; surface the error and reset state
                    state.session_task = None
                    err = state.connect_error
                    state.connect_error = None
                    raise err
            state.nesting_counter += 1
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        state = self._session_state
        session_task_to_await: asyncio.Task[None] | None = None
        async with self._enter_lock:
            if state.nesting_counter == 0:
                raise ValueError('MCPServer.__aexit__ called more times than __aenter__')
            state.nesting_counter -= 1
            if state.nesting_counter > 0:
                return None
            if state.session_task is None:
                return None
            assert state.stop_event is not None
            state.stop_event.set()
            session_task_to_await = state.session_task
            state.session_task = None
            self._cached_tools = None
            self._cached_prompts = None
            self._cached_resources = None
        # Await outside the lock: the session task's cancel scopes unwind inside the
        # task itself, so this await can safely happen from any caller. Bound the
        # wait so a transport whose `__aexit__` deadlocks (hung subprocess, server
        # that never closes the connection) cannot block our own shutdown forever;
        # `move_on_after` cancels this `await`, which propagates the cancel through
        # to `session_task_to_await` itself, so the runner gets torn down too.
        with anyio.move_on_after(_SHUTDOWN_GRACE_SECONDS):
            try:
                await session_task_to_await
            except BaseException:
                pass
        return None

    @property
    def is_running(self) -> bool:
        """Check if the MCP server is running."""
        return self._session_state.nesting_counter > 0

    async def _sampling_callback(
        self, context: RequestContext[ClientSession, Any], params: mcp_types.CreateMessageRequestParams
    ) -> mcp_types.CreateMessageResult | mcp_types.ErrorData:
        """MCP sampling callback."""
        if self.sampling_model is None:
            raise ValueError('Sampling model is not set')  # pragma: no cover

        pai_messages = _mcp.map_from_mcp_params(params)
        model_settings = ModelSettings(max_tokens=params.maxTokens)
        if (temperature := params.temperature) is not None:  # pragma: no branch
            model_settings['temperature'] = temperature
        if (stop_sequences := params.stopSequences) is not None:  # pragma: no branch
            model_settings['stop_sequences'] = stop_sequences

        model_response = await model_request(self.sampling_model, pai_messages, model_settings=model_settings)
        return mcp_types.CreateMessageResult(
            role='assistant',
            content=_mcp.map_from_model_response(model_response),
            model=self.sampling_model.model_name,
        )

    async def _handle_notification(
        self,
        message: RequestResponder[mcp_types.ServerRequest, mcp_types.ClientResult]
        | mcp_types.ServerNotification
        | Exception,
    ) -> None:
        """Handle notifications from the MCP server, invalidating caches as needed."""
        if isinstance(message, mcp_types.ServerNotification):  # pragma: no branch
            if isinstance(message.root, mcp_types.ToolListChangedNotification):
                self._cached_tools = None
            elif isinstance(message.root, mcp_types.ResourceListChangedNotification):
                self._cached_resources = None
            elif isinstance(message.root, mcp_types.PromptListChangedNotification):
                self._cached_prompts = None

    async def _map_tool_result_part(
        self, part: mcp_types.ContentBlock
    ) -> str | messages.BinaryContent | dict[str, Any] | list[Any]:
        # See https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#return-values
        # The only path that differs from the module-level `_map_mcp_tool_result` is `ResourceLink`:
        # here we have a live session and can read the resource; the module-level helper used by
        # `MCPToolset` is sync, so it returns the URI as a string. All other branches share logic.
        if isinstance(part, mcp_types.ResourceLink):
            return await self.read_resource(str(part.uri))
        return _map_mcp_tool_result(part)

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, MCPServer) and self.id == value.id and self.tool_prefix == value.tool_prefix


@deprecated(
    '`MCPServerStdio` is deprecated and will be removed in v2. '
    "Use `MCPToolset('path/to/script.py')` for Python scripts, `MCPToolset('script.js')` for Node "
    "scripts, or `MCPToolset(fastmcp.client.transports.StdioTransport(command='...', args=[...]))` "
    'for arbitrary commands.'
)
class MCPServerStdio(MCPServer):
    """Runs an MCP server in a subprocess and communicates with it over stdin/stdout.

    This class implements the stdio transport from the MCP specification.
    See <https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#stdio> for more information.

    !!! note
        Using this class as an async context manager will start the server as a subprocess when entering the context,
        and stop it when exiting the context.

    Example:
    ```python {py="3.10"}
    from pydantic_ai import Agent
    from pydantic_ai.mcp import MCPServerStdio

    server = MCPServerStdio(  # (1)!
        'uv', args=['run', 'mcp-run-python', 'stdio'], timeout=10
    )
    agent = Agent('openai:gpt-5.2', toolsets=[server])
    ```

    1. See [MCP Run Python](https://github.com/pydantic/mcp-run-python) for more information.
    """

    command: str
    """The command to run."""

    args: Sequence[str]
    """The arguments to pass to the command."""

    env: dict[str, str] | None
    """The environment variables the CLI server will have access to.

    By default the subprocess will not inherit any environment variables from the parent process.
    If you want to inherit the environment variables from the parent process, use `env=os.environ`.
    """

    cwd: str | Path | None
    """The working directory to use when spawning the process."""

    # last fields are re-defined from the parent class so they appear as fields
    tool_prefix: str | None
    log_level: mcp_types.LoggingLevel | None
    log_handler: LoggingFnT | None
    timeout: float
    read_timeout: float
    process_tool_call: ProcessToolCallback | None
    allow_sampling: bool
    sampling_model: models.Model | None
    max_retries: int
    elicitation_callback: ElicitationFnT | None = None
    cache_prompts: bool
    cache_tools: bool
    cache_resources: bool
    include_instructions: bool

    def __init__(
        self,
        command: str,
        args: Sequence[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
        tool_prefix: str | None = None,
        log_level: mcp_types.LoggingLevel | None = None,
        log_handler: LoggingFnT | None = None,
        timeout: float = 5,
        read_timeout: float = 5 * 60,
        process_tool_call: ProcessToolCallback | None = None,
        allow_sampling: bool = True,
        sampling_model: models.Model | None = None,
        max_retries: int = 1,
        elicitation_callback: ElicitationFnT | None = None,
        cache_prompts: bool = True,
        cache_tools: bool = True,
        cache_resources: bool = True,
        include_instructions: bool = False,
        include_return_schema: bool | None = None,
        id: str | None = None,
        client_info: mcp_types.Implementation | None = None,
    ):
        """Build a new MCP server.

        Args:
            command: The command to run.
            args: The arguments to pass to the command.
            env: The environment variables to set in the subprocess.
            cwd: The working directory to use when spawning the process.
            tool_prefix: A prefix to add to all tools that are registered with the server.
            log_level: The log level to set when connecting to the server, if any.
            log_handler: A handler for logging messages from the server.
            timeout: The timeout in seconds to wait for the client to initialize.
            read_timeout: Maximum time in seconds to wait for new messages before timing out.
            process_tool_call: Hook to customize tool calling and optionally pass extra metadata.
            allow_sampling: Whether to allow MCP sampling through this client.
            sampling_model: The model to use for sampling.
            max_retries: The maximum number of times to retry a tool call.
            elicitation_callback: Callback function to handle elicitation requests from the server.
            cache_prompts: Whether to cache the list of prompts.
                See [`MCPServer.cache_prompts`][pydantic_ai.mcp.MCPServer.cache_prompts].
            cache_tools: Whether to cache the list of tools.
                See [`MCPServer.cache_tools`][pydantic_ai.mcp.MCPServer.cache_tools].
            cache_resources: Whether to cache the list of resources.
                See [`MCPServer.cache_resources`][pydantic_ai.mcp.MCPServer.cache_resources].
            include_instructions: Whether to include the server's instructions in the agent's instructions.
                See [`MCPServer.include_instructions`][pydantic_ai.mcp.MCPServer.include_instructions].
            include_return_schema: Whether to include return schemas in tool definitions.
                See [`MCPServer.include_return_schema`][pydantic_ai.mcp.MCPServer.include_return_schema].
            id: An optional unique ID for the MCP server. An MCP server needs to have an ID in order to be used in a durable execution environment like Temporal, in which case the ID will be used to identify the server's activities within the workflow.
            client_info: Information describing the MCP client implementation.
        """
        self.command = command
        self.args = args
        self.env = env
        self.cwd = cwd

        super().__init__(
            tool_prefix,
            log_level,
            log_handler,
            timeout,
            read_timeout,
            process_tool_call,
            allow_sampling,
            sampling_model,
            max_retries,
            elicitation_callback,
            cache_prompts=cache_prompts,
            cache_tools=cache_tools,
            cache_resources=cache_resources,
            id=id,
            include_instructions=include_instructions,
            include_return_schema=include_return_schema,
            client_info=client_info,
        )

    @classmethod
    def __get_pydantic_core_schema__(cls, _: Any, __: Any) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            lambda dct: MCPServerStdio(**dct),  # pyright: ignore[reportDeprecated]
            core_schema.typed_dict_schema(
                {
                    'command': core_schema.typed_dict_field(core_schema.str_schema()),
                    'args': core_schema.typed_dict_field(core_schema.list_schema(core_schema.str_schema())),
                    'env': core_schema.typed_dict_field(
                        core_schema.dict_schema(core_schema.str_schema(), core_schema.str_schema()),
                        required=False,
                    ),
                }
            ),
        )

    @asynccontextmanager
    async def client_streams(
        self,
    ) -> AsyncGenerator[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
        ]
    ]:
        server = StdioServerParameters(command=self.command, args=list(self.args), env=self.env, cwd=self.cwd)
        async with stdio_client(server=server) as (read_stream, write_stream):
            yield read_stream, write_stream

    def __repr__(self) -> str:
        repr_args = [
            f'command={self.command!r}',
            f'args={self.args!r}',
        ]
        if self.id:
            repr_args.append(f'id={self.id!r}')  # pragma: lax no cover
        return f'{self.__class__.__name__}({", ".join(repr_args)})'

    def __eq__(self, value: object, /) -> bool:
        return (
            super().__eq__(value)
            and isinstance(value, MCPServerStdio)  # pyright: ignore[reportDeprecated]
            and self.command == value.command
            and self.args == value.args
            and self.env == value.env
            and self.cwd == value.cwd
        )


class _MCPServerHTTP(MCPServer):
    url: str
    """The URL of the endpoint on the MCP server."""

    headers: dict[str, Any] | None
    """Optional HTTP headers to be sent with each request to the endpoint.

    These headers will be passed directly to the underlying `httpx.AsyncClient`.
    Useful for authentication, custom headers, or other HTTP-specific configurations.

    !!! note
        You can either pass `headers` or `http_client`, but not both.

        See [`MCPServerHTTP.http_client`][pydantic_ai.mcp.MCPServerHTTP.http_client] for more information.
    """

    http_client: httpx.AsyncClient | None
    """An `httpx.AsyncClient` to use with the endpoint.

    This client may be configured to use customized connection parameters like self-signed certificates.

    !!! note
        You can either pass `headers` or `http_client`, but not both.

        If you want to use both, you can pass the headers to the `http_client` instead.

        ```python {py="3.10" test="skip"}
        import httpx

        from pydantic_ai.mcp import MCPServerSSE

        http_client = httpx.AsyncClient(headers={'Authorization': 'Bearer ...'})
        server = MCPServerSSE('http://localhost:3001/sse', http_client=http_client)
        ```
    """

    # last fields are re-defined from the parent class so they appear as fields
    tool_prefix: str | None
    log_level: mcp_types.LoggingLevel | None
    log_handler: LoggingFnT | None
    timeout: float
    read_timeout: float
    process_tool_call: ProcessToolCallback | None
    allow_sampling: bool
    sampling_model: models.Model | None
    max_retries: int
    elicitation_callback: ElicitationFnT | None = None
    cache_prompts: bool
    cache_tools: bool
    cache_resources: bool
    include_instructions: bool

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
        id: str | None = None,
        tool_prefix: str | None = None,
        log_level: mcp_types.LoggingLevel | None = None,
        log_handler: LoggingFnT | None = None,
        timeout: float = 5,
        read_timeout: float | None = None,
        process_tool_call: ProcessToolCallback | None = None,
        allow_sampling: bool = True,
        sampling_model: models.Model | None = None,
        max_retries: int = 1,
        elicitation_callback: ElicitationFnT | None = None,
        cache_prompts: bool = True,
        cache_tools: bool = True,
        cache_resources: bool = True,
        include_instructions: bool = False,
        include_return_schema: bool | None = None,
        client_info: mcp_types.Implementation | None = None,
        **_deprecated_kwargs: Any,
    ):
        """Build a new MCP server.

        Args:
            url: The URL of the endpoint on the MCP server.
            headers: Optional HTTP headers to be sent with each request to the endpoint.
            http_client: An `httpx.AsyncClient` to use with the endpoint.
            id: An optional unique ID for the MCP server. An MCP server needs to have an ID in order to be used in a durable execution environment like Temporal, in which case the ID will be used to identify the server's activities within the workflow.
            tool_prefix: A prefix to add to all tools that are registered with the server.
            log_level: The log level to set when connecting to the server, if any.
            log_handler: A handler for logging messages from the server.
            timeout: The timeout in seconds to wait for the client to initialize.
            read_timeout: Maximum time in seconds to wait for new messages before timing out.
            process_tool_call: Hook to customize tool calling and optionally pass extra metadata.
            allow_sampling: Whether to allow MCP sampling through this client.
            sampling_model: The model to use for sampling.
            max_retries: The maximum number of times to retry a tool call.
            elicitation_callback: Callback function to handle elicitation requests from the server.
            cache_prompts: Whether to cache the list of prompts.
                See [`MCPServer.cache_prompts`][pydantic_ai.mcp.MCPServer.cache_prompts].
            cache_tools: Whether to cache the list of tools.
                See [`MCPServer.cache_tools`][pydantic_ai.mcp.MCPServer.cache_tools].
            cache_resources: Whether to cache the list of resources.
                See [`MCPServer.cache_resources`][pydantic_ai.mcp.MCPServer.cache_resources].
            include_instructions: Whether to include the server's instructions in the agent's instructions.
                See [`MCPServer.include_instructions`][pydantic_ai.mcp.MCPServer.include_instructions].
            include_return_schema: Whether to include return schemas in tool definitions.
                See [`MCPServer.include_return_schema`][pydantic_ai.mcp.MCPServer.include_return_schema].
            client_info: Information describing the MCP client implementation.
        """
        if 'sse_read_timeout' in _deprecated_kwargs:
            if read_timeout is not None:
                raise TypeError("'read_timeout' and 'sse_read_timeout' cannot be set at the same time.")

            warnings.warn(
                "'sse_read_timeout' is deprecated, use 'read_timeout' instead.", DeprecationWarning, stacklevel=2
            )
            read_timeout = _deprecated_kwargs.pop('sse_read_timeout')

        _utils.validate_empty_kwargs(_deprecated_kwargs)

        if read_timeout is None:
            read_timeout = 5 * 60

        self.url = url
        self.headers = headers
        self.http_client = http_client

        super().__init__(
            tool_prefix=tool_prefix,
            log_level=log_level,
            log_handler=log_handler,
            timeout=timeout,
            read_timeout=read_timeout,
            process_tool_call=process_tool_call,
            allow_sampling=allow_sampling,
            sampling_model=sampling_model,
            max_retries=max_retries,
            elicitation_callback=elicitation_callback,
            cache_prompts=cache_prompts,
            cache_tools=cache_tools,
            cache_resources=cache_resources,
            include_instructions=include_instructions,
            include_return_schema=include_return_schema,
            id=id,
            client_info=client_info,
        )

    def __repr__(self) -> str:  # pragma: no cover
        repr_args = [
            f'url={self.url!r}',
        ]
        if self.id:
            repr_args.append(f'id={self.id!r}')
        return f'{self.__class__.__name__}({", ".join(repr_args)})'


@deprecated(
    '`MCPServerSSE` is deprecated and will be removed in v2. '
    "Use `MCPToolset('http://.../sse')` instead — the SSE transport is automatically inferred "
    'from URLs ending in `/sse`.'
)
class MCPServerSSE(_MCPServerHTTP):
    """An MCP server that connects over streamable HTTP connections.

    This class implements the SSE transport from the MCP specification.
    See <https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#http-with-sse> for more information.

    !!! note
        Using this class as an async context manager will create a new pool of HTTP connections to connect
        to a server which should already be running.

    Example:
    ```python {py="3.10"}
    from pydantic_ai import Agent
    from pydantic_ai.mcp import MCPServerSSE

    server = MCPServerSSE('http://localhost:3001/sse')
    agent = Agent('openai:gpt-5.2', toolsets=[server])
    ```
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, _: Any, __: Any) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            lambda dct: MCPServerSSE(**dct),  # pyright: ignore[reportDeprecated]
            core_schema.typed_dict_schema(
                {
                    'url': core_schema.typed_dict_field(core_schema.str_schema()),
                    'headers': core_schema.typed_dict_field(
                        core_schema.dict_schema(core_schema.str_schema(), core_schema.str_schema()), required=False
                    ),
                }
            ),
        )

    # sse_client has a hang bug (https://github.com/modelcontextprotocol/python-sdk/issues/1811)
    # that prevents testing SSE transport in CI.
    # TODO: Remove pragma and add a test
    # once https://github.com/modelcontextprotocol/python-sdk/pull/1838 is released.
    @asynccontextmanager
    async def client_streams(  # pragma: no cover
        self,
    ) -> AsyncGenerator[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
        ]
    ]:
        if self.http_client and self.headers:
            raise ValueError('`http_client` is mutually exclusive with `headers`.')

        if self.http_client is not None:

            def httpx_client_factory(
                headers: dict[str, str] | None = None,
                timeout: httpx.Timeout | None = None,
                auth: httpx.Auth | None = None,
            ) -> httpx.AsyncClient:
                assert self.http_client is not None
                return self.http_client

            async with sse_client(
                url=self.url,
                timeout=self.timeout,
                sse_read_timeout=self.read_timeout,
                httpx_client_factory=httpx_client_factory,
            ) as (read_stream, write_stream, *_):
                yield read_stream, write_stream
        else:
            async with sse_client(
                url=self.url,
                timeout=self.timeout,
                sse_read_timeout=self.read_timeout,
                headers=self.headers,
            ) as (read_stream, write_stream, *_):
                yield read_stream, write_stream

    def __eq__(self, value: object, /) -> bool:
        return super().__eq__(value) and isinstance(value, MCPServerSSE) and self.url == value.url  # pyright: ignore[reportDeprecated]


# Subclassing a `@deprecated` class emits a `DeprecationWarning` at class-creation time, which is
# fired the moment `pydantic_ai.mcp` is imported. Suppress it locally — the deprecation is
# intentional and `MCPServerHTTP` itself is also `@deprecated`, so users still see the warning
# when *they* construct or import it.
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    @deprecated('The `MCPServerHTTP` class is deprecated, use `MCPServerSSE` instead.')
    class MCPServerHTTP(MCPServerSSE):  # pyright: ignore[reportDeprecated]
        """An MCP server that connects over HTTP using the old SSE transport.

        This class implements the SSE transport from the MCP specification.
        See <https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#http-with-sse> for more information.

        !!! note
            Using this class as an async context manager will create a new pool of HTTP connections to connect
            to a server which should already be running.

        Example:
        ```python {py="3.10" test="skip"}
        from pydantic_ai import Agent
        from pydantic_ai.mcp import MCPServerHTTP

        server = MCPServerHTTP('http://localhost:3001/sse')
        agent = Agent('openai:gpt-5.2', toolsets=[server])
        ```
        """


@deprecated(
    '`MCPServerStreamableHTTP` is deprecated and will be removed in v2. '
    "Use `MCPToolset('http://.../mcp')` instead — Streamable HTTP is the default for HTTP URLs."
)
class MCPServerStreamableHTTP(_MCPServerHTTP):
    """An MCP server that connects over HTTP using the Streamable HTTP transport.

    This class implements the Streamable HTTP transport from the MCP specification.
    See <https://modelcontextprotocol.io/introduction#streamable-http> for more information.

    !!! note
        Using this class as an async context manager will create a new pool of HTTP connections to connect
        to a server which should already be running.

    Example:
    ```python {py="3.10"}
    from pydantic_ai import Agent
    from pydantic_ai.mcp import MCPServerStreamableHTTP

    server = MCPServerStreamableHTTP('http://localhost:8000/mcp')
    agent = Agent('openai:gpt-5.2', toolsets=[server])
    ```
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, _: Any, __: Any) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            lambda dct: MCPServerStreamableHTTP(**dct),  # pyright: ignore[reportDeprecated]
            core_schema.typed_dict_schema(
                {
                    'url': core_schema.typed_dict_field(core_schema.str_schema()),
                    'headers': core_schema.typed_dict_field(
                        core_schema.dict_schema(core_schema.str_schema(), core_schema.str_schema()), required=False
                    ),
                }
            ),
        )

    @asynccontextmanager
    async def client_streams(
        self,
    ) -> AsyncGenerator[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
        ]
    ]:
        if self.http_client and self.headers:
            raise ValueError('`http_client` is mutually exclusive with `headers`.')

        aexit_stack = AsyncExitStack()
        http_client = self.http_client or await aexit_stack.enter_async_context(
            httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, read=self.read_timeout), headers=self.headers)
        )
        read_stream, write_stream, *_ = await aexit_stack.enter_async_context(
            streamable_http_client(self.url, http_client=http_client)
        )
        try:
            yield read_stream, write_stream
        finally:
            await aexit_stack.aclose()

    def __eq__(self, value: object, /) -> bool:
        return super().__eq__(value) and isinstance(value, MCPServerStreamableHTTP) and self.url == value.url  # pyright: ignore[reportDeprecated]


ToolResult = (
    str
    | messages.BinaryContent
    | dict[str, Any]
    | list[Any]
    | Sequence[str | messages.BinaryContent | dict[str, Any] | list[Any]]
)
"""The result type of an MCP tool call."""


class CallToolFunc(Protocol):
    """A callable that invokes an MCP tool — typically `MCPToolset.direct_call_tool` or its legacy equivalent.

    Passed to user-defined [`ProcessToolCallback`][pydantic_ai.mcp.ProcessToolCallback] functions as
    the underlying call hook. `metadata` is keyword-only — pass it as
    `await call_tool(name, args, metadata=...)`.
    """

    async def __call__(
        self,
        name: str,
        args: dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult: ...


ProcessToolCallback = Callable[
    [
        RunContext[Any],
        CallToolFunc,
        str,
        dict[str, Any],
    ],
    Awaitable[ToolResult],
]
"""A process tool callback.

It accepts a run context, the original tool call function, a tool name, and arguments.

Allows wrapping an MCP server tool call to customize it, including adding extra request
metadata.
"""


# String forward-reference: the union references names that are only resolvable at runtime when
# fastmcp is installed, and `TypeAlias = ...` is evaluated eagerly at module-import time regardless
# of `from __future__ import annotations`. Stringifying the RHS lets `pydantic_ai.mcp` keep loading
# (so the legacy `MCPServer*` classes stay importable on bare-`mcp`-SDK installs without fastmcp).
MCPToolsetClient: TypeAlias = 'FastMCPClient[Any] | ClientTransport | FastMCP | FastMCP1Server | AnyUrl | Path | str'
"""Anything `MCPToolset` accepts as its `client` argument — a pre-built `fastmcp.Client`, a FastMCP
`ClientTransport`, an in-process `FastMCP` server, an `AnyUrl`/URL string, a script `Path`, or a
URL/path/script string.

For multi-server JSON config files, use [`load_mcp_toolsets`][pydantic_ai.mcp.load_mcp_toolsets]
instead — it expands env vars and constructs one `MCPToolset` per server entry."""


_UNSET: Any = object()
"""Sentinel for `MCPToolset.__init__` to distinguish "not passed" from "passed `None`/default value"
when validating that no kwargs were passed alongside a pre-built `fastmcp.Client`. Using a sentinel
keeps the conflict checks in sync with the actual default values, so changing a default doesn't
silently break the conflict check."""


@dataclass(init=False, repr=False)
class MCPToolset(AbstractToolset[AgentDepsT]):
    """A toolset for connecting to an MCP server.

    `MCPToolset` is the recommended way to use [Model Context Protocol](https://modelcontextprotocol.io)
    servers in Pydantic AI. It is built on the [FastMCP](https://gofastmcp.com) `Client`, which
    supports the full MCP protocol — tools, resources, sampling, elicitation, OAuth — and a wide
    range of transports (HTTP, SSE, stdio, in-process FastMCP servers, multi-server configs).

    Pass any input that FastMCP can build a transport from — a URL, a script path, a `FastMCP`
    server instance for in-process testing — or a pre-built `fastmcp.Client` for full control over
    its configuration. For multi-server JSON config files, use
    [`load_mcp_toolsets`][pydantic_ai.mcp.load_mcp_toolsets] instead.

    Example — connect to a streamable-HTTP MCP server:

    ```python {test="skip"}
    from pydantic_ai import Agent
    from pydantic_ai.mcp import MCPToolset

    toolset = MCPToolset('http://localhost:8000/mcp')
    agent = Agent('openai:gpt-5', toolsets=[toolset])
    ```

    Example — connect to a local stdio MCP server:

    ```python {test="skip"}
    from pydantic_ai.mcp import MCPToolset

    toolset = MCPToolset('my_mcp_server.py')
    ```

    Example — pass a pre-built FastMCP Client for full configuration control:

    ```python {test="skip"}
    from fastmcp.client import Client
    from fastmcp.client.transports import StreamableHttpTransport

    from pydantic_ai.mcp import MCPToolset

    client = Client(StreamableHttpTransport('http://localhost:8000/mcp'), auth='oauth')
    toolset = MCPToolset(client)
    ```
    """

    client: FastMCPClient[Any]
    """The underlying FastMCP `Client`. Always normalized to a `fastmcp.Client` regardless of how
    the toolset was constructed."""

    tool_error_behavior: Literal['retry', 'error']
    """How to handle tool errors raised by the server.

    `'retry'` (default) raises [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] so the model can
    self-correct; `'error'` propagates the underlying `fastmcp.exceptions.ToolError` to the caller.
    """

    max_retries: int | None
    """Maximum number of times a tool call may be retried after a `ModelRetry`.

    `None` (default) inherits the agent's retry count at runtime. Set explicitly to override.
    """

    cache_tools: bool
    """Whether to cache the list of tools across `get_tools()` calls.

    When enabled (default), tools are fetched once and cached until either:

    - The server sends a `notifications/tools/list_changed` notification
    - The toolset is fully exited (last `__aexit__` matches the first `__aenter__`)

    Set to `False` for servers that change tools dynamically without sending notifications, or when
    passing a pre-built FastMCP Client (the cache-invalidation message handler isn't installed in
    that case, so caches are only invalidated by session close).
    """

    cache_resources: bool
    """Whether to cache the list of resources across `list_resources()` calls.

    Same semantics as [`cache_tools`][pydantic_ai.mcp.MCPToolset.cache_tools] but for
    `notifications/resources/list_changed` notifications.
    """

    cache_prompts: bool
    """Whether to cache the list of prompts across `list_prompts()` calls.

    Same semantics as [`cache_tools`][pydantic_ai.mcp.MCPToolset.cache_tools] but for
    `notifications/prompts/list_changed` notifications.
    """

    include_instructions: bool
    """Whether to include the server's `initialize` instructions string in the agent's instruction set.

    Defaults to `False` for backward compatibility. When `True`, the instructions returned by the
    server during initialization are added to the agent's instructions.
    """

    include_return_schema: bool | None
    """Whether to include each tool's `outputSchema` in the schema sent to the model.

    When `None` (the default), defaults to `False` unless the
    [`IncludeToolReturnSchemas`][pydantic_ai.capabilities.IncludeToolReturnSchemas] capability is
    used.
    """

    process_tool_call: ProcessToolCallback | None
    """Hook to wrap tool calls — useful for adding request-level metadata, custom retry policies,
    or telemetry. See [`ProcessToolCallback`][pydantic_ai.mcp.ProcessToolCallback].
    """

    sampling_model: models.Model | None
    """A Pydantic AI model that the server may sample from via the MCP `sampling/createMessage` flow.

    When set (and no explicit `sampling_handler` is passed), Pydantic AI builds a sampling handler
    that delegates to this model with the request's `maxTokens`/`temperature`/`stopSequences`
    settings applied. If both `sampling_model` and `sampling_handler` are passed, an error is raised.
    """

    log_level: mcp_types.LoggingLevel | None
    """Log level requested from the server via `logging/setLevel` after initialization.

    `None` (default) leaves the server's default log level alone. Combine with `log_handler` to
    receive log messages.
    """

    _id: str | None
    _server_info: mcp_types.Implementation | None
    _server_capabilities: ServerCapabilities | None
    _instructions: str | None
    _cached_tools: list[mcp_types.Tool] | None
    _cached_resources: list[Resource] | None
    _cached_prompts: list[Prompt] | None
    _running_count: int
    _exit_stack: AsyncExitStack | None
    _user_message_handler: MessageHandlerT | None

    @functools.cached_property
    def _enter_lock(self) -> anyio.Lock:
        # `anyio.Lock` binds to the event loop on which it's first used; deferring creation to first
        # access ensures it binds to the running loop and avoids issues with Temporal's workflow sandbox.
        return anyio.Lock()

    def __init__(
        self,
        client: MCPToolsetClient,
        *,
        # Pydantic AI-layer config
        id: str | None = None,
        max_retries: int | None = None,
        tool_error_behavior: Literal['retry', 'error'] = 'retry',
        process_tool_call: ProcessToolCallback | None = None,
        cache_tools: bool = True,
        cache_resources: bool = True,
        cache_prompts: bool = True,
        include_instructions: bool = False,
        include_return_schema: bool | None = None,
        # Sampling — high-level shortcut and low-level escape hatch
        sampling_model: models.Model | None = None,
        sampling_handler: SamplingHandler[Any, Any] | None = None,
        # MCP protocol kwargs (forwarded to a default FastMCP Client when one isn't passed)
        elicitation_handler: ElicitationHandler[Any, Any] | None = None,
        log_handler: LogHandler | None = None,
        log_level: mcp_types.LoggingLevel | None = None,
        progress_handler: ProgressHandler | None = None,
        message_handler: MessageHandlerT | None = None,
        client_info: mcp_types.Implementation | None = None,
        init_timeout: float | None = _UNSET,
        read_timeout: float | None = _UNSET,
        roots: RootsList | RootsHandler[Any] | None = None,
        # HTTP-specific (only used when constructing a default transport from a URL)
        auth: httpx.Auth | Literal['oauth'] | str | None = None,
        verify: ssl.SSLContext | bool | str | None = None,
        headers: dict[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        """Build a new `MCPToolset`.

        Args:
            client: How to connect to the MCP server. See the class docstring for accepted shapes.
            id: An optional unique identifier for this toolset. Required for use in durable execution
                environments like Temporal or DBOS, where it identifies the toolset's activities/steps
                within a workflow.
            max_retries: Maximum number of times a tool call may be retried after a `ModelRetry`.
                `None` inherits the agent's retry count at runtime.
            tool_error_behavior: `'retry'` (default) raises
                [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] on tool errors so the model can
                self-correct; `'error'` propagates the underlying exception.
            process_tool_call: Hook to wrap tool calls. See
                [`ProcessToolCallback`][pydantic_ai.mcp.ProcessToolCallback].
            cache_tools: Whether to cache the list of tools. See
                [`MCPToolset.cache_tools`][pydantic_ai.mcp.MCPToolset.cache_tools].
            cache_resources: Whether to cache the list of resources. See
                [`MCPToolset.cache_resources`][pydantic_ai.mcp.MCPToolset.cache_resources].
            cache_prompts: Whether to cache the list of prompts. See
                [`MCPToolset.cache_prompts`][pydantic_ai.mcp.MCPToolset.cache_prompts].
            include_instructions: Whether to include the server's instructions in the agent's
                instructions. See
                [`MCPToolset.include_instructions`][pydantic_ai.mcp.MCPToolset.include_instructions].
            include_return_schema: Whether to include return schemas in tool definitions. See
                [`MCPToolset.include_return_schema`][pydantic_ai.mcp.MCPToolset.include_return_schema].
            sampling_model: A Pydantic AI model the server may sample from. Mutually exclusive with
                `sampling_handler`.
            sampling_handler: A FastMCP-shaped sampling handler. Use for full control over the
                sampling response.
            elicitation_handler: A FastMCP-shaped elicitation handler that receives MCP
                `elicitation/create` requests from the server.
            log_handler: A FastMCP-shaped log handler that receives log messages from the server.
            log_level: Log level requested from the server via `logging/setLevel` after
                initialization.
            progress_handler: A FastMCP-shaped progress handler.
            message_handler: A FastMCP-shaped message handler called for every server-sent message.
                Pydantic AI installs its own message handler internally to invalidate caches on
                `list_changed` notifications; if you provide one, both run (yours after ours).
            client_info: Information describing the MCP client implementation, sent to the server
                during initialization.
            init_timeout: Timeout in seconds for the initial connection and `initialize` handshake.
            read_timeout: Maximum time in seconds to wait for new messages on the long-lived
                connection. Defaults to 5 minutes.
            roots: Filesystem roots advertised to the server.
            auth: HTTP authentication for HTTP transports — an `httpx.Auth`, the literal string
                `'oauth'` to enable FastMCP's OAuth flow, or a bearer-token string.
            verify: SSL verification mode for HTTP transports — an `ssl.SSLContext`, a CA bundle
                path string, or a bool.
            headers: Extra HTTP headers for HTTP transports. Mutually exclusive with `http_client`.
            http_client: A pre-configured `httpx.AsyncClient` to use for HTTP transports — useful
                for self-signed certificates or custom connection pooling. Mutually exclusive with
                `headers`.

        Raises:
            ValueError: If a pre-built `fastmcp.Client` is passed alongside any of the kwargs that
                would otherwise build a default Client (sampling, elicitation, headers, etc.), or
                if `sampling_model` and `sampling_handler` are both passed, or if `headers` and
                `http_client` are both passed.
            ImportError: If the fastmcp client isn't installed. Install the `mcp` extra (which pulls
                `fastmcp-slim[client]`): `pip install "pydantic-ai-slim[mcp]"`.
        """
        _require_fastmcp()
        if isinstance(client, FastMCPClient):
            forwarded_values: dict[str, Any] = {
                'sampling_handler': sampling_handler,
                'sampling_model': sampling_model,
                'elicitation_handler': elicitation_handler,
                'log_handler': log_handler,
                'progress_handler': progress_handler,
                'message_handler': message_handler,
                'client_info': client_info,
                'roots': roots,
                'auth': auth,
                'verify': verify,
                'headers': headers,
                'http_client': http_client,
            }
            conflicts = [name for name, value in forwarded_values.items() if value is not None]
            # `init_timeout`/`read_timeout` use `_UNSET` as their default so we can detect "passed
            # explicitly" vs "default" without coupling to the literal default values.
            if init_timeout is not _UNSET:
                conflicts.append('init_timeout')
            if read_timeout is not _UNSET:
                conflicts.append('read_timeout')
            if conflicts:
                names = ', '.join(repr(n) for n in conflicts)
                raise ValueError(
                    f'Cannot pass {names} alongside a pre-built `fastmcp.Client` — '
                    'configure these on the Client itself instead.'
                )
            self.client = client
            self._user_message_handler = None
        else:
            if sampling_handler is not None and sampling_model is not None:
                raise ValueError('Pass either `sampling_model` or `sampling_handler`, not both.')
            if headers is not None and http_client is not None:
                raise ValueError(
                    '`headers` and `http_client` are mutually exclusive — set headers on the `http_client` instead.'
                )

            # Resolve sentinels to actual defaults now that the conflict check has run.
            if init_timeout is _UNSET:
                init_timeout = 5
            if read_timeout is _UNSET:
                read_timeout = 5 * 60

            transport = _build_transport(
                client,
                headers=headers,
                http_client=http_client,
                auth=auth,
                verify=verify,
                read_timeout=read_timeout,
            )
            resolved_sampling_handler = sampling_handler
            if resolved_sampling_handler is None and sampling_model is not None:
                resolved_sampling_handler = _build_sampling_handler(sampling_model)

            wrapped_message_handler = _build_message_handler(self, message_handler)

            self.client = FastMCPClient[Any](
                transport=transport,
                sampling_handler=resolved_sampling_handler,
                elicitation_handler=elicitation_handler,
                log_handler=log_handler,
                progress_handler=progress_handler,
                message_handler=wrapped_message_handler,
                client_info=client_info,
                init_timeout=init_timeout,
                timeout=read_timeout,
                roots=roots,
            )
            self._user_message_handler = message_handler

        self._id = id
        self.max_retries = max_retries
        self.tool_error_behavior = tool_error_behavior
        self.process_tool_call = process_tool_call
        self.cache_tools = cache_tools
        self.cache_resources = cache_resources
        self.cache_prompts = cache_prompts
        self.include_instructions = include_instructions
        self.include_return_schema = include_return_schema
        self.sampling_model = sampling_model
        self.log_level = log_level

        self._server_info = None
        self._server_capabilities = None
        self._instructions = None
        self._cached_tools = None
        self._cached_resources = None
        self._cached_prompts = None
        self._running_count = 0
        self._exit_stack = None

    @property
    def id(self) -> str | None:
        return self._id

    @id.setter
    def id(self, value: str | None) -> None:
        self._id = value

    @property
    def label(self) -> str:
        if self.id:
            return super().label  # pragma: no cover
        return repr(self)

    @property
    def tool_name_conflict_hint(self) -> str:
        return 'Wrap the toolset with `.prefixed("...")` to disambiguate tool names from multiple MCP servers.'

    @property
    def server_info(self) -> mcp_types.Implementation:
        """The server-implementation info sent during initialization.

        Raises [`AttributeError`][AttributeError] when accessed before the toolset has been entered.
        """
        if self._server_info is None:
            raise AttributeError(f'`{self.__class__.__name__}.server_info` is only available after initialization.')
        return self._server_info

    @property
    def capabilities(self) -> ServerCapabilities:
        """The capabilities advertised by the server during initialization.

        Raises [`AttributeError`][AttributeError] when accessed before the toolset has been entered.
        """
        if self._server_capabilities is None:
            raise AttributeError(f'`{self.__class__.__name__}.capabilities` is only available after initialization.')
        return self._server_capabilities

    @property
    def instructions(self) -> str | None:
        """The instructions sent by the server during initialization.

        Raises [`AttributeError`][AttributeError] when accessed before the toolset has been entered.
        """
        if not self._initialized:
            raise AttributeError(f'`{self.__class__.__name__}.instructions` is only available after initialization.')
        return self._instructions

    @property
    def is_running(self) -> bool:
        """Whether the toolset is currently entered (the FastMCP session is open)."""
        return self._running_count > 0

    @property
    def _initialized(self) -> bool:
        return self._server_info is not None

    def _invalidate_tools_cache(self) -> None:
        self._cached_tools = None

    def _invalidate_resources_cache(self) -> None:
        self._cached_resources = None

    def _invalidate_prompts_cache(self) -> None:
        self._cached_prompts = None

    async def __aenter__(self) -> Self:
        async with self._enter_lock:
            if self._running_count == 0:
                # Build the exit stack inside an `async with` so any failure after
                # `enter_async_context(self.client)` cleans up the open session — only commit the
                # stack and write `_server_info`/`_server_capabilities`/`_instructions` to `self`
                # once initialization fully succeeds, so `_initialized` can't see stale data from a
                # session that got torn down mid-setup.
                async with AsyncExitStack() as exit_stack:
                    await exit_stack.enter_async_context(self.client)
                    init_result = self.client.initialize_result
                    assert init_result is not None, 'FastMCP Client initialization returned no result'
                    server_info = init_result.serverInfo
                    server_capabilities = ServerCapabilities.from_mcp_sdk(init_result.capabilities)
                    instructions = init_result.instructions
                    if self.log_level is not None:
                        await self.client.session.set_logging_level(self.log_level)
                    self._exit_stack = exit_stack.pop_all()
                    self._server_info = server_info
                    self._server_capabilities = server_capabilities
                    self._instructions = instructions
            self._running_count += 1
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        async with self._enter_lock:
            if self._running_count == 0:
                raise ValueError(f'`{self.__class__.__name__}.__aexit__` called more times than `__aenter__`')
            self._running_count -= 1
            if self._running_count == 0 and self._exit_stack is not None:
                await self._exit_stack.aclose()
                self._exit_stack = None
                self._server_info = None
                self._server_capabilities = None
                self._instructions = None
                self._cached_tools = None
                self._cached_resources = None
                self._cached_prompts = None
        return None

    async def get_instructions(self, ctx: RunContext[AgentDepsT]) -> messages.InstructionPart | None:
        """Return the server's instructions if `include_instructions` is enabled."""
        if not self.include_instructions:
            return None
        if not self._initialized or self._instructions is None:
            return None
        # Instructions are captured once during `__aenter__` and don't change across runs while
        # the toolset stays entered — so they're static from the agent's perspective, not dynamic.
        return messages.InstructionPart(content=self._instructions, dynamic=False)

    async def list_tools(self) -> list[mcp_types.Tool]:
        """Retrieve the tools currently exposed by the server.

        When [`cache_tools`][pydantic_ai.mcp.MCPToolset.cache_tools] is enabled (default), results
        are cached and invalidated by `notifications/tools/list_changed` or the toolset's last
        `__aexit__`.
        """
        if self.cache_tools and self._cached_tools is not None:
            return self._cached_tools
        async with self:
            tools = await self.client.list_tools()
            if self.cache_tools:
                self._cached_tools = tools
            return tools

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        max_retries = self.max_retries if self.max_retries is not None else ctx.max_retries
        tools: dict[str, ToolsetTool[AgentDepsT]] = {}
        for mcp_tool in await self.list_tools():
            task_support = mcp_tool.execution.taskSupport if mcp_tool.execution else None
            tools[mcp_tool.name] = ToolsetTool[AgentDepsT](
                toolset=self,
                tool_def=ToolDefinition(
                    name=mcp_tool.name,
                    description=mcp_tool.description,
                    parameters_json_schema=mcp_tool.inputSchema,
                    metadata={
                        'meta': mcp_tool.meta,
                        'annotations': mcp_tool.annotations.model_dump() if mcp_tool.annotations else None,
                        'task': task_support in ('required', 'optional'),
                    },
                    return_schema=mcp_tool.outputSchema or None,
                    include_return_schema=self.include_return_schema,
                ),
                max_retries=max_retries,
                args_validator=TOOL_SCHEMA_VALIDATOR,
            )
        return tools

    def tool_for_tool_def(self, tool_def: ToolDefinition) -> ToolsetTool[AgentDepsT]:
        return ToolsetTool[AgentDepsT](
            toolset=self,
            tool_def=tool_def,
            max_retries=self.max_retries if self.max_retries is not None else 1,
            args_validator=TOOL_SCHEMA_VALIDATOR,
        )

    async def direct_call_tool(
        self,
        name: str,
        args: dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
        use_task: bool = False,
    ) -> Any:
        """Call a tool on the server directly.

        Args:
            name: The name of the tool to call.
            args: The arguments to pass to the tool.
            metadata: Optional request-level `_meta` payload sent alongside the call.
            use_task: When `True`, send the call with `task=True` per MCP
                [SEP-1686](https://modelcontextprotocol.io/specification/2025-11-25/basic/utilities/tasks) so
                the server wraps execution in a durable, cancelable, pollable task; the result is awaited via
                `tasks/result`. Only valid for tools whose `execution.taskSupport` is `'required'` or `'optional'`.

        Raises:
            ModelRetry: If the tool errors and `tool_error_behavior='retry'` (the default).
            fastmcp.exceptions.ToolError: If the tool errors and `tool_error_behavior='error'`.
        """
        async with self:
            try:
                if use_task:
                    tool_task: ToolTask = await self.client.call_tool(
                        name=name, arguments=args, task=True, meta=metadata
                    )
                    result: CallToolResult = await tool_task.result()
                else:
                    result = await self.client.call_tool(name=name, arguments=args, meta=metadata)
            except ToolError as e:
                if self.tool_error_behavior == 'retry':
                    raise exceptions.ModelRetry(message=str(e)) from e
                raise

        # Prefer structured content if all parts are text (per the docs they contain the JSON-encoded
        # structured content for backward compatibility).
        # See https://github.com/modelcontextprotocol/python-sdk#structured-output
        if (structured := result.structured_content) and all(
            isinstance(part, mcp_types.TextContent) for part in result.content
        ):
            # The MCP SDK wraps primitives and generic types like list in a `result` key, but we want
            # the raw value returned by the tool function.
            if isinstance(structured, dict) and len(structured) == 1 and 'result' in structured:
                return structured['result']
            return structured

        return _map_mcp_tool_results(result.content)

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
    ) -> Any:
        # Server-side task-augmented execution per MCP SEP-1686 is governed entirely by the tool's
        # `execution.taskSupport`: 'required'/'optional' → task path; 'forbidden' or absent → regular path.
        use_task = bool((tool.tool_def.metadata or {}).get('task'))
        if self.process_tool_call is not None:
            return await self.process_tool_call(
                ctx, functools.partial(self.direct_call_tool, use_task=use_task), name, tool_args
            )
        return await self.direct_call_tool(name, tool_args, use_task=use_task)

    async def list_prompts(self) -> list[Prompt]:
        """Retrieve the prompts currently exposed by the server.

        When [`cache_prompts`][pydantic_ai.mcp.MCPToolset.cache_prompts] is enabled (default),
        results are cached and invalidated by `notifications/prompts/list_changed` or the
        toolset's last `__aexit__`.

        Returns an empty list if the server does not advertise the `prompts` capability.

        Raises:
            MCPError: If the server returns an error.
        """
        if self.cache_prompts and self._cached_prompts is not None:
            return self._cached_prompts
        async with self:
            if not self.capabilities.prompts:
                return []
            try:
                mcp_prompts = await self.client.list_prompts()
            except mcp_exceptions.McpError as e:
                raise MCPError.from_mcp_sdk(e) from e
            prompts = [Prompt.from_mcp_sdk(p) for p in mcp_prompts]
            if self.cache_prompts:
                self._cached_prompts = prompts
            return prompts

    async def get_prompt(self, name: str, arguments: dict[str, str] | None = None) -> PromptResult:
        """Retrieve a specific prompt from the server, optionally parameterized.

        Args:
            name: The name of the prompt to retrieve.
            arguments: Arguments to parameterize the prompt, if applicable.

        Raises:
            MCPError: If the server doesn't advertise the `prompts` capability, or if it returns
                an error response.
        """
        async with self:
            if not self.capabilities.prompts:
                raise MCPError(
                    message=f'Server does not advertise the `prompts` capability; cannot get prompt {name!r}.',
                    code=-32601,
                )
            try:
                result = await self.client.get_prompt(name, arguments)
            except mcp_exceptions.McpError as e:
                raise MCPError.from_mcp_sdk(e) from e
            return PromptResult(
                description=result.description,
                metadata=result.meta,
                messages=[
                    PromptMessage(role=msg.role, content=_map_mcp_prompt_part(msg.content)) for msg in result.messages
                ],
            )

    async def list_resources(self) -> list[Resource]:
        """Retrieve the resources currently exposed by the server.

        When [`cache_resources`][pydantic_ai.mcp.MCPToolset.cache_resources] is enabled (default),
        results are cached and invalidated by `notifications/resources/list_changed` or the
        toolset's last `__aexit__`.

        Returns an empty list if the server does not advertise the `resources` capability.

        Raises:
            MCPError: If the server returns an error.
        """
        if self.cache_resources and self._cached_resources is not None:
            return self._cached_resources
        async with self:
            if not self.capabilities.resources:
                return []
            try:
                mcp_resources = await self.client.list_resources()
            except mcp_exceptions.McpError as e:
                raise MCPError.from_mcp_sdk(e) from e
            resources = [Resource.from_mcp_sdk(r) for r in mcp_resources]
            if self.cache_resources:
                self._cached_resources = resources
            return resources

    async def list_resource_templates(self) -> list[ResourceTemplate]:
        """Retrieve the resource templates currently exposed by the server.

        Returns an empty list if the server does not advertise the `resources` capability.

        Raises:
            MCPError: If the server returns an error.
        """
        async with self:
            if not self.capabilities.resources:
                return []
            try:
                mcp_templates = await self.client.list_resource_templates()
            except mcp_exceptions.McpError as e:
                raise MCPError.from_mcp_sdk(e) from e
        return [ResourceTemplate.from_mcp_sdk(t) for t in mcp_templates]

    @overload
    async def read_resource(self, uri: str) -> str | messages.BinaryContent | list[str | messages.BinaryContent]: ...

    @overload
    async def read_resource(
        self, uri: Resource
    ) -> str | messages.BinaryContent | list[str | messages.BinaryContent]: ...

    async def read_resource(
        self, uri: str | Resource
    ) -> str | messages.BinaryContent | list[str | messages.BinaryContent]:
        """Read the contents of a specific resource by URI.

        Args:
            uri: The URI of the resource to read, or a [`Resource`][pydantic_ai.mcp.Resource] object.

        Returns:
            The resource contents — a single value if the resource has one content item, or a list
            otherwise. Text content is returned as `str`, binary content as
            [`BinaryContent`][pydantic_ai.messages.BinaryContent].

        Raises:
            MCPError: If the server returns an error.
        """
        resource_uri = uri if isinstance(uri, str) else uri.uri
        async with self:
            try:
                contents = await self.client.read_resource(AnyUrl(resource_uri))
            except mcp_exceptions.McpError as e:
                raise MCPError.from_mcp_sdk(e) from e

        return (
            _resource_content_to_pai(contents[0])
            if len(contents) == 1
            else [_resource_content_to_pai(c) for c in contents]
        )

    def __repr__(self) -> str:
        repr_args = [f'client={self.client!r}']
        if self._id is not None:
            repr_args.append(f'id={self._id!r}')
        return f'{self.__class__.__name__}({", ".join(repr_args)})'

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, MCPToolset) and self._id == value._id and self.client is value.client

    def __hash__(self) -> int:
        return hash((self._id, id(self.client)))


def _build_message_handler(toolset: MCPToolset[Any], user_handler: MessageHandlerT | None) -> MessageHandlerT:
    """Wrap a user message handler so we invalidate `MCPToolset` caches on `list_changed` notifications.

    The toolset's own cache invalidation runs first, then the user-supplied handler (if any).
    """

    async def handler(message: Any) -> None:
        if isinstance(message, mcp_types.ServerNotification):
            if isinstance(message.root, mcp_types.ToolListChangedNotification):
                toolset._invalidate_tools_cache()  # pyright: ignore[reportPrivateUsage]
            elif isinstance(message.root, mcp_types.ResourceListChangedNotification):
                toolset._invalidate_resources_cache()  # pyright: ignore[reportPrivateUsage]
            elif isinstance(message.root, mcp_types.PromptListChangedNotification):
                toolset._invalidate_prompts_cache()  # pyright: ignore[reportPrivateUsage]
        if user_handler is not None:
            await user_handler(message)

    return handler


def _build_transport(
    client: MCPToolsetClient,
    *,
    headers: dict[str, str] | None,
    http_client: httpx.AsyncClient | None,
    auth: httpx.Auth | Literal['oauth'] | str | None,
    verify: ssl.SSLContext | bool | str | None,
    read_timeout: float | None,
) -> MCPToolsetClient:
    """Build a FastMCP transport from a flexible input.

    For URL-shaped inputs combined with HTTP-specific kwargs, we construct the transport explicitly
    so the kwargs take effect (FastMCP's `Client(url, ...)` doesn't forward HTTP kwargs to its
    auto-inferred transport). For everything else, we pass the input through and let FastMCP's
    `Client` infer the transport.
    """
    needs_explicit_http = headers is not None or http_client is not None or auth is not None or verify is not None
    is_url = isinstance(client, AnyUrl) or (isinstance(client, str) and client.startswith(('http://', 'https://')))
    if needs_explicit_http and not is_url:
        raise ValueError(
            '`headers`, `http_client`, `auth`, and `verify` only apply to HTTP transports built '
            'from a URL string. Pass them on your transport / `fastmcp.Client` directly instead.'
        )
    if not needs_explicit_http:
        return client
    url = str(client)
    # FastMCP's HTTP transports accept `httpx_client_factory`; adapt `http_client` to that shape.
    factory = _make_httpx_client_factory(http_client) if http_client is not None else None
    if infer_transport_type_from_url(url) == 'sse':
        return SSETransport(
            url=url,
            headers=headers,
            auth=auth,
            verify=verify,
            # SSE keeps its own read timeout for the long-lived event stream.
            sse_read_timeout=read_timeout if read_timeout is not None else 5 * 60,
            httpx_client_factory=factory,
        )
    # `sse_read_timeout` is deprecated on StreamableHttpTransport; the read timeout for the
    # long-lived session is configured via the FastMCP `Client(timeout=...)` instead.
    return StreamableHttpTransport(
        url=url,
        headers=headers,
        auth=auth,
        verify=verify,
        httpx_client_factory=factory,
    )


def _make_httpx_client_factory(
    http_client: httpx.AsyncClient,
) -> Callable[..., httpx.AsyncClient]:
    """Return an `httpx_client_factory` that always returns the user-supplied `http_client`."""

    def factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        return http_client

    return factory


def _build_sampling_handler(sampling_model: models.Model) -> SamplingHandler[Any, Any]:
    """Build a FastMCP-shaped sampling handler that delegates to a Pydantic AI model."""

    async def handler(
        sampling_messages: list[mcp_types.SamplingMessage],
        params: mcp_types.CreateMessageRequestParams,
        ctx: Any,
    ) -> mcp_types.CreateMessageResult:
        pai_messages = _mcp.map_from_mcp_params(params)
        model_settings = ModelSettings(max_tokens=params.maxTokens)
        if (temperature := params.temperature) is not None:  # pragma: no branch
            model_settings['temperature'] = temperature
        if (stop_sequences := params.stopSequences) is not None:  # pragma: no branch
            model_settings['stop_sequences'] = stop_sequences

        model_response = await model_request(sampling_model, pai_messages, model_settings=model_settings)
        return mcp_types.CreateMessageResult(
            role='assistant',
            content=_mcp.map_from_model_response(model_response),
            model=sampling_model.model_name,
        )

    return handler


def _map_mcp_tool_results(
    parts: Sequence[mcp_types.ContentBlock],
) -> (
    str
    | messages.BinaryContent
    | dict[str, Any]
    | list[Any]
    | list[str | messages.BinaryContent | dict[str, Any] | list[Any]]
):
    mapped = [_map_mcp_tool_result(part) for part in parts]
    return mapped[0] if len(mapped) == 1 else mapped


def _map_mcp_tool_result(part: mcp_types.ContentBlock) -> str | messages.BinaryContent | dict[str, Any] | list[Any]:
    # Tool results don't preserve MCP annotations/`_meta` onto `BinaryContent.vendor_metadata`;
    # only `_map_mcp_prompt_part` does that via `_map_mcp_binary_content`. The PR that added prompts
    # made this asymmetric on purpose (tool returns flow to the model; prompt content flows to the
    # user). Revisit if a future PR decides tool returns should also surface MCP annotations.
    if isinstance(part, mcp_types.TextContent):
        text = part.text
        if text.startswith(('[', '{')):
            try:
                return pydantic_core.from_json(text)
            except ValueError:
                pass
        return text
    elif isinstance(part, mcp_types.ImageContent):
        return messages.BinaryImage(data=base64.b64decode(part.data), media_type=part.mimeType)
    elif isinstance(part, mcp_types.AudioContent):
        return messages.BinaryContent(data=base64.b64decode(part.data), media_type=part.mimeType)  # pragma: no cover
    elif isinstance(part, mcp_types.EmbeddedResource):
        return _resource_content_to_pai(part.resource)
    elif isinstance(part, mcp_types.ResourceLink):
        # Reading the linked resource requires a session reference; fall back to returning the URI.
        # For inline reading, callers can use `MCPToolset.read_resource(part.uri)` directly.
        return str(part.uri)
    else:
        assert_never(part)


def _mcp_part_metadata(
    part: mcp_types.TextContent | mcp_types.ImageContent | mcp_types.AudioContent,
) -> dict[str, Any] | None:
    metadata: dict[str, Any] = {}
    if part.annotations:
        metadata['mcp_annotations'] = ResourceAnnotations.from_mcp_sdk(part.annotations)
    if part.meta:
        metadata['mcp_meta'] = part.meta
    return metadata or None


def _map_mcp_binary_content(part: mcp_types.ImageContent | mcp_types.AudioContent) -> messages.BinaryContent:
    data = base64.b64decode(part.data)
    vendor_metadata = _mcp_part_metadata(part)
    if isinstance(part, mcp_types.ImageContent):
        return messages.BinaryImage(data=data, media_type=part.mimeType, vendor_metadata=vendor_metadata)
    return messages.BinaryContent(data=data, media_type=part.mimeType, vendor_metadata=vendor_metadata)


def _map_mcp_prompt_part(part: mcp_types.ContentBlock) -> ContentBlock:
    if isinstance(part, mcp_types.TextContent):
        return messages.TextContent(content=part.text, metadata=_mcp_part_metadata(part))
    elif isinstance(part, (mcp_types.ImageContent, mcp_types.AudioContent)):
        return _map_mcp_binary_content(part)
    elif isinstance(part, mcp_types.EmbeddedResource):
        return EmbeddedResource.from_mcp_sdk(part, _resource_content_to_pai(part.resource))
    elif isinstance(part, mcp_types.ResourceLink):
        return ResourceLink.from_mcp_sdk(part)
    else:
        assert_never(part)


def _resource_content_to_pai(
    resource: mcp_types.TextResourceContents | mcp_types.BlobResourceContents,
) -> str | messages.BinaryContent:
    if isinstance(resource, mcp_types.TextResourceContents):
        return resource.text
    elif isinstance(resource, mcp_types.BlobResourceContents):
        return messages.BinaryContent.narrow_type(
            messages.BinaryContent(
                data=base64.b64decode(resource.blob),
                media_type=resource.mimeType or 'application/octet-stream',
            )
        )
    else:
        assert_never(resource)


def _mcp_server_discriminator(value: dict[str, Any]) -> str | None:
    if 'url' in value:
        if value['url'].endswith('/sse'):
            return 'sse'
        return 'streamable-http'
    return 'stdio'


class _MCPServerConfig(BaseModel):
    """Internal config model for `load_mcp_servers` / `load_mcp_toolsets`.

    Exposed as the deprecated `pydantic_ai.mcp.MCPServerConfig` via this module's `__getattr__`.
    """

    mcp_servers: Annotated[
        dict[
            str,
            Annotated[
                Annotated[MCPServerStdio, Tag('stdio')]  # pyright: ignore[reportDeprecated]
                | Annotated[MCPServerStreamableHTTP, Tag('streamable-http')]  # pyright: ignore[reportDeprecated]
                | Annotated[MCPServerSSE, Tag('sse')],  # pyright: ignore[reportDeprecated]
                Discriminator(_mcp_server_discriminator),
            ],
        ],
        Field(alias='mcpServers'),
    ]


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand environment variables in a JSON structure.

    Environment variables can be referenced using `${VAR_NAME}` syntax,
    or `${VAR_NAME:-default}` syntax to provide a default value if the variable is not set.

    Args:
        value: The value to expand (can be str, dict, list, or other JSON types).

    Returns:
        The value with all environment variables expanded.

    Raises:
        ValueError: If an environment variable is not defined and no default value is provided.
    """
    if isinstance(value, str):
        # Find all environment variable references in the string
        # Supports both ${VAR_NAME} and ${VAR_NAME:-default} syntax
        def replace_match(match: re.Match[str]) -> str:
            var_name = match.group(1)
            has_default = match.group(2) is not None
            default_value = match.group(3) if has_default else None

            # Check if variable exists in environment
            if var_name in os.environ:
                return os.environ[var_name]
            elif has_default:
                # Use default value if the :- syntax was present (even if empty string)
                return default_value or ''
            else:
                # No default value and variable not set - raise error
                raise ValueError(f'Environment variable ${{{var_name}}} is not defined')

        value = _ENV_VAR_PATTERN.sub(replace_match, value)

        return value
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}  # type: ignore[misc]
    elif isinstance(value, list):
        return [_expand_env_vars(item) for item in value]  # type: ignore[misc]
    else:
        return value


@deprecated(
    '`load_mcp_servers` is deprecated and will be removed in v2. '
    'Use `pydantic_ai.mcp.load_mcp_toolsets` instead — same JSON config shape, returns `MCPToolset` '
    'instances wrapped with their server name as a tool prefix.'
)
def load_mcp_servers(
    config_path: str | Path,
) -> list[MCPServerStdio | MCPServerStreamableHTTP | MCPServerSSE]:  # pyright: ignore[reportDeprecated]
    """Load MCP servers from a configuration file.

    Environment variables can be referenced in the configuration file using:
    - `${VAR_NAME}` syntax - expands to the value of VAR_NAME, raises error if not defined
    - `${VAR_NAME:-default}` syntax - expands to VAR_NAME if set, otherwise uses the default value

    Args:
        config_path: The path to the configuration file.

    Returns:
        A list of MCP servers.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValidationError: If the configuration file does not match the schema.
        ValueError: If an environment variable referenced in the configuration is not defined and no default value is provided.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f'Config file {config_path} not found')

    config_data = pydantic_core.from_json(config_path.read_bytes())
    expanded_config_data = _expand_env_vars(config_data)
    # Discriminator constructs deprecated `MCPServer*` instances; suppressing the warnings here
    # is intentional — `load_mcp_servers` is itself deprecated and returns these classes.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'`MCPServer\w+` is deprecated', DeprecationWarning)
        config = _MCPServerConfig.model_validate(expanded_config_data)

    servers: list[MCPServerStdio | MCPServerStreamableHTTP | MCPServerSSE] = []  # pyright: ignore[reportDeprecated]
    for name, server in config.mcp_servers.items():
        server.id = name
        server.tool_prefix = name
        servers.append(server)

    return servers


def load_mcp_toolsets(config_path: str | Path) -> list[AbstractToolset[Any]]:
    """Load `MCPToolset`s from a configuration file.

    The configuration file uses the same `mcpServers` JSON shape as Claude Desktop, Cursor, and the
    MCP specification. Each server entry produces one [`MCPToolset`][pydantic_ai.mcp.MCPToolset],
    wrapped in a [`PrefixedToolset`][pydantic_ai.toolsets.PrefixedToolset] using the server's name
    as prefix to disambiguate tools across multiple servers.

    Environment variables can be referenced in the configuration file using:

    - `${VAR_NAME}` syntax — expands to the value of `VAR_NAME`, raises if not defined
    - `${VAR_NAME:-default}` syntax — expands to `VAR_NAME` if set, otherwise the default

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        A list of toolsets, one per server in the config file, each prefixed with the server name.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValidationError: If the configuration file does not match the schema.
        ValueError: If an environment variable referenced in the configuration is not defined and
            no default is provided.
        ImportError: If the fastmcp client isn't installed. Install the `mcp` extra (which pulls
            `fastmcp-slim[client]`): `pip install "pydantic-ai-slim[mcp]"`.
    """
    _require_fastmcp()
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f'Config file {config_path} not found')

    config_data = pydantic_core.from_json(config_path.read_bytes())
    expanded_config_data = _expand_env_vars(config_data)
    # `_MCPServerConfig` validates into deprecated `MCPServer*` subclasses; we only use them to
    # extract `command`/`args`/`url` and build fresh `MCPToolset`s below.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'`MCPServer\w+` is deprecated', DeprecationWarning)
        config = _MCPServerConfig.model_validate(expanded_config_data)

    toolsets: list[AbstractToolset[Any]] = []
    for name, server in config.mcp_servers.items():
        toolset: MCPToolset[Any]
        if isinstance(server, MCPServerStdio):  # pyright: ignore[reportDeprecated]
            transport = StdioTransport(
                command=server.command,
                args=list(server.args),
                env=server.env,
                cwd=str(server.cwd) if server.cwd is not None else None,
            )
            toolset = MCPToolset(transport, id=name)
        elif isinstance(server, _MCPServerHTTP):
            toolset = MCPToolset(server.url, id=name, headers=server.headers)
        else:  # pragma: no cover
            assert_never(server)
        toolsets.append(toolset.prefixed(name))

    return toolsets


# Module-level deprecation shim for names removed in v2. Internal code references the renamed
# private symbols (e.g. `_MCPServerConfig`) so it doesn't trigger its own deprecation warning.
_DEPRECATED_NAMES: dict[str, tuple[str, Any]] = {
    'MCPServerConfig': (
        'Pass the JSON config to `load_mcp_toolsets(...)` directly, or build `MCPToolset`s '
        'inline from `fastmcp.client.transports.StdioTransport` / URLs.',
        _MCPServerConfig,
    ),
}


def __getattr__(name: str) -> Any:
    if name in _DEPRECATED_NAMES:
        from ._warnings import PydanticAIDeprecationWarning

        message, target = _DEPRECATED_NAMES[name]
        warnings.warn(
            f'`pydantic_ai.mcp.{name}` is deprecated and will be removed in v2. {message}',
            PydanticAIDeprecationWarning,
            stacklevel=2,
        )
        return target
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
