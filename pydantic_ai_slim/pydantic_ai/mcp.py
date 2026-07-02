from __future__ import annotations

import base64
import functools
import os
import re
import ssl
from abc import ABC
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol, TypeAlias, cast, overload

import anyio
import httpx
import pydantic_core
from pydantic import AnyUrl, Field
from typing_extensions import Self, assert_never

from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition

from .direct import model_request
from .toolsets.abstract import AbstractToolset, ToolsetTool

try:
    from mcp import types as mcp_types
    from mcp.shared import exceptions as mcp_exceptions
except ImportError as _import_error:
    raise ImportError(
        'Please install the `mcp` package to use `MCPToolset`, '
        'you can use the `mcp` optional group — `pip install "pydantic-ai-slim[mcp]"`'
    ) from _import_error

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
except ImportError as _fastmcp_import_error:  # pragma: no cover
    raise ImportError(
        'Please install the fastmcp client to use `MCPToolset` — '
        '`pip install "pydantic-ai-slim[mcp]"` pulls `fastmcp-slim[client]`, '
        'or install the full `fastmcp` package directly.'
    ) from _fastmcp_import_error

# In-process MCP servers (`FastMCP` / `FastMCP1Server`) live in the *server* halves of fastmcp /
# the MCP SDK respectively. The lightweight `[mcp]` install (`fastmcp-slim[client]`) does NOT ship
# them, so guard those imports separately — `MCPToolsetClient` widens to `Any` for the missing
# names, and code that takes an in-process server is unreachable in that environment.
if TYPE_CHECKING:
    from fastmcp.client.client import CallToolResult
    from fastmcp.client.tasks import ToolTask
    from fastmcp.server import FastMCP
    from mcp.server.fastmcp import FastMCP as FastMCP1Server
else:
    try:
        from fastmcp.server import FastMCP
    except ImportError:  # pragma: no cover
        FastMCP = Any
    try:
        from mcp.server.fastmcp import FastMCP as FastMCP1Server
    except ImportError:  # pragma: no cover
        FastMCP1Server = Any


# after mcp imports so any import error maps to this file, not _mcp.py
from . import _mcp, _utils, exceptions, messages, models
from .settings import ModelSettings

__all__ = (
    'MCPToolset',
    'MCPToolsetClient',
    'load_mcp_toolsets',
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
    """The result of a [`get_prompt`][pydantic_ai.mcp.MCPToolset.get_prompt] request."""

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


MCPToolsetClient: TypeAlias = FastMCPClient[Any] | ClientTransport | FastMCP | FastMCP1Server | AnyUrl | Path | str
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
        """
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

    def set_sampling_model(self, model: models.Model) -> None:
        """Set the [`sampling_model`][pydantic_ai.mcp.MCPToolset.sampling_model] on an already-constructed toolset.

        Swaps both the public attribute and the underlying FastMCP client's sampling callback.
        Takes effect on the next session opened by the client; calls already in flight on an
        existing session continue using the previously configured handler.
        """
        self.sampling_model = model
        self.client.set_sampling_callback(_build_sampling_handler(model))  # pyright: ignore[reportUnknownMemberType]

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
            except _utils.BaseExceptionGroup as eg:
                # The FastMCP client runs the MCP session in an anyio task group, so a tool/protocol
                # error can surface wrapped in an `ExceptionGroup` rather than as a bare
                # `ToolError`/`McpError`. This has been observed in production (an empty-bodied tool
                # error racing with the session's GET-stream teardown), though the exact frame it
                # unwinds from is not pinned down — so this is a best-effort guard: when the group
                # contains only tool/protocol errors, treat it like the bare case above; otherwise
                # re-raise unchanged so a concurrent cancellation grouped alongside is never swallowed.
                if self.tool_error_behavior != 'retry':
                    raise
                matched, rest = eg.split((ToolError, mcp_exceptions.McpError))
                if matched is None or rest is not None:
                    raise
                # `matched` holds only tool/protocol errors; descend through any nesting to a leaf.
                error: BaseException = matched
                while isinstance(error, _utils.BaseExceptionGroup):
                    error = error.exceptions[0]
                raise exceptions.ModelRetry(message=str(error)) from eg

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
        # FastMCP's StreamableHttpTransport calls the factory with `follow_redirects`,
        # which the mcp SDK's `McpHttpClientFactory` protocol doesn't declare.
        follow_redirects: bool = True,
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
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f'Config file {config_path} not found')

    config_data = pydantic_core.from_json(config_path.read_bytes())
    expanded_config_data = _expand_env_vars(config_data)
    if not isinstance(expanded_config_data, dict):
        raise ValueError(f'Expected JSON object at root of {config_path}, got {type(expanded_config_data).__name__}')
    servers = cast(dict[str, Any], expanded_config_data).get('mcpServers')
    if not isinstance(servers, dict):
        raise ValueError(f'Expected `mcpServers` object in {config_path}')

    toolsets: list[AbstractToolset[Any]] = []
    for name, server in cast(dict[str, Any], servers).items():
        if 'command' in server:
            transport = StdioTransport(
                command=server['command'],
                args=list(server.get('args') or []),
                env=server.get('env'),
                cwd=str(server['cwd']) if server.get('cwd') is not None else None,
            )
            toolset = MCPToolset(transport, id=name)
        elif 'url' in server:
            toolset = MCPToolset(server['url'], id=name, headers=server.get('headers'))
        else:
            raise ValueError(f'MCP server config {name!r} must have either `command` or `url`')
        toolsets.append(toolset.prefixed(name))

    return toolsets
