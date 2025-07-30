from __future__ import annotations

import base64
import functools
import warnings
from abc import ABC, abstractmethod
from asyncio import Lock
from collections.abc import AsyncIterator, Awaitable, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import anyio
import httpx
import pydantic_core
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from typing_extensions import Self, assert_never, deprecated

from pydantic_ai._run_context import RunContext
from pydantic_ai.tools import ToolDefinition

from .toolsets.abstract import AbstractToolset, ToolsetTool

try:
    from mcp import types as mcp_types
    from mcp.client.session import ClientSession, LoggingFnT
    from mcp.client.sse import sse_client
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.client.streamable_http import GetSessionIdCallback, streamablehttp_client
    from mcp.shared.context import RequestContext
    from mcp.shared.exceptions import McpError
    from mcp.shared.message import SessionMessage
except ImportError as _import_error:
    raise ImportError(
        'Please install the `mcp` package to use the MCP server, '
        'you can use the `mcp` optional group — `pip install "pydantic-ai-slim[mcp]"`'
    ) from _import_error

# after mcp imports so any import error maps to this file, not _mcp.py
from . import _mcp, _utils, exceptions, messages, models

__all__ = 'MCPServer', 'MCPServerStdio', 'MCPServerHTTP', 'MCPServerSSE', 'MCPServerStreamableHTTP'

TOOL_SCHEMA_VALIDATOR = pydantic_core.SchemaValidator(
    schema=pydantic_core.core_schema.dict_schema(
        pydantic_core.core_schema.str_schema(), pydantic_core.core_schema.any_schema()
    )
)


class MCPServer(AbstractToolset[Any], ABC):
    """Base class for attaching agents to MCP servers.

    See <https://modelcontextprotocol.io> for more information.
    """

    # these fields should be re-defined by dataclass subclasses so they appear as fields {
    tool_prefix: str | None = None
    log_level: mcp_types.LoggingLevel | None = None
    log_handler: LoggingFnT | None = None
    timeout: float = 5
    read_timeout: float = 5 * 60
    process_tool_call: ProcessToolCallback | None = None
    allow_sampling: bool = True
    max_retries: int = 1
    sampling_model: models.Model | None = None
    # } end of "abstract fields"

    _enter_lock: Lock = field(compare=False)
    _running_count: int
    _exit_stack: AsyncExitStack | None

    _client: ClientSession
    _read_stream: MemoryObjectReceiveStream[SessionMessage | Exception]
    _write_stream: MemoryObjectSendStream[SessionMessage]

    def __post_init__(self):
        self._enter_lock = Lock()
        self._running_count = 0
        self._exit_stack = None

    @abstractmethod
    @asynccontextmanager
    async def client_streams(
        self,
    ) -> AsyncIterator[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
        ]
    ]:
        """Create the streams for the MCP server."""
        raise NotImplementedError('MCP Server subclasses must implement this method.')
        yield

    @property
    def name(self) -> str:
        return repr(self)

    @property
    def tool_name_conflict_hint(self) -> str:
        return 'Consider setting `tool_prefix` to avoid name conflicts.'

    async def list_tools(self) -> list[mcp_types.Tool]:
        """Retrieve tools that are currently active on the server.

        Note:
        - We don't cache tools as they might change.
        - We also don't subscribe to the server to avoid complexity.
        """
        async with self:  # Ensure server is running
            result = await self._client.list_tools()
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
                result = await self._client.send_request(
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
            except McpError as e:
                raise exceptions.ModelRetry(e.error.message)

        content = [await self._map_tool_result_part(part) for part in result.content]

        if result.isError:
            text = '\n'.join(str(part) for part in content)
            raise exceptions.ModelRetry(text)
        else:
            return content[0] if len(content) == 1 else content

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
            name: ToolsetTool(
                toolset=self,
                tool_def=ToolDefinition(
                    name=name,
                    description=mcp_tool.description,
                    parameters_json_schema=mcp_tool.inputSchema,
                ),
                max_retries=self.max_retries,
                args_validator=TOOL_SCHEMA_VALIDATOR,
            )
            for mcp_tool in await self.list_tools()
            if (name := f'{self.tool_prefix}_{mcp_tool.name}' if self.tool_prefix else mcp_tool.name)
        }

    async def __aenter__(self) -> Self:
        """Enter the MCP server context.

        This will initialize the connection to the server.
        If this server is an [`MCPServerStdio`][pydantic_ai.mcp.MCPServerStdio], the server will first be started as a subprocess.

        This is a no-op if the MCP server has already been entered.
        """
        async with self._enter_lock:
            if self._running_count == 0:
                async with AsyncExitStack() as exit_stack:
                    self._read_stream, self._write_stream = await exit_stack.enter_async_context(self.client_streams())
                    client = ClientSession(
                        read_stream=self._read_stream,
                        write_stream=self._write_stream,
                        sampling_callback=self._sampling_callback if self.allow_sampling else None,
                        logging_callback=self.log_handler,
                        read_timeout_seconds=timedelta(seconds=self.read_timeout),
                    )
                    self._client = await exit_stack.enter_async_context(client)

                    with anyio.fail_after(self.timeout):
                        await self._client.initialize()

                        if log_level := self.log_level:
                            await self._client.set_logging_level(log_level)

                    self._exit_stack = exit_stack.pop_all()
            self._running_count += 1
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        async with self._enter_lock:
            self._running_count -= 1
            if self._running_count == 0 and self._exit_stack is not None:
                await self._exit_stack.aclose()
                self._exit_stack = None

    @property
    def is_running(self) -> bool:
        """Check if the MCP server is running."""
        return bool(self._running_count)

    async def _sampling_callback(
        self, context: RequestContext[ClientSession, Any], params: mcp_types.CreateMessageRequestParams
    ) -> mcp_types.CreateMessageResult | mcp_types.ErrorData:
        """MCP sampling callback."""
        if self.sampling_model is None:
            raise ValueError('Sampling model is not set')  # pragma: no cover

        pai_messages = _mcp.map_from_mcp_params(params)
        model_settings = models.ModelSettings()
        if max_tokens := params.maxTokens:  # pragma: no branch
            model_settings['max_tokens'] = max_tokens
        if temperature := params.temperature:  # pragma: no branch
            model_settings['temperature'] = temperature
        if stop_sequences := params.stopSequences:  # pragma: no branch
            model_settings['stop_sequences'] = stop_sequences

        model_response = await self.sampling_model.request(
            pai_messages,
            model_settings,
            models.ModelRequestParameters(),
        )
        return mcp_types.CreateMessageResult(
            role='assistant',
            content=_mcp.map_from_model_response(model_response),
            model=self.sampling_model.model_name,
        )

    async def _map_tool_result_part(
        self, part: mcp_types.ContentBlock
    ) -> str | messages.BinaryContent | dict[str, Any] | list[Any]:
        # See https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#return-values

        if isinstance(part, mcp_types.TextContent):
            text = part.text
            if text.startswith(('[', '{')):
                try:
                    return pydantic_core.from_json(text)
                except ValueError:
                    pass
            return text
        elif isinstance(part, mcp_types.ImageContent):
            return messages.BinaryContent(data=base64.b64decode(part.data), media_type=part.mimeType)
        elif isinstance(part, mcp_types.AudioContent):
            # NOTE: The FastMCP server doesn't support audio content.
            # See <https://github.com/modelcontextprotocol/python-sdk/issues/952> for more details.
            return messages.BinaryContent(
                data=base64.b64decode(part.data), media_type=part.mimeType
            )  # pragma: no cover
        elif isinstance(part, mcp_types.EmbeddedResource):
            resource = part.resource
            return self._get_content(resource)
        elif isinstance(part, mcp_types.ResourceLink):
            resource_result: mcp_types.ReadResourceResult = await self._client.read_resource(part.uri)
            return (
                self._get_content(resource_result.contents[0])
                if len(resource_result.contents) == 1
                else [self._get_content(resource) for resource in resource_result.contents]
            )
        else:
            assert_never(part)

    def _get_content(
        self, resource: mcp_types.TextResourceContents | mcp_types.BlobResourceContents
    ) -> str | messages.BinaryContent:
        if isinstance(resource, mcp_types.TextResourceContents):
            return resource.text
        elif isinstance(resource, mcp_types.BlobResourceContents):
            return messages.BinaryContent(
                data=base64.b64decode(resource.blob), media_type=resource.mimeType or 'application/octet-stream'
            )
        else:
            assert_never(resource)


@dataclass
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
        'deno',
        args=[
            'run',
            '-N',
            '-R=node_modules',
            '-W=node_modules',
            '--node-modules-dir=auto',
            'jsr:@pydantic/mcp-run-python',
            'stdio',
        ]
    )
    agent = Agent('openai:gpt-4o', toolsets=[server])

    async def main():
        async with agent:  # (2)!
            ...
    ```

    1. See [MCP Run Python](../mcp/run-python.md) for more information.
    2. This will start the server as a subprocess and connect to it.
    """

    command: str
    """The command to run."""

    args: Sequence[str]
    """The arguments to pass to the command."""

    env: dict[str, str] | None = None
    """The environment variables the CLI server will have access to.

    By default the subprocess will not inherit any environment variables from the parent process.
    If you want to inherit the environment variables from the parent process, use `env=os.environ`.
    """

    cwd: str | Path | None = None
    """The working directory to use when spawning the process."""

    # last fields are re-defined from the parent class so they appear as fields
    tool_prefix: str | None = None
    """A prefix to add to all tools that are registered with the server.

    If not empty, will include a trailing underscore(`_`).

    e.g. if `tool_prefix='foo'`, then a tool named `bar` will be registered as `foo_bar`
    """

    log_level: mcp_types.LoggingLevel | None = None
    """The log level to set when connecting to the server, if any.

    See <https://modelcontextprotocol.io/specification/2025-03-26/server/utilities/logging#logging> for more details.

    If `None`, no log level will be set.
    """

    log_handler: LoggingFnT | None = None
    """A handler for logging messages from the server."""

    timeout: float = 5
    """The timeout in seconds to wait for the client to initialize."""

    process_tool_call: ProcessToolCallback | None = None
    """Hook to customize tool calling and optionally pass extra metadata."""

    allow_sampling: bool = True
    """Whether to allow MCP sampling through this client."""

    max_retries: int = 1
    """The maximum number of times to retry a tool call."""

    sampling_model: models.Model | None = None
    """The model to use for sampling."""

    @asynccontextmanager
    async def client_streams(
        self,
    ) -> AsyncIterator[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
        ]
    ]:
        server = StdioServerParameters(command=self.command, args=list(self.args), env=self.env, cwd=self.cwd)
        async with stdio_client(server=server) as (read_stream, write_stream):
            yield read_stream, write_stream

    def __repr__(self) -> str:
        return f'MCPServerStdio(command={self.command!r}, args={self.args!r}, tool_prefix={self.tool_prefix!r})'


@dataclass(init=False)
class _MCPServerHTTP(MCPServer):
    url: str
    """The URL of the endpoint on the MCP server."""

    headers: dict[str, Any] | None = None
    """Optional HTTP headers to be sent with each request to the endpoint.

    These headers will be passed directly to the underlying `httpx.AsyncClient`.
    Useful for authentication, custom headers, or other HTTP-specific configurations.

    !!! note
        You can either pass `headers` or `http_client`, but not both.

        See [`MCPServerHTTP.http_client`][pydantic_ai.mcp.MCPServerHTTP.http_client] for more information.
    """

    http_client: httpx.AsyncClient | None = None
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

    read_timeout: float = 5 * 60
    """Maximum time in seconds to wait for new messages before timing out.

    This timeout applies to the long-lived connection after it's established.
    If no new messages are received within this time, the connection will be considered stale
    and may be closed. Defaults to 5 minutes (300 seconds).
    """

    # last fields are re-defined from the parent class so they appear as fields
    tool_prefix: str | None = None
    """A prefix to add to all tools that are registered with the server.

    If not empty, will include a trailing underscore (`_`).

    For example, if `tool_prefix='foo'`, then a tool named `bar` will be registered as `foo_bar`
    """

    log_level: mcp_types.LoggingLevel | None = None
    """The log level to set when connecting to the server, if any.

    See <https://modelcontextprotocol.io/introduction#logging> for more details.

    If `None`, no log level will be set.
    """

    log_handler: LoggingFnT | None = None
    """A handler for logging messages from the server."""

    timeout: float = 5
    """Initial connection timeout in seconds for establishing the connection.

    This timeout applies to the initial connection setup and handshake.
    If the connection cannot be established within this time, the operation will fail.
    """

    process_tool_call: ProcessToolCallback | None = None
    """Hook to customize tool calling and optionally pass extra metadata."""

    allow_sampling: bool = True
    """Whether to allow MCP sampling through this client."""

    max_retries: int = 1
    """The maximum number of times to retry a tool call."""

    sampling_model: models.Model | None = None
    """The model to use for sampling."""

    def __init__(
        self,
        *,
        url: str,
        headers: dict[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
        read_timeout: float | None = None,
        tool_prefix: str | None = None,
        log_level: mcp_types.LoggingLevel | None = None,
        log_handler: LoggingFnT | None = None,
        timeout: float = 5,
        process_tool_call: ProcessToolCallback | None = None,
        allow_sampling: bool = True,
        max_retries: int = 1,
        sampling_model: models.Model | None = None,
        **kwargs: Any,
    ):
        # Handle deprecated sse_read_timeout parameter
        if 'sse_read_timeout' in kwargs:
            if read_timeout is not None:
                raise TypeError("'read_timeout' and 'sse_read_timeout' cannot be set at the same time.")

            warnings.warn(
                "'sse_read_timeout' is deprecated, use 'read_timeout' instead.", DeprecationWarning, stacklevel=2
            )
            read_timeout = kwargs.pop('sse_read_timeout')

        _utils.validate_empty_kwargs(kwargs)

        if read_timeout is None:
            read_timeout = 5 * 60

        self.url = url
        self.headers = headers
        self.http_client = http_client
        self.tool_prefix = tool_prefix
        self.log_level = log_level
        self.log_handler = log_handler
        self.timeout = timeout
        self.process_tool_call = process_tool_call
        self.allow_sampling = allow_sampling
        self.max_retries = max_retries
        self.sampling_model = sampling_model
        self.read_timeout = read_timeout
        self.__post_init__()

    @property
    @abstractmethod
    def _transport_client(
        self,
    ) -> Callable[
        ...,
        AbstractAsyncContextManager[
            tuple[
                MemoryObjectReceiveStream[SessionMessage | Exception],
                MemoryObjectSendStream[SessionMessage],
                GetSessionIdCallback,
            ],
        ]
        | AbstractAsyncContextManager[
            tuple[
                MemoryObjectReceiveStream[SessionMessage | Exception],
                MemoryObjectSendStream[SessionMessage],
            ]
        ],
    ]: ...

    @asynccontextmanager
    async def client_streams(
        self,
    ) -> AsyncIterator[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
        ]
    ]:  # pragma: no cover
        if self.http_client and self.headers:
            raise ValueError('`http_client` is mutually exclusive with `headers`.')

        transport_client_partial = functools.partial(
            self._transport_client,
            url=self.url,
            timeout=self.timeout,
            sse_read_timeout=self.read_timeout,
        )

        if self.http_client is not None:

            def httpx_client_factory(
                headers: dict[str, str] | None = None,
                timeout: httpx.Timeout | None = None,
                auth: httpx.Auth | None = None,
            ) -> httpx.AsyncClient:
                assert self.http_client is not None
                return self.http_client

            async with transport_client_partial(httpx_client_factory=httpx_client_factory) as (
                read_stream,
                write_stream,
                *_,
            ):
                yield read_stream, write_stream
        else:
            async with transport_client_partial(headers=self.headers) as (read_stream, write_stream, *_):
                yield read_stream, write_stream

    def __repr__(self) -> str:  # pragma: no cover
        return f'{self.__class__.__name__}(url={self.url!r}, tool_prefix={self.tool_prefix!r})'


@dataclass(init=False)
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

    server = MCPServerSSE('http://localhost:3001/sse')  # (1)!
    agent = Agent('openai:gpt-4o', toolsets=[server])

    async def main():
        async with agent:  # (2)!
            ...
    ```

    1. E.g. you might be connecting to a server run with [`mcp-run-python`](../mcp/run-python.md).
    2. This will connect to a server running on `localhost:3001`.
    """

    @property
    def _transport_client(self):
        return sse_client  # pragma: no cover


@deprecated('The `MCPServerHTTP` class is deprecated, use `MCPServerSSE` instead.')
@dataclass
class MCPServerHTTP(MCPServerSSE):
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

    server = MCPServerHTTP('http://localhost:3001/sse')  # (1)!
    agent = Agent('openai:gpt-4o', toolsets=[server])

    async def main():
        async with agent:  # (2)!
            ...
    ```

    1. E.g. you might be connecting to a server run with [`mcp-run-python`](../mcp/run-python.md).
    2. This will connect to a server running on `localhost:3001`.
    """


@dataclass
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

    server = MCPServerStreamableHTTP('http://localhost:8000/mcp')  # (1)!
    agent = Agent('openai:gpt-4o', toolsets=[server])

    async def main():
        async with agent:  # (2)!
            ...
    ```
    """

    @property
    def _transport_client(self):
        return streamablehttp_client  # pragma: no cover


ToolResult = (
    str
    | messages.BinaryContent
    | dict[str, Any]
    | list[Any]
    | Sequence[str | messages.BinaryContent | dict[str, Any] | list[Any]]
)
"""The result type of an MCP tool call."""

CallToolFunc = Callable[[str, dict[str, Any], dict[str, Any] | None], Awaitable[ToolResult]]
"""A function type that represents a tool call."""

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
