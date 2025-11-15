from __future__ import annotations

import base64
import functools
import os
import re
import warnings
from abc import ABC, abstractmethod
from asyncio import Lock
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from dataclasses import field, replace
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Any

import anyio
import httpx
import pydantic_core
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from pydantic import BaseModel, Discriminator, Field, Tag
from pydantic_core import CoreSchema, core_schema
from typing_extensions import Self, assert_never, deprecated

from pydantic_ai.tools import RunContext, ToolDefinition

from .direct import model_request
from .toolsets.abstract import AbstractToolset, ToolsetTool

try:
    from mcp import types as mcp_types
    from mcp.client.session import ClientSession, ElicitationFnT, LoggingFnT
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

__all__ = 'MCPServer', 'MCPServerStdio', 'MCPServerHTTP', 'MCPServerSSE', 'MCPServerStreamableHTTP', 'load_mcp_servers'

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


class MCPServer(AbstractToolset[Any], ABC):
    """Base class for attaching agents to MCP servers.

    See <https://modelcontextprotocol.io> for more information.
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

    _id: str | None

    _enter_lock: Lock = field(compare=False)
    _running_count: int
    _exit_stack: AsyncExitStack | None

    _client: ClientSession
    _read_stream: MemoryObjectReceiveStream[SessionMessage | Exception]
    _write_stream: MemoryObjectSendStream[SessionMessage]
    _server_info: mcp_types.Implementation

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
        id: str | None = None,
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

        self._id = id or tool_prefix

        self.__post_init__()

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
                        elicitation_callback=self.elicitation_callback,
                        logging_callback=self.log_handler,
                        read_timeout_seconds=timedelta(seconds=self.read_timeout),
                    )
                    self._client = await exit_stack.enter_async_context(client)

                    with anyio.fail_after(self.timeout):
                        result = await self._client.initialize()
                        self._server_info = result.serverInfo
                        if log_level := self.log_level:
                            await self._client.set_logging_level(log_level)

                    self._exit_stack = exit_stack.pop_all()
            self._running_count += 1
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        if self._running_count == 0:
            raise ValueError('MCPServer.__aexit__ called more times than __aenter__')
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

        model_response = await model_request(self.sampling_model, pai_messages, model_settings=model_settings)
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

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, MCPServer) and self.id == value.id and self.tool_prefix == value.tool_prefix


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
    agent = Agent('openai:gpt-4o', toolsets=[server])
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
        id: str | None = None,
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
            id: An optional unique ID for the MCP server. An MCP server needs to have an ID in order to be used in a durable execution environment like Temporal, in which case the ID will be used to identify the server's activities within the workflow.
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
            id=id,
        )

    @classmethod
    def __get_pydantic_core_schema__(cls, _: Any, __: Any) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            lambda dct: MCPServerStdio(**dct),
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
            and isinstance(value, MCPServerStdio)
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
            id=id,
        )

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
    ]:
        if self.http_client and self.headers:
            raise ValueError('`http_client` is mutually exclusive with `headers`.')  # pragma: no cover

        transport_client_partial = functools.partial(
            self._transport_client,
            url=self.url,
            timeout=self.timeout,
            sse_read_timeout=self.read_timeout,
        )

        if self.http_client is not None:  # pragma: no cover

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
        repr_args = [
            f'url={self.url!r}',
        ]
        if self.id:
            repr_args.append(f'id={self.id!r}')
        return f'{self.__class__.__name__}({", ".join(repr_args)})'


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
    agent = Agent('openai:gpt-4o', toolsets=[server])
    ```
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, _: Any, __: Any) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            lambda dct: MCPServerSSE(**dct),
            core_schema.typed_dict_schema(
                {
                    'url': core_schema.typed_dict_field(core_schema.str_schema()),
                    'headers': core_schema.typed_dict_field(
                        core_schema.dict_schema(core_schema.str_schema(), core_schema.str_schema()), required=False
                    ),
                }
            ),
        )

    @property
    def _transport_client(self):
        return sse_client  # pragma: no cover

    def __eq__(self, value: object, /) -> bool:
        return super().__eq__(value) and isinstance(value, MCPServerSSE) and self.url == value.url


@deprecated('The `MCPServerHTTP` class is deprecated, use `MCPServerSSE` instead.')
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

    server = MCPServerHTTP('http://localhost:3001/sse')
    agent = Agent('openai:gpt-4o', toolsets=[server])
    ```
    """


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
    agent = Agent('openai:gpt-4o', toolsets=[server])
    ```
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, _: Any, __: Any) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            lambda dct: MCPServerStreamableHTTP(**dct),
            core_schema.typed_dict_schema(
                {
                    'url': core_schema.typed_dict_field(core_schema.str_schema()),
                    'headers': core_schema.typed_dict_field(
                        core_schema.dict_schema(core_schema.str_schema(), core_schema.str_schema()), required=False
                    ),
                }
            ),
        )

    @property
    def _transport_client(self):
        return streamablehttp_client

    def __eq__(self, value: object, /) -> bool:
        return super().__eq__(value) and isinstance(value, MCPServerStreamableHTTP) and self.url == value.url


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


def _mcp_server_discriminator(value: dict[str, Any]) -> str | None:
    if 'url' in value:
        if value['url'].endswith('/sse'):
            return 'sse'
        return 'streamable-http'
    return 'stdio'


class MCPServerConfig(BaseModel):
    """Configuration for MCP servers."""

    mcp_servers: Annotated[
        dict[
            str,
            Annotated[
                Annotated[MCPServerStdio, Tag('stdio')]
                | Annotated[MCPServerStreamableHTTP, Tag('streamable-http')]
                | Annotated[MCPServerSSE, Tag('sse')],
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


def load_mcp_servers(config_path: str | Path) -> list[MCPServerStdio | MCPServerStreamableHTTP | MCPServerSSE]:
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
    config = MCPServerConfig.model_validate(expanded_config_data)

    servers: list[MCPServerStdio | MCPServerStreamableHTTP | MCPServerSSE] = []
    for name, server in config.mcp_servers.items():
        server.id = name
        server.tool_prefix = name
        servers.append(server)

    return servers
