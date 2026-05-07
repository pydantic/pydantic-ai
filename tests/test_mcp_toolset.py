"""Tests for `pydantic_ai.mcp.MCPToolset`.

`TestMCPToolsetConstruction` covers the unique construction logic the new class adds on top of the
legacy `FastMCPToolset` and `MCPServer*` paths — kwarg conflict detection, HTTP transport adapter
for `http_client=`, sampling shortcut, the cache-invalidating message handler.

`TestMCPToolsetIntegration` exercises lifecycle, tool calling, resource methods, and caching
against an in-process FastMCP server. The fixture mirrors the one in `test_fastmcp.py` so the new
class is validated against the same surface area as the legacy `FastMCPToolset`.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import httpx
import pytest

from pydantic_ai import models
from pydantic_ai._run_context import RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from .conftest import try_import

with try_import() as imports_successful:
    from fastmcp.client import Client
    from fastmcp.client.transports import (
        SSETransport,
        StreamableHttpTransport,
    )
    from fastmcp.exceptions import ToolError
    from fastmcp.server import FastMCP
    from mcp import types as mcp_types

    from pydantic_ai.mcp import (
        MCPError,
        MCPToolset,
        ResourceAnnotations,
        ResourceTemplate,
        load_mcp_toolsets,
    )


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='fastmcp not installed'),
    pytest.mark.anyio,
]


# Construction tests don't need a server and don't take async fixtures.


class TestMCPToolsetConstruction:
    def test_url_builds_streamable_http_transport(self):
        toolset = MCPToolset('https://example.com/mcp')
        assert isinstance(toolset.client.transport, StreamableHttpTransport)

    def test_sse_url_builds_sse_transport_with_headers(self):
        toolset = MCPToolset('https://example.com/sse', headers={'X-Key': 'foo'})
        assert isinstance(toolset.client.transport, SSETransport)
        assert toolset.client.transport.headers == {'X-Key': 'foo'}

    def test_url_with_headers_routes_through_explicit_transport(self):
        toolset = MCPToolset('https://example.com/mcp', headers={'X-Key': 'foo'})
        assert isinstance(toolset.client.transport, StreamableHttpTransport)
        assert toolset.client.transport.headers == {'X-Key': 'foo'}

    def test_http_client_kwarg_uses_factory(self):
        client = httpx.AsyncClient()
        toolset = MCPToolset('https://example.com/mcp', http_client=client)
        assert isinstance(toolset.client.transport, StreamableHttpTransport)
        assert toolset.client.transport.httpx_client_factory is not None
        assert toolset.client.transport.httpx_client_factory() is client

    def test_sse_url_with_http_client_uses_factory(self):
        client = httpx.AsyncClient()
        toolset = MCPToolset('https://example.com/sse', http_client=client)
        assert isinstance(toolset.client.transport, SSETransport)
        assert toolset.client.transport.httpx_client_factory is not None
        assert toolset.client.transport.httpx_client_factory() is client

    def test_headers_and_http_client_conflict_raises(self):
        with pytest.raises(ValueError, match='mutually exclusive'):
            MCPToolset(
                'https://example.com/mcp',
                headers={'X-Key': 'foo'},
                http_client=httpx.AsyncClient(),
            )

    def test_pre_built_client_with_handler_kwargs_raises(self):
        client = Client('https://example.com/mcp')
        with pytest.raises(ValueError, match='pre-built `fastmcp.Client`'):
            MCPToolset(client, headers={'X-Key': 'foo'})

    def test_pre_built_client_with_overridden_init_timeout_raises(self):
        client = Client('https://example.com/mcp')
        with pytest.raises(ValueError, match='init_timeout'):
            MCPToolset(client, init_timeout=30)

    def test_pre_built_client_with_overridden_read_timeout_raises(self):
        client = Client('https://example.com/mcp')
        with pytest.raises(ValueError, match='read_timeout'):
            MCPToolset(client, read_timeout=30)

    def test_pre_built_client_used_as_is(self):
        client = Client('https://example.com/mcp')
        toolset = MCPToolset(client)
        assert toolset.client is client

    def test_sampling_model_and_handler_conflict(self):
        with pytest.raises(ValueError, match='sampling_model.*sampling_handler'):
            MCPToolset(
                'https://example.com/mcp',
                sampling_model=models.infer_model('test'),
                sampling_handler=lambda *_: None,  # type: ignore[arg-type,return-value]
            )

    def test_sampling_model_installs_handler(self):
        toolset = MCPToolset('https://example.com/mcp', sampling_model=models.infer_model('test'))
        assert toolset.client._session_kwargs.get('sampling_callback') is not None  # pyright: ignore[reportPrivateUsage]

    def test_id_property(self):
        toolset = MCPToolset('https://example.com/mcp', id='example')
        assert toolset.id == 'example'

    def test_repr_includes_id(self):
        toolset = MCPToolset('https://example.com/mcp', id='example')
        assert "id='example'" in repr(toolset)

    def test_repr_without_id(self):
        toolset = MCPToolset('https://example.com/mcp')
        assert 'MCPToolset(client=' in repr(toolset)

    def test_pre_init_property_access_raises(self):
        toolset = MCPToolset('https://example.com/mcp')
        with pytest.raises(AttributeError, match='only available after initialization'):
            _ = toolset.server_info
        with pytest.raises(AttributeError, match='only available after initialization'):
            _ = toolset.capabilities
        with pytest.raises(AttributeError, match='only available after initialization'):
            _ = toolset.instructions
        assert toolset.is_running is False

    def test_tool_name_conflict_hint_mentions_prefixed(self):
        toolset = MCPToolset('https://example.com/mcp')
        assert '.prefixed' in toolset.tool_name_conflict_hint

    def test_eq_and_hash(self):
        client = Client('https://example.com/mcp')
        a = MCPToolset(client, id='same')
        b = MCPToolset(client, id='same')
        c = MCPToolset(client, id='other')
        assert a == b
        assert a != c
        assert hash(a) == hash(b)

    def test_id_setter(self):
        toolset = MCPToolset('https://example.com/mcp')
        toolset.id = 'new'
        assert toolset.id == 'new'


class TestResourceTypeMapping:
    """The PAI-shaped `Resource` / `ResourceTemplate` / `MCPError` types are kept under
    `pydantic_ai.mcp.*`. They were ported from the deprecated `MCPServer*` path; these tests pin
    the wire-level field mapping so drifts from the MCP SDK schema are caught."""

    def test_resource_template_from_mcp_sdk(self):
        sdk_template = mcp_types.ResourceTemplate(
            uriTemplate='file:///{path}',
            name='file',
            title='File',
            description='Read a file',
            mimeType='application/octet-stream',
            annotations=mcp_types.Annotations(audience=['user'], priority=0.7),
            _meta={'origin': 'test'},
        )
        template = ResourceTemplate.from_mcp_sdk(sdk_template)
        assert template.uri_template == 'file:///{path}'
        assert template.name == 'file'
        assert template.title == 'File'
        assert template.description == 'Read a file'
        assert template.mime_type == 'application/octet-stream'
        assert isinstance(template.annotations, ResourceAnnotations)
        assert template.annotations.audience == ['user']
        assert template.annotations.priority == 0.7
        assert template.metadata == {'origin': 'test'}

    def test_mcp_error_str_includes_code_and_data(self):
        err = MCPError('boom', code=-32002, data={'extra': 1})
        assert 'boom' in str(err)
        assert '-32002' in str(err)
        assert 'extra' in str(err)

    def test_mcp_error_str_without_data(self):
        err = MCPError('boom', code=-32002)
        assert 'boom' in str(err)
        assert '-32002' in str(err)


@pytest.fixture
async def fastmcp_server() -> FastMCP[None]:
    """In-process FastMCP server with a representative tool/resource surface."""
    server: FastMCP[None] = FastMCP('test_server', instructions='You are an MCP test server.')

    @server.tool()
    async def echo(message: str) -> str:
        """Echo a message back."""
        return f'Echo: {message}'

    @server.tool()
    async def add(a: int, b: int) -> dict[str, int]:
        """Add two numbers and return the result."""
        return {'sum': a + b}

    @server.tool()
    async def boom() -> str:
        """A tool that always raises an error — used to test error handling."""
        raise ValueError('boom')

    @server.resource('resource://greeting.txt')
    async def greeting() -> str:
        return 'Hello, world!'

    @server.resource('resource://{name}/profile.json')
    async def profile(name: str) -> str:
        return f'{{"name": "{name}"}}'

    return server


@pytest.fixture
def run_context() -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
    )


class TestMCPToolsetIntegration:
    """End-to-end coverage against an in-process FastMCP server."""

    async def test_lifecycle_exposes_init_state(self, fastmcp_server: FastMCP[None]):
        toolset = MCPToolset(fastmcp_server)
        assert toolset.is_running is False
        async with toolset:
            assert toolset.is_running is True
            assert toolset.server_info.name == 'test_server'
            assert toolset.capabilities.tools is True
            assert toolset.instructions == 'You are an MCP test server.'
        assert toolset.is_running is False

    async def test_aexit_extra_call_raises(self, fastmcp_server: FastMCP[None]):
        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            pass
        with pytest.raises(ValueError, match='called more times than'):
            await toolset.__aexit__()

    async def test_get_tools_caches_and_lists(self, fastmcp_server: FastMCP[None], run_context: RunContext[None]):
        toolset = MCPToolset(fastmcp_server, include_instructions=True)
        async with toolset:
            tools_first = await toolset.get_tools(run_context)
            tools_second = await toolset.get_tools(run_context)
            assert set(tools_first) == {'echo', 'add', 'boom'}
            # Second call should hit the cache (covers the cached-return branch).
            assert tools_first['echo'].tool_def.description == tools_second['echo'].tool_def.description

    async def test_get_instructions_when_enabled(self, fastmcp_server: FastMCP[None], run_context: RunContext[None]):
        toolset = MCPToolset(fastmcp_server, include_instructions=True)
        async with toolset:
            part = await toolset.get_instructions(run_context)
        assert part is not None
        assert part.content == 'You are an MCP test server.'
        assert part.dynamic is True

    async def test_get_instructions_returns_none_when_disabled(
        self, fastmcp_server: FastMCP[None], run_context: RunContext[None]
    ):
        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            assert await toolset.get_instructions(run_context) is None

    async def test_get_instructions_returns_none_pre_init(
        self, fastmcp_server: FastMCP[None], run_context: RunContext[None]
    ):
        toolset = MCPToolset(fastmcp_server, include_instructions=True)
        # Without entering, instructions aren't populated yet.
        assert await toolset.get_instructions(run_context) is None

    async def test_tools_no_caching_when_disabled(self, fastmcp_server: FastMCP[None], run_context: RunContext[None]):
        toolset = MCPToolset(fastmcp_server, cache_tools=False)
        async with toolset:
            await toolset.get_tools(run_context)
            assert toolset._cached_tools is None  # pyright: ignore[reportPrivateUsage]

    async def test_call_tool_returns_structured_content(
        self, fastmcp_server: FastMCP[None], run_context: RunContext[None]
    ):
        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            tools = await toolset.get_tools(run_context)
            result = await toolset.call_tool('add', {'a': 2, 'b': 3}, run_context, tools['add'])
        assert result == {'sum': 5}

    async def test_call_tool_returns_text(self, fastmcp_server: FastMCP[None], run_context: RunContext[None]):
        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            tools = await toolset.get_tools(run_context)
            result = await toolset.call_tool('echo', {'message': 'hi'}, run_context, tools['echo'])
        assert result == 'Echo: hi'

    async def test_tool_error_raises_model_retry(self, fastmcp_server: FastMCP[None], run_context: RunContext[None]):
        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            tools = await toolset.get_tools(run_context)
            with pytest.raises(ModelRetry):
                await toolset.call_tool('boom', {}, run_context, tools['boom'])

    async def test_tool_error_raises_tool_error_when_configured(
        self, fastmcp_server: FastMCP[None], run_context: RunContext[None]
    ):
        toolset = MCPToolset(fastmcp_server, tool_error_behavior='error')
        async with toolset:
            tools = await toolset.get_tools(run_context)
            with pytest.raises(ToolError):
                await toolset.call_tool('boom', {}, run_context, tools['boom'])

    async def test_process_tool_call_hook_runs(self, fastmcp_server: FastMCP[None], run_context: RunContext[None]):
        seen: list[tuple[str, dict[str, Any]]] = []

        async def hook(ctx: RunContext[Any], call_tool: Any, name: str, args: dict[str, Any]) -> Any:
            seen.append((name, args))
            return await call_tool(name, args, None)

        toolset = MCPToolset(fastmcp_server, process_tool_call=hook)
        async with toolset:
            tools = await toolset.get_tools(run_context)
            result = await toolset.call_tool('echo', {'message': 'hi'}, run_context, tools['echo'])
        assert result == 'Echo: hi'
        assert seen == [('echo', {'message': 'hi'})]

    async def test_list_resources_returns_pai_types(self, fastmcp_server: FastMCP[None]):
        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            resources = await toolset.list_resources()
            cached = await toolset.list_resources()
        assert any(r.name == 'greeting' for r in resources)
        assert resources == cached

    async def test_list_resources_no_caching_when_disabled(self, fastmcp_server: FastMCP[None]):
        toolset = MCPToolset(fastmcp_server, cache_resources=False)
        async with toolset:
            await toolset.list_resources()
            assert toolset._cached_resources is None  # pyright: ignore[reportPrivateUsage]

    async def test_list_resource_templates(self, fastmcp_server: FastMCP[None]):
        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            templates = await toolset.list_resource_templates()
        # The `profile` resource has a `{name}` placeholder so it's a template.
        assert any('{name}' in t.uri_template for t in templates)

    async def test_read_resource_text(self, fastmcp_server: FastMCP[None]):
        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            content = await toolset.read_resource('resource://greeting.txt')
        assert content == 'Hello, world!'

    async def test_read_resource_via_resource_object(self, fastmcp_server: FastMCP[None]):
        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            resources = await toolset.list_resources()
            greeting = next(r for r in resources if r.name == 'greeting')
            content = await toolset.read_resource(greeting)
        assert content == 'Hello, world!'

    async def test_resource_methods_without_capability(self):
        """When the server doesn't advertise resources, the methods return empty lists."""
        server: FastMCP[None] = FastMCP('no_resources_server')

        @server.tool()
        async def noop() -> str:
            return 'ok'

        toolset = MCPToolset(server)
        async with toolset:
            assert await toolset.list_resources() == []
            assert await toolset.list_resource_templates() == []

    async def test_message_handler_invalidates_caches(
        self, fastmcp_server: FastMCP[None], run_context: RunContext[None]
    ):
        from pydantic_ai.mcp import _CacheInvalidatingMessageHandler  # type: ignore[attr-defined]

        toolset = MCPToolset(fastmcp_server)
        handler = _CacheInvalidatingMessageHandler(toolset, user_handler=None)
        toolset._cached_tools = []  # pyright: ignore[reportPrivateUsage]
        toolset._cached_resources = []  # pyright: ignore[reportPrivateUsage]

        await handler(
            mcp_types.ServerNotification(
                root=mcp_types.ToolListChangedNotification(method='notifications/tools/list_changed')
            )
        )
        assert toolset._cached_tools is None  # pyright: ignore[reportPrivateUsage]

        toolset._cached_tools = []  # pyright: ignore[reportPrivateUsage]
        await handler(
            mcp_types.ServerNotification(
                root=mcp_types.ResourceListChangedNotification(method='notifications/resources/list_changed')
            )
        )
        assert toolset._cached_resources is None  # pyright: ignore[reportPrivateUsage]

    async def test_message_handler_forwards_to_user_handler(self, fastmcp_server: FastMCP[None]):
        from pydantic_ai.mcp import _CacheInvalidatingMessageHandler  # type: ignore[attr-defined]

        seen: list[Any] = []

        async def user_handler(message: Any) -> None:
            seen.append(message)

        toolset = MCPToolset(fastmcp_server)
        handler = _CacheInvalidatingMessageHandler(toolset, user_handler=user_handler)
        notification = mcp_types.ServerNotification(
            root=mcp_types.ToolListChangedNotification(method='notifications/tools/list_changed')
        )
        await handler(notification)
        assert seen == [notification]


class TestLoadMCPToolsets:
    async def test_loads_toolsets_from_config_with_prefix(self):
        config = {
            'mcpServers': {
                'alpha': {'command': 'python', 'args': ['-m', 'tests.mcp_server']},
            }
        }
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / 'mcp.json'
            config_path.write_text(json.dumps(config))
            toolsets = load_mcp_toolsets(config_path)
        # Single server entry, wrapped with `.prefixed('alpha')`.
        assert len(toolsets) == 1
        # The wrapped toolset is a `PrefixedToolset`, not an `MCPToolset` directly.
        from pydantic_ai.toolsets.prefixed import PrefixedToolset

        assert isinstance(toolsets[0], PrefixedToolset)
        assert isinstance(toolsets[0].wrapped, MCPToolset)

    async def test_load_mcp_toolsets_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_mcp_toolsets('/nonexistent/path/to/config.json')

    async def test_load_mcp_toolsets_http_entry(self):
        config = {
            'mcpServers': {
                'beta': {'url': 'http://localhost:8000/mcp', 'headers': {'X-Key': 'foo'}},
            }
        }
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / 'mcp.json'
            config_path.write_text(json.dumps(config))
            toolsets = load_mcp_toolsets(config_path)
        from pydantic_ai.toolsets.prefixed import PrefixedToolset

        assert len(toolsets) == 1
        assert isinstance(toolsets[0], PrefixedToolset)
        wrapped = toolsets[0].wrapped
        assert isinstance(wrapped, MCPToolset)
        # Headers flowed through to the FastMCP transport.
        assert isinstance(wrapped.client.transport, StreamableHttpTransport)
        assert wrapped.client.transport.headers == {'X-Key': 'foo'}


def test_construction_does_not_emit_warnings(recwarn: Any) -> None:
    """Building an `MCPToolset` from a URL must not emit `FastMCPDeprecationWarning` for the
    `sse_read_timeout` parameter — the StreamableHttp path migrated off it (the FastMCP `Client`
    `timeout` carries the read timeout instead)."""
    MCPToolset('https://example.com/mcp', headers={'X-Key': 'foo'})
    deprecation_messages = [str(w.message) for w in recwarn if 'sse_read_timeout' in str(w.message)]
    assert deprecation_messages == [], deprecation_messages
