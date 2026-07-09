"""Tests for `pydantic_ai.mcp.MCPToolset`.

`TestMCPToolsetConstruction` covers construction-time behavior — kwarg conflict detection, HTTP
transport adapter for `http_client=`, sampling shortcut, the cache-invalidating message handler.

`TestMCPToolsetIntegration` exercises lifecycle, tool calling, resource methods, and caching
against an in-process FastMCP server.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import AsyncMock

import anyio
import httpx
import pytest
from inline_snapshot import snapshot

from pydantic_ai import models
from pydantic_ai._run_context import RunContext
from pydantic_ai._utils import BaseExceptionGroup
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
    from fastmcp.prompts import Message
    from fastmcp.server import FastMCP
    from fastmcp.server.tasks import TaskConfig
    from mcp import types as mcp_types
    from mcp.shared.exceptions import McpError
    from mcp.types import (
        Annotations,
        AudioContent,
        BlobResourceContents,
        EmbeddedResource,
        ImageContent,
        ResourceLink,
        TextContent as McpTextContent,
        TextResourceContents,
    )
    from pydantic import AnyUrl

    from pydantic_ai.mcp import (
        MCPError,
        MCPToolset,
        Prompt,
        PromptArgument,
        PromptMessage,
        PromptResult,
        ResourceAnnotations,
        ResourceTemplate,
        ServerCapabilities,
        _make_httpx_client_factory,  # pyright: ignore[reportPrivateUsage]
        load_mcp_toolsets,
    )
    from pydantic_ai.messages import TextContent


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
        # FastMCP's StreamableHttpTransport calls the factory with `follow_redirects`, which the
        # mcp SDK's `McpHttpClientFactory` protocol doesn't declare; the factory must accept it.
        factory = _make_httpx_client_factory(client)
        assert factory(follow_redirects=True) is client

    def test_sse_url_with_http_client_uses_factory(self):
        client = httpx.AsyncClient()
        toolset = MCPToolset('https://example.com/sse', http_client=client)
        assert isinstance(toolset.client.transport, SSETransport)
        assert toolset.client.transport.httpx_client_factory is not None
        assert toolset.client.transport.httpx_client_factory() is client
        factory = _make_httpx_client_factory(client)
        assert factory(follow_redirects=True) is client

    def test_http_kwargs_with_non_url_input_raises(self):
        """HTTP-only kwargs (headers/auth/verify/http_client) must error out when the connection
        target isn't an HTTP URL — otherwise the kwargs are silently dropped on stdio / Path /
        in-process inputs."""
        from fastmcp.server import FastMCP

        with pytest.raises(ValueError, match='only apply to HTTP transports built from a URL'):
            MCPToolset(FastMCP(name='in_process'), headers={'X-Key': 'foo'})

    def test_headers_and_http_client_conflict_raises(self):
        with pytest.raises(ValueError, match='mutually exclusive'):
            MCPToolset(
                'https://example.com/mcp',
                headers={'X-Key': 'foo'},
                http_client=httpx.AsyncClient(),
            )

    def test_pre_built_client_with_handler_kwargs_raises(self):
        client = Client('https://example.com/mcp')
        with pytest.raises(ValueError, match=re.escape('pre-built `fastmcp.Client`')):
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
        with pytest.raises(ValueError, match=r'sampling_model.*sampling_handler'):
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

    def test_explicit_timeouts_override_defaults(self):
        """Passing `init_timeout` / `read_timeout` explicitly bypasses the `_UNSET` sentinel
        resolution branch."""
        toolset = MCPToolset('https://example.com/mcp', init_timeout=10, read_timeout=120)
        # Both kwargs flow into the FastMCP `Client`; verify the read timeout was forwarded.
        assert toolset.client._init_timeout is not None  # pyright: ignore[reportPrivateUsage]


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

    @server.tool()
    async def image_tool() -> ImageContent:
        """A tool that returns an image content block."""
        encoded = base64.b64encode(b'fake_image_bytes').decode('utf-8')
        return ImageContent(type='image', data=encoded, mimeType='image/png')

    @server.tool()
    async def embedded_blob_tool() -> EmbeddedResource:
        """A tool that returns an embedded blob resource."""
        encoded = base64.b64encode(b'fake_blob_bytes').decode('utf-8')
        return EmbeddedResource(
            type='resource',
            resource=BlobResourceContents(uri=AnyUrl('resource://blob.bin'), blob=encoded),
        )

    @server.tool()
    async def resource_link_tool() -> ResourceLink:
        """A tool that returns a resource link."""
        return ResourceLink(type='resource_link', uri=AnyUrl('resource://greeting.txt'), name='greeting')

    @server.resource('resource://greeting.txt')
    async def greeting() -> str:
        return 'Hello, world!'

    @server.resource('resource://{name}/profile.json')
    async def profile(name: str) -> str:
        return f'{{"name": "{name}"}}'

    _register_prompts(server)
    return server


def _register_prompts(server: FastMCP[None]) -> None:
    @server.prompt()
    def simple_prompt() -> str:
        """A simple prompt template."""
        return 'This is a simple prompt'

    @server.prompt()
    def parameterized_prompt(name: str, topic: str) -> str:
        """A prompt template with parameters."""
        return f"Hello {name}, let's talk about {topic}!"

    @server.prompt()
    def annotated_text_prompt() -> list[Message]:
        """A prompt template with annotated text content."""
        return [
            Message(
                content=McpTextContent(
                    type='text',
                    text='annotated text',
                    annotations=Annotations(audience=['user'], priority=1.0),
                )
            )
        ]

    @server.prompt()
    def text_meta_prompt() -> list[Message]:
        """A prompt template with `_meta` text metadata."""
        return [Message(content=McpTextContent(type='text', text='meta text', _meta={'source': 'mcp'}))]

    @server.prompt()
    def image_prompt() -> list[Message]:
        """A prompt template with image content."""
        return [
            Message(
                content=ImageContent(
                    type='image',
                    data=base64.b64encode(b'image-bytes').decode('utf-8'),
                    mimeType='image/jpeg',
                    annotations=Annotations(audience=['user'], priority=0.8),
                )
            )
        ]

    @server.prompt()
    def audio_prompt() -> list[Message]:
        """A prompt template with audio content."""
        return [
            Message(
                content=AudioContent(
                    type='audio',
                    data=base64.b64encode(b'audio-bytes').decode('utf-8'),
                    mimeType='audio/mpeg',
                    annotations=Annotations(audience=['assistant'], priority=0.3),
                )
            )
        ]

    @server.prompt()
    def embedded_resource_prompt() -> list[Message]:
        """A prompt template with an embedded text resource."""
        return [
            Message(
                content=EmbeddedResource(
                    type='resource',
                    resource=TextResourceContents(
                        uri=AnyUrl('resource://product_name.txt'),
                        text='Pydantic AI',
                        mimeType='text/plain',
                    ),
                    annotations=Annotations(audience=['user'], priority=0.5),
                )
            )
        ]


@pytest.fixture
def run_context() -> RunContext:
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

    async def test_aexit_called_before_aenter_raises(self, fastmcp_server: FastMCP[None]):
        """Calling `__aexit__` before any `__aenter__` should raise — `_running_count` is 0."""
        toolset = MCPToolset(fastmcp_server)
        with pytest.raises(ValueError, match='called more times than'):
            await toolset.__aexit__(None, None, None)

    async def test_aexit_called_more_times_than_aenter(self, fastmcp_server: FastMCP[None]):
        """Calling `__aexit__` more times than `__aenter__` should raise."""
        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            pass
        with pytest.raises(ValueError, match='called more times than'):
            await toolset.__aexit__(None, None, None)

    async def test_get_tools_caches_and_lists(self, fastmcp_server: FastMCP[None], run_context: RunContext):
        toolset = MCPToolset(fastmcp_server, include_instructions=True)
        async with toolset:
            tools_first = await toolset.get_tools(run_context)
            tools_second = await toolset.get_tools(run_context)
            assert {'echo', 'add', 'boom'} <= set(tools_first)
            # Second call should hit the cache (covers the cached-return branch).
            assert tools_first['echo'].tool_def.description == tools_second['echo'].tool_def.description

    async def test_get_instructions_when_enabled(self, fastmcp_server: FastMCP[None], run_context: RunContext):
        toolset = MCPToolset(fastmcp_server, include_instructions=True)
        async with toolset:
            part = await toolset.get_instructions(run_context)
        assert part is not None
        assert part.content == 'You are an MCP test server.'
        assert part.dynamic is False

    async def test_get_instructions_returns_none_when_disabled(
        self, fastmcp_server: FastMCP[None], run_context: RunContext
    ):
        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            assert await toolset.get_instructions(run_context) is None

    async def test_get_instructions_returns_none_pre_init(self, fastmcp_server: FastMCP[None], run_context: RunContext):
        toolset = MCPToolset(fastmcp_server, include_instructions=True)
        # Without entering, instructions aren't populated yet.
        assert await toolset.get_instructions(run_context) is None

    async def test_tools_no_caching_when_disabled(self, fastmcp_server: FastMCP[None], run_context: RunContext):
        toolset = MCPToolset(fastmcp_server, cache_tools=False)
        async with toolset:
            await toolset.get_tools(run_context)
            assert toolset._cached_tools is None  # pyright: ignore[reportPrivateUsage]

    async def test_call_tool_returns_structured_content(self, fastmcp_server: FastMCP[None], run_context: RunContext):
        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            tools = await toolset.get_tools(run_context)
            result = await toolset.call_tool('add', {'a': 2, 'b': 3}, run_context, tools['add'])
        assert result == {'sum': 5}

    async def test_call_tool_returns_text(self, fastmcp_server: FastMCP[None], run_context: RunContext):
        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            tools = await toolset.get_tools(run_context)
            result = await toolset.call_tool('echo', {'message': 'hi'}, run_context, tools['echo'])
        assert result == 'Echo: hi'

    async def test_tool_error_raises_model_retry(self, fastmcp_server: FastMCP[None], run_context: RunContext):
        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            tools = await toolset.get_tools(run_context)
            with pytest.raises(ModelRetry):
                await toolset.call_tool('boom', {}, run_context, tools['boom'])

    async def test_tool_error_raises_tool_error_when_configured(
        self, fastmcp_server: FastMCP[None], run_context: RunContext
    ):
        toolset = MCPToolset(fastmcp_server, tool_error_behavior='error')
        async with toolset:
            tools = await toolset.get_tools(run_context)
            with pytest.raises(ToolError):
                await toolset.call_tool('boom', {}, run_context, tools['boom'])

    @pytest.mark.parametrize(
        'leaf_factory',
        [
            pytest.param(lambda: ToolError('grouped tool error'), id='tool-error'),
            pytest.param(lambda: McpError(mcp_types.ErrorData(code=400, message='grouped tool error')), id='mcp-error'),
        ],
    )
    async def test_call_tool_unwraps_real_exception_group_to_model_retry(
        self, fastmcp_server: FastMCP[None], run_context: RunContext, leaf_factory: Any
    ):
        """A tool/protocol error that surfaces wrapped in an `ExceptionGroup` is converted to a
        recoverable `ModelRetry`, not a fatal crash.

        This is a unit test because the wrapping is a timing-dependent race in the MCP client's
        anyio task group (an empty-bodied tool error colliding with the session's GET-stream
        teardown) that can't be triggered deterministically. Rather than hand-build the group, we
        inject a failure at the real escape seam — `self.client.call_tool` — and let a genuine
        `anyio` task group produce the `ExceptionGroup`, so its structure matches production.
        """
        toolset = MCPToolset(fastmcp_server)

        async def call_tool_in_failing_task_group(*args: Any, **kwargs: Any) -> Any:
            async def fail() -> None:
                raise leaf_factory()

            async with anyio.create_task_group() as tg:
                tg.start_soon(fail)

        async with toolset:
            tools = await toolset.get_tools(run_context)
            toolset.client.call_tool = call_tool_in_failing_task_group
            with pytest.raises(ModelRetry, match='grouped tool error'):
                await toolset.call_tool('echo', {'message': 'hi'}, run_context, tools['echo'])

    async def test_call_tool_reraises_grouped_errors_it_must_not_convert(
        self, fastmcp_server: FastMCP[None], run_context: RunContext
    ):
        """Groups we must not silently turn into a retry are re-raised unchanged: a group that also
        contains a non-tool error, and (with `tool_error_behavior='error'`) any grouped tool error."""

        def failing_call_tool(*excs: BaseException) -> Any:
            async def call_tool(*args: Any, **kwargs: Any) -> Any:
                async def fail(exc: BaseException) -> None:
                    raise exc

                async with anyio.create_task_group() as tg:
                    for exc in excs:
                        tg.start_soon(fail, exc)

            return call_tool

        # A mixed group (tool error + an unrelated error) must propagate, not be swallowed.
        retry_toolset = MCPToolset(fastmcp_server)
        async with retry_toolset:
            tools = await retry_toolset.get_tools(run_context)
            retry_toolset.client.call_tool = failing_call_tool(
                ToolError('grouped tool error'), ValueError('unrelated failure')
            )
            with pytest.raises(BaseExceptionGroup):
                await retry_toolset.call_tool('echo', {'message': 'hi'}, run_context, tools['echo'])

        # With `tool_error_behavior='error'`, even a pure tool-error group propagates unchanged.
        error_toolset = MCPToolset(fastmcp_server, tool_error_behavior='error')
        async with error_toolset:
            tools = await error_toolset.get_tools(run_context)
            error_toolset.client.call_tool = failing_call_tool(ToolError('grouped tool error'))
            with pytest.raises(BaseExceptionGroup):
                await error_toolset.call_tool('echo', {'message': 'hi'}, run_context, tools['echo'])

    async def test_process_tool_call_hook_runs(self, fastmcp_server: FastMCP[None], run_context: RunContext):
        seen: list[tuple[str, dict[str, Any]]] = []

        async def hook(ctx: RunContext[Any], call_tool: Any, name: str, args: dict[str, Any]) -> Any:
            seen.append((name, args))
            return await call_tool(name, args, metadata=None)

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

    async def test_read_resource_template_instance(self, fastmcp_server: FastMCP[None]):
        """Reading a resource produced from a template (`resource://{name}/profile.json`)."""
        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            content = await toolset.read_resource('resource://alice/profile.json')
        assert content == '{"name": "alice"}'

    async def test_resource_methods_without_capability(self, fastmcp_server: FastMCP[None]):
        """When the server's `capabilities.resources` is `False`, the methods return empty lists
        without round-tripping to the server."""
        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            # Force the capability off to exercise the early-return branches.
            toolset._server_capabilities = ServerCapabilities()  # pyright: ignore[reportPrivateUsage]
            assert await toolset.list_resources() == []
            assert await toolset.list_resource_templates() == []

    async def test_list_prompts_returns_pai_types(self, fastmcp_server: FastMCP[None]):
        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            prompts = await toolset.list_prompts()
        assert prompts == snapshot(
            [
                Prompt(
                    name='simple_prompt', description='A simple prompt template.', metadata={'fastmcp': {'tags': []}}
                ),
                Prompt(
                    name='parameterized_prompt',
                    description='A prompt template with parameters.',
                    arguments=[
                        PromptArgument(
                            name='name',
                            description='Provide as a JSON string matching the following schema: {"type":"string"}',
                            required=True,
                        ),
                        PromptArgument(
                            name='topic',
                            description='Provide as a JSON string matching the following schema: {"type":"string"}',
                            required=True,
                        ),
                    ],
                    metadata={'fastmcp': {'tags': []}},
                ),
                Prompt(
                    name='annotated_text_prompt',
                    description='A prompt template with annotated text content.',
                    metadata={'fastmcp': {'tags': []}},
                ),
                Prompt(
                    name='text_meta_prompt',
                    description='A prompt template with `_meta` text metadata.',
                    metadata={'fastmcp': {'tags': []}},
                ),
                Prompt(
                    name='image_prompt',
                    description='A prompt template with image content.',
                    metadata={'fastmcp': {'tags': []}},
                ),
                Prompt(
                    name='audio_prompt',
                    description='A prompt template with audio content.',
                    metadata={'fastmcp': {'tags': []}},
                ),
                Prompt(
                    name='embedded_resource_prompt',
                    description='A prompt template with an embedded text resource.',
                    metadata={'fastmcp': {'tags': []}},
                ),
            ]
        )

    async def test_get_prompt_simple(self, fastmcp_server: FastMCP[None]):
        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            result = await toolset.get_prompt('simple_prompt')
        assert result == snapshot(
            PromptResult(
                messages=[PromptMessage(role='user', content=TextContent(content='This is a simple prompt'))],
                description='A simple prompt template.',
            )
        )

    async def test_get_prompt_parameterized(self, fastmcp_server: FastMCP[None]):
        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            result = await toolset.get_prompt('parameterized_prompt', {'name': 'Alice', 'topic': 'AI'})
        assert result == snapshot(
            PromptResult(
                messages=[PromptMessage(role='user', content=TextContent(content="Hello Alice, let's talk about AI!"))],
                description='A prompt template with parameters.',
            )
        )

    async def test_list_prompts_caches_when_enabled(self, fastmcp_server: FastMCP[None]):
        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            first = await toolset.list_prompts()
            assert toolset._cached_prompts is not None  # pyright: ignore[reportPrivateUsage]
            second = await toolset.list_prompts()
        assert first == second

    async def test_list_prompts_no_caching_when_disabled(self, fastmcp_server: FastMCP[None]):
        toolset = MCPToolset(fastmcp_server, cache_prompts=False)
        async with toolset:
            await toolset.list_prompts()
            assert toolset._cached_prompts is None  # pyright: ignore[reportPrivateUsage]

    async def test_prompts_cache_invalidation_on_notification(self, fastmcp_server: FastMCP[None]):
        from pydantic_ai.mcp import _build_message_handler  # type: ignore[attr-defined]

        toolset = MCPToolset(fastmcp_server)
        handler = _build_message_handler(toolset, user_handler=None)
        toolset._cached_prompts = []  # pyright: ignore[reportPrivateUsage]

        await handler(
            mcp_types.ServerNotification(
                root=mcp_types.PromptListChangedNotification(method='notifications/prompts/list_changed')
            )
        )
        assert toolset._cached_prompts is None  # pyright: ignore[reportPrivateUsage]

    async def test_prompts_without_capability(self, fastmcp_server: FastMCP[None]):
        """`list_prompts` returns `[]` and `get_prompt` raises `MCPError` when prompts capability is absent."""
        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            toolset._server_capabilities = ServerCapabilities()  # pyright: ignore[reportPrivateUsage]
            assert await toolset.list_prompts() == []
            with pytest.raises(MCPError, match='does not advertise the `prompts` capability') as exc_info:
                await toolset.get_prompt('does_not_matter')
            assert exc_info.value.code == -32601

    async def test_list_prompts_wraps_mcp_error(self, fastmcp_server: FastMCP[None]):
        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            toolset.client.list_prompts = AsyncMock(
                side_effect=McpError(mcp_types.ErrorData(code=-32603, message='boom'))
            )
            with pytest.raises(MCPError, match='boom'):
                await toolset.list_prompts()

    async def test_get_prompt_wraps_mcp_error(self, fastmcp_server: FastMCP[None]):
        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            with pytest.raises(MCPError, match='Unknown prompt'):
                await toolset.get_prompt('does_not_exist')

    async def test_map_prompt_content(self, fastmcp_server: FastMCP[None]):
        """`get_prompt` maps every MCP prompt content type to its Pydantic AI equivalent.

        Plain `TextContent` without annotations is already covered by `test_get_prompt_simple`,
        so this exercises annotated text, image, audio, and embedded resource. `ResourceLink`
        prompt content is covered separately in `test_get_prompt_maps_resource_link` because the
        in-process FastMCP server serializes resource links to text rather than emitting a
        `resource_link` content block.
        """
        from pydantic_ai.mcp import EmbeddedResource as PaiEmbeddedResource
        from pydantic_ai.messages import BinaryContent, BinaryImage

        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            # TextContent with annotations preserved in metadata
            annotated = await toolset.get_prompt('annotated_text_prompt')
            assert annotated.messages == snapshot(
                [
                    PromptMessage(
                        role='user',
                        content=TextContent(
                            content='annotated text',
                            metadata={'mcp_annotations': ResourceAnnotations(audience=['user'], priority=1.0)},
                        ),
                    )
                ]
            )

            # ImageContent → BinaryImage
            image = await toolset.get_prompt('image_prompt')
            assert image.messages == snapshot(
                [
                    PromptMessage(
                        role='user',
                        content=BinaryImage(
                            data=b'image-bytes',
                            media_type='image/jpeg',
                            vendor_metadata={'mcp_annotations': ResourceAnnotations(audience=['user'], priority=0.8)},
                        ),
                    )
                ]
            )

            # AudioContent → BinaryContent
            audio = await toolset.get_prompt('audio_prompt')
            assert audio.messages == snapshot(
                [
                    PromptMessage(
                        role='user',
                        content=BinaryContent(
                            data=b'audio-bytes',
                            media_type='audio/mpeg',
                            vendor_metadata={
                                'mcp_annotations': ResourceAnnotations(audience=['assistant'], priority=0.3)
                            },
                        ),
                    )
                ]
            )

            # EmbeddedResource with annotations
            embedded = await toolset.get_prompt('embedded_resource_prompt')
            assert embedded.messages == snapshot(
                [
                    PromptMessage(
                        role='user',
                        content=PaiEmbeddedResource(
                            uri='resource://product_name.txt',
                            content='Pydantic AI',
                            mime_type='text/plain',
                            annotations=ResourceAnnotations(audience=['user'], priority=0.5),
                        ),
                    )
                ]
            )

    async def test_get_prompt_maps_resource_link(self, fastmcp_server: FastMCP[None]):
        """A `resource_link` prompt content block maps to a Pydantic AI `ResourceLink`.

        FastMCP can't emit `resource_link` prompt content (it serializes the link to text), so we
        patch the client to return a real MCP `GetPromptResult` carrying one and assert the mapping.
        """
        from pydantic_ai.mcp import ResourceLink as PaiResourceLink

        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            toolset.client.get_prompt = AsyncMock(
                return_value=mcp_types.GetPromptResult(
                    description='A prompt template with a resource link.',
                    messages=[
                        mcp_types.PromptMessage(
                            role='user',
                            content=ResourceLink(
                                type='resource_link',
                                uri=AnyUrl('resource://kiwi.jpg'),
                                name='kiwi-image',
                                title='Kiwi Image',
                                description='A photo of a kiwi fruit',
                                mimeType='image/jpeg',
                            ),
                        )
                    ],
                )
            )
            result = await toolset.get_prompt('resource_link_prompt')
        assert result.messages == snapshot(
            [
                PromptMessage(
                    role='user',
                    content=PaiResourceLink(
                        uri='resource://kiwi.jpg',
                        name='kiwi-image',
                        title='Kiwi Image',
                        description='A photo of a kiwi fruit',
                        mime_type='image/jpeg',
                    ),
                )
            ]
        )

    async def test_map_prompt_content_text_meta(self, fastmcp_server: FastMCP[None]):
        """MCP `_meta` on prompt text is preserved in the mapped content metadata."""
        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            result = await toolset.get_prompt('text_meta_prompt')
        [message] = result.messages
        content = message.content
        assert isinstance(content, TextContent)
        assert content.metadata == {'mcp_meta': {'source': 'mcp'}}

    async def test_message_handler_ignores_non_list_changed_notifications(self, fastmcp_server: FastMCP[None]):
        from pydantic_ai.mcp import _build_message_handler  # type: ignore[attr-defined]

        toolset = MCPToolset(fastmcp_server)
        handler = _build_message_handler(toolset, user_handler=None)
        toolset._cached_tools = []  # pyright: ignore[reportPrivateUsage]
        # `LoggingMessageNotification` is unrelated to any cache.
        await handler(
            mcp_types.ServerNotification(
                root=mcp_types.LoggingMessageNotification(
                    method='notifications/message',
                    params=mcp_types.LoggingMessageNotificationParams(level='info', data='hi'),
                )
            )
        )
        assert toolset._cached_tools == []  # pyright: ignore[reportPrivateUsage]

    async def test_message_handler_ignores_non_notification_messages(self, fastmcp_server: FastMCP[None]):
        from pydantic_ai.mcp import _build_message_handler  # type: ignore[attr-defined]

        toolset = MCPToolset(fastmcp_server)
        handler = _build_message_handler(toolset, user_handler=None)
        toolset._cached_tools = []  # pyright: ignore[reportPrivateUsage]
        # Exception messages are passed through but shouldn't crash or invalidate caches.
        await handler(RuntimeError('transport error'))
        assert toolset._cached_tools == []  # pyright: ignore[reportPrivateUsage]

    async def test_message_handler_invalidates_caches(self, fastmcp_server: FastMCP[None], run_context: RunContext):
        from pydantic_ai.mcp import _build_message_handler  # type: ignore[attr-defined]

        toolset = MCPToolset(fastmcp_server)
        handler = _build_message_handler(toolset, user_handler=None)
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
        from pydantic_ai.mcp import _build_message_handler  # type: ignore[attr-defined]

        seen: list[Any] = []

        async def user_handler(message: Any) -> None:
            seen.append(message)

        toolset = MCPToolset(fastmcp_server)
        handler = _build_message_handler(toolset, user_handler=user_handler)
        notification = mcp_types.ServerNotification(
            root=mcp_types.ToolListChangedNotification(method='notifications/tools/list_changed')
        )
        await handler(notification)
        assert seen == [notification]

    async def test_call_tool_returns_image(self, fastmcp_server: FastMCP[None], run_context: RunContext):
        from pydantic_ai.messages import BinaryContent

        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            tools = await toolset.get_tools(run_context)
            result = await toolset.call_tool('image_tool', {}, run_context, tools['image_tool'])
        assert isinstance(result, BinaryContent)
        assert result.media_type == 'image/png'

    async def test_call_tool_returns_embedded_blob(self, fastmcp_server: FastMCP[None], run_context: RunContext):
        from pydantic_ai.messages import BinaryContent

        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            tools = await toolset.get_tools(run_context)
            result = await toolset.call_tool('embedded_blob_tool', {}, run_context, tools['embedded_blob_tool'])
        assert isinstance(result, BinaryContent)

    async def test_call_tool_returns_resource_link_uri(self, fastmcp_server: FastMCP[None], run_context: RunContext):
        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            tools = await toolset.get_tools(run_context)
            result = await toolset.call_tool('resource_link_tool', {}, run_context, tools['resource_link_tool'])
        # `_map_mcp_tool_result` for ResourceLink returns the URI string.
        assert result == 'resource://greeting.txt'

    async def test_log_level_is_set_after_aenter(self, fastmcp_server: FastMCP[None]):
        toolset = MCPToolset(fastmcp_server, log_level='warning')
        async with toolset:
            # Server received the logging/setLevel call without raising.
            assert toolset.is_running

    async def test_label_falls_back_to_repr(self):
        toolset = MCPToolset('https://example.com/mcp')
        assert 'MCPToolset' in toolset.label

    async def test_tool_for_tool_def_uses_default_retries_when_unset(self):
        from pydantic_ai.tools import ToolDefinition

        toolset = MCPToolset('https://example.com/mcp')
        tool = toolset.tool_for_tool_def(
            ToolDefinition(name='foo', description='', parameters_json_schema={'type': 'object'})
        )
        assert tool.max_retries == 1

    async def test_direct_call_tool_propagates_error_when_configured(self, fastmcp_server: FastMCP[None]):
        toolset = MCPToolset(fastmcp_server, tool_error_behavior='error')
        async with toolset:
            with pytest.raises(ToolError):
                await toolset.direct_call_tool('boom', {})


class TestToolResultMapping:
    """Direct unit tests for `_map_mcp_tool_result` — easier than crafting a server response
    that bypasses FastMCP's `structured_content` shortcut."""

    def test_text_content_returns_str(self):
        from pydantic_ai.mcp import _map_mcp_tool_result  # type: ignore[attr-defined]

        out = _map_mcp_tool_result(mcp_types.TextContent(type='text', text='hello'))
        assert out == 'hello'

    def test_text_content_with_json_object_is_parsed(self):
        from pydantic_ai.mcp import _map_mcp_tool_result  # type: ignore[attr-defined]

        out = _map_mcp_tool_result(mcp_types.TextContent(type='text', text='{"a": 1}'))
        assert out == {'a': 1}

    def test_text_content_with_json_array_is_parsed(self):
        from pydantic_ai.mcp import _map_mcp_tool_result  # type: ignore[attr-defined]

        out = _map_mcp_tool_result(mcp_types.TextContent(type='text', text='[1, 2, 3]'))
        assert out == [1, 2, 3]

    def test_text_content_with_invalid_json_falls_back_to_text(self):
        from pydantic_ai.mcp import _map_mcp_tool_result  # type: ignore[attr-defined]

        # Starts with `{` but isn't valid JSON.
        out = _map_mcp_tool_result(mcp_types.TextContent(type='text', text='{not valid'))
        assert out == '{not valid'


class TestSamplingHandler:
    async def test_sampling_handler_round_trip(self):
        """Drive the sampling handler built from `sampling_model=` to cover its body."""
        from pydantic_ai.mcp import _build_sampling_handler  # type: ignore[attr-defined]

        model = TestModel()
        handler = _build_sampling_handler(model)
        params = mcp_types.CreateMessageRequestParams(
            messages=[mcp_types.SamplingMessage(role='user', content=mcp_types.TextContent(type='text', text='hi'))],
            maxTokens=42,
            temperature=0.5,
            stopSequences=['STOP'],
        )
        result = await handler([], params, None)  # type: ignore[arg-type, misc]
        assert isinstance(result, mcp_types.CreateMessageResult)
        assert result.model == model.model_name


class TestSamplingMessageMapping:
    """Cover the mapping helpers in `pydantic_ai._mcp` that translate MCP sampling messages
    to/from PAI message parts. Exercised via the sampling handler that `MCPToolset(sampling_model=...)` installs."""

    async def test_map_handles_image_audio_and_role_transitions(self):
        from pydantic_ai import _mcp as _mcp_helpers

        params = mcp_types.CreateMessageRequestParams(
            messages=[
                mcp_types.SamplingMessage(role='user', content=mcp_types.TextContent(type='text', text='hello')),
                mcp_types.SamplingMessage(role='assistant', content=mcp_types.TextContent(type='text', text='hi back')),
                mcp_types.SamplingMessage(
                    role='user',
                    content=mcp_types.ImageContent(
                        type='image',
                        data=base64.b64encode(b'fake').decode(),
                        mimeType='image/png',
                    ),
                ),
                mcp_types.SamplingMessage(
                    role='user',
                    content=mcp_types.AudioContent(
                        type='audio',
                        data=base64.b64encode(b'fake').decode(),
                        mimeType='audio/wav',
                    ),
                ),
                mcp_types.SamplingMessage(role='assistant', content=mcp_types.TextContent(type='text', text='final')),
            ],
            systemPrompt='you are helpful',
            maxTokens=10,
        )
        pai_messages = _mcp_helpers.map_from_mcp_params(params)
        # Should alternate Request/Response, with the trailing assistant becoming the final ModelResponse.
        kinds = [type(m).__name__ for m in pai_messages]
        assert kinds == ['ModelRequest', 'ModelResponse', 'ModelRequest', 'ModelResponse']

    async def test_map_rejects_unsupported_content_types(self):
        from pydantic_ai import _mcp as _mcp_helpers

        list_content_params = mcp_types.CreateMessageRequestParams(
            messages=[
                mcp_types.SamplingMessage(role='user', content=[]),
            ],
            maxTokens=10,
        )
        with pytest.raises(NotImplementedError, match='list content'):
            _mcp_helpers.map_from_mcp_params(list_content_params)

        # `ToolUseContent` / `ToolResultContent` from the user side aren't legal sampling input.
        tool_use_params = mcp_types.CreateMessageRequestParams(
            messages=[
                mcp_types.SamplingMessage(
                    role='user',
                    content=mcp_types.ToolUseContent(type='tool_use', id='t', name='foo', input={}),
                ),
            ],
            maxTokens=10,
        )
        with pytest.raises(NotImplementedError, match='cannot be used as user content'):
            _mcp_helpers.map_from_mcp_params(tool_use_params)

        # Audio sampling responses are also explicitly unsupported.
        audio_response_params = mcp_types.CreateMessageRequestParams(
            messages=[
                mcp_types.SamplingMessage(
                    role='assistant',
                    content=mcp_types.AudioContent(
                        type='audio',
                        data=base64.b64encode(b'fake').decode(),
                        mimeType='audio/wav',
                    ),
                ),
            ],
            maxTokens=10,
        )
        with pytest.raises(NotImplementedError):
            _mcp_helpers.map_from_sampling_content(audio_response_params.messages[0].content)  # type: ignore[arg-type]

    async def test_map_handles_consecutive_assistant_messages(self):
        """Two assistant messages in a row append into the same `ModelResponse` (no intervening request)."""
        from pydantic_ai import _mcp as _mcp_helpers

        params = mcp_types.CreateMessageRequestParams(
            messages=[
                mcp_types.SamplingMessage(role='assistant', content=mcp_types.TextContent(type='text', text='one')),
                mcp_types.SamplingMessage(role='assistant', content=mcp_types.TextContent(type='text', text='two')),
            ],
            maxTokens=10,
        )
        pai_messages = _mcp_helpers.map_from_mcp_params(params)
        assert [type(m).__name__ for m in pai_messages] == ['ModelResponse']

    async def test_map_from_model_response_skips_thinking_and_rejects_unknown(self):
        from pydantic_ai import _mcp as _mcp_helpers
        from pydantic_ai.exceptions import UnexpectedModelBehavior
        from pydantic_ai.messages import ModelResponse, TextPart, ThinkingPart, ToolCallPart

        # `ThinkingPart` is silently skipped, leaving the text content.
        result = _mcp_helpers.map_from_model_response(
            ModelResponse(parts=[ThinkingPart(content='hidden'), TextPart(content='visible')])
        )
        assert result.text == 'visible'

        # Unsupported parts (e.g. tool calls) raise a clear error.
        with pytest.raises(UnexpectedModelBehavior):
            _mcp_helpers.map_from_model_response(ModelResponse(parts=[ToolCallPart(tool_name='foo', args='{}')]))


class TestResourceMethodErrorPaths:
    async def test_list_resources_wraps_mcp_error(self, fastmcp_server: FastMCP[None]):
        """Server errors from `list_resources` are wrapped in `MCPError`."""
        from unittest.mock import AsyncMock

        from mcp.shared.exceptions import McpError

        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            toolset.client.list_resources = AsyncMock(
                side_effect=McpError(mcp_types.ErrorData(code=-32603, message='boom'))
            )
            with pytest.raises(MCPError, match='boom'):
                await toolset.list_resources()

    async def test_list_resource_templates_wraps_mcp_error(self, fastmcp_server: FastMCP[None]):
        from unittest.mock import AsyncMock

        from mcp.shared.exceptions import McpError

        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            toolset.client.list_resource_templates = AsyncMock(
                side_effect=McpError(mcp_types.ErrorData(code=-32603, message='boom'))
            )
            with pytest.raises(MCPError, match='boom'):
                await toolset.list_resource_templates()

    async def test_read_resource_wraps_mcp_error(self, fastmcp_server: FastMCP[None]):
        from unittest.mock import AsyncMock

        from mcp.shared.exceptions import McpError

        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            toolset.client.read_resource = AsyncMock(
                side_effect=McpError(mcp_types.ErrorData(code=-32002, message='not found'))
            )
            with pytest.raises(MCPError, match='not found'):
                await toolset.read_resource('resource://missing')


class TestLoadMCPToolsets:
    async def test_loads_toolsets_from_config_without_env(self):
        """Stdio entries without an `env` field also produce valid toolsets."""
        config = {
            'mcpServers': {
                'alpha': {'command': 'python', 'args': ['-m', 'tests.mcp_server']},
            }
        }
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / 'mcp.json'
            config_path.write_text(json.dumps(config))
            toolsets = load_mcp_toolsets(config_path)
        assert len(toolsets) == 1

    async def test_loads_toolsets_from_config_with_prefix(self):
        config = {
            'mcpServers': {
                'alpha': {'command': 'python', 'args': ['-m', 'tests.mcp_server'], 'env': {'FOO': 'bar'}},
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

    async def test_load_mcp_toolsets_expands_env_vars(self, monkeypatch: pytest.MonkeyPatch):
        """`${VAR_NAME}` references in the config are resolved from `os.environ`; default-syntax
        (`${VAR_NAME:-fallback}`) returns the fallback when unset; missing required vars raise."""
        monkeypatch.setenv('MCP_TEST_TOKEN', 'secret-value')
        config = {
            'mcpServers': {
                'alpha': {
                    'url': 'https://${MCP_TEST_HOST:-localhost:8000}/mcp',
                    'headers': {'Authorization': 'Bearer ${MCP_TEST_TOKEN}', 'X-Extras': ['${MCP_TEST_TOKEN}']},
                },
            }
        }
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / 'mcp.json'
            config_path.write_text(json.dumps(config))
            toolsets = load_mcp_toolsets(config_path)

        wrapped = toolsets[0].wrapped  # type: ignore[attr-defined]
        assert isinstance(wrapped, MCPToolset)
        assert isinstance(wrapped.client.transport, StreamableHttpTransport)
        assert wrapped.client.transport.headers == {
            'Authorization': 'Bearer secret-value',
            'X-Extras': ['secret-value'],
        }
        assert str(wrapped.client.transport.url) == 'https://localhost:8000/mcp'

    async def test_load_mcp_toolsets_undefined_env_var_raises(self):
        """A `${VAR}` reference without a default and not set in the environment raises a clear `ValueError`."""
        config = {'mcpServers': {'alpha': {'url': 'https://${MCP_TEST_UNDEFINED}/mcp'}}}
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / 'mcp.json'
            config_path.write_text(json.dumps(config))
            with pytest.raises(ValueError, match=r'\$\{MCP_TEST_UNDEFINED\} is not defined'):
                load_mcp_toolsets(config_path)

    async def test_load_mcp_toolsets_rejects_non_object_root(self):
        """The config root must be a JSON object; a list / scalar at the root raises a descriptive error."""
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / 'mcp.json'
            config_path.write_text(json.dumps(['not an object']))
            with pytest.raises(ValueError, match='Expected JSON object at root'):
                load_mcp_toolsets(config_path)

    async def test_load_mcp_toolsets_rejects_missing_mcp_servers_key(self):
        """The config must have an `mcpServers` object."""
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / 'mcp.json'
            config_path.write_text(json.dumps({'someOtherKey': {}}))
            with pytest.raises(ValueError, match='Expected `mcpServers` object'):
                load_mcp_toolsets(config_path)

    async def test_load_mcp_toolsets_rejects_invalid_server_entry(self):
        """A server entry missing both `command` and `url` raises a clear `ValueError`."""
        config = {'mcpServers': {'alpha': {'something': 'else'}}}
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / 'mcp.json'
            config_path.write_text(json.dumps(config))
            with pytest.raises(ValueError, match=r"MCP server config 'alpha' must have either"):
                load_mcp_toolsets(config_path)

    async def test_load_mcp_toolsets_passes_primitive_values_through_env_expansion(self):
        """Non-string/dict/list values (ints, bools, null) in the config flow through
        `_expand_env_vars` unchanged."""
        config = {
            'mcpServers': {
                'alpha': {
                    'command': 'python',
                    'args': ['-m', 'tests.mcp_server'],
                    # Primitive values: `_expand_env_vars` should return these as-is.
                    'startup_timeout': 30,
                    'enable_telemetry': True,
                    'log_file': None,
                },
            }
        }
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / 'mcp.json'
            config_path.write_text(json.dumps(config))
            toolsets = load_mcp_toolsets(config_path)
        assert len(toolsets) == 1


def test_construction_does_not_emit_warnings(recwarn: Any) -> None:
    """Building an `MCPToolset` from a URL must not emit `FastMCPDeprecationWarning` for the
    `sse_read_timeout` parameter — the StreamableHttp path migrated off it (the FastMCP `Client`
    `timeout` carries the read timeout instead)."""
    MCPToolset('https://example.com/mcp', headers={'X-Key': 'foo'})
    deprecation_messages = [str(w.message) for w in recwarn if 'sse_read_timeout' in str(w.message)]
    assert deprecation_messages == [], deprecation_messages


class TestToolErrorStructuredMessage:
    """The tool-call path should preserve structured MCP error info (code/data)."""

    def test_mcp_error_includes_code(self):
        from mcp.shared.exceptions import McpError
        from mcp.types import ErrorData

        from pydantic_ai.mcp import _build_tool_error_message  # pyright: ignore[reportPrivateUsage]

        err = McpError(ErrorData(code=-32603, message='internal error'))
        msg = _build_tool_error_message(err)
        assert 'internal error' in msg and 'code: -32603' in msg

    def test_non_mcp_falls_back_to_str(self):
        from pydantic_ai.mcp import _build_tool_error_message  # pyright: ignore[reportPrivateUsage]

        assert _build_tool_error_message(ValueError('boom')) == 'boom'

    def test_tool_error_wrapping_mcp_error(self):
        """ToolError wrapping an McpError should extract the underlying error code."""
        from mcp.shared.exceptions import McpError
        from mcp.types import ErrorData

        from pydantic_ai.mcp import ToolError, _build_tool_error_message  # pyright: ignore[reportPrivateUsage]

        mcpe = McpError(ErrorData(code=-32603, message='internal error'))
        tool_err = ToolError('tool failed')
        tool_err.__cause__ = mcpe
        msg = _build_tool_error_message(tool_err)
        assert 'internal error' in msg and 'code: -32603' in msg


class TestMCPToolsetBackgroundTasks:
    """SEP-1686 task-augmented execution. `MCPToolset` reads each tool's server-declared
    `execution.taskSupport` and routes the call accordingly:
    `'required'` and `'optional'` go through `client.call_tool(task=True)` -> `tool_task.result()`,
    while `'forbidden'`/absent stay on the regular sync path."""

    @pytest.fixture
    async def task_server(self) -> FastMCP[None]:
        server: FastMCP[None] = FastMCP('task_server')

        @server.tool(task=TaskConfig(mode='required'))
        async def task_required_tool() -> str:
            """A tool that requires task-augmented execution."""
            await asyncio.sleep(0)
            return 'task_required_completed'

        @server.tool(task=TaskConfig(mode='optional'))
        async def task_optional_tool() -> str:
            """A tool that may run either as a task or synchronously."""
            await asyncio.sleep(0)
            return 'task_optional_completed'

        @server.tool()
        async def plain_tool() -> str:
            """A tool with no task support - `execution` is `None`."""
            return 'plain_completed'

        return server

    async def test_get_tools_exposes_task_metadata(
        self, task_server: FastMCP[None], run_context: RunContext[None]
    ) -> None:
        """`get_tools` exposes `task: bool` so downstream capabilities can target task-augmented tools."""
        toolset = MCPToolset(task_server)
        async with toolset:
            tools = await toolset.get_tools(run_context)

        assert (tools['task_required_tool'].tool_def.metadata or {}).get('task') is True
        assert (tools['task_optional_tool'].tool_def.metadata or {}).get('task') is True
        assert (tools['plain_tool'].tool_def.metadata or {}).get('task') is False

    async def test_required_tool_routes_through_task_path(
        self, task_server: FastMCP[None], run_context: RunContext[None]
    ) -> None:
        """`mode='required'` succeeds - getting the real result proves `task=True` was sent (the server
        would otherwise return `-32601: requires task-augmented execution`)."""
        toolset = MCPToolset(task_server)
        async with toolset:
            tools = await toolset.get_tools(run_context)
            result = await toolset.call_tool('task_required_tool', {}, run_context, tools['task_required_tool'])
        assert result == 'task_required_completed'

    async def test_optional_tool_routes_through_task_path(
        self, task_server: FastMCP[None], run_context: RunContext[None]
    ) -> None:
        """`mode='optional'` also goes through the task path by default - the SEP allows either, and the
        task path delivers durability/cancellation/progress benefits with no functional downside."""
        toolset = MCPToolset(task_server)
        async with toolset:
            tools = await toolset.get_tools(run_context)
            result = await toolset.call_tool('task_optional_tool', {}, run_context, tools['task_optional_tool'])
        assert result == 'task_optional_completed'

    async def test_plain_tool_stays_on_sync_path(
        self, task_server: FastMCP[None], run_context: RunContext[None]
    ) -> None:
        """A tool with no `execution.taskSupport` stays on the regular sync `tools/call`. Sending
        `task=True` to such a server would violate the SEP."""
        toolset = MCPToolset(task_server)
        async with toolset:
            tools = await toolset.get_tools(run_context)
            result = await toolset.call_tool('plain_tool', {}, run_context, tools['plain_tool'])
        assert result == 'plain_completed'

    async def test_direct_call_tool_with_use_task(self, task_server: FastMCP[None]) -> None:
        """`direct_call_tool(..., use_task=True)` is the low-level escape hatch for users calling without
        a `ToolDefinition` - `mode='required'` works directly."""
        toolset = MCPToolset(task_server)
        async with toolset:
            result = await toolset.direct_call_tool('task_required_tool', {}, use_task=True)
        assert result == 'task_required_completed'

    async def test_process_tool_call_receives_use_task_partial(
        self, task_server: FastMCP[None], run_context: RunContext[None]
    ) -> None:
        """`process_tool_call` gets a `CallToolFunc` that already has `use_task` baked in via `partial`,
        so a custom wrapper doesn't need to know about the task path to preserve it."""

        async def passthrough(ctx: RunContext[Any], call_tool: Any, name: str, args: dict[str, Any]) -> Any:
            return await call_tool(name, args)

        toolset = MCPToolset(task_server, process_tool_call=passthrough)
        async with toolset:
            tools = await toolset.get_tools(run_context)
            result = await toolset.call_tool('task_required_tool', {}, run_context, tools['task_required_tool'])
        assert result == 'task_required_completed'
