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
import subprocess
import sys
import warnings
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, cast
from unittest.mock import AsyncMock

import anyio
import httpx
import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel, ValidationError

from pydantic_ai import Agent, ToolsetTool, models
from pydantic_ai._run_context import RunContext
from pydantic_ai._utils import BaseExceptionGroup
from pydantic_ai.exceptions import ModelRetry, ToolFailed, UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RunUsage

from .conftest import IsStr, try_import

with try_import() as imports_successful:
    from fastmcp.client import Client
    from fastmcp.client.client import CallToolResult
    from fastmcp.client.transports import (
        SSETransport,
        StdioTransport,
        StreamableHttpTransport,
    )
    from fastmcp.exceptions import McpError, ToolError
    from fastmcp.prompts import Message
    from fastmcp.server import Context, FastMCP

    try:
        from fastmcp.server.tasks import TaskConfig
    except ImportError:
        # FastMCP 4 moved `TaskConfig`.
        from fastmcp.utilities.tasks import TaskConfig
    # `mcp.types` serves either SDK generation: v2 keeps it as an exact re-export of `mcp_types`.
    from mcp import types as mcp_types

    from pydantic_ai import mcp as mcp_module

    # `fastmcp_tasks` is never installed in the typecheck environment, so pyright only gets a declaration.
    if TYPE_CHECKING:
        TasksExtension: Any
    else:
        try:
            from fastmcp_tasks import TasksExtension
        except ImportError:
            TasksExtension = None

    Annotations = mcp_types.Annotations
    AudioContent = mcp_types.AudioContent
    BlobResourceContents = mcp_types.BlobResourceContents
    EmbeddedResource = mcp_types.EmbeddedResource
    ImageContent = mcp_types.ImageContent
    ResourceLink = mcp_types.ResourceLink
    McpTextContent = mcp_types.TextContent
    TextResourceContents = mcp_types.TextResourceContents
    from pydantic import AnyUrl, TypeAdapter

    from pydantic_ai._mcp_compat import is_mcp_sdk_v2, wire_name
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
    from pydantic_ai.toolsets.prefixed import PrefixedToolset


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='fastmcp not installed'),
    pytest.mark.anyio,
]

MCP_SDK_V2 = imports_successful() and is_mcp_sdk_v2()


def make_mcp_error(code: int, message: str) -> McpError:
    """Construct an MCP protocol error with either SDK generation.

    SDK v1 wraps an `ErrorData`; v2 takes the fields directly.
    """
    if MCP_SDK_V2:
        # `cast` because the typecheck environment only knows the SDK v1 constructor signature.
        return cast(Any, McpError)(code=code, message=message)
    return McpError(mcp_types.ErrorData(code=code, message=message))


def wrap_server_notification(notification: Any) -> Any:
    """Wrap a notification in the SDK v1 root model; SDK v2 delivers the value unwrapped."""
    if MCP_SDK_V2:
        return notification
    return mcp_types.ServerNotification(root=notification)


def make_legacy_client(server: FastMCP[None]) -> Client[Any]:
    """Build a client pinned to a legacy (handshake-era) session.

    FastMCP 4 negotiates a modern session by default and takes `mode='legacy'` to pin the
    handshake era; FastMCP 3 only speaks the handshake era and has no such parameter.
    """
    return cast(Any, Client)(server, **({'mode': 'legacy'} if MCP_SDK_V2 else {}))


@pytest.mark.parametrize(
    'installed,expected',
    [('1.26.0', False), ('2.0.0a1', True), ('2.0.0', True), ('10.0.0', True)],
)
def test_is_mcp_sdk_v2_reads_the_installed_distribution_version(
    installed: str, expected: bool, monkeypatch: pytest.MonkeyPatch
):
    """The generation is read off the installed `mcp` distribution version. SDK v2.0.0 restored
    `mcp.types` as an exact re-export of the standalone package, so a v2 install serves the v2
    classes under the v1 module name and neither the module nor its class shapes are a public
    contract to detect the generation by.
    """
    import pydantic_ai._mcp_compat as _mcp_compat

    def fake_package_version(distribution_name: str) -> str:
        assert distribution_name == 'mcp'
        return installed

    monkeypatch.setattr(_mcp_compat, 'version', fake_package_version)
    assert _mcp_compat.is_mcp_sdk_v2() is expected


MCP_FIELD_READS = [
    ('Annotations', 'last_modified'),
    ('AudioContent', 'mime_type'),
    ('BlobResourceContents', 'mime_type'),
    ('CreateMessageRequestParams', 'max_tokens'),
    ('CreateMessageRequestParams', 'stop_sequences'),
    ('CreateMessageRequestParams', 'system_prompt'),
    ('Icon', 'mime_type'),
    ('ImageContent', 'mime_type'),
    ('InitializeResult', 'server_info'),
    ('PromptsCapability', 'list_changed'),
    ('Resource', 'mime_type'),
    ('ResourceLink', 'mime_type'),
    ('ResourceTemplate', 'mime_type'),
    ('ResourceTemplate', 'uri_template'),
    ('ResourcesCapability', 'list_changed'),
    ('TextResourceContents', 'mime_type'),
    ('Tool', 'input_schema'),
    ('Tool', 'output_schema'),
    ('ToolExecution', 'task_support'),
    ('ToolsCapability', 'list_changed'),
]
"""Every `(class, field)` the compat readers in `pydantic_ai._mcp_compat` read across
`pydantic_ai.mcp` and `pydantic_ai._mcp`."""


def test_compat_readers_name_real_sdk_v2_fields():
    """Each field the compat readers read is a real SDK v2 field whose wire alias is the derived
    camelCase spelling.

    Nothing else pins this: `[mcp]` installs SDK v1, so the v2 half of every reader is unreachable in
    the suite, and a wrong snake_case spelling reads as `None` instead of raising — a mistyped
    `mime_type` would silently drop every media type once the pin widens. The standalone `mcp-types`
    distribution depends only on Pydantic, so it installs alongside SDK v1 and the fields can be
    checked against the real v2 models. Its `alias` is the wire (camelCase) name — exactly what
    `wire_name` derives and the SDK v1 attribute the readers fall back to — so one lookup validates
    the spelling and the derivation at once.
    """
    v2_types = pytest.importorskip('mcp_types', reason='the `mcp-types` dev dependency is not installed')

    for class_name, field_name in MCP_FIELD_READS:
        model: type[BaseModel] = getattr(v2_types, class_name)
        field = model.model_fields.get(field_name)
        assert field is not None, f'`{class_name}.{field_name}` is not a field on the SDK v2 model'
        assert field.alias == wire_name(field_name), (
            f'`{class_name}.{field_name}` has wire alias `{field.alias}`, not `{wire_name(field_name)}`'
        )


def test_sdk_v2_image_content_accepts_wire_field_name():
    """The v1 `mimeType` spelling remains a valid constructor alias under SDK v2."""
    v2_types = pytest.importorskip('mcp_types', reason='the `mcp-types` dev dependency is not installed')

    image = v2_types.ImageContent(type='image', data='eA==', mimeType='image/png')

    assert image.mime_type == 'image/png'


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

    @server.tool(annotations={'title': 'Echo', 'readOnlyHint': True})
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
        # SDK v2 retypes every `uri` from `AnyUrl` to `str` and rejects an `AnyUrl` instance, so the
        # URIs here are plain strings, cast to satisfy the v1 annotation these tests type-check against.
        return EmbeddedResource(
            type='resource',
            resource=BlobResourceContents(uri=cast(AnyUrl, 'resource://blob.bin'), blob=encoded),
        )

    @server.tool()
    async def resource_link_tool() -> ResourceLink:
        """A tool that returns a resource link."""
        return ResourceLink(type='resource_link', uri=cast(AnyUrl, 'resource://greeting.txt'), name='greeting')

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
                        uri=cast(AnyUrl, 'resource://product_name.txt'),
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

    async def test_tool_annotations_keep_the_wire_spelling(
        self, fastmcp_server: FastMCP[None], run_context: RunContext
    ):
        """Tool filters read `metadata['annotations']` by key, so the keys track the wire spelling
        rather than the attribute names of whichever MCP SDK generation happens to be installed."""
        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            tools = await toolset.get_tools(run_context)
        assert (tools['echo'].tool_def.metadata or {})['annotations'] == snapshot(
            {
                'title': 'Echo',
                'readOnlyHint': True,
                'destructiveHint': None,
                'idempotentHint': None,
                'openWorldHint': None,
            }
        )

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

    async def test_tool_error_raises_tool_failed_when_configured(
        self, fastmcp_server: FastMCP[None], run_context: RunContext
    ):
        toolset = MCPToolset(fastmcp_server, tool_error_behavior='failed')
        async with toolset:
            tools = await toolset.get_tools(run_context)
            with pytest.raises(ToolFailed, match='boom'):
                await toolset.call_tool('boom', {}, run_context, tools['boom'])

    @pytest.mark.parametrize(
        ('tool_error_behavior', 'expected_exception'),
        [
            pytest.param('retry', ModelRetry, id='retry'),
            pytest.param('failed', ToolFailed, id='failed'),
        ],
    )
    async def test_direct_tool_error_is_converted(
        self,
        fastmcp_server: FastMCP[None],
        run_context: RunContext,
        tool_error_behavior: Literal['retry', 'failed'],
        expected_exception: type[ModelRetry] | type[ToolFailed],
    ):
        """A direct FastMCP `ToolError` still follows the configured model-visible behavior."""
        toolset = MCPToolset(fastmcp_server, tool_error_behavior=tool_error_behavior)

        async with toolset:
            tools = await toolset.get_tools(run_context)
            toolset.client.call_tool = AsyncMock(side_effect=ToolError('direct tool error'))
            with pytest.raises(expected_exception, match='direct tool error'):
                await toolset.call_tool('echo', {'message': 'hi'}, run_context, tools['echo'])

    async def test_tool_failed_preserves_structured_error_content(
        self, fastmcp_server: FastMCP[None], run_context: RunContext
    ):
        """Structured MCP error details remain model-visible instead of collapsing to the first text block.

        This uses an in-process client result because no external model or HTTP provider boundary is involved.
        """
        toolset = MCPToolset(fastmcp_server, tool_error_behavior='failed')
        structured_error = {'errorCategory': 'validation', 'isRetryable': False, 'detail': 'bad input'}

        async def call_tool(*args: Any, **kwargs: Any) -> Any:
            assert kwargs['raise_on_error'] is False
            return CallToolResult(
                content=[
                    mcp_types.TextContent(type='text', text='bad input'),
                    mcp_types.ImageContent(
                        type='image', data=base64.b64encode(b'image').decode(), mimeType='image/png'
                    ),
                ],
                structured_content=structured_error,
                meta=None,
                is_error=True,
            )

        async with toolset:
            tools = await toolset.get_tools(run_context)
            toolset.client.call_tool = call_tool
            with pytest.raises(ToolFailed) as exc_info:
                await toolset.call_tool('echo', {'message': 'hi'}, run_context, tools['echo'])

        assert json.loads(exc_info.value.message) == structured_error

    @pytest.mark.parametrize(
        'content',
        [
            pytest.param([], id='empty'),
            pytest.param(
                [mcp_types.ImageContent(type='image', data=base64.b64encode(b'image').decode(), mimeType='image/png')],
                id='non-text',
            ),
        ],
    )
    async def test_tool_failed_without_text_uses_fallback_message(
        self,
        fastmcp_server: FastMCP[None],
        run_context: RunContext,
        content: list[mcp_types.ContentBlock],
    ):
        """An MCP error without text keeps FastMCP's informative fallback instead of an empty message."""
        toolset = MCPToolset(fastmcp_server, tool_error_behavior='failed')

        async with toolset:
            tools = await toolset.get_tools(run_context)
            toolset.client.call_tool = AsyncMock(
                return_value=CallToolResult(content=content, structured_content=None, meta=None, is_error=True)
            )
            with pytest.raises(ToolFailed, match="Tool 'echo' returned an error"):
                await toolset.call_tool('echo', {'message': 'hi'}, run_context, tools['echo'])

    @pytest.mark.parametrize(
        'leaf_factory',
        [
            pytest.param(lambda: ToolError('grouped tool error'), id='tool-error'),
            pytest.param(lambda: make_mcp_error(400, 'grouped tool error'), id='mcp-error'),
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

    @pytest.mark.parametrize(
        ('leaf_factory', 'expected_exception'),
        [
            pytest.param(lambda: ToolError('grouped tool error'), ToolFailed, id='tool-error'),
            pytest.param(
                lambda: make_mcp_error(400, 'grouped protocol error'),
                ModelRetry,
                id='mcp-error',
            ),
        ],
    )
    async def test_call_tool_unwraps_real_exception_group_with_failed_behavior(
        self,
        fastmcp_server: FastMCP[None],
        run_context: RunContext,
        leaf_factory: Any,
        expected_exception: type[ToolFailed] | type[ModelRetry],
    ):
        """Failed behavior converts completed tool errors but keeps grouped protocol errors retryable."""
        toolset = MCPToolset(fastmcp_server, tool_error_behavior='failed')

        async def call_tool_in_failing_task_group(*args: Any, **kwargs: Any) -> Any:
            async def fail() -> None:
                raise leaf_factory()

            async with anyio.create_task_group() as tg:
                tg.start_soon(fail)

        async with toolset:
            tools = await toolset.get_tools(run_context)
            toolset.client.call_tool = call_tool_in_failing_task_group
            with pytest.raises(expected_exception, match=r'grouped (tool|protocol) error'):
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

    @pytest.mark.parametrize('behavior', ['retry', 'failed'])
    async def test_call_tool_converts_bare_mcp_error_to_model_retry(
        self, fastmcp_server: FastMCP[None], run_context: RunContext, behavior: Literal['retry', 'failed']
    ):
        """A bare, un-grouped `McpError` — a JSON-RPC error the server/gateway returns for a call it
        refuses, rather than a `ToolError` or a result-level error — is a protocol error, so like the
        grouped protocol-error case it stays a recoverable `ModelRetry` even under
        `tool_error_behavior='failed'`, instead of escaping the toolset and crashing the run.

        A unit test because it injects at the real escape seam (`self.client.call_tool`): no in-process
        FastMCP tool surfaces a bare protocol `McpError` instead of a `ToolError`.
        """
        toolset = MCPToolset(fastmcp_server, tool_error_behavior=behavior)

        async def call_tool_raising_bare_mcp_error(*args: Any, **kwargs: Any) -> Any:
            raise make_mcp_error(400, 'bare protocol error')

        async with toolset:
            tools = await toolset.get_tools(run_context)
            toolset.client.call_tool = call_tool_raising_bare_mcp_error
            with pytest.raises(ModelRetry, match='bare protocol error'):
                await toolset.call_tool('echo', {'message': 'hi'}, run_context, tools['echo'])

    async def test_call_tool_propagates_bare_mcp_error_when_configured(
        self, fastmcp_server: FastMCP[None], run_context: RunContext
    ):
        """With `tool_error_behavior='error'`, a bare `McpError` propagates unchanged to the caller."""
        toolset = MCPToolset(fastmcp_server, tool_error_behavior='error')

        async def call_tool_raising_bare_mcp_error(*args: Any, **kwargs: Any) -> Any:
            raise make_mcp_error(400, 'bare protocol error')

        async with toolset:
            tools = await toolset.get_tools(run_context)
            toolset.client.call_tool = call_tool_raising_bare_mcp_error
            with pytest.raises(McpError, match='bare protocol error'):
                await toolset.call_tool('echo', {'message': 'hi'}, run_context, tools['echo'])

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
                            # FastMCP generates this text and has reworded it across releases; pin only
                            # that the argument's JSON schema is conveyed.
                            description=IsStr(regex=r'.*\{"type":"string"\}.*'),
                            required=True,
                        ),
                        PromptArgument(
                            name='topic',
                            # FastMCP generates this text and has reworded it across releases; pin only
                            # that the argument's JSON schema is conveyed.
                            description=IsStr(regex=r'.*\{"type":"string"\}.*'),
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
        if MCP_SDK_V2:
            assert result.metadata == {
                'io.modelcontextprotocol/serverInfo': {'name': 'test_server', 'version': IsStr()}
            }
        else:
            assert result.metadata is None
        assert replace(result, metadata=None) == snapshot(
            PromptResult(
                messages=[PromptMessage(role='user', content=TextContent(content='This is a simple prompt'))],
                description='A simple prompt template.',
            )
        )

    async def test_get_prompt_parameterized(self, fastmcp_server: FastMCP[None]):
        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            result = await toolset.get_prompt('parameterized_prompt', {'name': 'Alice', 'topic': 'AI'})
        if MCP_SDK_V2:
            assert result.metadata == {
                'io.modelcontextprotocol/serverInfo': {'name': 'test_server', 'version': IsStr()}
            }
        else:
            assert result.metadata is None
        assert replace(result, metadata=None) == snapshot(
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
            wrap_server_notification(
                mcp_types.PromptListChangedNotification(method='notifications/prompts/list_changed')
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
            toolset.client.list_prompts = AsyncMock(side_effect=make_mcp_error(-32603, 'boom'))
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
                                uri=cast(AnyUrl, 'resource://kiwi.jpg'),
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
            wrap_server_notification(
                mcp_types.LoggingMessageNotification(
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
            wrap_server_notification(mcp_types.ToolListChangedNotification(method='notifications/tools/list_changed'))
        )
        assert toolset._cached_tools is None  # pyright: ignore[reportPrivateUsage]

        toolset._cached_tools = []  # pyright: ignore[reportPrivateUsage]
        await handler(
            wrap_server_notification(
                mcp_types.ResourceListChangedNotification(method='notifications/resources/list_changed')
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
        notification = wrap_server_notification(
            mcp_types.ToolListChangedNotification(method='notifications/tools/list_changed')
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
        client = make_legacy_client(fastmcp_server)
        toolset = MCPToolset(client, log_level='warning')
        async with toolset:
            assert toolset.is_running

    async def test_log_level_warns_but_connects_on_modern_session(
        self, fastmcp_server: FastMCP[None], as_modern_mcp_session: None
    ):
        """`logging/setLevel` left the protocol in the 2026-07-28 revision. An unapplied log level
        doesn't stop an application from working, so the session opens and warns rather than
        failing — raising would turn an upstream protocol removal into a break on our side."""
        toolset = MCPToolset(fastmcp_server, log_level='warning')
        with pytest.warns(UserWarning, match='`log_level` was not applied'):
            async with toolset:
                assert toolset.is_running

    async def test_sampling_and_elicitation_warn_on_modern_session(
        self, fastmcp_server: FastMCP[None], as_modern_mcp_session: None, monkeypatch: pytest.MonkeyPatch
    ):
        """A modern session refuses server-initiated requests, so a handler configured against one
        can never fire. The names are reported together, so a user setting both learns about
        both. Pinned to no task extension: with `fastmcp-tasks` loaded the elicitation handler
        still fires for task input and is not reported."""

        async def elicitation_handler(message: str, response_type: Any, params: Any, ctx: Any) -> Any:
            raise AssertionError('elicitation handler should never be called')  # pragma: no cover

        monkeypatch.setattr(mcp_module, '_load_call_tool_task', lambda: None)
        toolset = MCPToolset(
            fastmcp_server,
            sampling_model=TestModel(custom_output_text='sampled'),
            elicitation_handler=elicitation_handler,
        )
        with pytest.warns(UserWarning, match=r'`sampling_model`, `elicitation_handler` will never be called'):
            async with toolset:
                assert toolset.is_running

    async def test_elicitation_handler_not_reported_dead_when_tasks_extension_is_loaded(
        self, fastmcp_server: FastMCP[None], as_modern_mcp_session: None, monkeypatch: pytest.MonkeyPatch
    ):
        """With `fastmcp-tasks` loaded, a task parked on `input_required` is answered through the
        elicitation handler, so on a modern session the handler can still fire and must not be
        reported as dead — while sampling has no task path and is still warned about."""

        async def elicitation_handler(message: str, response_type: Any, params: Any, ctx: Any) -> Any:
            raise AssertionError('not exercised by this test')  # pragma: no cover

        async def fake_call_tool_task(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError('not exercised by this test')  # pragma: no cover

        monkeypatch.setattr(mcp_module, '_load_call_tool_task', lambda: fake_call_tool_task)
        toolset = MCPToolset(
            fastmcp_server,
            sampling_model=TestModel(custom_output_text='sampled'),
            elicitation_handler=elicitation_handler,
        )
        with pytest.warns(UserWarning, match=r'^`sampling_model` will never be called') as caught:
            async with toolset:
                assert toolset.is_running
        assert 'elicitation_handler' not in str(caught[0].message)

    async def test_no_dead_handler_warning_for_elicitation_alone_when_tasks_extension_is_loaded(
        self, fastmcp_server: FastMCP[None], as_modern_mcp_session: None, monkeypatch: pytest.MonkeyPatch
    ):
        """When the elicitation handler is the only server-initiated option configured and the
        task extension is loaded, there is nothing dead to warn about at all."""

        async def elicitation_handler(message: str, response_type: Any, params: Any, ctx: Any) -> Any:
            raise AssertionError('not exercised by this test')  # pragma: no cover

        async def fake_call_tool_task(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError('not exercised by this test')  # pragma: no cover

        monkeypatch.setattr(mcp_module, '_load_call_tool_task', lambda: fake_call_tool_task)
        toolset = MCPToolset(fastmcp_server, elicitation_handler=elicitation_handler)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            async with toolset:
                assert toolset.is_running

    async def test_set_sampling_model_replaces_the_reported_handler(
        self, fastmcp_server: FastMCP[None], as_modern_mcp_session: None
    ):
        """`set_sampling_model` swaps out the callback a `sampling_handler=` argument installed, so
        the warning names only the option still in effect rather than both."""

        async def sampling_handler(messages: Any, params: Any, ctx: Any) -> Any:
            raise AssertionError('sampling handler should never be called')  # pragma: no cover

        toolset = MCPToolset(fastmcp_server, sampling_handler=sampling_handler)
        toolset.set_sampling_model(TestModel(custom_output_text='sampled'))

        with pytest.warns(UserWarning, match=r'^`sampling_model` will never be called') as caught:
            async with toolset:
                assert toolset.is_running
        assert 'sampling_handler' not in str(caught[0].message)

    async def test_sampling_and_elicitation_do_not_warn_on_legacy_session(
        self, fastmcp_server: FastMCP[None], as_legacy_mcp_session: None
    ):
        """The warning is specific to a modern session — a legacy one still delivers both."""
        toolset = MCPToolset(fastmcp_server, sampling_model=TestModel(custom_output_text='sampled'))
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            async with toolset:
                assert toolset.is_running

    async def test_server_metadata_read_from_era_neutral_properties(
        self, fastmcp_server: FastMCP[None], monkeypatch: pytest.MonkeyPatch, as_mcp_sdk_v2: None
    ):
        """A modern session has no `initialize` handshake, so server metadata comes off the
        client's era-neutral properties instead of `initialize_result`."""
        present_as_modern_session(
            monkeypatch,
            server_info=mcp_types.Implementation(name='era-neutral', version='9.9'),
            server_capabilities=mcp_types.ServerCapabilities(),
            instructions='from the era-neutral property',
        )

        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            assert toolset.server_info.name == 'era-neutral'
            assert toolset.instructions == 'from the era-neutral property'

    async def test_missing_server_info_is_tolerated_on_modern_session(
        self, fastmcp_server: FastMCP[None], monkeypatch: pytest.MonkeyPatch, as_mcp_sdk_v2: None
    ):
        """`serverInfo` is an optional display-only stamp on modern sessions, so a server that
        omits it still connects; only reading `server_info` fails, and says why."""
        present_as_modern_session(monkeypatch, server_capabilities=mcp_types.ServerCapabilities())

        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            assert toolset.is_running
            with pytest.raises(AttributeError, match='did not send implementation info'):
                toolset.server_info

    async def test_non_string_era_neutral_instructions_are_dropped(
        self, fastmcp_server: FastMCP[None], monkeypatch: pytest.MonkeyPatch, as_mcp_sdk_v2: None
    ):
        """A malformed, non-string value in the era-neutral property is dropped rather than stored."""
        present_as_modern_session(
            monkeypatch,
            server_info=mcp_types.Implementation(name='era-neutral', version='9.9'),
            server_capabilities=mcp_types.ServerCapabilities(),
            instructions=1234,
        )

        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            assert toolset.instructions is None

    async def test_modern_session_without_era_neutral_metadata_is_rejected(
        self, fastmcp_server: FastMCP[None], monkeypatch: pytest.MonkeyPatch, as_mcp_sdk_v2: None
    ):
        """`initialize_result` alone decides the era, so a client that reports a modern session
        without capabilities to read fails with an actionable error. Deriving the era from the
        properties instead would send this case down the legacy path, where the missing
        `initialize_result` surfaces as a bare assertion."""
        present_as_modern_session(monkeypatch)

        toolset = MCPToolset(fastmcp_server)
        with pytest.raises(UserError, match=r'exposes no `server_capabilities`'):
            await toolset.list_tools()

    async def test_uninitialized_fastmcp3_client_gets_actionable_error(
        self, fastmcp_server: FastMCP[None], monkeypatch: pytest.MonkeyPatch
    ):
        """On FastMCP 3 a missing `initialize_result` means the client never initialized (e.g. it
        was built with `auto_initialize=False`), not a modern session — the error must say so
        rather than blame a session generation that doesn't exist in that line."""
        monkeypatch.setattr(mcp_module, '_MCP_SDK_V2', False)
        monkeypatch.setattr(Client, 'initialize_result', property(lambda self: None))

        toolset = MCPToolset(fastmcp_server)
        with pytest.raises(UserError, match=r'was it built with `auto_initialize=False`'):
            await toolset.list_tools()

    async def test_label_falls_back_to_repr(self):
        toolset = MCPToolset('https://example.com/mcp')
        assert 'MCPToolset' in toolset.label

    @pytest.mark.parametrize(
        'toolset_max_retries,ctx_max_retries,expected',
        [
            pytest.param(None, 5, 5, id='inherits-ctx'),
            pytest.param(2, 5, 2, id='explicit-wins'),
        ],
    )
    async def test_tool_for_tool_def_retry_budget(
        self, run_context: RunContext, toolset_max_retries: int | None, ctx_max_retries: int, expected: int
    ):
        """Resolution table for `tool_for_tool_def`: an explicit `max_retries` wins, else `ctx.max_retries`.

        A unit rather than an agent-run test because `ToolsetTool.max_retries` is only observable
        end-to-end as a retry *count*; the durable end-to-end proof lives in
        `tests/durable_exec/test_dbos.py::test_dbos_mcp_tool_inherits_agent_retries`.
        """
        toolset = MCPToolset('https://example.com/mcp', max_retries=toolset_max_retries)
        tool = toolset.tool_for_tool_def(
            ToolDefinition(name='foo', description='', parameters_json_schema={'type': 'object'}),
            ctx=replace(run_context, max_retries=ctx_max_retries),
        )
        assert tool.max_retries == expected

    @pytest.mark.parametrize(
        'toolset_max_retries,ctx_max_retries,expected',
        [
            pytest.param(None, 5, 5, id='inherits-ctx'),
            pytest.param(2, 5, 2, id='explicit-wins'),
        ],
    )
    async def test_get_tools_resolves_retries_via_tool_for_tool_def(
        self,
        fastmcp_server: FastMCP[None],
        run_context: RunContext,
        toolset_max_retries: int | None,
        ctx_max_retries: int,
        expected: int,
    ):
        """`get_tools` builds every tool through `tool_for_tool_def`, so the two agree by construction.

        A unit because the retry budget is only observable end-to-end as a count. Pinned because these
        used to be separate `ToolsetTool` construction sites, and their divergence was the #5180 bug.
        """
        toolset = MCPToolset(fastmcp_server, max_retries=toolset_max_retries)
        tools = await toolset.get_tools(replace(run_context, max_retries=ctx_max_retries))
        assert tools
        assert {tool.max_retries for tool in tools.values()} == {expected}

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

        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            toolset.client.list_resources = AsyncMock(side_effect=make_mcp_error(-32603, 'boom'))
            with pytest.raises(MCPError, match='boom'):
                await toolset.list_resources()

    async def test_list_resource_templates_wraps_mcp_error(self, fastmcp_server: FastMCP[None]):
        from unittest.mock import AsyncMock

        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            toolset.client.list_resource_templates = AsyncMock(side_effect=make_mcp_error(-32603, 'boom'))
            with pytest.raises(MCPError, match='boom'):
                await toolset.list_resource_templates()

    async def test_read_resource_wraps_mcp_error(self, fastmcp_server: FastMCP[None]):
        from unittest.mock import AsyncMock

        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            toolset.client.read_resource = AsyncMock(side_effect=make_mcp_error(-32002, 'not found'))
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
            config_path.write_text(json.dumps(config), encoding='utf-8')
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
            config_path.write_text(json.dumps(config), encoding='utf-8')
            toolsets = load_mcp_toolsets(config_path)
        # Single server entry, wrapped with `.prefixed('alpha')`.
        assert len(toolsets) == 1
        # The wrapped toolset is a `PrefixedToolset`, not an `MCPToolset` directly.
        assert isinstance(toolsets[0], PrefixedToolset)
        assert isinstance(toolsets[0].wrapped, MCPToolset)

    async def test_load_mcp_toolsets_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_mcp_toolsets('/nonexistent/path/to/config.json')

    async def test_load_mcp_toolsets_http_entry(self):
        """A URL-configured toolset completes a real Streamable HTTP tool call."""
        process = await anyio.open_process(
            [
                sys.executable,
                '-c',
                (
                    'import socket\n'
                    'import uvicorn\n'
                    'from fastmcp import FastMCP\n'
                    "mcp = FastMCP('test_server')\n"
                    "mcp.tool(name='get_weather_forecast')"
                    "(lambda location: f'The weather in {location} is sunny and 26 degrees Celsius.')\n"
                    "server_socket = socket.create_server(('127.0.0.1', 0))\n"
                    'print(server_socket.getsockname()[1], flush=True)\n'
                    "uvicorn.run(mcp.http_app(), fd=server_socket.fileno(), lifespan='on', log_level='warning')"
                ),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        async def read_stderr() -> str:  # pragma: no cover
            assert process.stderr is not None
            try:
                return (await process.stderr.receive()).decode()
            except anyio.EndOfStream:  # pragma: no cover
                return '<no stderr>'

        try:
            assert process.stdout is not None
            try:
                port = int((await process.stdout.receive()).decode().strip())
            except anyio.EndOfStream:  # pragma: no cover
                raise AssertionError(f'HTTP MCP test server exited during startup: {await read_stderr()}') from None
            config = {
                'mcpServers': {
                    'beta': {'url': f'http://127.0.0.1:{port}/mcp', 'headers': {'X-Key': 'foo'}},
                }
            }
            with TemporaryDirectory() as tmp:
                config_path = Path(tmp) / 'mcp.json'
                config_path.write_text(json.dumps(config), encoding='utf-8')
                toolsets = load_mcp_toolsets(config_path)

            assert len(toolsets) == 1
            assert isinstance(toolsets[0], PrefixedToolset)
            wrapped = toolsets[0].wrapped
            assert isinstance(wrapped, MCPToolset)
            assert isinstance(wrapped.client.transport, StreamableHttpTransport)
            assert wrapped.client.transport.headers == {'X-Key': 'foo'}

            try:
                with anyio.fail_after(10):
                    agent = Agent(TestModel(call_tools=['beta_get_weather_forecast']), toolsets=toolsets)
                    result = await agent.run('weather')
            except BaseException:  # pragma: no cover
                if process.returncode is not None:
                    raise AssertionError(f'HTTP MCP test server exited during startup: {await read_stderr()}') from None
                raise
        finally:
            with anyio.CancelScope(shield=True):
                if process.returncode is None:  # pragma: no branch
                    process.terminate()
                    with anyio.move_on_after(5) as graceful_shutdown:
                        await process.wait()
                    if graceful_shutdown.cancel_called:  # pragma: lax no cover
                        process.kill()
                if process.returncode is None:  # pragma: no branch
                    await process.wait()  # pragma: lax no cover

        assert result.output == snapshot(
            '{"beta_get_weather_forecast":"The weather in a is sunny and 26 degrees Celsius."}'
        )

    async def test_load_mcp_toolsets_accepts_explicit_null_optional_values(self):
        """Explicit JSON `null` keeps working for optional values used by shared MCP configs."""
        config = {
            'mcpServers': {
                'stdio': {'command': 'python', 'args': None, 'env': None, 'cwd': None},
                'http': {'url': 'https://example.com/mcp', 'headers': None},
            }
        }
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / 'mcp.json'
            config_path.write_text(json.dumps(config), encoding='utf-8')
            toolsets = load_mcp_toolsets(config_path)

        assert len(toolsets) == snapshot(2)
        assert isinstance(toolsets[0], PrefixedToolset)
        stdio = toolsets[0].wrapped
        assert isinstance(stdio, MCPToolset)
        assert isinstance(stdio.client.transport, StdioTransport)
        assert stdio.client.transport.args == []
        assert stdio.client.transport.env is None
        assert stdio.client.transport.cwd is None
        assert isinstance(toolsets[1], PrefixedToolset)
        http = toolsets[1].wrapped
        assert isinstance(http, MCPToolset)
        assert isinstance(http.client.transport, StreamableHttpTransport)
        assert http.client.transport.headers == {}

    async def test_load_mcp_toolsets_expands_env_vars(self, monkeypatch: pytest.MonkeyPatch):
        """`${VAR_NAME}` references in the config are resolved from `os.environ`; default-syntax
        (`${VAR_NAME:-fallback}`) returns the fallback when unset; missing required vars raise."""
        monkeypatch.setenv('MCP_TEST_TOKEN', 'secret-value')
        config = {
            'mcpServers': {
                'alpha': {
                    'url': 'https://${MCP_TEST_HOST:-localhost:8000}/mcp',
                    'headers': {'Authorization': 'Bearer ${MCP_TEST_TOKEN}'},
                },
                # `args` is the list-valued field, so it covers expansion inside a list.
                'beta': {'command': 'python', 'args': ['-m', '${MCP_TEST_TOKEN}']},
            }
        }
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / 'mcp.json'
            config_path.write_text(json.dumps(config), encoding='utf-8')
            toolsets = load_mcp_toolsets(config_path)

        wrapped = toolsets[0].wrapped  # type: ignore[attr-defined]
        assert isinstance(wrapped, MCPToolset)
        assert isinstance(wrapped.client.transport, StreamableHttpTransport)
        assert wrapped.client.transport.headers == {'Authorization': 'Bearer secret-value'}
        assert str(wrapped.client.transport.url) == 'https://localhost:8000/mcp'

        stdio = toolsets[1].wrapped  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
        assert isinstance(stdio, MCPToolset)
        assert isinstance(stdio.client.transport, StdioTransport)
        assert stdio.client.transport.args == ['-m', 'secret-value']

    async def test_load_mcp_toolsets_undefined_env_var_raises(self):
        """A `${VAR}` reference without a default and not set in the environment raises a clear `ValueError`."""
        config = {'mcpServers': {'alpha': {'url': 'https://${MCP_TEST_UNDEFINED}/mcp'}}}
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / 'mcp.json'
            config_path.write_text(json.dumps(config), encoding='utf-8')
            with pytest.raises(ValueError, match=r'\$\{MCP_TEST_UNDEFINED\} is not defined'):
                load_mcp_toolsets(config_path)

    @pytest.mark.parametrize(
        'config,loc',
        [
            pytest.param(['not an object'], (), id='root-not-an-object'),
            pytest.param({'someOtherKey': {}}, ('mcpServers',), id='missing-mcp-servers-key'),
            pytest.param({'mcpServers': {'alpha': None}}, ('mcpServers', 'alpha'), id='entry-not-an-object'),
            pytest.param(
                {'mcpServers': {'alpha': {'command': 123}}}, ('mcpServers', 'alpha', 'command'), id='command-not-a-str'
            ),
            pytest.param(
                {'mcpServers': {'alpha': {'command': 'echo', 'args': 'not-a-list'}}},
                ('mcpServers', 'alpha', 'args'),
                id='args-not-a-list',
            ),
            pytest.param(
                {'mcpServers': {'alpha': {'command': 'echo', 'env': 'not-a-mapping'}}},
                ('mcpServers', 'alpha', 'env'),
                id='env-not-a-mapping',
            ),
            pytest.param(
                {'mcpServers': {'alpha': {'url': 'http://localhost:8000/sse', 'headers': 'not-a-mapping'}}},
                ('mcpServers', 'alpha', 'headers'),
                id='headers-not-a-mapping',
            ),
            # The `Raises:` block and the PR body name these inner types specifically: each used to
            # load without complaint and only misbehave once the server was contacted.
            pytest.param(
                {'mcpServers': {'alpha': {'command': 'echo', 'args': ['-p', 8080]}}},
                ('mcpServers', 'alpha', 'args', 1),
                id='args-item-not-a-str',
            ),
            pytest.param(
                {'mcpServers': {'alpha': {'command': 'echo', 'cwd': 123}}},
                ('mcpServers', 'alpha', 'cwd'),
                id='cwd-not-a-str',
            ),
            pytest.param(
                {'mcpServers': {'alpha': {'command': 'echo', 'env': {'PORT': 8080}}}},
                ('mcpServers', 'alpha', 'env', 'PORT'),
                id='env-value-not-a-str',
            ),
            pytest.param(
                {'mcpServers': {'alpha': {'url': 'http://localhost:8000/mcp', 'headers': {'X-Count': 5}}}},
                ('mcpServers', 'alpha', 'headers', 'X-Count'),
                id='headers-value-not-a-str',
            ),
        ],
    )
    async def test_load_mcp_toolsets_rejects_config_not_matching_the_schema(
        self, config: object, loc: tuple[str | int, ...]
    ):
        """A config that doesn't match the `mcpServers` shape raises `ValidationError`, pointing at the field.

        Covers the inner types as well as the fields themselves — an item inside `args`, a `cwd`, and
        the values inside `env` and `headers`. Those are the shapes that used to load without
        complaint and only misbehave later, once the server was contacted, so `loc` is asserted down
        to the offending index or key.
        """
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / 'mcp.json'
            config_path.write_text(json.dumps(config), encoding='utf-8')
            with pytest.raises(ValidationError) as exc_info:
                load_mcp_toolsets(config_path)

        assert [error['loc'] for error in exc_info.value.errors()] == [loc]
        # `ValidationError` subclasses `ValueError`, so callers catching `ValueError` still work.
        assert isinstance(exc_info.value, ValueError)

    async def test_load_mcp_toolsets_ignores_keys_other_mcp_clients_write(self):
        """A config shared with another MCP client still loads; its extra keys are ignored."""
        config = {
            'mcpServers': {
                'alpha': {'command': 'python', 'args': ['-m', 'x'], 'type': 'stdio', 'disabled': False, 'timeout': 30}
            }
        }
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / 'mcp.json'
            config_path.write_text(json.dumps(config), encoding='utf-8')
            assert len(load_mcp_toolsets(config_path)) == 1

    async def test_load_mcp_toolsets_rejects_invalid_server_entry(self):
        """A server entry with neither `command` nor `url` is a `ValueError`, not a schema problem."""
        config = {'mcpServers': {'alpha': {'something': 'else'}}}
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / 'mcp.json'
            config_path.write_text(json.dumps(config), encoding='utf-8')
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
            config_path.write_text(json.dumps(config), encoding='utf-8')
            toolsets = load_mcp_toolsets(config_path)
        assert len(toolsets) == 1


def test_construction_does_not_emit_warnings(recwarn: Any) -> None:
    """Building an `MCPToolset` from a URL must not emit `FastMCPDeprecationWarning` for the
    `sse_read_timeout` parameter — the StreamableHttp path migrated off it (the FastMCP `Client`
    `timeout` carries the read timeout instead)."""
    MCPToolset('https://example.com/mcp', headers={'X-Key': 'foo'})
    deprecation_messages = [str(w.message) for w in recwarn if 'sse_read_timeout' in str(w.message)]
    assert deprecation_messages == [], deprecation_messages


@pytest.fixture
def as_mcp_sdk_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Take the MCP SDK v2 branches.

    The SDK generation is read once at import into a module-level constant, so tests that synthesize
    v2-only client state use this fixture even when a different supported stack is installed.
    """
    monkeypatch.setattr(mcp_module, '_MCP_SDK_V2', True)


def present_as_modern_session(monkeypatch: pytest.MonkeyPatch, **properties: Any) -> None:
    """Present the client as a modern (sessionless) one: `initialize_result` is the era signal,
    and the era-neutral properties named in `properties` are the only metadata available."""
    monkeypatch.setattr(Client, 'initialize_result', property(lambda self: None))
    era_neutral_properties = {
        'server_info': None,
        'server_capabilities': None,
        'instructions': None,
        **properties,
    }
    for name, value in era_neutral_properties.items():
        monkeypatch.setattr(Client, name, property(lambda self, value=value: value), raising=False)


@pytest.fixture
def as_modern_mcp_session(monkeypatch: pytest.MonkeyPatch, as_mcp_sdk_v2: None) -> None:
    """Additionally present the client as a modern (sessionless) session, with the metadata a
    happy-path server would expose."""
    present_as_modern_session(
        monkeypatch,
        server_info=mcp_types.Implementation(name='modern', version='9.9'),
        server_capabilities=mcp_types.ServerCapabilities(),
    )


@pytest.fixture
def as_legacy_mcp_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Present the connected client as a handshake-era session."""
    init_result = mcp_types.InitializeResult(
        protocolVersion='2025-11-25',
        capabilities=mcp_types.ServerCapabilities(),
        serverInfo=mcp_types.Implementation(name='legacy', version='1.0'),
    )
    monkeypatch.setattr(Client, 'initialize_result', property(lambda self: init_result))


class TestMCPToolsetBackgroundTasks:
    """Task-augmented execution across both generations.

    FastMCP 3 speaks SEP-1686, where the server declares per-tool `execution.taskSupport` and
    `MCPToolset` routes the call according to that declaration and the client's task preference.
    FastMCP 4 speaks SEP-2663, where the client no longer routes at all — an ordinary call to a
    task-only tool is driven to completion transparently — and `use_task=True` explicitly routes
    through the tasks extension while still waiting for the completed result.
    """

    @pytest.fixture
    async def task_server(self) -> FastMCP[None]:
        server: FastMCP[None] = FastMCP('task_server')
        if MCP_SDK_V2:
            # The FastMCP 4 compatibility environment installs the task extra so this integration
            # is exercised rather than skipped.
            assert TasksExtension is not None
            # FastMCP 4 moved task execution into an optional extension package.
            getattr(server, 'add_extension')(TasksExtension())

        @server.tool(task=TaskConfig(mode='required'))
        async def task_required_tool() -> str:
            """A tool that requires task-augmented execution."""
            await asyncio.sleep(0)
            return 'task_required_completed'

        @server.tool(task=TaskConfig(mode='optional'))
        async def task_optional_tool(ctx: Context) -> str:
            """A tool that may run either as a task or synchronously."""
            await asyncio.sleep(0)
            mode = 'task' if ctx.is_background_task else 'sync'
            return f'task_optional_{mode}'

        @server.tool(task=TaskConfig(mode='forbidden'))
        async def task_forbidden_tool() -> str:
            """A tool that forbids task-augmented execution."""
            return 'task_forbidden_completed'

        @server.tool()
        async def plain_tool() -> str:
            """A tool with no task support - `execution` is `None`."""
            return 'plain_completed'

        return server

    async def test_get_tools_exposes_client_task_routing(
        self, task_server: FastMCP[None], run_context: RunContext[None]
    ) -> None:
        """FastMCP 3 exposes its client-side routing choice; FastMCP 4 routes server-side."""
        toolset = MCPToolset(task_server)
        async with toolset:
            tools = await toolset.get_tools(run_context)

        client_routes_tasks = not MCP_SDK_V2
        assert (tools['task_required_tool'].tool_def.metadata or {}).get('task') is client_routes_tasks
        assert (tools['task_optional_tool'].tool_def.metadata or {}).get('task') is client_routes_tasks
        assert (tools['task_forbidden_tool'].tool_def.metadata or {}).get('task') is False
        assert (tools['plain_tool'].tool_def.metadata or {}).get('task') is False

        toolset = MCPToolset(task_server, prefer_tasks=False)
        async with toolset:
            tools = await toolset.get_tools(run_context)

        assert (tools['task_required_tool'].tool_def.metadata or {}).get('task') is client_routes_tasks
        assert (tools['task_optional_tool'].tool_def.metadata or {}).get('task') is False

    async def test_explicit_forbidden_task_support_stays_on_sync_path(
        self, run_context: RunContext[None], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The in-process FastMCP server omits `execution` for forbidden tools, so supply the
        explicit protocol value to cover that distinct wire shape."""
        toolset = MCPToolset('https://example.com/mcp')
        monkeypatch.setattr(
            toolset,
            'list_tools',
            AsyncMock(
                return_value=[
                    mcp_types.Tool(
                        name='forbidden_tool',
                        inputSchema={'type': 'object'},
                        execution=mcp_types.ToolExecution(taskSupport='forbidden'),
                    )
                ]
            ),
        )
        direct_call_tool = AsyncMock(return_value='completed')
        monkeypatch.setattr(toolset, 'direct_call_tool', direct_call_tool)

        tools = await toolset.get_tools(run_context)
        metadata = tools['forbidden_tool'].tool_def.metadata
        assert metadata is not None
        assert metadata['task'] is False
        result = await toolset.call_tool('forbidden_tool', {}, run_context, tools['forbidden_tool'])

        assert result == 'completed'
        direct_call_tool.assert_awaited_once_with('forbidden_tool', {}, use_task=False)

    async def test_required_tool_follows_generation_task_semantics(
        self, task_server: FastMCP[None], run_context: RunContext[None]
    ) -> None:
        """FastMCP 3 routes required tasks explicitly; FastMCP 4 drives them transparently."""
        toolset = MCPToolset(task_server, prefer_tasks=False)
        async with toolset:
            tools = await toolset.get_tools(run_context)
            assert (tools['task_required_tool'].tool_def.metadata or {}).get('task') is (not MCP_SDK_V2)
            result = await toolset.call_tool('task_required_tool', {}, run_context, tools['task_required_tool'])
        assert result == 'task_required_completed'

    async def test_optional_tool_follows_default_generation_task_semantics(
        self, task_server: FastMCP[None], run_context: RunContext[None]
    ) -> None:
        """Both generations run optional tools as tasks by default, but only v1 routes client-side."""
        toolset = MCPToolset(task_server)
        async with toolset:
            tools = await toolset.get_tools(run_context)
            assert (tools['task_optional_tool'].tool_def.metadata or {}).get('task') is (not MCP_SDK_V2)
            result = await toolset.call_tool('task_optional_tool', {}, run_context, tools['task_optional_tool'])
        assert result == 'task_optional_task'

    async def test_optional_tool_follows_generation_semantics_when_tasks_disabled(
        self, task_server: FastMCP[None], run_context: RunContext[None]
    ) -> None:
        toolset = MCPToolset(task_server, prefer_tasks=False)
        async with toolset:
            tools = await toolset.get_tools(run_context)
            result = await toolset.call_tool('task_optional_tool', {}, run_context, tools['task_optional_tool'])
        assert result == ('task_optional_task' if MCP_SDK_V2 else 'task_optional_sync')

    async def test_agent_follows_generation_semantics_when_tasks_disabled(self, task_server: FastMCP[None]) -> None:
        toolset = MCPToolset(task_server, prefer_tasks=False)
        agent = Agent(TestModel(call_tools=['task_optional_tool']), toolsets=[toolset])

        result = await agent.run('Call the optional task tool')

        expected_result = 'task_optional_task' if MCP_SDK_V2 else 'task_optional_sync'
        assert result.output == f'{{"task_optional_tool":"{expected_result}"}}'

    async def test_reconstructed_tool_definition_preserves_task_routing(
        self,
        task_server: FastMCP[None],
        run_context: RunContext[None],
    ) -> None:
        """Serialized tool metadata preserves the effective routing choice for durable workers.

        Both flag directions are asserted: the `True` direction is the one that fails loudly if
        `tool_for_tool_def` ever drops metadata (a sync call to the required tool is rejected by
        the server), while the `False` direction alone would be indistinguishable from that."""
        toolset = MCPToolset(task_server, prefer_tasks=False)
        async with toolset:
            tools = await toolset.get_tools(run_context)
            adapter = TypeAdapter(ToolDefinition)

            def round_trip(name: str) -> ToolsetTool[None]:
                tool_def = adapter.validate_json(adapter.dump_json(tools[name].tool_def))
                return toolset.tool_for_tool_def(tool_def, ctx=run_context)

            optional_result = await toolset.call_tool(
                'task_optional_tool', {}, run_context, round_trip('task_optional_tool')
            )
            required_result = await toolset.call_tool(
                'task_required_tool', {}, run_context, round_trip('task_required_tool')
            )

        assert optional_result == ('task_optional_task' if MCP_SDK_V2 else 'task_optional_sync')
        assert required_result == 'task_required_completed'

    async def test_forbidden_tool_stays_on_sync_path(
        self, task_server: FastMCP[None], run_context: RunContext[None]
    ) -> None:
        toolset = MCPToolset(task_server)
        async with toolset:
            tools = await toolset.get_tools(run_context)
            result = await toolset.call_tool('task_forbidden_tool', {}, run_context, tools['task_forbidden_tool'])
        assert result == 'task_forbidden_completed'

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

    async def test_task_call_on_a_legacy_session_is_rejected(
        self, fastmcp_server: FastMCP[None], as_mcp_sdk_v2: None
    ) -> None:
        """Under MCP SDK v2, tasks ride a client extension that a legacy session never negotiates,
        so a legacy client has no task path at all."""
        client = make_legacy_client(fastmcp_server)
        toolset = MCPToolset(client)
        async with toolset:
            with pytest.raises(UserError, match='not supported by FastMCP 4 clients using legacy protocol mode'):
                await toolset.direct_call_tool('echo', {'message': 'hi'}, use_task=True)

    async def test_task_call_without_fastmcp_tasks_installed_is_rejected(
        self, fastmcp_server: FastMCP[None], as_modern_mcp_session: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MCP SDK v2 moved tasks out of fastmcp core into the separate `fastmcp-tasks` package;
        the error names the `mcp-tasks` extra that installs it."""
        monkeypatch.setattr(mcp_module, '_load_call_tool_task', lambda: None)

        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            with pytest.raises(ImportError, match=r'`mcp-tasks` optional group'):
                await toolset.direct_call_tool('echo', {'message': 'hi'}, use_task=True)

    async def test_sdk_v2_disables_client_task_routing(
        self, task_server: FastMCP[None], run_context: RunContext[None], as_mcp_sdk_v2: None
    ) -> None:
        """Client-side task routing is a SEP-1686 (SDK v1) concept, so a server-declared
        `taskSupport` must not route SDK v2 calls into the client task path: a legacy session has
        no such path at all, and a modern one completes task tools on an ordinary call anyway."""
        client = make_legacy_client(task_server)
        toolset = MCPToolset(client)
        async with toolset:
            tools = await toolset.get_tools(run_context)
            for name in ('task_required_tool', 'task_optional_tool', 'task_forbidden_tool', 'plain_tool'):
                assert (tools[name].tool_def.metadata or {}).get('task') is False, name

    async def test_task_call_dispatches_through_fastmcp_tasks(
        self, fastmcp_server: FastMCP[None], as_modern_mcp_session: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On a modern session the call goes through `fastmcp_tasks.call_tool_task`, which hands back
        a handle whose `result()` resolves to the same `CallToolResult` the direct path returns."""
        calls: list[str] = []

        async def call_tool_task(client: Any, **kwargs: Any) -> Any:
            calls.append(kwargs['name'])
            result = await client.call_tool(**kwargs)
            return SimpleNamespace(result=AsyncMock(return_value=result))

        monkeypatch.setattr(mcp_module, '_load_call_tool_task', lambda: call_tool_task)

        toolset = MCPToolset(fastmcp_server)
        async with toolset:
            result = await toolset.direct_call_tool('echo', {'message': 'hi'}, use_task=True)
        assert result == 'Echo: hi'
        assert calls == ['echo']

    @pytest.mark.parametrize('prefer_tasks', [True, False])
    async def test_process_tool_call_preserves_task_preference(
        self,
        task_server: FastMCP[None],
        run_context: RunContext[None],
        prefer_tasks: bool,
    ) -> None:
        """The delegate carries the effective client-side preference when the wrapper invokes it."""

        async def passthrough(ctx: RunContext[Any], call_tool: Any, name: str, args: dict[str, Any]) -> Any:
            return await call_tool(name, args)

        toolset = MCPToolset(
            task_server,
            process_tool_call=passthrough,
            prefer_tasks=prefer_tasks,
        )
        async with toolset:
            tools = await toolset.get_tools(run_context)
            result = await toolset.call_tool('task_optional_tool', {}, run_context, tools['task_optional_tool'])
        assert result == ('task_optional_task' if MCP_SDK_V2 or prefer_tasks else 'task_optional_sync')

    async def test_process_tool_call_preserves_metadata_and_task_preference(
        self,
        fastmcp_server: FastMCP[None],
        run_context: RunContext[None],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Mocks `direct_call_tool` to pin the exact delegate kwargs: a wrapper's `metadata` must
        arrive alongside the baked-in `use_task`. The task-preference half is covered end-to-end
        against the real server above; this pins the internal call shape that carries both."""

        async def add_metadata(ctx: RunContext[Any], call_tool: Any, name: str, args: dict[str, Any]) -> Any:
            return await call_tool(name, args, metadata={'trace_id': '123'})

        toolset = MCPToolset(
            fastmcp_server,
            process_tool_call=add_metadata,
            prefer_tasks=False,
        )
        direct_call_tool = AsyncMock(return_value='completed')
        monkeypatch.setattr(toolset, 'direct_call_tool', direct_call_tool)
        tools = await toolset.get_tools(run_context)

        result = await toolset.call_tool('echo', {'message': 'hi'}, run_context, tools['echo'])

        assert result == 'completed'
        direct_call_tool.assert_awaited_once_with(
            'echo',
            {'message': 'hi'},
            metadata={'trace_id': '123'},
            use_task=False,
        )

    async def test_process_tool_call_can_short_circuit_without_calling_server(
        self, run_context: RunContext[None], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A callback can return a cached result without making an MCP tool call.

        Mocks `direct_call_tool` because the property under test is that no server call happens at
        all (`assert_not_awaited`), which no real-server response could demonstrate."""

        async def short_circuit(ctx: RunContext[Any], call_tool: Any, name: str, args: dict[str, Any]) -> Any:
            return 'cached'

        toolset = MCPToolset('https://example.com/mcp', process_tool_call=short_circuit)
        direct_call_tool = AsyncMock(side_effect=AssertionError('tool call should not run'))
        monkeypatch.setattr(toolset, 'direct_call_tool', direct_call_tool)
        tool = toolset.tool_for_tool_def(ToolDefinition(name='durable_tool'), ctx=run_context)

        result = await toolset.call_tool('durable_tool', {}, run_context, tool)

        assert result == 'cached'
        direct_call_tool.assert_not_awaited()
