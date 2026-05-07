"""Construction-path tests for `pydantic_ai.mcp.MCPToolset`.

These cover the unique logic that the new class adds on top of the legacy `FastMCPToolset` and
`MCPServer*` paths — kwarg conflict detection, HTTP transport adapter for `http_client=`, sampling
shortcut, and the cache-invalidating message handler. End-to-end behavior is exercised by the
existing tests in `test_fastmcp.py` and `test_mcp.py`, both of which cover the deprecated paths
that share data flow with this class.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from pydantic_ai import models

from .conftest import try_import

with try_import() as imports_successful:
    from fastmcp.client import Client
    from fastmcp.client.transports import (
        SSETransport,
        StreamableHttpTransport,
    )
    from mcp import types as mcp_types

    from pydantic_ai.mcp import MCPError, MCPToolset, ResourceAnnotations, ResourceTemplate


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='fastmcp not installed'),
    pytest.mark.anyio,
]


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
        # Factory installed; passing through any kwargs returns the user's client
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

    def test_pre_built_client_with_overridden_timeout_raises(self):
        client = Client('https://example.com/mcp')
        with pytest.raises(ValueError, match='init_timeout'):
            MCPToolset(client, init_timeout=30)

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
        # Handler is wired into the underlying Client via session_kwargs
        assert toolset.client._session_kwargs.get('sampling_callback') is not None  # pyright: ignore[reportPrivateUsage]


class TestMCPToolsetIdentity:
    def test_id_property(self):
        toolset = MCPToolset('https://example.com/mcp', id='example')
        assert toolset.id == 'example'

    def test_repr(self):
        toolset = MCPToolset('https://example.com/mcp', id='example')
        assert "id='example'" in repr(toolset)

    def test_pre_init_property_access_raises(self):
        toolset = MCPToolset('https://example.com/mcp')
        with pytest.raises(AttributeError, match='only available after initialization'):
            _ = toolset.server_info
        with pytest.raises(AttributeError, match='only available after initialization'):
            _ = toolset.capabilities
        with pytest.raises(AttributeError, match='only available after initialization'):
            _ = toolset.instructions
        assert toolset.is_running is False


class TestMCPToolsetCacheInvalidation:
    def test_cache_invalidation_helpers_clear_state(self):
        """Smoke test for the cache-invalidation helpers used by the message handler."""
        toolset = MCPToolset('https://example.com/mcp')
        # Pre-populate so we can verify clearing
        toolset._cached_tools = []  # pyright: ignore[reportPrivateUsage]
        toolset._cached_resources = []  # pyright: ignore[reportPrivateUsage]
        toolset._invalidate_tools_cache()  # pyright: ignore[reportPrivateUsage]
        toolset._invalidate_resources_cache()  # pyright: ignore[reportPrivateUsage]
        assert toolset._cached_tools is None  # pyright: ignore[reportPrivateUsage]
        assert toolset._cached_resources is None  # pyright: ignore[reportPrivateUsage]


class TestResourceTypeMapping:
    """The PAI-shaped `Resource` / `ResourceTemplate` / `MCPError` types are kept under
    `pydantic_ai.mcp.*`. They were ported from the deprecated `MCPServer*` path; these tests pin
    the wire-level field mapping so that drifts from the MCP SDK schema are caught."""

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


def test_construction_does_not_emit_warnings(recwarn: Any) -> None:
    """Building an `MCPToolset` from a URL must not emit `FastMCPDeprecationWarning` for the
    `sse_read_timeout` parameter — the StreamableHttp path migrated off it (the FastMCP `Client`
    `timeout` carries the read timeout instead)."""
    MCPToolset('https://example.com/mcp', headers={'X-Key': 'foo'})
    deprecation_messages = [str(w.message) for w in recwarn if 'sse_read_timeout' in str(w.message)]
    assert deprecation_messages == [], deprecation_messages
