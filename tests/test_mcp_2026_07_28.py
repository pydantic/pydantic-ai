"""Contracts around the sessionless Streamable HTTP revision.

The first test exercises Pydantic AI's ownership boundary today. The strict expected failures are
dependency gates, not wire tests: once the MCP SDK can speak the revision, each must be replaced
with a real request-level test against a 2026-07-28 server.
"""

from __future__ import annotations

import pytest

from .conftest import try_import

with try_import() as imports_successful:
    from fastmcp.client.transports import StreamableHttpTransport
    from mcp.shared.version import SUPPORTED_PROTOCOL_VERSIONS

    from pydantic_ai.mcp import MCPToolset


pytestmark = pytest.mark.skipif(not imports_successful(), reason='fastmcp not installed')

_MODERN_PROTOCOL = '2026-07-28'


def _requires_modern_protocol(contract: str) -> pytest.MarkDecorator:
    """Mark a future wire contract unavailable on the currently locked MCP SDK."""
    return pytest.mark.xfail(
        strict=True,
        reason=(
            f'Dependency precondition for the {contract} wire contract: the locked MCP SDK supports '
            'Streamable HTTP through 2025-11-25. MCP 2026-07-28 support requires the FastMCP 4 / '
            'MCP SDK v2 compatibility work in #6738; replace this gate with a real wire test on XPASS.'
        ),
    )


def test_http_toolsets_keep_credentials_and_transport_state_isolated() -> None:
    """Per-user authentication must not share a client, transport, or configured credentials.

    This is a construction-level contract because making a real server request would only test
    FastMCP's legacy protocol behavior. It protects Pydantic AI's documented per-user isolation
    boundary while the protocol implementation remains delegated to FastMCP.
    """
    alice = MCPToolset('https://example.com/mcp', headers={'Authorization': 'Bearer alice'})
    bob = MCPToolset('https://example.com/mcp', headers={'Authorization': 'Bearer bob'})

    assert alice.client is not bob.client
    assert isinstance(alice.client.transport, StreamableHttpTransport)
    assert isinstance(bob.client.transport, StreamableHttpTransport)
    assert alice.client.transport is not bob.client.transport
    assert alice.client.transport.headers == {'Authorization': 'Bearer alice'}
    assert bob.client.transport.headers == {'Authorization': 'Bearer bob'}


@_requires_modern_protocol('no-initialize')
def test_2026_07_28_requires_no_initialize_handshake() -> None:
    """Gate the future test that proves modern requests never send `initialize`."""
    assert _MODERN_PROTOCOL in SUPPORTED_PROTOCOL_VERSIONS


@_requires_modern_protocol('no-session-id')
def test_2026_07_28_requires_no_mcp_session_id() -> None:
    """Gate the future test that proves modern requests never send `Mcp-Session-Id`."""
    assert _MODERN_PROTOCOL in SUPPORTED_PROTOCOL_VERSIONS


@_requires_modern_protocol('POST-only, request-scoped SSE, and request metadata')
def test_2026_07_28_requires_post_only_request_scoped_sse_and_metadata() -> None:
    """Gate POST-only delivery, version/method headers, and operation-specific name headers."""
    assert _MODERN_PROTOCOL in SUPPORTED_PROTOCOL_VERSIONS


@_requires_modern_protocol('per-request authorization')
def test_2026_07_28_requires_authorization_on_every_post() -> None:
    """Gate the future test that proves credentials are sent on each independent HTTP request."""
    assert _MODERN_PROTOCOL in SUPPORTED_PROTOCOL_VERSIONS
