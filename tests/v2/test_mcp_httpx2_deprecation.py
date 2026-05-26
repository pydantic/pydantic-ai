"""`pydantic_ai.mcp` prefers `httpx2`; HTTP MCP server construction (legacy `MCPServerSSE` /
`MCPServerStreamableHTTP` and the new `MCPToolset` HTTP path) emits a
`PydanticAIDeprecationWarning` when only `httpx` is installed. The fallback is removed in v2."""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import pytest

from pydantic_ai._warnings import PydanticAIDeprecationWarning

from ..conftest import try_import

with try_import() as imports_successful:
    import pydantic_ai.mcp


pytestmark = pytest.mark.skipif(not imports_successful(), reason='mcp not installed')

skip_if_httpx2_installed = pytest.mark.skipif(
    importlib.util.find_spec('httpx2') is not None,
    reason='httpx2 installed; fallback not exercised',
)


@skip_if_httpx2_installed
@pytest.mark.parametrize('cls_name', ['MCPServerSSE', 'MCPServerStreamableHTTP'])
def test_legacy_http_mcp_server_warns_on_httpx_fallback(cls_name: str):
    """Constructing a legacy HTTP MCP server (deprecated path) warns when `httpx2` isn't installed.

    The class itself is also `@deprecated` in v2, so we filter the unrelated stdlib
    `DeprecationWarning` to assert only on the httpx2 migration warning.
    """
    cls = getattr(pydantic_ai.mcp, cls_name)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        with pytest.warns(
            PydanticAIDeprecationWarning,
            match=r'Using `httpx` with `pydantic_ai\.mcp` is deprecated; install `httpx2` instead\.',
        ):
            cls('http://localhost:3001/sse')


@skip_if_httpx2_installed
@pytest.mark.parametrize('url', ['http://localhost:3001/mcp', 'http://localhost:3001/sse'])
def test_mcptoolset_http_warns_on_httpx_fallback(url: str):
    """Constructing `MCPToolset` with an HTTP URL warns when `httpx2` isn't installed."""
    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'Using `httpx` with `pydantic_ai\.mcp` is deprecated; install `httpx2` instead\.',
    ):
        pydantic_ai.mcp.MCPToolset(url)


@skip_if_httpx2_installed
def test_mcptoolset_stdio_does_not_warn(tmp_path: Path):
    """Constructing `MCPToolset` with a non-URL client does not warn — no httpx is involved."""
    script = tmp_path / 'fake_mcp_server.py'
    script.write_text('# placeholder')
    with warnings.catch_warnings():
        warnings.simplefilter('error', PydanticAIDeprecationWarning)
        pydantic_ai.mcp.MCPToolset(script)
