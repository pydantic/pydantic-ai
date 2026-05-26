"""`pydantic_ai.mcp` prefers `httpx2`; constructing an HTTP MCP server while only `httpx` is
installed emits a `PydanticAIDeprecationWarning`. The fallback is removed in v2."""

from __future__ import annotations

import importlib.util
import warnings

import pytest

from pydantic_ai._warnings import PydanticAIDeprecationWarning

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.mcp import (
        MCPServerSSE,  # pyright: ignore[reportDeprecated]
        MCPServerStreamableHTTP,  # pyright: ignore[reportDeprecated]
        _MCPServerHTTP,  # pyright: ignore[reportPrivateUsage]
    )


pytestmark = pytest.mark.skipif(not imports_successful(), reason='mcp not installed')


@pytest.mark.skipif(importlib.util.find_spec('httpx2') is not None, reason='httpx2 installed; fallback not exercised')
@pytest.mark.parametrize(
    'cls',
    [
        MCPServerSSE,  # pyright: ignore[reportDeprecated]
        MCPServerStreamableHTTP,  # pyright: ignore[reportDeprecated]
    ],
)
def test_http_mcp_server_warns_on_httpx_fallback(cls: type[_MCPServerHTTP]):
    """Constructing an HTTP MCP server warns when `httpx2` isn't installed.

    The class itself is also `@deprecated` in v2, so we filter the unrelated stdlib
    `DeprecationWarning` to assert only on the httpx2 migration warning.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        with pytest.warns(
            PydanticAIDeprecationWarning,
            match=r'Using `httpx` with `pydantic_ai\.mcp` is deprecated; install `httpx2` instead\.',
        ):
            cls('http://localhost:3001/sse')
