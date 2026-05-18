"""Regression test for issue #5512.

`pydantic-ai-slim[mcp]` pulls `fastmcp-slim[client]`, which does NOT ship
`fastmcp.server`. The `pydantic_ai.mcp` module must still import cleanly and
let users construct an `MCPToolset` in this configuration — the
`fastmcp.server.FastMCP` symbol is only used in the *stringified*
`MCPToolsetClient` type alias (resolved by type-checkers via the
`TYPE_CHECKING` import).
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest

from .conftest import try_import

with try_import() as imports_successful:
    # The test simulates a slim install (no `fastmcp.server`) but still needs the
    # rest of fastmcp.client.* and the bare `mcp` SDK available to exercise the
    # actual import path; skip if the dev env doesn't have any fastmcp at all.
    import fastmcp.client  # noqa: F401


pytestmark = pytest.mark.skipif(
    not imports_successful(),
    reason='fastmcp not installed; this test simulates the slim install path',
)


def test_mcptoolset_importable_and_constructible_without_fastmcp_server():
    """Regression test for #5512.

    With `fastmcp.server` made unimportable (the `fastmcp-slim[client]` reality):

    1. `from pydantic_ai.mcp import MCPToolset` must succeed.
    2. `MCPToolset('https://example.com/mcp')` must construct without raising
       `ImportError` from `_require_fastmcp()`.

    Prior to the fix, importing the module set `_fastmcp_import_error` because of
    a runtime `from fastmcp.server import FastMCP` statement, and any subsequent
    `MCPToolset(...)` call raised the documented `pip install ...` message.
    """
    # Drop any cached pydantic_ai.mcp so the next import re-runs the try/except.
    saved_modules: dict[str, object] = {}
    for name in list(sys.modules):
        if name == 'pydantic_ai.mcp' or name.startswith('pydantic_ai.mcp.'):
            saved_modules[name] = sys.modules.pop(name)

    try:
        # `sys.modules[name] = None` makes `import name` raise ImportError —
        # the standard trick for simulating an absent module without uninstalling it.
        with patch.dict(sys.modules, {'fastmcp.server': None}):
            mcp_mod = importlib.import_module('pydantic_ai.mcp')
            MCPToolset = mcp_mod.MCPToolset

            # Construction exercises `_require_fastmcp()`; before the fix this raised.
            toolset = MCPToolset('https://example.com/mcp')
            assert toolset.client is not None
    finally:
        # Restore previously-cached pydantic_ai.mcp module(s) so other tests don't
        # pick up the version imported under the patched sys.modules.
        for name in list(sys.modules):
            if name == 'pydantic_ai.mcp' or name.startswith('pydantic_ai.mcp.'):
                del sys.modules[name]
        sys.modules.update(saved_modules)
