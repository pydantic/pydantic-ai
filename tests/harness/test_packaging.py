"""The generic `mcp` extra installs the MCP client shared by hosted MCP integrations.

Only metadata checks belong here: this file stays collected on base (no-extras) installs.
"""

from __future__ import annotations

import importlib.metadata


def test_mcp_extra_installs_the_slim_mcp_extra() -> None:
    metadata = importlib.metadata.metadata('pydantic-ai-harness')
    assert 'mcp' in (metadata.get_all('Provides-Extra') or [])
    mcp_requirements = [
        req for req in metadata.get_all('Requires-Dist') or [] if 'extra == "mcp"' in req or "extra == 'mcp'" in req
    ]
    assert mcp_requirements
    assert all(req.startswith('pydantic-ai-slim[mcp]') for req in mcp_requirements)
