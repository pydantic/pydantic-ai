"""Connect Pydantic AI agents to Pylon's hosted MCP server.

Requires the `pylon` extra: `uv add "pydantic-ai-harness[pylon]"`.
"""

from pydantic_ai_harness.pylon._capability import Pylon

__all__ = ['Pylon']
