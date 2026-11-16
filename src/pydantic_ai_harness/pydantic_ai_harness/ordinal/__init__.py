"""Connect Pydantic AI agents to Ordinal's hosted MCP server.

Requires the `ordinal` extra: `uv add "pydantic-ai-harness[ordinal]"`.
"""

from pydantic_ai_harness.ordinal._capability import Ordinal

__all__ = ['Ordinal']
