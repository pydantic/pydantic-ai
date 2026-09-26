"""Connect Pydantic AI agents to Grain's hosted MCP server.

Requires the `grain` extra: `uv add "pydantic-ai-harness[grain]"`.
"""

from pydantic_ai_harness.grain._capability import Grain

__all__ = ['Grain']
