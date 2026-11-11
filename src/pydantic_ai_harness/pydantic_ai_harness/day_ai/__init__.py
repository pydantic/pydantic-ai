"""Connect Pydantic AI agents to Day AI's hosted MCP server.

Requires the `day-ai` extra: `uv add "pydantic-ai-harness[day-ai]"`.
"""

from pydantic_ai_harness.day_ai._capability import DayAI

__all__ = ['DayAI']
