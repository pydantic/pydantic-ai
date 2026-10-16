"""Connect Pydantic AI agents to PostHog's hosted MCP server.

Requires the `posthog` extra: `uv add "pydantic-ai-harness[posthog]"`.
"""

from pydantic_ai_harness.posthog._capability import PostHog

__all__ = ['PostHog']
