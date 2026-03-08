"""Agentic Chat feature."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic_ai import Agent
from pydantic_ai.ui.ag_ui.app import AGUIApp

agent = Agent('openai:gpt-5-mini')


@agent.tool_plain
async def current_time(timezone: str = 'UTC') -> str:
    """Get the current time in ISO format.

    Args:
        timezone: The timezone to use.

    Returns:
        The current time in ISO format string.
    """
    tz: ZoneInfo = ZoneInfo(timezone)
    return datetime.now(tz=tz).isoformat()


app = AGUIApp(agent)
