"""Agentic Chat feature."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from pydantic_ai import Agent
from pydantic_ai.ag_ui import AGUIApp

# Ensure environment variables are loaded.
load_dotenv()

agent: Agent = Agent(
    'openai:gpt-4o-mini',
    output_type=str,
)

app: AGUIApp = agent.to_ag_ui()


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
