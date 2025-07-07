"""Agentic Chat feature."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic_ai.ag_ui import FastAGUI

from .agent import agent

app: FastAGUI = agent()


@app.adapter.agent.tool_plain
async def current_time(timezone: str = 'UTC') -> str:
    """Get the current time in ISO format.

    Args:
        timezone: The timezone to use.

    Returns:
        The current time in ISO format string.
    """
    tz: ZoneInfo = ZoneInfo(timezone)
    return datetime.now(tz=tz).isoformat()
