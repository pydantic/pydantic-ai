"""An ACP agent with a slow tool, used by the over-the-wire cancellation test."""

from __future__ import annotations

import asyncio

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.experimental.acp import run_acp_stdio_sync


def build_agent() -> Agent[None, str]:
    agent = Agent(TestModel())

    @agent.tool_plain
    async def slow() -> str:
        await asyncio.sleep(30)
        return 'done'

    return agent


if __name__ == '__main__':
    run_acp_stdio_sync(build_agent())
