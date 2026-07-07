"""A minimal ACP agent process used by the end-to-end stdio test."""

from __future__ import annotations

import asyncio

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.experimental.acp import run_acp_stdio


def build_agent() -> Agent[None, str]:
    agent = Agent(TestModel())

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return f'Sunny in {city}'

    return agent


if __name__ == '__main__':
    asyncio.run(run_acp_stdio(build_agent()))
