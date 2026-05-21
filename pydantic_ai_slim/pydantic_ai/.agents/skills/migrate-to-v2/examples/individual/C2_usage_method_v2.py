"""v2: result.usage is a property (drop the parentheses)."""
import asyncio

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

_agent = Agent(TestModel())


async def _drive():
    r = await _agent.run('hi')
    _ = r.usage


def trigger():
    asyncio.run(_drive())


if __name__ == '__main__':
    trigger()
