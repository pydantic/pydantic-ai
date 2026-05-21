"""v1: result.usage() called as a method is deprecated; access as a property in v2."""
import asyncio

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

_agent = Agent(TestModel())


async def _drive():
    r = await _agent.run('hi')
    # DEPRECATION: C2_usage_method
    _ = r.usage()


def trigger():
    asyncio.run(_drive())


EXPECT = '`AgentRunResult.usage` is no longer a method'

if __name__ == '__main__':
    trigger()
