"""v2: use stream_response() (singular). Yields ModelResponse snapshots directly."""
import asyncio

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

_agent = Agent(TestModel())


async def _drive():
    async with _agent.iter('hi') as run_ctx:
        async for node in run_ctx:
            if Agent.is_model_request_node(node):
                async with node.stream(run_ctx.ctx) as stream:
                    async for _msg in stream.stream_response():
                        pass


def trigger():
    asyncio.run(_drive())


if __name__ == '__main__':
    trigger()
