"""v1: AgentStream.stream_responses() (plural) is deprecated; use stream_response() (singular)."""
import asyncio

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

_agent = Agent(TestModel())


async def _drive():
    async with _agent.iter('hi') as run_ctx:
        async for node in run_ctx:
            if Agent.is_model_request_node(node):
                async with node.stream(run_ctx.ctx) as stream:
                    # DEPRECATION: C1_stream_responses — plural form
                    async for _msg in stream.stream_responses():
                        pass


def trigger():
    asyncio.run(_drive())


EXPECT = '`AgentStream.stream_responses()` is deprecated'

if __name__ == '__main__':
    trigger()
