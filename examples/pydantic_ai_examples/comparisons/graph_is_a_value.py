"""The agent loop is a value you can drive — no graph DSL required.

Pydantic AI doesn't need you to adopt a graph abstraction for structure.
The run is plain async code over a typed value; when you do want the
graph, it's the same value, iterated node by node.
"""
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart


async def model(messages, info):
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('double', {'n': 21})])
    return ModelResponse(parts=[TextPart('42')])


agent = Agent(FunctionModel(model))


@agent.tool
def double(ctx, n: int) -> int:
    return n * 2


async def main():
    nodes = []
    async with agent.iter('what is 21*2?') as run:
        async for node in run:
            nodes.append(type(node).__name__)
    print('nodes in one run:', ' -> '.join(nodes))
    print('the loop is a plain value: iterate it, drive it manually, or let a capability transform it')


asyncio.run(main())
