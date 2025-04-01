import asyncio

from pydantic_ai import Agent

agent = Agent(model='anthropic:claude-3-7-sonnet-latest')


@agent.tool_plain
def sum(a: int, b: int) -> int:
    """Get the sum of two numbers.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        The sum of the two numbers.
    """
    return a + b


async def main():
    async with agent.iter('Get me the sum of 1 and 2, using the sum tool.') as agent_run:
        async for node in agent_run:
            print(node)
            print()
    print(agent_run.result)


if __name__ == '__main__':
    asyncio.run(main())
