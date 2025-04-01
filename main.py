import asyncio

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent(
    model='anthropic:claude-3-7-sonnet-latest',
    model_settings=AnthropicModelSettings(anthropic_thinking={'budget_tokens': 1024, 'type': 'enabled'}),
)


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
    result = await agent.run('Get me the sum of 1 and 2, using the sum tool.')
    print(result.data)

    result = await agent.run(
        'Sum the previous result with 3.', model='openai:gpt-4o', message_history=result.all_messages()
    )
    print(result.data)

    from rich.pretty import pprint

    pprint(result.all_messages())
    # async with agent.iter('Get me the sum of 1 and 2, using the sum tool.') as agent_run:
    #     async for node in agent_run:
    #         print(node)
    #         print()
    # print(agent_run.result)


if __name__ == '__main__':
    asyncio.run(main())
