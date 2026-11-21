"""Ask a real model a question that needs a tool.

Run it from the repository root, with `OPENAI_API_KEY` set:
`uv run python experiments/pydantic-ai-core/python/example.py`.
"""

from __future__ import annotations

import asyncio

from pydantic_ai_core import Agent

agent = Agent('gpt-5.4-mini', instructions='Answer in one short sentence.')


@agent.tool_plain
def get_temperature(city: str) -> dict[str, str | int]:
    """Get the current temperature in a city."""
    return {'city': city, 'celsius': 21}


async def main() -> None:
    """Run the agent once and print what it answered."""
    result = await agent.run('What is the temperature in Tokyo?')
    print(result.output)
    print(result.usage)
    print([type(message).__name__ for message in result.messages])


if __name__ == '__main__':
    asyncio.run(main())
