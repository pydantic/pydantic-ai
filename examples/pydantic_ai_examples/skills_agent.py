"""Skills integration example demonstrating progressive skill discovery with Pydantic AI.

This example shows how to create an agent with skills that can:
- List available skills
- Load detailed skill instructions on demand
- Read additional resources
- Execute skill scripts
"""

import asyncio
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.toolsets import SkillsToolset


async def main() -> None:
    """Pydantic AI with Agent Skills."""
    # Get the skills directory (examples/pydantic_ai_examples/skills)
    skills_dir = Path(__file__).parent / 'skills'

    # Initialize Skills Toolset
    skills_toolset = SkillsToolset(directories=[skills_dir])

    # Create agent with skills
    agent = Agent(
        model='openai:gpt-4o',
        instructions='You are a helpful research assistant.',
        toolsets=[skills_toolset],
    )

    # Add skills system prompt (includes skill descriptions and usage instructions)
    @agent.system_prompt
    async def add_skills_prompt() -> str:  # pyright: ignore[reportUnusedFunction]
        return skills_toolset.get_skills_system_prompt()

    # Use agent - skills tools are available for the agent to call
    result = await agent.run('What are the main features of Pydantic AI framework?')
    print(f'\nResponse:\n{result.output}')


if __name__ == '__main__':
    asyncio.run(main())
