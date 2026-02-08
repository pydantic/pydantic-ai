import asyncio

from pydantic_ai import Agent
from pydantic_ai._acp import run_stdio

# Helper script to run an agent in stdio mode (e.g. for IDE integration)

agent = Agent(
    'openai:gpt-4o',
    system_prompt='You are a helpful coding assistant.',
)

if __name__ == '__main__':
    # This script reads from stdin and writes to stdout
    asyncio.run(run_stdio(agent, name='stdio-agent'))
