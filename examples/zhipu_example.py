"""Example of using Zhipu AI with Pydantic AI.

This example demonstrates how to use Zhipu AI models with the Pydantic AI framework.

To run this example, you need to:
1. Install pydantic-ai with openai support: pip install "pydantic-ai-slim[openai]"
2. Set your Zhipu API key: export ZHIPU_API_KEY='your-api-key'
3. Run the script: python examples/zhipu_example.py
"""

from __future__ import annotations as _annotations

import asyncio

from pydantic_ai import Agent


async def main():
    """Run a simple example with Zhipu AI."""
    # Create an agent using Zhipu AI's GLM-4.5 model
    model_spec = 'zhipu:glm-4.5'
    agent = Agent(model_spec, system_prompt='You are a helpful assistant.')

    # Run a simple query
    result = await agent.run('What is the capital of China?')
    print(f'Response: {result.output}')
    # Access the configured model name directly from the agent's model to avoid relying on
    # message internals (keeps static typing happy if message schema varies by provider).
    print(f'Model used: {model_spec}')


if __name__ == '__main__':
    asyncio.run(main())
