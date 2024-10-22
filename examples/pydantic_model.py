"""Simple example of using Pydantic AI to construct a Pydantic model from a text input.

Run with:

    uv run -m examples.pydantic_model
"""

import os
from typing import cast

# if you don't want to use logfire, just comment out these lines
import logfire
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.agent import KnownModelName

logfire.configure()


class MyModel(BaseModel):
    city: str
    country: str


model = cast(KnownModelName, os.getenv('PYDANTIC_AI_MODEL', 'openai:gpt-4o'))
agent = Agent(model, result_type=MyModel, deps=None)

if __name__ == '__main__':
    result = agent.run_sync('The windy city in the US of A.')
    print(result.response)
