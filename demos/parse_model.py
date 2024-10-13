from pydantic import BaseModel

from pydantic_ai import Agent


class MyModel(BaseModel):
    city: str
    country: str


agent = Agent('openai:gpt-4o', result_type=MyModel, deps=None)

result = agent.run_sync('The windy city in the US of A.')

print(result.response)
