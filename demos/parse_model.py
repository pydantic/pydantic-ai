from devtools import debug
from pydantic import BaseModel

from pydantic_ai import Agent


class MyModel(BaseModel):
    city: str
    country: str


agent = Agent('openai:gpt-4o', response_type=MyModel)

# debug(agent.result_schema.json_schema)
result = agent.run_sync('The windy city in the US of A.')

debug(result.response)
