import json

from rich.pretty import pprint

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel

model = OpenAIResponsesModel(model_name='gpt-4o')
agent = Agent(model)


@agent.instructions
def simple_instructions():
    return 'You play soccer.'


result = agent.run_sync('What sport do you play?')
pprint(json.loads(result.all_messages_json().decode('utf-8')))

print()
agent2 = Agent(model, instructions='You play volleyball.')
result2 = agent2.run_sync('What sport do you play?', message_history=result.all_messages())
pprint(json.loads(result2.all_messages_json().decode('utf-8')))
