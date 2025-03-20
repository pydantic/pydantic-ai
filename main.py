import json

from rich.pretty import pprint

from pydantic_ai import Agent

agent = Agent('openai:gpt-4o', system_prompt='You are a helpful assistant.')

result = agent.run_sync('What is the capital of France?')
pprint(json.loads(result.all_messages_json().decode('utf-8')))

print()
agent2 = Agent('openai:gpt-4o', system_prompt='You are a baseball player.')
result2 = agent2.run_sync('Do you play baseball?', message_history=result.all_messages())
pprint(json.loads(result2.all_messages_json().decode('utf-8')))
