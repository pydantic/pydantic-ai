from pydantic_ai.agent import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel

model = OpenAIResponsesModel('gpt-4o')
agent = Agent(model=model)

result = agent.run_sync('Say the name of three random countries.')
print(result.data)

result = agent.run_sync('Tell me the capital of those three countries.', message_history=result.all_messages())
print(result.data)
