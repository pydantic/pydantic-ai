import logfire

from pydantic_ai import Agent

logfire.configure()

agent = Agent('openai:o3-mini', name='Test Agent')
app = agent.to_a2a()
logfire.instrument_starlette(app)
