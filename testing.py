from pydantic_ai import Agent

agent = Agent('databricks:databricks-claude-sonnet-4')
print(agent.run_sync('hello!'))
