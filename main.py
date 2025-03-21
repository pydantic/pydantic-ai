from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

provider = OpenAIProvider(base_url='http://127.0.0.1:11434/v1')
model = OpenAIModel('llama3.2', provider=provider)

agent = Agent(model=model)


@agent.tool_plain
async def my_tool(a: int, b: int) -> int:
    """Sum two numbers.

    Args:
        a: First number.
        b: Second number:

    Returns:
        int: The sum between a and b.
    """
    return a + b


result = agent.run_sync('Sum 10 + 15!')
print(result.data)
