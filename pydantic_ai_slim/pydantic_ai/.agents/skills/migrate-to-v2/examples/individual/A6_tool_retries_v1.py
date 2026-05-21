"""v1: tool_retries=/output_retries= ctor kwargs."""
from pydantic_ai import Agent


def trigger():
    # DEPRECATION: A6_tool_retries
    return Agent('openai-chat:gpt-4o', tool_retries=3, output_retries=2)


EXPECT = "`Agent(tool_retries=...)` is deprecated"

if __name__ == '__main__':
    trigger()
