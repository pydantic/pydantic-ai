"""v1: Agent.to_a2a()."""
from pydantic_ai import Agent


def trigger():
    a = Agent('openai-chat:gpt-4o')
    # DEPRECATION: E1_to_a2a
    return a.to_a2a()


EXPECT = '`Agent.to_a2a()` is deprecated'

if __name__ == '__main__':
    trigger()
