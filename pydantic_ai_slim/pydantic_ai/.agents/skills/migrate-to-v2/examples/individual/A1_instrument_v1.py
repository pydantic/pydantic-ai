"""v1: Agent(instrument=True) — deprecated, use capabilities=[Instrumentation()]."""
from pydantic_ai import Agent


def trigger():
    # DEPRECATION: A1_instrument
    return Agent('openai-chat:gpt-4o', instrument=True)


EXPECT = '`Agent(instrument=...)` is deprecated'

if __name__ == '__main__':
    trigger()
