"""v1: history_processors= kwarg."""
from pydantic_ai import Agent


def _proc(messages):
    return messages


def trigger():
    # DEPRECATION: A2_history_processors
    return Agent('openai-chat:gpt-4o', history_processors=[_proc])


EXPECT = '`Agent(history_processors=[fn, ...])` is deprecated'

if __name__ == '__main__':
    trigger()
