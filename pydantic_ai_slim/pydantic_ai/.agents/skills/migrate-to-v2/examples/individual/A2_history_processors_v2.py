"""v2 form: ProcessHistory capability (one per processor)."""
from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessHistory


def _proc(messages):
    return messages


def trigger():
    return Agent('openai-chat:gpt-4o', capabilities=[ProcessHistory(_proc)])


if __name__ == '__main__':
    trigger()
