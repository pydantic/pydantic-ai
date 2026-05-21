"""v2 form: explicit provider prefix; use openai-chat: to keep v1 chat semantics."""
from pydantic_ai import Agent


def trigger():
    return Agent('openai-chat:gpt-4o')


if __name__ == '__main__':
    trigger()
