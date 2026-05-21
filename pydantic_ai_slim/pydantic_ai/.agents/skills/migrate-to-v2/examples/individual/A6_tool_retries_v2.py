"""v2 form: retries={'tools': N, 'output': M} dict."""
from pydantic_ai import Agent


def trigger():
    return Agent('openai-chat:gpt-4o', retries={'tools': 3, 'output': 2})


if __name__ == '__main__':
    trigger()
