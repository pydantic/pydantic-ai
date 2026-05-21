"""v2 form: Instrumentation capability."""
from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation


def trigger():
    return Agent('openai-chat:gpt-4o', capabilities=[Instrumentation()])


if __name__ == '__main__':
    trigger()
