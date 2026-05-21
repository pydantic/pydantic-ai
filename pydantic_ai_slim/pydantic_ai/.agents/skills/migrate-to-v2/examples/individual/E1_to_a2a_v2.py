"""v2 form: fasta2a.pydantic_ai.agent_to_a2a (external package)."""
from pydantic_ai import Agent
from fasta2a.pydantic_ai import agent_to_a2a


def trigger():
    a = Agent('openai-chat:gpt-4o')
    return agent_to_a2a(a)


if __name__ == '__main__':
    trigger()
