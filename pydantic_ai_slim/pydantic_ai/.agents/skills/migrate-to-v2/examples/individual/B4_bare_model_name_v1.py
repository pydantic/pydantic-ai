"""v1: bare model name without provider prefix."""
from pydantic_ai import Agent


def trigger():
    # DEPRECATION: B4_bare_model_name
    return Agent('gpt-4o')


EXPECT = "Specifying a model name without a provider prefix is deprecated"

if __name__ == '__main__':
    trigger()
