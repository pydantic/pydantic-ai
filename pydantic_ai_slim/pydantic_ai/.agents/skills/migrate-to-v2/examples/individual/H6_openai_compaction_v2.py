"""v2: OpenAICompaction no longer accepts instructions=. Drop the kwarg."""
from pydantic_ai.models.openai import OpenAICompaction


def trigger():
    OpenAICompaction()


if __name__ == '__main__':
    trigger()
