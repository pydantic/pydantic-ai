"""v1: OpenAIModel — renamed to OpenAIChatModel."""
from pydantic_ai.models.openai import OpenAIModel


def trigger():
    # DEPRECATION: B1_openai_model
    return OpenAIModel('gpt-4o')


EXPECT = '`OpenAIModel` was renamed to `OpenAIChatModel`'

if __name__ == '__main__':
    trigger()
