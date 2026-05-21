"""v2 form: OpenAIChatModel."""
from pydantic_ai.models.openai import OpenAIChatModel


def trigger():
    return OpenAIChatModel('gpt-4o')


if __name__ == '__main__':
    trigger()
