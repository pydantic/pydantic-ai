"""v2 form: XaiProvider + XaiModel."""
from pydantic_ai.providers.xai import XaiProvider
from pydantic_ai.models.xai import XaiModel


def trigger():
    return XaiModel('grok-4', provider=XaiProvider(api_key='dummy'))


if __name__ == '__main__':
    trigger()
