"""v1: GrokProvider."""
from pydantic_ai.providers.grok import GrokProvider


def trigger():
    # DEPRECATION: B5_grok_provider
    return GrokProvider(api_key='dummy')


EXPECT = '`GrokProvider` is deprecated'

if __name__ == '__main__':
    trigger()
