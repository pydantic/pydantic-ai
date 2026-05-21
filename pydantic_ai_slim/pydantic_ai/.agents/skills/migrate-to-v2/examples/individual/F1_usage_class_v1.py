"""v1: Usage class."""
from pydantic_ai.usage import Usage


def trigger():
    # DEPRECATION: F1_usage_class
    return Usage()


EXPECT = '`Usage` is deprecated, use `RunUsage` instead'

if __name__ == '__main__':
    trigger()
