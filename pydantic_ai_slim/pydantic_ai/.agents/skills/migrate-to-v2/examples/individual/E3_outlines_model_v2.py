"""v2: pydantic_ai.models.outlines is removed. No drop-in replacement.

Users on Outlines should either drop the integration or track
https://github.com/dottxt-ai/outlines/issues for a community-maintained adapter.
This file documents the absence: importing the v1 path now raises ImportError.
"""


def trigger():
    try:
        import pydantic_ai.models.outlines  # noqa: F401
    except ImportError:
        # Expected: module removed in v2.
        pass


if __name__ == '__main__':
    trigger()
