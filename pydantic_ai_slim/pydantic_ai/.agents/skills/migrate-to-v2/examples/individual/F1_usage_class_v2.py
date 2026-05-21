"""v2 form: RunUsage (aggregate) and RequestUsage (per-call)."""
from pydantic_ai.usage import RunUsage


def trigger():
    return RunUsage()


if __name__ == '__main__':
    trigger()
