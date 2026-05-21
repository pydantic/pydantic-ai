"""v1: OpenAICompaction(instructions=...) emits a DeprecationWarning."""
from pydantic_ai.models.openai import OpenAICompaction


def trigger():
    # DEPRECATION: H6_openai_compaction
    OpenAICompaction(instructions='Summarize tightly.')


EXPECT = '`OpenAICompaction(instructions=...)` is deprecated'

if __name__ == '__main__':
    trigger()
