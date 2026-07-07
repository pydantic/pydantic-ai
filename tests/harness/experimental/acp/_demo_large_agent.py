"""An ACP agent that streams a large message, used by the large-payload stdio test."""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.experimental.acp import run_acp_stdio_sync

LARGE_OUTPUT = 'x' * 200_000

if __name__ == '__main__':
    run_acp_stdio_sync(Agent(TestModel(custom_output_text=LARGE_OUTPUT)))
