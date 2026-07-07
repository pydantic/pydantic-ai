"""An ACP agent advertising switchable models, used by the stdio model-config test."""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.experimental.acp import run_acp_stdio_sync

if __name__ == '__main__':
    run_acp_stdio_sync(Agent(TestModel(custom_output_text='hi')), models=['test', 'openai:gpt-4o'])
