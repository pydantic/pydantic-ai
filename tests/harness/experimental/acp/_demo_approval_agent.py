"""An ACP agent with an approval-gated tool, used by the end-to-end permission test."""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.experimental.acp import run_acp_stdio_sync


def build_agent() -> Agent[None, str]:
    agent = Agent(TestModel())

    @agent.tool_plain(requires_approval=True)
    def delete_file(path: str) -> str:
        return f'deleted {path}'

    return agent


if __name__ == '__main__':
    run_acp_stdio_sync(build_agent())
