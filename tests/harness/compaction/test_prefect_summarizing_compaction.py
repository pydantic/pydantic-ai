"""`SummarizingCompaction` across a Prefect flow retry: the summary replays exactly, so nothing after it runs again."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import (
    CompactionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.compaction import SummarizingCompaction
from tests.conftest import try_import

with try_import() as imports_successful:
    from prefect import flow
    from prefect.settings import PREFECT_LOCAL_STORAGE_PATH, temporary_settings

    from pydantic_ai.durable_exec.prefect import PrefectDurability

pytestmark = [
    pytest.mark.xdist_group(name='prefect'),
    pytest.mark.skipif(not imports_successful(), reason='prefect not installed'),
]


def _history() -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart('first')]),
        ModelResponse(parts=[TextPart('response')]),
        ModelRequest(parts=[UserPromptPart('second')]),
        ModelResponse(parts=[TextPart('response')]),
    ]


async def test_flow_retry_replays_the_summary_and_runs_nothing_again(prefect_test_server: None, tmp_path: Path) -> None:
    """The summary's metadata, written time included, is the same on the retry, so the model requests and the tool
    keyed on the history after it replay rather than run again."""
    summaries = 0
    requests = 0
    tool_runs: list[str] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal summaries, requests
        if any(
            isinstance(part, UserPromptPart) and isinstance(part.content, str) and '<messages>' in part.content
            for message in messages
            for part in message.parts
        ):
            summaries += 1
            return ModelResponse(parts=[TextPart(f'summary {summaries}')])
        requests += 1
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart('record_side_effect', tool_call_id='effect')])

    async def record_side_effect() -> str:
        """Stands in for a charge, an email, or any other non-idempotent effect."""
        tool_runs.append('record_side_effect')
        return 'ok'

    agent = Agent(
        FunctionModel(model),
        name='summary_retry_agent',
        tools=[record_side_effect],
        capabilities=[
            SummarizingCompaction(max_messages=1, keep_messages=1, preserve_first_user_message=False),
            PrefectDurability(),
        ],
    )

    metadata: list[list[dict[str, Any] | None]] = []
    counts: list[tuple[int, int, int]] = []

    @flow(retries=1, retry_delay_seconds=0)
    async def flaky() -> str:
        result = await agent.run('go', message_history=_history())
        metadata.append(
            [
                part.metadata
                for message in result.all_messages()
                for part in message.parts
                if isinstance(part, CompactionPart)
            ]
        )
        counts.append((summaries, requests, len(tool_runs)))
        # Fail after the agent run, so the retry has every task's result to replay.
        if len(metadata) == 1:
            raise RuntimeError('boom')
        return result.output

    with temporary_settings({PREFECT_LOCAL_STORAGE_PATH: tmp_path}):
        assert await flaky() == 'done'

    first, retried = metadata
    assert first and all(first) and retried == first
    assert counts[1] == counts[0]
    assert tool_runs == ['record_side_effect']
