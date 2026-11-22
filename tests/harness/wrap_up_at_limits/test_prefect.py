"""`WrapUpAtLimits` under `PrefectDurability`: the time trigger runs in flow code."""

from __future__ import annotations

from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.wrap_up_at_limits import DEFAULT_WRAP_UP_PROMPT, WrapUpAtLimits
from tests.conftest import try_import

from ._messages import wrap_ups

with try_import() as imports_successful:
    from prefect import flow
    from prefect.settings import PREFECT_LOCAL_STORAGE_PATH, temporary_settings

    from pydantic_ai.durable_exec.prefect import PrefectDurability

pytestmark = [
    pytest.mark.xdist_group(name='prefect'),
    pytest.mark.skipif(not imports_successful(), reason='prefect not installed'),
]


def _model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if info.model_settings and info.model_settings.get('tool_choice') == 'none':
        return ModelResponse(parts=[TextPart('what I found so far')])
    return ModelResponse(parts=[ToolCallPart('look_up', {'topic': 'a'})])  # pragma: no cover


async def test_flow_wraps_up_for_timeout(prefect_test_server: None, tmp_path: Path) -> None:
    # A 60s reserve on a 30s timeout makes the first request the last, whatever the steps take.
    agent = Agent(
        FunctionModel(_model),
        name='wrap_up_prefect',
        capabilities=[WrapUpAtLimits(timeout_margin=60), PrefectDurability()],
    )

    @agent.tool_plain
    def look_up(topic: str) -> str:  # pragma: no cover
        return f'notes on {topic}'

    @flow
    async def run() -> list[ModelMessage]:
        return (await agent.run('Research', timeout=30)).all_messages()

    with temporary_settings({PREFECT_LOCAL_STORAGE_PATH: tmp_path}):
        messages = await run()

    assert wrap_ups(messages) == [(DEFAULT_WRAP_UP_PROMPT, 'timeout')]
    assert isinstance(messages[-1], ModelResponse)
    assert messages[-1].text == 'what I found so far'
