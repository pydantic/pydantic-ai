"""`WrapUpAtLimits` under `DBOSDurability`: recovering a workflow wraps up at the same request over the journal."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Generator
from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.settings import ToolChoice
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness import WrapUpAtLimits
from pydantic_ai_harness.wrap_up_at_limits import DEFAULT_WRAP_UP_PROMPT
from tests.conftest import try_import

from ._messages import wrap_ups

with try_import() as imports_successful:
    from dbos import DBOS, DBOSConfig, SetWorkflowID

    from pydantic_ai.durable_exec.dbos import DBOSDurability

pytestmark = [
    pytest.mark.xdist_group(name='harness-dbos'),
    pytest.mark.skipif(not imports_successful(), reason='dbos not installed'),
]


class _Crash(BaseException):
    """Leave the first attempt pending, so resuming it replays the journaled model requests."""


@pytest.fixture
def dbos(tmp_path: Path) -> Generator[DBOS, None, None]:
    config: DBOSConfig = {
        'name': 'durable_wrap_up_at_limits',
        'system_database_url': f'sqlite:///{tmp_path / "dbos.sqlite"}',
    }
    instance = DBOS(config=config)
    DBOS.launch()
    try:
        yield instance
    finally:
        DBOS.destroy()


_model_calls: list[ToolChoice] = []


def _model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    tool_choice = info.model_settings.get('tool_choice') if info.model_settings else None
    _model_calls.append(tool_choice)
    if tool_choice == 'none':
        return ModelResponse(parts=[TextPart('what I found so far')])
    step = len(_model_calls)
    return ModelResponse(
        parts=[ToolCallPart('look_up', {'topic': f'a{step}'}), ToolCallPart('look_up', {'topic': f'b{step}'})]
    )


def test_recovery_wraps_up_at_the_same_request(dbos: DBOS) -> None:
    agent = Agent(FunctionModel(_model), name='wrap_up_dbos', capabilities=[WrapUpAtLimits(), DBOSDurability()])

    @agent.tool_plain
    def look_up(topic: str) -> str:
        return f'notes on {topic}'

    attempts: list[tuple[str, list[ModelMessage]]] = []

    @DBOS.workflow(name='wrap_up_dbos')
    def workflow() -> None:
        result = agent.run_sync('Research', usage_limits=UsageLimits(request_limit=3))
        attempts.append((result.output, result.all_messages()))
        if len(attempts) == 1:
            raise _Crash
        # Recovery runs on a DBOS executor thread, where `run_sync` sets a new event loop that nothing else closes.
        asyncio.get_event_loop().close()

    _model_calls.clear()
    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id), pytest.raises(_Crash):
        workflow()

    DBOS.resume_workflow(workflow_id).get_result()

    (first_output, first), (recovered_output, recovered) = attempts
    assert first_output == recovered_output == 'what I found so far'
    expected = [(DEFAULT_WRAP_UP_PROMPT, 'request_limit')]
    assert wrap_ups(first) == wrap_ups(recovered) == expected
    # Recovery read the model responses from the journal; the hooks ran again over them.
    assert _model_calls == [None, None, 'none']
