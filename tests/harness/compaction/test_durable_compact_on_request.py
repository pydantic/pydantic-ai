from __future__ import annotations

import uuid
from collections.abc import Generator
from pathlib import Path

import pytest

from tests.conftest import detach_dbos_logging

try:
    from dbos import DBOS, DBOSConfig, SetWorkflowID

    from pydantic_ai.durable_exec.dbos import DBOSDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('dbos not installed', allow_module_level=True)

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
from pydantic_ai_harness.compaction import CompactOnRequest, SummarizingCompaction

pytestmark = [pytest.mark.xdist_group(name='harness-dbos')]


@pytest.fixture
def dbos(tmp_path: Path) -> Generator[DBOS, None, None]:
    config: DBOSConfig = {
        'name': 'durable_compact_on_request',
        'system_database_url': f'sqlite:///{tmp_path / "dbos.sqlite"}',
        'run_admin_server': False,
    }
    instance = DBOS(config=config)
    DBOS.launch()
    try:
        yield instance
    finally:
        DBOS.destroy()
        detach_dbos_logging()


def _history() -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart('explore')]),
        ModelResponse(parts=[TextPart('explored')]),
    ]


_summaries: list[str] = []


def _summary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    _summaries.append(str(messages[-1].parts[-1]))
    return ModelResponse(parts=[TextPart('requested summary')])


def _respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
        return ModelResponse(parts=[TextPart('done')])
    return ModelResponse(parts=[ToolCallPart('compact_context', {'focus': 'the plan'}, tool_call_id='c1')])


_agent: Agent[None, str] = Agent(
    FunctionModel(_respond),
    name='durable_compact_on_request',
    deps_type=type(None),
    capabilities=[
        CompactOnRequest(
            SummarizingCompaction(
                model=FunctionModel(_summary), max_messages=1, keep_messages=2, preserve_first_user_message=False
            )
        ),
        DBOSDurability[None](),
    ],
)


@DBOS.workflow(name='durable_compact_on_request')
async def _workflow() -> list[str]:
    result = await _agent.run('continue', message_history=_history())
    return [
        part.content
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, CompactionPart) and part.content is not None
    ]


async def test_dbos_replays_the_requested_compaction(dbos: DBOS) -> None:
    _summaries.clear()
    workflow_id = str(uuid.uuid4())

    with SetWorkflowID(workflow_id):
        assert await _workflow() == ['requested summary']
    with SetWorkflowID(workflow_id):
        assert await _workflow() == ['requested summary']

    assert len(_summaries) == 1
    assert 'the plan' in _summaries[0]
    steps = {step['function_name'] for step in await dbos.list_workflow_steps_async(workflow_id)}
    assert 'durable_compact_on_request__capability__compact_on_request.compact' in steps
