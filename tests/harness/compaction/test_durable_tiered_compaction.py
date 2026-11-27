"""Regression tests for #14743: nested summarizers must bind their durable operation.

A `SummarizingCompaction` nested as a `TieredCompaction` tier must have its `summarize`
operation bound by `DBOSDurability`; direct attachment must keep working unchanged.
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from pathlib import Path

import pytest

try:
    from dbos import DBOS, DBOSConfig, SetWorkflowID

    from pydantic_ai.durable_exec.dbos import DBOSDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('dbos not installed', allow_module_level=True)

from pydantic_ai import Agent
from pydantic_ai.messages import CompactionPart, ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.compaction import SummarizingCompaction, TieredCompaction

pytestmark = [pytest.mark.xdist_group(name='harness-dbos')]


@pytest.fixture
def dbos(tmp_path: Path) -> Generator[DBOS, None, None]:
    config: DBOSConfig = {
        'name': 'durable_tiered_compaction',
        'system_database_url': f'sqlite:///{tmp_path / "dbos.sqlite"}',
    }
    instance = DBOS(config=config)
    DBOS.launch()
    try:
        yield instance
    finally:
        DBOS.destroy()


def _history() -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart('first')]),
        ModelResponse(parts=[TextPart('response')]),
        ModelRequest(parts=[UserPromptPart('second')]),
        ModelResponse(parts=[TextPart('response')]),
    ]


_nested_summary_calls = 0


async def _respond_nested(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    del info
    global _nested_summary_calls
    if any(
        isinstance(part, UserPromptPart) and isinstance(part.content, str) and '<messages>' in part.content
        for message in messages
        for part in message.parts
    ):
        _nested_summary_calls += 1
        return ModelResponse(parts=[TextPart(f'summary {_nested_summary_calls}')])
    return ModelResponse(parts=[TextPart('done')])


_nested_compaction: TieredCompaction[None] = TieredCompaction(
    tiers=[SummarizingCompaction(max_messages=1, keep_messages=1, preserve_first_user_message=False)],
    target_tokens=1,
)
_nested_agent: Agent[None, str] = Agent(
    FunctionModel(_respond_nested),
    name='durable_nested_summary',
    deps_type=type(None),
    capabilities=[_nested_compaction, DBOSDurability[None]()],
)


@DBOS.workflow(name='durable_nested_summary')
async def _nested_workflow() -> tuple[str, int]:
    result = await _nested_agent.run('continue', message_history=_history())
    first = result.all_messages()[0]
    assert isinstance(first, ModelRequest)
    [summary_part] = first.parts
    assert isinstance(summary_part, CompactionPart) and summary_part.content is not None
    return summary_part.content, result.usage.requests


_direct_summary_calls = 0


async def _respond_direct(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    del info
    global _direct_summary_calls
    if any(
        isinstance(part, UserPromptPart) and isinstance(part.content, str) and '<messages>' in part.content
        for message in messages
        for part in message.parts
    ):
        _direct_summary_calls += 1
        return ModelResponse(parts=[TextPart(f'summary {_direct_summary_calls}')])
    return ModelResponse(parts=[TextPart('done')])


_direct_compaction: SummarizingCompaction[None] = SummarizingCompaction(
    max_messages=1, keep_messages=1, preserve_first_user_message=False
)
_direct_agent: Agent[None, str] = Agent(
    FunctionModel(_respond_direct),
    name='durable_direct_summary',
    deps_type=type(None),
    capabilities=[_direct_compaction, DBOSDurability[None]()],
)


@DBOS.workflow(name='durable_direct_summary')
async def _direct_workflow() -> tuple[str, int]:
    result = await _direct_agent.run('continue', message_history=_history())
    first = result.all_messages()[0]
    assert isinstance(first, ModelRequest)
    [summary_part] = first.parts
    assert isinstance(summary_part, CompactionPart) and summary_part.content is not None
    return summary_part.content, result.usage.requests


async def test_nested_summarizing_tier_schedules_summarize_activity(dbos: DBOS) -> None:
    global _nested_summary_calls
    _nested_summary_calls = 0
    workflow_id = str(uuid.uuid4())

    with SetWorkflowID(workflow_id):
        assert await _nested_workflow() == ('summary 1', 2)
    with SetWorkflowID(workflow_id):
        assert await _nested_workflow() == ('summary 1', 2)

    assert _nested_summary_calls == 1
    steps = await dbos.list_workflow_steps_async(workflow_id)
    assert 'durable_nested_summary__capability__summarizing_compaction.summarize' in {
        step['function_name'] for step in steps
    }


async def test_direct_summarizer_binding_unchanged(dbos: DBOS) -> None:
    global _direct_summary_calls
    _direct_summary_calls = 0
    workflow_id = str(uuid.uuid4())

    with SetWorkflowID(workflow_id):
        assert await _direct_workflow() == ('summary 1', 2)
    with SetWorkflowID(workflow_id):
        assert await _direct_workflow() == ('summary 1', 2)

    assert _direct_summary_calls == 1
    steps = await dbos.list_workflow_steps_async(workflow_id)
    assert 'durable_direct_summary__capability__summarizing_compaction.summarize' in {
        step['function_name'] for step in steps
    }
