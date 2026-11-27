"""Regression tests for #14743: a summarizer nested in a composing strategy is a durable operation.

A strategy nested as a `TieredCompaction` tier or in a `FallbackCompaction` chain is not registered
on the agent, so its own durable operations don't bind; the composing capability runs it inside its
`compact` operation instead. A direct summarizer keeps its own `summarize` operation, and a
composition without durable operations adds no step.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Coroutine, Generator
from pathlib import Path
from typing import Any

import pytest

try:
    from dbos import DBOS, DBOSConfig, SetWorkflowID

    from pydantic_ai.durable_exec.dbos import DBOSDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('dbos not installed', allow_module_level=True)

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import CompactionPart, ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.compaction import (
    ClearToolResults,
    FallbackCompaction,
    SummarizingCompaction,
    TieredCompaction,
)

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


_summary_calls: list[str] = []


def _summarizer(label: str, **kwargs: Any) -> SummarizingCompaction[None]:
    def summarize(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        _summary_calls.append(label)
        return ModelResponse(parts=[TextPart(f'{label} summary')])

    settings: dict[str, Any] = {'max_messages': 1, 'keep_messages': 1, 'preserve_first_user_message': False, **kwargs}
    return SummarizingCompaction(model=FunctionModel(summarize), **settings)


def _respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('done')])


def _agent(name: str, *capabilities: AbstractCapability[None]) -> Agent[None, str]:
    return Agent(
        FunctionModel(_respond),
        name=name,
        deps_type=type(None),
        capabilities=[*capabilities, DBOSDurability[None]()],
    )


def _workflow(name: str, agent: Agent[None, str]) -> Callable[[], Coroutine[Any, Any, list[str]]]:
    @DBOS.workflow(name=name)
    async def run() -> list[str]:
        result = await agent.run('continue', message_history=_history())
        return [
            part.content
            for message in result.all_messages()
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, CompactionPart) and part.content is not None
        ]

    return run


_tiered = _workflow(
    'nested_tiered',
    _agent('nested_tiered', TieredCompaction(tiers=[ClearToolResults(), _summarizer('tier')], target_tokens=1)),
)
_fallback = _workflow(
    'nested_fallback',
    _agent('nested_fallback', FallbackCompaction([_summarizer('chain')], max_tokens=1)),
)
_tiered_fallback = _workflow(
    'nested_tiered_fallback',
    _agent(
        'nested_tiered_fallback',
        TieredCompaction(tiers=[FallbackCompaction([_summarizer('inner')])], target_tokens=1),
    ),
)
_two_tiers = _workflow(
    'nested_two_tiers',
    _agent(
        'nested_two_tiers',
        TieredCompaction(
            tiers=[_summarizer('cheap', keep_messages=3), _summarizer('strong')],
            target_tokens=1,
        ),
    ),
)
_direct = _workflow(
    'nested_tier_beside_direct',
    _agent(
        'nested_tier_beside_direct',
        TieredCompaction(tiers=[_summarizer('tier')], target_tokens=1),
        _summarizer('direct', max_messages=1000),
    ),
)
_direct = _workflow('direct_summary', _agent('direct_summary', _summarizer('direct')))
_cheap_tiers_only = _workflow(
    'cheap_tiers_only',
    _agent('cheap_tiers_only', TieredCompaction(tiers=[ClearToolResults()], target_tokens=1)),
)


async def _run_twice(dbos: DBOS, workflow: Callable[[], Coroutine[Any, Any, list[str]]]) -> tuple[list[str], set[str]]:
    _summary_calls.clear()
    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id):
        first = await workflow()
    with SetWorkflowID(workflow_id):
        assert await workflow() == first
    steps = {step['function_name'] for step in await dbos.list_workflow_steps_async(workflow_id)}
    return first, {step.split('__capability__')[1] for step in steps if '__capability__' in step}


async def test_tiered_runs_a_nested_summarizer_as_its_compact_step(dbos: DBOS) -> None:
    summaries, operations = await _run_twice(dbos, _tiered)

    assert summaries == ['tier summary']
    assert _summary_calls == ['tier']
    assert operations == {'tiered_compaction.compact'}


async def test_fallback_runs_a_nested_summarizer_as_its_compact_step(dbos: DBOS) -> None:
    summaries, operations = await _run_twice(dbos, _fallback)

    assert summaries == ['chain summary']
    assert _summary_calls == ['chain']
    assert operations == {'fallback_compaction.compact'}


async def test_fallback_nested_in_tiered_runs_inside_the_tiered_step(dbos: DBOS) -> None:
    summaries, operations = await _run_twice(dbos, _tiered_fallback)

    assert summaries == ['inner summary']
    assert _summary_calls == ['inner']
    assert operations == {'tiered_compaction.compact'}


async def test_two_summarizer_tiers_keep_escalating(dbos: DBOS) -> None:
    """The cheap tier keeps more messages, so the target of 1 token escalates to the strong tier on its result."""
    summaries, operations = await _run_twice(dbos, _two_tiers)

    assert summaries == ['strong summary']
    assert _summary_calls == ['cheap', 'strong']
    assert operations == {'tiered_compaction.compact'}


async def test_direct_summarizer_keeps_its_summarize_step(dbos: DBOS) -> None:
    summaries, operations = await _run_twice(dbos, _direct)

    assert summaries == ['direct summary']
    assert _summary_calls == ['direct']
    assert operations == {'summarizing_compaction.summarize'}


async def test_tiers_without_durable_operations_add_no_step(dbos: DBOS) -> None:
    summaries, operations = await _run_twice(dbos, _cheap_tiers_only)

    assert summaries == []
    assert operations == set()
