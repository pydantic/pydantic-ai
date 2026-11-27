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
from pydantic_ai.profiles import ModelProfile
from pydantic_ai_harness.compaction import SummarizingCompaction

try:
    from logfire.testing import CaptureLogfire

    logfire_installed = True
except ImportError:  # pragma: no cover
    logfire_installed = False

pytestmark = [pytest.mark.xdist_group(name='harness-dbos')]


@pytest.fixture
def dbos(tmp_path: Path) -> Generator[DBOS, None, None]:
    config: DBOSConfig = {
        'name': 'durable_summarizing_compaction',
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


_summary_calls = 0


async def _respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    del info
    global _summary_calls
    if any(
        isinstance(part, UserPromptPart) and isinstance(part.content, str) and '<messages>' in part.content
        for message in messages
        for part in message.parts
    ):
        _summary_calls += 1
        return ModelResponse(parts=[TextPart(f'summary {_summary_calls}')])
    return ModelResponse(parts=[TextPart('done')])


_compaction: SummarizingCompaction[None] = SummarizingCompaction(
    max_messages=1, keep_messages=1, preserve_first_user_message=False
)
_agent: Agent[None, str] = Agent(
    FunctionModel(_respond),
    name='durable_summary',
    deps_type=type(None),
    capabilities=[_compaction, DBOSDurability[None]()],
)


@DBOS.workflow(name='durable_summary')
async def _workflow() -> tuple[str, int]:
    result = await _agent.run('continue', message_history=_history())
    first = result.all_messages()[0]
    assert isinstance(first, ModelRequest)
    [summary_part] = first.parts
    assert isinstance(summary_part, CompactionPart) and summary_part.content is not None
    return summary_part.content, result.usage.requests


_custom_compaction: SummarizingCompaction[None] = SummarizingCompaction(
    id='custom_summary', max_messages=1, keep_messages=1, preserve_first_user_message=False
)
_custom_agent: Agent[None, str] = Agent(
    FunctionModel(_respond),
    name='durable_custom_summary',
    deps_type=type(None),
    capabilities=[_custom_compaction, DBOSDurability[None]()],
)


@DBOS.workflow(name='durable_custom_summary')
async def _custom_workflow() -> str:
    return (await _custom_agent.run('continue', message_history=_history())).output


def test_default_id_is_stable() -> None:
    assert _compaction.id == 'summarizing_compaction'


async def test_dbos_replays_the_recorded_summary(dbos: DBOS) -> None:
    global _summary_calls
    _summary_calls = 0
    workflow_id = str(uuid.uuid4())

    with SetWorkflowID(workflow_id):
        assert await _workflow() == ('summary 1', 2)
    with SetWorkflowID(workflow_id):
        assert await _workflow() == ('summary 1', 2)

    assert _summary_calls == 1
    steps = await dbos.list_workflow_steps_async(workflow_id)
    assert 'durable_summary__capability__summarizing_compaction.summarize' in {step['function_name'] for step in steps}


async def test_dbos_uses_custom_id_for_durable_summary(dbos: DBOS) -> None:
    global _summary_calls
    _summary_calls = 0
    workflow_id = str(uuid.uuid4())

    with SetWorkflowID(workflow_id):
        assert await _custom_workflow() == 'done'

    assert _summary_calls == 1
    steps = await dbos.list_workflow_steps_async(workflow_id)
    assert 'durable_custom_summary__capability__custom_summary.summarize' in {step['function_name'] for step in steps}


def _long_history() -> list[ModelMessage]:
    """Twelve turns, too many for one request to `_chunking_agent`'s 2,000-token summarizer window."""
    history: list[ModelMessage] = []
    for index in range(12):
        history += [
            ModelRequest(parts=[UserPromptPart(f'question {index}')]),
            ModelResponse(parts=[TextPart(f'answer {index} ' + 'x' * 1_200)]),
        ]
    return history


_chunking_agent: Agent[None, str] = Agent(
    FunctionModel(_respond, profile=ModelProfile(context_window=2_000)),
    name='durable_chunked_summary',
    deps_type=type(None),
    capabilities=[
        SummarizingCompaction(max_messages=1, keep_messages=1, preserve_first_user_message=False),
        DBOSDurability[None](),
    ],
)


@DBOS.workflow(name='durable_chunked_summary')
async def _chunking_workflow() -> str:
    return (await _chunking_agent.run('continue', message_history=_long_history())).output


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.usefixtures('instrument_all_agents')
async def test_dbos_records_the_summary_calls_on_the_span_on_replay(dbos: DBOS, capfire: CaptureLogfire) -> None:
    """The call count comes back with the recorded summary, so a replay that skips the requests still records it."""
    global _summary_calls
    _summary_calls = 0
    workflow_id = str(uuid.uuid4())

    with SetWorkflowID(workflow_id):
        assert await _chunking_workflow() == 'done'
    # A fork from the step after the summary reuses the recorded summary without running it again.
    steps = await dbos.list_workflow_steps_async(workflow_id)
    [summarize] = [step['function_id'] for step in steps if step['function_name'].endswith('.summarize')]
    fork = await dbos.fork_workflow_async(workflow_id, summarize + 1)
    assert await fork.get_result() == 'done'

    assert _summary_calls == 4
    spans = [span for span in capfire.exporter.exported_spans_as_dict() if span['name'] == 'compact_messages']
    assert [span['attributes'].get('compaction.summary_calls') for span in spans] == [4, 4]
