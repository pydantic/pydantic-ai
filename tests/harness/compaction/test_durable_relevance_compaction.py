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
from pydantic_ai_harness.compaction import RelevanceCompaction, SummarizingCompaction

pytestmark = [pytest.mark.xdist_group(name='harness-dbos')]


@pytest.fixture
def dbos(tmp_path: Path) -> Generator[DBOS, None, None]:
    config: DBOSConfig = {
        'name': 'durable_relevance_compaction',
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
    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart('do the task')])]
    for index, text in enumerate(['KEEP a ', 'GIST b ', 'last c ']):
        messages.append(ModelResponse(parts=[ToolCallPart('read', {'n': index}, tool_call_id=f'c{index}')]))
        messages.append(ModelRequest(parts=[ToolReturnPart('read', text * 400, tool_call_id=f'c{index}')]))
    messages.append(ModelResponse(parts=[TextPart('done')]))
    return messages


_calls = {'judge': 0, 'summary': 0}


def _judge(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    _calls['judge'] += 1
    request = messages[-1]
    assert isinstance(request, ModelRequest)
    needed = 0.9 if 'KEEP' in str(request.parts[-1]) else 0.1
    return ModelResponse(
        parts=[ToolCallPart(info.output_tools[0].name, {'needed_verbatim': needed, 'still_matters': 0.5})]
    )


def _summary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    _calls['summary'] += 1
    return ModelResponse(parts=[TextPart('nested summary')])


def _respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('ok')])


_agent: Agent[None, str] = Agent(
    FunctionModel(_respond),
    name='durable_relevance',
    deps_type=type(None),
    capabilities=[
        RelevanceCompaction(
            FunctionModel(_judge),
            summarizer=SummarizingCompaction(model=FunctionModel(_summary), max_messages=1),
            max_tokens=1_500,
            target_tokens=3_000,
            keep_tokens=10,
        ),
        # A registered summarizer under the nested one's default id, which must not take its summary over.
        SummarizingCompaction(
            model=FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('registered summary')])),
            max_messages=1_000,
        ),
        DBOSDurability[None](),
    ],
)


@DBOS.workflow(name='durable_relevance')
async def _workflow() -> list[str]:
    result = await _agent.run('continue', message_history=_history())
    return [
        part.content
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, CompactionPart) and part.content is not None
    ]


async def test_dbos_replays_the_recorded_judgements_and_summary(dbos: DBOS) -> None:
    _calls.update(judge=0, summary=0)
    workflow_id = str(uuid.uuid4())

    with SetWorkflowID(workflow_id):
        assert await _workflow() == ['nested summary']
    with SetWorkflowID(workflow_id):
        assert await _workflow() == ['nested summary']

    assert _calls == {'judge': 2, 'summary': 1}
    steps = {step['function_name'] for step in await dbos.list_workflow_steps_async(workflow_id)}
    assert 'durable_relevance__capability__relevance_compaction.judge' in steps
    assert 'durable_relevance__capability__relevance_compaction.summarize' in steps
