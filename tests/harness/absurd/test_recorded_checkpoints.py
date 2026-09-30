"""Pins the stored checkpoint format against recorded runs, on a real Absurd PostgreSQL schema.

The fixture holds checkpoints recorded from runs of the agent built by `_agent` below (a logical
string model ID resolved by a capability, function toolsets, a capability-contributed toolset, an MCP
server, and structured output). Model payloads are trimmed to the fields a replay reads; tool payloads
are verbatim.

A recorded run must resume without re-running any checkpointed step, and a fresh run must write the
same step names and payloads, so tasks in flight across a deploy resume without repeating work.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest

pytest.importorskip('absurd_sdk')
pytest.importorskip('fastmcp')

from absurd_sdk import AsyncAbsurd, AsyncTaskContext, JsonValue
from fastmcp import FastMCP
from psycopg import AsyncConnection
from psycopg.rows import TupleRow
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, ResolveModelId
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelMessage, ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness.absurd import AbsurdDurability

from ._task import checkpoints

GOLDEN: dict[str, dict[str, JsonValue]] = json.loads(
    (Path(__file__).parent / 'fixtures' / 'recorded_checkpoints.json').read_text()
)
# Fixture key -> whether the first `report_finding` call raises `ModelRetry`.
CASES = {'plain': False, 'with_retry': True}


class Report(BaseModel):
    outcome: Literal['completed', 'partial', 'failed']
    report: str


def _agent(retry_first: bool, executions: list[str]) -> Agent[object, Report]:
    def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        executions.append('model')
        done = [p.tool_name for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
        retried = any(isinstance(p, RetryPromptPart) for m in messages for p in m.parts)
        if not done and retry_first and not retried:
            return ModelResponse(
                parts=[ToolCallPart('report_finding', {'title': 'p99 up', 'severity': 'bogus-high'}, 'c0')]
            )
        if not done:
            return ModelResponse(
                parts=[
                    ToolCallPart('report_finding', {'title': 'p99 up', 'severity': 'high'}, 'c1'),
                    ToolCallPart('query_metrics', {'query': 'errors'}, 'c2'),
                ]
            )
        if 'add' not in done:
            return ModelResponse(
                parts=[
                    ToolCallPart('add', {'a': 2, 'b': 3}, 'c3'),
                    ToolCallPart('search_knowledge', {'term': 'deploys'}, 'c4'),
                ]
            )
        return ModelResponse(
            parts=[
                TextPart('wrapping up'),
                ToolCallPart('final_result', {'outcome': 'completed', 'report': f'saw {sorted(done)}'}, 'c5'),
            ]
        )

    model = FunctionModel(script, model_name='fn')

    findings = FunctionToolset[object](id='findings')

    @findings.tool_plain
    def report_finding(title: str, severity: str) -> str:
        if severity.startswith('bogus'):
            executions.append('report_finding:retry')
            raise ModelRetry('severity must be low or high')
        executions.append('report_finding')
        return f'recorded {title} ({severity})'

    metrics = FunctionToolset[object](id='metrics')

    @metrics.tool_plain
    def query_metrics(query: str) -> dict[str, int]:
        executions.append('query_metrics')
        return {'rows': 7}

    knowledge = FunctionToolset[object](id='knowledge')

    @knowledge.tool_plain
    def search_knowledge(term: str) -> dict[str, list[str]]:
        executions.append('search_knowledge')
        return {'hits': [f'{term}-1', f'{term}-2']}

    class ProjectKnowledge(AbstractCapability[object]):
        def get_toolset(self) -> FunctionToolset[object]:
            return knowledge

    server: FastMCP[object] = FastMCP(name='calc')

    @server.tool
    def add(a: int, b: int) -> int:
        executions.append('add')
        return a + b

    return Agent(
        'custom:analyst',
        name='analyst',
        output_type=Report,
        toolsets=[findings, metrics, MCPToolset[object](server, id='calc')],
        capabilities=[
            ResolveModelId(lambda ctx, model_id: model if model_id == 'custom:analyst' else None),
            ProjectKnowledge(),
            AbsurdDurability(),
        ],
    )


EXPECTED_OUTPUT = {
    'outcome': 'completed',
    'report': "saw ['add', 'query_metrics', 'report_finding', 'search_knowledge']",
}


def _register(absurd: AsyncAbsurd, agent: Agent[object, Report]) -> None:
    @absurd.register_task(name='analyse')
    async def analyse(params: JsonValue, ctx: AsyncTaskContext) -> JsonValue:
        result = await agent.run('Investigate the latency spike.')
        return result.output.model_dump(mode='json')


@pytest.mark.parametrize('case', CASES)
class TestRecordedCheckpoints:
    async def test_resumes_a_recorded_run(
        self, case: str, absurd: AsyncAbsurd, async_conn: AsyncConnection[TupleRow], queue_name: str
    ) -> None:
        executions: list[str] = []
        _register(absurd, _agent(CASES[case], executions))
        spawned = await absurd.spawn(
            'analyse', None, max_attempts=2, retry_strategy={'kind': 'fixed', 'base_seconds': 0}
        )
        [claimed] = await absurd.claim_tasks(batch_size=1)
        for name, state in GOLDEN[case].items():
            await async_conn.execute(
                'SELECT absurd.set_task_checkpoint_state(%s, %s, %s, %s, %s)',
                (queue_name, spawned['task_id'], name, json.dumps(state), claimed['run_id']),
            )
        # The recording worker died after its last checkpoint; the next attempt resumes the task.
        await async_conn.execute(
            'SELECT absurd.fail_run(%s, %s, %s)', (queue_name, claimed['run_id'], '{"type": "crash"}')
        )
        await absurd.work_batch(batch_size=1)

        result = await absurd.fetch_task_result(spawned['task_id'])
        assert result is not None and result.state == 'completed'
        assert result.result == EXPECTED_OUTPUT
        # Only the `ModelRetry` call re-runs: it is never checkpointed.
        assert executions == (['report_finding:retry'] if CASES[case] else [])

    async def test_fresh_run_writes_the_same_checkpoints(self, case: str, absurd: AsyncAbsurd) -> None:
        _register(absurd, _agent(CASES[case], []))
        spawned = await absurd.spawn('analyse', None)
        await absurd.work_batch(batch_size=1)

        stored = await checkpoints(absurd, spawned['task_id'])
        golden = GOLDEN[case]
        assert sorted(stored) == sorted(golden)
        # Tool results are compared verbatim. Model responses and MCP tool listings also carry fields
        # (timestamps, usage, fields added in later releases) that differ run to run or release to
        # release, so those are compared on the fields a replay reads.
        for name, expected in golden.items():
            if '.get_tools' in name:
                assert _tool_schemas(stored[name]) == _tool_schemas(expected), name
            else:
                assert _project(stored[name], expected) == expected, name


def _project(value: JsonValue, like: JsonValue) -> JsonValue:
    """Keep only the parts of `value` that `like` has."""
    if isinstance(like, dict):
        assert isinstance(value, dict)
        return {key: _project(value[key], sub) for key, sub in like.items()}
    if isinstance(like, list):
        assert isinstance(value, list) and len(value) == len(like)
        return [_project(item, sub) for item, sub in zip(value, like)]
    return value


def _tool_schemas(listing: JsonValue) -> JsonValue:
    assert isinstance(listing, dict)
    keys = ('name', 'description', 'kind', 'parameters_json_schema')
    return {name: _project(tool, {key: None for key in keys}) for name, tool in listing.items()}
