"""Checkpoint compatibility with `pydantic-ai-absurd` 0.8.0, against a real Absurd PostgreSQL schema.

The fixture holds checkpoints recorded by real `pydantic-ai-absurd` 0.8.0 runs of the agent built by
`_agent` below (shaped like a production workflow agent: a logical string model ID resolved by a
capability, id'd function toolsets, a capability-contributed toolset, an MCP server, and structured
output). Model payloads are trimmed to the fields a replay reads; tool payloads are verbatim.

A run started under `pydantic-ai-absurd` must resume under `AbsurdDurability` without re-running any
checkpointed step, and a fresh run must write the same step names and payloads, so tasks in flight
when a worker fleet switches packages resume without repeating work.
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
    (Path(__file__).parent / 'fixtures' / 'pydantic_ai_absurd_0.8.0_checkpoints.json').read_text()
)
# Fixture key -> whether the first `report_finding` call raises `ModelRetry`. The retry case was recorded
# with an id-less MCP server; its two MCP step names were renamed by hand to the id'd form.
CASES = {'id_mcp': False, 'id_mcp_with_retry': True}


class WorkflowOutput(BaseModel):
    """Structured output of the recorded workflow agent."""

    outcome: Literal['completed', 'partial', 'failed']
    report: str


def _agent(retry_first: bool, executions: list[str]) -> Agent[object, WorkflowOutput]:
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
                    ToolCallPart('posthog_query', {'query': 'errors'}, 'c2'),
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

    model = FunctionModel(script, model_name='minimax-m3')

    findings = FunctionToolset[object](id='findings')

    @findings.tool_plain
    def report_finding(title: str, severity: str) -> str:
        if severity.startswith('bogus'):
            executions.append('report_finding:retry')
            raise ModelRetry('severity must be low or high')
        executions.append('report_finding')
        return f'recorded {title} ({severity})'

    posthog = FunctionToolset[object](id='posthog')

    @posthog.tool_plain
    def posthog_query(query: str) -> dict[str, int]:
        executions.append('posthog_query')
        return {'rows': 7}

    knowledge = FunctionToolset[object](id='knowledge')

    @knowledge.tool_plain
    def search_knowledge(term: str) -> dict[str, list[str]]:
        executions.append('search_knowledge')
        return {'hits': [f'{term}-1', f'{term}-2']}

    class ProjectKnowledge(AbstractCapability[object]):
        def get_toolset(self) -> FunctionToolset[object]:
            return knowledge

    server: FastMCP[object] = FastMCP(name='logfire-mcp')

    @server.tool
    def add(a: int, b: int) -> int:
        executions.append('add')
        return a + b

    return Agent(
        'logfire:sre',
        name='sherlockberto',
        output_type=WorkflowOutput,
        toolsets=[findings, posthog, MCPToolset[object](server, id='logfire_mcp')],
        capabilities=[
            ResolveModelId(lambda ctx, model_id: model if model_id == 'logfire:sre' else None),
            ProjectKnowledge(),
            AbsurdDurability(),
        ],
    )


EXPECTED_OUTPUT = {
    'outcome': 'completed',
    'report': "saw ['add', 'posthog_query', 'report_finding', 'search_knowledge']",
}


def _register(absurd: AsyncAbsurd, agent: Agent[object, WorkflowOutput]) -> None:
    @absurd.register_task(name='workflow')
    async def workflow(params: JsonValue, ctx: AsyncTaskContext) -> JsonValue:
        result = await agent.run('Run the SRE investigation now.')
        return result.output.model_dump(mode='json')


@pytest.mark.parametrize('case', CASES)
class TestPydanticAiAbsurdCheckpoints:
    """Checkpoints recorded by `pydantic-ai-absurd` 0.8.0 match the ones written here."""

    async def test_resumes_a_run_recorded_by_pydantic_ai_absurd(
        self, case: str, absurd: AsyncAbsurd, async_conn: AsyncConnection[TupleRow], queue_name: str
    ) -> None:
        executions: list[str] = []
        _register(absurd, _agent(CASES[case], executions))
        spawned = await absurd.spawn(
            'workflow', None, max_attempts=2, retry_strategy={'kind': 'fixed', 'base_seconds': 0}
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
        # Only the `ModelRetry` call re-runs: `pydantic-ai-absurd` never checkpointed it either.
        assert executions == (['report_finding:retry'] if CASES[case] else [])

    async def test_fresh_run_writes_the_same_checkpoints(self, case: str, absurd: AsyncAbsurd) -> None:
        _register(absurd, _agent(CASES[case], []))
        spawned = await absurd.spawn('workflow', None)
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
