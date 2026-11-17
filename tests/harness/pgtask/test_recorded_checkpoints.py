"""Pins the stored checkpoint format against recorded runs, on a real pgtask schema.

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

pytest.importorskip('pgtask')
pytest.importorskip('fastmcp')

from fastmcp import FastMCP
from pgtask import JSONValue
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, ResolveModelId
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness.pgtask import PGTaskDurability

from ._task import PGTaskDatabase, StepKey, checkpoints, run_in_task

# Case -> the recorded checkpoints, as `[step name, occurrence, value]` rows.
GOLDEN: dict[str, list[tuple[str, int, JSONValue]]] = json.loads(
    (Path(__file__).parent / 'fixtures' / 'recorded_checkpoints.json').read_text()
)
# Case -> whether the first `report_finding` call raises `ModelRetry`.
CASES = {'plain': False, 'with_retry': True}


class Report(BaseModel):
    outcome: Literal['completed', 'partial', 'failed']
    report: str


def _agent(retry_first: bool, executions: list[str]) -> Agent[object, Report]:
    def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        executions.append('model')
        returns = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
        done = [p.tool_name for p in returns if p.outcome != 'retried']
        retried = any(p.outcome == 'retried' for p in returns)
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
            PGTaskDurability(),
        ],
    )


EXPECTED_OUTPUT = {
    'outcome': 'completed',
    'report': "saw ['add', 'query_metrics', 'report_finding', 'search_knowledge']",
}


def _golden(case: str) -> dict[StepKey, JSONValue]:
    return {(name, occurrence): value for name, occurrence, value in GOLDEN[case]}


@pytest.mark.parametrize('case', CASES)
class TestRecordedCheckpoints:
    async def test_resumes_a_recorded_run(self, case: str, pgtask_db: PGTaskDatabase) -> None:
        executions: list[str] = []
        agent = _agent(CASES[case], executions)
        outputs: list[JSONValue] = []

        async def body(attempt: int) -> None:
            outputs.append((await agent.run('Investigate the latency spike.')).output.model_dump(mode='json'))

        # The recording worker died after its last checkpoint; this attempt resumes the task.
        await run_in_task(pgtask_db, body, seed=_golden(case))

        assert outputs == [EXPECTED_OUTPUT]
        assert executions == []

    async def test_fresh_run_writes_the_same_checkpoints(self, case: str, pgtask_db: PGTaskDatabase) -> None:
        agent = _agent(CASES[case], [])

        async def body(attempt: int) -> None:
            await agent.run('Investigate the latency spike.')

        stored = await checkpoints(pgtask_db, await run_in_task(pgtask_db, body))
        golden = _golden(case)
        assert sorted(stored) == sorted(golden)
        # Tool results are compared verbatim. Model responses and MCP tool listings also carry fields
        # (timestamps, usage, fields added in later releases) that differ run to run or release to
        # release, so those are compared on the fields a replay reads.
        for key, expected in golden.items():
            if '.get_tools' in key[0]:
                assert _tool_schemas(stored[key]) == _tool_schemas(expected), key
            else:
                assert _project(stored[key], expected) == expected, key


def _project(value: JSONValue, like: JSONValue) -> JSONValue:
    """Keep only the parts of `value` that `like` has."""
    if isinstance(like, dict):
        assert isinstance(value, dict)
        return {key: _project(value[key], sub) for key, sub in like.items()}
    if isinstance(like, list):
        assert isinstance(value, list) and len(value) == len(like)
        return [_project(item, sub) for item, sub in zip(value, like)]
    return value


def _tool_schemas(listing: JSONValue) -> JSONValue:
    assert isinstance(listing, dict)
    keys = ('name', 'description', 'kind', 'parameters_json_schema')
    return {name: _project(tool, {key: None for key in keys}) for name, tool in listing.items()}
