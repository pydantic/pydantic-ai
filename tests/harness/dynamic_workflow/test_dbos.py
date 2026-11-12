"""`DynamicWorkflow` under `DBOSDurability`: in a workflow, a script's sub-agent calls run one at a time as workflow code."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable, Generator
from pathlib import Path

import pytest

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from tests.conftest import detach_dbos_logging, try_import, without_admin_server
from tests.harness.dynamic_workflow._concurrent_sub_agents import EXPECTED, build_orchestrator, summarize_model

with try_import() as imports_successful:
    from dbos import DBOS, DBOSConfig, DBOSContextSetAuth, SetWorkflowID

    from pydantic_ai.durable_exec.dbos import DBOSDurability

pytestmark = [
    pytest.mark.xdist_group(name='harness-dbos'),
    pytest.mark.skipif(not imports_successful(), reason='dbos not installed'),
]

CATCHING_SCRIPT = (
    "try:\n    result = await summarize(task='alpha')\nexcept RuntimeError as error:\n    result = str(error)\nresult"
)


@pytest.fixture
def dbos(tmp_path: Path) -> Generator[DBOS, None, None]:
    config: DBOSConfig = {
        'name': 'dynamic_workflow_dbos',
        'system_database_url': f'sqlite:///{tmp_path / "dbos.sqlite"}',
    }
    instance = DBOS(config=without_admin_server(config))
    DBOS.launch()
    try:
        yield instance
    finally:
        DBOS.destroy()
        detach_dbos_logging()


@pytest.fixture
def model_calls() -> list[str]:
    return []


@pytest.fixture
def tool_runs() -> list[str]:
    return []


def record_model(calls: list[str]) -> FunctionModel:
    """A sub-agent model that records its requests in `calls` as `'<task> <message count>'` and calls `record` once."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        (task,) = [part.content for part in messages[0].parts if isinstance(part, UserPromptPart)]
        calls.append(f'{task} {len(messages)}')
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('record')])
        return ModelResponse(parts=[TextPart(f'summary of {task}')])

    return FunctionModel(respond)


def record_tool(step: Callable[[str], Awaitable[str]]) -> Tool[None]:
    """The sub-agents' `record` tool: workflow code that calls the DBOS `step` with the sub-agent's task."""

    async def record(ctx: RunContext[None]) -> str:
        assert isinstance(ctx.prompt, str)
        return await step(ctx.prompt)

    return Tool(record)


def dbos_workflow(orchestrator: Agent[None, str], name: str) -> Callable[[], Awaitable[str]]:
    @DBOS.workflow(name=name)
    async def run() -> str:
        return (await orchestrator.run('Summarize alpha and beta')).output

    return run


@pytest.fixture
def workflow(dbos: DBOS, model_calls: list[str], tool_runs: list[str]) -> Callable[[], Awaitable[str]]:
    @DBOS.step(name='record')
    async def record(task: str) -> str:
        tool_runs.append(task)
        return f'recorded {task}'

    # Built outside the event loop: registering a DBOS workflow reads its source.
    orchestrator = build_orchestrator(DBOSDurability, record_model(model_calls), tools=[record_tool(record)])
    return dbos_workflow(orchestrator, 'dynamic_workflow_dbos')


@pytest.fixture
def admin_workflow(dbos: DBOS, model_calls: list[str], tool_runs: list[str]) -> Callable[[], Awaitable[str]]:
    # A class-level default is the role DBOS checks only when a step is called from workflow code.
    @DBOS.default_required_roles(['admin'])
    class AdminSteps:
        @staticmethod
        @DBOS.step(name='admin_record')
        async def record(task: str) -> str:
            tool_runs.append(task)
            return f'recorded {task}'

    orchestrator = build_orchestrator(
        DBOSDurability, record_model(model_calls), script=CATCHING_SCRIPT, tools=[record_tool(AdminSteps.record)]
    )
    return dbos_workflow(orchestrator, 'dynamic_workflow_dbos_admin')


async def test_gathered_sub_agents_run_one_at_a_time_and_recover_from_their_own_steps(
    dbos: DBOS, workflow: Callable[[], Awaitable[str]], model_calls: list[str], tool_runs: list[str]
) -> None:
    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id):
        assert await workflow() == EXPECTED
    # The script gathers both calls, but alpha runs to completion before beta starts.
    assert model_calls == ['alpha 1', 'alpha 3', 'beta 1', 'beta 3']
    assert tool_runs == ['alpha', 'beta']

    # Each sub-agent's tool call is a step of its own, in dispatch order.
    steps = await dbos.list_workflow_steps_async(workflow_id)
    record_steps = [step for step in steps if step['function_name'] == 'record']
    assert [step['output'] for step in record_steps] == ['recorded alpha', 'recorded beta']

    # Re-execute the workflow from just after beta's tool step, the way recovery does after a crash there:
    # alpha and beta's tool step replay, and only beta's last model request runs again.
    handle = await DBOS.fork_workflow_async(workflow_id, record_steps[1]['function_id'] + 1)
    assert await handle.get_result(polling_interval_sec=0.01) == EXPECTED
    assert tool_runs == ['alpha', 'beta']
    assert model_calls == ['alpha 1', 'alpha 3', 'beta 1', 'beta 3', 'beta 3']


async def test_sub_agent_step_tools_keep_their_role_checks(
    admin_workflow: Callable[[], Awaitable[str]], tool_runs: list[str]
) -> None:
    assert await admin_workflow() == "done: sub-agent 'summarize' raised DBOSNotAuthorizedError"
    assert tool_runs == []

    with DBOSContextSetAuth('alice', ['admin']):
        assert await admin_workflow() == 'done: summary of alpha'
    assert tool_runs == ['alpha']


@pytest.fixture
def orchestrator(dbos: DBOS, model_calls: list[str]) -> Agent[None, str]:
    return build_orchestrator(DBOSDurability, summarize_model(model_calls))


async def test_gathered_sub_agents_outside_a_workflow_run_concurrently(
    orchestrator: Agent[None, str], model_calls: list[str]
) -> None:
    assert (await orchestrator.run('Summarize alpha and beta')).output == EXPECTED
    # Alpha, dispatched first, finished last.
    assert sorted(model_calls) == ['alpha 1', 'alpha 3', 'beta 1', 'beta 3']
    assert model_calls[-1] == 'alpha 3'
