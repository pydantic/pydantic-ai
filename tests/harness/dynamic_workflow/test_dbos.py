"""`DynamicWorkflow` under `DBOSDurability`: gathered sub-agents run in parallel, each recorded in a step drawn in dispatch order."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Generator
from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from tests.conftest import detach_dbos_logging, try_import, without_admin_server
from tests.harness.dynamic_workflow._concurrent_sub_agents import EXPECTED, build_orchestrator

with try_import() as imports_successful:
    from dbos import DBOS, DBOSConfig, SetWorkflowID

    from pydantic_ai.durable_exec.dbos import DBOSDurability

pytestmark = [
    pytest.mark.xdist_group(name='harness-dbos'),
    pytest.mark.skipif(not imports_successful(), reason='dbos not installed'),
]


class _ProcessCrash(BaseException):
    """Stands in for the process dying: DBOS leaves the workflow `PENDING` for recovery."""


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
def orchestrator(dbos: DBOS, model_calls: list[str]) -> Agent[None, str]:
    beta_done = asyncio.Event()

    async def summarize(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        (task,) = [part.content for part in messages[0].parts if isinstance(part, UserPromptPart)]
        model_calls.append(f'{task} {len(messages)}')
        if len(messages) > 1:
            if task == 'beta':
                beta_done.set()
            return ModelResponse(parts=[TextPart(f'summary of {task}')])
        if task == 'alpha':
            # Alpha, dispatched first, finishes only after beta: the sub-agents run concurrently.
            # The timeout fails a dispatch that runs them one at a time, rather than hanging it.
            await asyncio.wait_for(beta_done.wait(), timeout=5)
        return ModelResponse(parts=[ToolCallPart('noop')])

    # Built outside the event loop: registering a DBOS workflow reads its source.
    return build_orchestrator(DBOSDurability, summarize_model=FunctionModel(summarize))


@pytest.fixture
def workflow(orchestrator: Agent[None, str]) -> Callable[[], Awaitable[str]]:
    attempts: list[str] = []

    @DBOS.workflow(name='dynamic_workflow_dbos')
    async def run() -> str:
        output = (await orchestrator.run('Summarize alpha and beta')).output
        attempts.append(output)
        if len(attempts) == 1:
            raise _ProcessCrash
        return output

    return run


async def test_parallel_sub_agents_get_stable_steps_and_recover(
    dbos: DBOS, workflow: Callable[[], Awaitable[str]], model_calls: list[str]
) -> None:
    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id), pytest.raises(_ProcessCrash):
        await workflow()
    # Alpha, dispatched first, finished last.
    assert sorted(model_calls) == ['alpha 1', 'alpha 3', 'beta 1', 'beta 3']
    assert model_calls[-1] == 'alpha 3'

    # Each sub-agent run is one step, numbered in dispatch order rather than completion order.
    steps = await dbos.list_workflow_steps_async(workflow_id)
    sub_agent_steps = [step for step in steps if step['function_name'] == 'run_workflow.summarize']
    assert [step['output'] for step in sub_agent_steps] == ['summary of alpha', 'summary of beta']
    assert sub_agent_steps[0]['function_id'] < sub_agent_steps[1]['function_id']

    # Resuming the pending workflow re-executes it the way recovery does: each call replays its own step.
    handle = await DBOS.resume_workflow_async(workflow_id)
    assert await handle.get_result(polling_interval_sec=0.01) == EXPECTED
    assert len(model_calls) == 4


async def test_parallel_sub_agents_outside_a_workflow_make_no_steps(
    orchestrator: Agent[None, str], model_calls: list[str]
) -> None:
    assert (await orchestrator.run('Summarize alpha and beta')).output == EXPECTED
    # Alpha, dispatched first, finished last.
    assert sorted(model_calls) == ['alpha 1', 'alpha 3', 'beta 1', 'beta 3']
    assert model_calls[-1] == 'alpha 3'
