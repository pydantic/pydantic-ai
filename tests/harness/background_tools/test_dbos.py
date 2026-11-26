"""DBOS integration tests for `BackgroundTools`."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Generator
from pathlib import Path

import pytest

try:
    from dbos import DBOS, DBOSConfig, SetWorkflowID, error as dbos_error

    from pydantic_ai.durable_exec.dbos import DBOSDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('dbos not installed', allow_module_level=True)

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai_harness import BackgroundTools

# DBOS launches once per process, like the other harness DBOS tests in this group.
pytestmark = pytest.mark.xdist_group(name='harness-dbos')


class _ProcessCrash(BaseException):
    """Stands in for the process dying: DBOS leaves the workflow `PENDING` for recovery."""


class _Gates:
    """Orders the first run's steps, so the background tool's second step starts while the run waits."""

    def __init__(self) -> None:
        self.may_sleep = asyncio.Event()
        """Set once `search` is under way: its body started (first run), or it returned its recorded result (replay)."""
        self.may_finish_search = asyncio.Event()
        """Set as the foreground tool starts its durable sleep."""
        self.summarized = asyncio.Event()


_gates = _Gates()


@DBOS.step()
async def search() -> str:
    _gates.may_sleep.set()
    await _gates.may_finish_search.wait()  # a slow step: it finishes only once the run is waiting
    return 'found'


@DBOS.step()
async def summarize() -> str:
    return 'summary'


async def research() -> str:
    """Research the topic."""
    await search()
    # On replay `search` returns its recorded result without running its body, so this lets the foreground tool sleep,
    # and `summarize` reserves its step number before that tool resumes: the race the replay loses, made deterministic.
    _gates.may_sleep.set()
    summary = await summarize()
    _gates.summarized.set()
    return summary


@DBOS.step()
async def prepare() -> str:
    return 'prepared'


async def pause() -> str:
    """Wait a while in workflow code."""
    await prepare()
    await _gates.may_sleep.wait()
    # `DBOS.sleep_async` reserves its step number before it first yields, so `search` finishes after that.
    _gates.may_finish_search.set()
    await DBOS.sleep_async(0.01)  # real-wait: DBOS has no clock to inject; the sleep is only the wait's step
    await _gates.summarized.wait()
    raise _ProcessCrash


def _model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Calls both tools in one response; the run dies before it asks again."""
    return ModelResponse(
        parts=[ToolCallPart('research', {}, tool_call_id='research'), ToolCallPart('pause', {}, tool_call_id='pause')]
    )


# Module scope builds the agent and registers its DBOS workflow, which DBOS requires before launch.
_AGENT = Agent(
    FunctionModel(_model),
    name='background_tools_dbos',
    deps_type=type(None),
    toolsets=[
        FunctionToolset[None](tools=[research], id='background', metadata={'background': True}),
        FunctionToolset[None](tools=[pause], id='pause'),
    ],
    capabilities=[BackgroundTools[None](), DBOSDurability[None]()],
)


@DBOS.workflow(name='background_tools_dbos_workflow')
async def _workflow() -> str:
    return (await _AGENT.run('go')).output


@pytest.fixture
def dbos(tmp_path: Path) -> Generator[DBOS, None, None]:
    config: DBOSConfig = {
        'name': 'background_tools_dbos',
        'system_database_url': f'sqlite:///{tmp_path / "dbos.sqlite"}',
    }
    instance = DBOS(config=config)
    DBOS.launch()
    try:
        yield instance
    finally:
        DBOS.destroy()


async def test_recovery_during_a_wait_in_workflow_code(dbos: DBOS) -> None:
    """A background tool's step that starts while the run waits in workflow code is renumbered on replay.

    DBOS numbers steps in the order they start. The first run records `prepare`, `search`, the durable sleep, then
    `summarize`, which starts once `search` finishes during the sleep. On replay `search` returns its recorded result
    at once, so `summarize` starts before the sleep and takes its step number.
    """
    global _gates
    _gates = _Gates()
    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id), pytest.raises(_ProcessCrash):
        await _workflow()
    steps = await dbos.list_workflow_steps_async(workflow_id)
    assert [(step['function_id'], step['function_name']) for step in steps[1:]] == [
        (2, 'prepare'),
        (3, 'search'),
        (4, 'DBOS.sleep'),
        (5, 'summarize'),
    ]

    _gates = _Gates()
    # Resuming the pending workflow re-executes it under the same ID, the way recovery does.
    handle = await DBOS.resume_workflow_async(workflow_id)
    # This pins today's failure, https://github.com/pydantic/pydantic-ai/issues/14500: once it's fixed, flip this to
    # expect a clean recovery. `summarize` and the sleep swap step numbers 4 and 5, and either one can notice first.
    with pytest.raises(
        dbos_error.DBOSUnexpectedStepError,
        match=r'step [45], function (summarize|DBOS\.sleep) was recorded when (DBOS\.sleep|summarize) was expected',
    ):
        await handle.get_result(polling_interval_sec=0.01)
