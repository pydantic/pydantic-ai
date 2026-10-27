"""End-to-end coverage: the same `Coder` agent runs its workspace tools on Temporal, DBOS and Prefect."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from pytest_examples import CodeExample

from pydantic_ai.workspaces import WorkspaceRef
from tests.docs_model_aliases import resolve_model_aliases

from .._docs_examples import run_block
from .._temporal import skip_temporal_sandbox_on_314

# `Coder` on each engine, run end-to-end the way the workspace guide describes.
# These are test scripts, not docs examples.
CODER_TEMPORAL = """\
import asyncio
import uuid

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from pydantic_ai_harness.coder import Coder
from temporalio import workflow
from temporalio.client import Client
from temporalio.worker import Worker

agent = Agent(
    '{{ anthropic:large }}',
    name='coder',  # names the agent's Temporal activities
    capabilities=[LocalWorkspace('.'), Coder(), TemporalDurability()],
)


@workflow.defn
class CoderWorkflow(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [agent]

    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await agent.run(prompt)).output


async def main() -> None:
    client = await Client.connect('localhost:7233', plugins=[PydanticAIPlugin()])
    async with Worker(client, task_queue='coder', workflows=[CoderWorkflow]):
        output = await client.execute_workflow(
            CoderWorkflow.run, 'Add a test for fizzbuzz.py and run it.', id=f'coder-{uuid.uuid4()}', task_queue='coder'
        )
        print(output)


if __name__ == '__main__':
    asyncio.run(main())
"""

CODER_DBOS = """\
import asyncio

from dbos import DBOS, DBOSConfig
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.durable_exec.dbos import DBOSDurability
from pydantic_ai_harness.coder import Coder

dbos_config: DBOSConfig = {'name': 'coder', 'system_database_url': 'sqlite:///coder.sqlite'}
DBOS(config=dbos_config)

agent = Agent(
    '{{ anthropic:large }}',
    name='coder',
    capabilities=[LocalWorkspace('.'), Coder(), DBOSDurability()],
)


@DBOS.workflow()
async def run(prompt: str) -> str:
    return (await agent.run(prompt)).output


async def main() -> None:
    DBOS.launch()
    print(await run('Add a test for fizzbuzz.py and run it.'))


if __name__ == '__main__':
    asyncio.run(main())
"""

CODER_PREFECT = """\
import asyncio

from prefect import flow
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.durable_exec.prefect import PrefectDurability
from pydantic_ai_harness.coder import Coder

agent = Agent(
    '{{ anthropic:large }}',
    name='coder',
    capabilities=[LocalWorkspace('.'), Coder(), PrefectDurability()],
)


@flow
async def run(prompt: str) -> str:
    return (await agent.run(prompt)).output


if __name__ == '__main__':
    print(asyncio.run(run('Add a test for fizzbuzz.py and run it.')))
"""


async def _cleanup(ref: WorkspaceRef) -> None:
    pass  # The workspace is the test's `tmp_path`.


def _run(source: str, workspace: Path) -> None:
    """Run `source` from `workspace`, the directory its `LocalWorkspace('.')` names."""
    example = CodeExample.create(resolve_model_aliases(source), path=Path('coder_durable_engine.py'))
    os.chdir(workspace)
    _, runs = run_block(example, cleanup=_cleanup)
    assert [(run.used_sandbox, set(run.outputs)) for run in runs] == [(True, {'shell', 'write_file', 'read_file'})]
    assert runs[0].ref == WorkspaceRef(provider='local', id=str(workspace.resolve()))


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)  # restores the working directory `_run` changes
    return tmp_path


@pytest.mark.temporal
@pytest.mark.xdist_group(name='harness-temporal')
@skip_temporal_sandbox_on_314
def test_temporal_example(workspace: Path) -> None:
    pytest.importorskip('temporalio')
    _run(CODER_TEMPORAL, workspace)


@pytest.fixture
def dbos_teardown() -> Iterator[None]:
    dbos = pytest.importorskip('dbos')
    from dbos._dbos import _get_or_create_dbos_registry  # pyright: ignore[reportPrivateUsage]

    # `dbos` 3 dropped `function_type_map`, leaving `workflow_info_map` the only map keyed by workflow name:
    # https://github.com/dbos-inc/dbos-transact-py/pull/850
    attrs = vars(_get_or_create_dbos_registry())
    maps = {name: dict(attrs[name]) for name in ('workflow_info_map', 'function_type_map') if name in attrs}
    yield
    dbos.DBOS.destroy()
    # Drop only the example's workflows: their source was `exec`d, so a later `DBOS.launch()` could not hash
    # it, while modules that register agents at import time still need theirs.
    attrs.update(maps)


def test_dbos_example(workspace: Path, dbos_teardown: None) -> None:
    _run(CODER_DBOS, workspace)


@pytest.mark.xdist_group(name='prefect')
def test_prefect_example(workspace: Path, prefect_test_server: None) -> None:
    _run(CODER_PREFECT, workspace)
