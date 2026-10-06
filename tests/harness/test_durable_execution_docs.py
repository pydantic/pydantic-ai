"""The durable-execution page's Temporal, DBOS and Prefect examples run their tools on each engine."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from pydantic_ai.workspaces import WorkspaceRef

from ._docs_examples import python_blocks, run_block
from ._temporal import skip_temporal_sandbox_on_314

_PAGE = 'docs/harness/durable-execution.md'


async def _cleanup(ref: WorkspaceRef) -> None:
    pass  # The workspace is the test's `tmp_path`.


def _run(title: str, workspace: Path) -> None:
    """Run the block titled `title` from `workspace`, the directory its `LocalWorkspace('.')` names."""
    # `python_blocks` moves to the repository root, so change directory only once the block is found.
    (example,) = [example for example in python_blocks(_PAGE) if example.prefix_settings().get('title') == title]
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
    _run('coder_temporal.py', workspace)


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
    _run('coder_dbos.py', workspace)


def test_prefect_example(workspace: Path) -> None:
    pytest.importorskip('prefect')
    from prefect.settings import PREFECT_SERVER_SERVICES_TASK_RUN_RECORDER_ENABLED, temporary_settings
    from prefect.testing.utilities import prefect_test_harness

    with temporary_settings({PREFECT_SERVER_SERVICES_TASK_RUN_RECORDER_ENABLED: False}):
        with prefect_test_harness(server_startup_timeout=120):
            _run('coder_prefect.py', workspace)
