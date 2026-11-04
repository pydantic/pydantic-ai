"""`ToolOutputLimits` under a real durable engine: a spill is recorded, not written again on replay."""

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
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.tool_output_limits import Band, Spill, ToolOutputLimits, Truncate
from tests.conftest import detach_dbos_logging

pytestmark = [pytest.mark.xdist_group(name='harness-dbos')]


@pytest.fixture
def dbos(tmp_path: Path) -> Generator[DBOS, None, None]:
    config: DBOSConfig = {
        'name': 'durable_tool_output_limits',
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


class _CountingStore:
    """Hands out a new handle per write, as an append-only blob store would; can fail the first write."""

    def __init__(self) -> None:
        self.writes: dict[str, bytes] = {}
        self.fail_first = False

    async def write(self, key: str, data: bytes) -> str:
        attempt = len(self.writes) + 1
        self.writes[f'blob/{attempt}'] = data
        if self.fail_first and attempt == 1:
            raise OSError('store unavailable')
        return f'blob/{attempt}'

    async def read(self, handle: str) -> bytes:
        return self.writes[handle]


_SPILL_STEP = 'durable_spill__capability__tool_output_limits.spill'
_store = _CountingStore()
_agent: Agent[None, str] = Agent(
    TestModel(call_tools=['large_output']),
    name='durable_spill',
    deps_type=type(None),
    capabilities=[
        ToolOutputLimits[None](bands=[Band(over=5, action=Spill(then=Truncate(max_chars=10)))], store=_store),
        DBOSDurability[None](),
    ],
)


_BINARY = b'\x00\xff' * 100
_binary_agent: Agent[None, str] = Agent(
    TestModel(call_tools=['large_binary_output']),
    name='durable_binary_spill',
    deps_type=type(None),
    capabilities=[ToolOutputLimits[None](bands=[Band(over=5, action=Spill())], store=_store), DBOSDurability[None]()],
)


@_agent.tool_plain
def large_output() -> str:
    return 'x' * 100


@_binary_agent.tool_plain
def large_binary_output() -> bytes:
    return _BINARY


async def _tool_return(agent: Agent[None, str]) -> str:
    result = await agent.run('call the tool')
    [part] = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
    assert isinstance(part.content, str)
    return part.content


@DBOS.workflow(name='durable_spill')
async def _workflow(binary: bool) -> str:
    return await _tool_return(_binary_agent if binary else _agent)


async def _run_and_replay(dbos: DBOS, *, binary: bool = False) -> tuple[str, str, set[str]]:
    """Run the workflow, then replay it; return both results and the recorded step names."""
    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id):
        first = await _workflow(binary)
    steps = await dbos.list_workflow_steps_async(workflow_id)

    # Re-execute the workflow function from its last step, the way recovery does.
    handle = await DBOS.fork_workflow_async(workflow_id, len(steps))
    # The default 1s result poll would add up to a second on top of the internal queue's own poll.
    replay = await handle.get_result(polling_interval_sec=0.01)
    return first, replay, {step['function_name'] for step in steps}


class TestDurableSpill:
    async def test_dbos_replays_the_recorded_spill(self, dbos: DBOS) -> None:
        """A replay reuses the recorded handle: no second write, same preview, and the handle still reads back."""
        _store.writes.clear()
        _store.fail_first = False

        first, replay, steps = await _run_and_replay(dbos)

        assert list(_store.writes) == ['blob/1'], 'the replay wrote the spill again'
        assert "'blob/1'" in first
        assert replay == first
        assert await _store.read('blob/1') == b'x' * 100
        assert _SPILL_STEP in steps

    async def test_dbos_replays_the_recorded_spill_failure(self, dbos: DBOS) -> None:
        """A spill that failed falls back to truncation, and a replay keeps that fallback instead of retrying."""
        _store.writes.clear()
        _store.fail_first = True

        first, replay, steps = await _run_and_replay(dbos)

        assert len(_store.writes) == 1, 'the replay retried the failed spill'
        assert first == 'xxxxxxxxxx'
        assert replay == first
        assert _SPILL_STEP in steps

    async def test_dbos_replays_the_recorded_binary_spill(self, dbos: DBOS) -> None:
        """A binary payload crosses the operation as base64 and is stored, and read back, byte for byte."""
        _store.writes.clear()
        _store.fail_first = False

        first, replay, steps = await _run_and_replay(dbos, binary=True)

        assert list(_store.writes) == ['blob/1'], 'the replay wrote the spill again'
        assert 'binary' in first
        assert replay == first
        assert await _store.read('blob/1') == _BINARY
        assert 'durable_binary_spill__capability__tool_output_limits.spill' in steps
