"""Shared fixtures for BoxdSandbox tests."""

from __future__ import annotations

import importlib.util
import logging
import os
import types
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import anyio
import pytest

_HAS_BOXD = importlib.util.find_spec('boxd') is not None
collect_ignore = (
    []
    if _HAS_BOXD
    else [
        'test_backend.py',
        'test_boxd_desktop.py',
        'test_boxd_live.py',
        'test_boxd_sandbox.py',
        'test_conformance.py',
        'test_live_sweep.py',
    ]
)

# Live machines are deleted by the tests that create them, and `_sweep_live_machines` deletes what their
# cleanup missed once the session ends. CI stops a live job after 15 minutes, before either can run, so
# each one also gets a short idle suspend: a machine left behind that way stops using its vCPUs instead
# of running until someone deletes it.
LIVE_AUTO_SUSPEND_TIMEOUT = 60

if TYPE_CHECKING or _HAS_BOXD:  # pragma: no branch - the SDK-installed and slim jobs take opposite branches
    import boxd
    from boxd.resources.machines import AsyncMachines

    from pydantic_ai.workspaces import WorkspaceRef
    from pydantic_ai_harness.boxd_sandbox import BoxdSandbox

    from .fake_boxd import FakeBoxd, FakeMachines

_Machines = TypeVar('_Machines', 'AsyncMachines', 'FakeMachines')
_SWEEP_REPORT = pytest.StashKey[str]()


def _live_tier_enabled() -> bool:
    return os.getenv('PYDANTIC_AI_HARNESS_BOXD_LIVE') == '1' and bool(os.getenv('BOXD_API_KEY'))


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Gate the live tier, before any live fixture can create a machine.

    It runs only with `PYDANTIC_AI_HARNESS_BOXD_LIVE=1` and a non-empty `BOXD_API_KEY`. Otherwise
    it skips, unless `BOXD_REQUIRE_LIVE=1` (CI's live job sets it), where a missing key fails
    instead: a live job that tested nothing must not pass as a silent skip.
    """
    if item.get_closest_marker('boxd_live') is None:
        return
    if _live_tier_enabled():  # pragma: no cover - live job
        return
    reason = 'requires PYDANTIC_AI_HARNESS_BOXD_LIVE=1 and a non-empty BOXD_API_KEY'
    if os.getenv('BOXD_REQUIRE_LIVE') == '1':  # pragma: no cover - live job
        pytest.fail(f'BOXD_REQUIRE_LIVE=1 but the live tier cannot run: it {reason}.')
    pytest.skip(reason)


@dataclass
class LiveMachines:
    """The machines a session's calls created or forked, until a call deletes them.

    A test's own cleanup can miss its machine: boxd can leave a machine it just made unlisted for over
    30 seconds, `BoxdSandbox().destroy` takes a machine it can't find as deleted already, and boxd can fail
    a delete with `INTERNAL`. `sweep` deletes what is left, by the ids those calls returned: other refs'
    live runs share the boxd account, so a name prefix would also match their machines.
    """

    poll: float = 1.0
    timeout: float = 120.0
    """How long `sweep` retries a machine; the tests ran first, so this comes on top of their run time."""
    pending: set[str] = field(default_factory=set[str])
    recorded: int = 0
    """How many machines `create` and `fork` returned, so a live run's report shows the watch saw them."""

    def watch(self, monkeypatch: pytest.MonkeyPatch, machines: type[_Machines]) -> None:
        """Record what `create` and `fork` on `machines` return, and forget each machine its `delete` deletes."""
        delete = machines.delete

        async def deleted(resource: _Machines, machine_id: str) -> None:
            await delete(resource, machine_id)
            self.pending.discard(machine_id)

        monkeypatch.setattr(machines, 'create', self._recorded(machines.create))
        monkeypatch.setattr(machines, 'fork', self._recorded(machines.fork))
        monkeypatch.setattr(machines, 'delete', deleted)

    def _recorded(self, make: Callable[..., Awaitable[boxd.Machine]]) -> Callable[..., Awaitable[boxd.Machine]]:
        async def made(*args: object, **kwargs: object) -> boxd.Machine:
            machine = await make(*args, **kwargs)
            self.pending.add(machine.id)
            self.recorded += 1
            return machine

        return made

    async def sweep(self, machines: AsyncMachines | FakeMachines) -> None:
        """Delete each pending machine, retrying for up to `timeout`; what is still pending after could not be."""

        async def delete(machine_id: str) -> None:
            # real-wait: boxd lists a new machine, or succeeds in deleting it, only after a delay of its own.
            while True:
                try:
                    return await machines.delete(machine_id)
                except boxd.AuthenticationError:
                    # No retry gets past refused credentials, and the session should fail on them.
                    raise
                except boxd.BoxdError:
                    # Not listed yet, or boxd failed to delete it: `INTERNAL`, or a dropped connection.
                    await anyio.sleep(self.poll)

        with anyio.move_on_after(self.timeout):
            async with anyio.create_task_group() as group:
                for machine_id in sorted(self.pending):
                    group.start_soon(delete, machine_id)


@pytest.fixture(scope='session', autouse=True)
def _sweep_live_machines(pytestconfig: pytest.Config) -> Iterator[None]:
    """In a live session, delete each machine its calls made that no call deleted, and report what it did."""
    if _live_tier_enabled():  # pragma: no cover - live job
        live = LiveMachines()
        with pytest.MonkeyPatch.context() as monkeypatch:
            live.watch(monkeypatch, AsyncMachines)
            yield
            missed = sorted(live.pending)

            async def sweep() -> None:
                client = boxd.AsyncBoxd()
                try:
                    await live.sweep(client.machines)
                finally:
                    await client.close()

            try:
                # A thread of its own, like `run_block`'s cleanup: the session's event loop may still be set.
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(anyio.run, sweep).result()
            finally:
                deleted = [machine_id for machine_id in missed if machine_id not in live.pending]
                pytestconfig.stash[_SWEEP_REPORT] = (
                    f'boxd live sweep: the session made {live.recorded} machine(s); '
                    f'deleted {len(deleted)} the tests left: {deleted}; '
                    f'could not delete {len(live.pending)}: {sorted(live.pending)}'
                )
    else:
        yield


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter, config: pytest.Config) -> None:
    """Print what `_sweep_live_machines` did after the test results, so a live run's log shows it."""
    if (report := config.stash.get(_SWEEP_REPORT, None)) is not None:  # pragma: no cover - live job
        terminalreporter.write_line(report)


async def delete_live_machine(ref: WorkspaceRef, client: boxd.AsyncBoxd | None = None) -> None:
    """Delete a live test's machine, or leave it to `_sweep_live_machines` when boxd fails the delete.

    The sweep retries what is left, so boxd's own failure, such as `INTERNAL` ("couldn't destroy that machine"),
    doesn't fail the test. Refused credentials raise `WorkspaceUnavailableError` and still do.
    """
    try:
        await BoxdSandbox(client=client).destroy(ref)
    except boxd.BoxdError as error:
        logging.getLogger(__name__).warning(
            'boxd did not delete machine %s, so the session sweep retries it: %r', ref.id, error
        )


class _PoisonedBoxd(types.ModuleType):
    """A `boxd` stand-in that fails loudly on any attribute access.

    The real SDK is installed in the dev venv for the live tier, so a unit test that forgets the
    `fake_boxd` fixture would otherwise reach production and, with a developer API key configured,
    create real billed machines.
    """

    def __getattr__(self, name: str) -> object:  # pragma: no cover - tripwire, hit only by a misbehaving test
        raise AssertionError(
            'A boxd_sandbox unit test touched the real `boxd` package. '
            'Use the `fake_boxd` fixture, or mark the test `boxd_live`.'
        )


@pytest.fixture(autouse=True)
def _no_real_boxd(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Poison `boxd` for every test here except the opt-in live tier."""
    if 'boxd_live' in request.keywords:  # pragma: no cover - live tier runs without coverage
        yield
        return
    monkeypatch.setattr('pydantic_ai_harness.boxd_sandbox._backend.boxd', _PoisonedBoxd('boxd'))
    yield


if _HAS_BOXD:  # pragma: no branch - the fixture cannot be defined without its SDK-backed fake

    @pytest.fixture
    async def fake_boxd(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[FakeBoxd]:
        """Swap the SDK's `AsyncBoxd` for the fake and return its control surface.

        The real module stays in place for its error types, which the backend matches on.
        """
        fake = FakeBoxd()
        monkeypatch.setattr('pydantic_ai_harness.boxd_sandbox._backend.boxd', boxd)
        monkeypatch.setattr(boxd, 'AsyncBoxd', fake.client)
        # Nothing here waits on a real control plane.
        monkeypatch.setattr('pydantic_ai_harness.boxd_sandbox._backend._POLL_INTERVAL', 0)
        yield fake
        await fake.reap()

    @pytest.fixture
    def host(fake_boxd: FakeBoxd, tmp_path: Path) -> FakeBoxd:
        """The fake with its machines' commands running in a temporary host directory."""
        fake_boxd.host_root = tmp_path.resolve()
        return fake_boxd
