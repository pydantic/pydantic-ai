"""Shared fixtures for PlatinumSandbox tests."""

from __future__ import annotations

import importlib.util
import logging
import os
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import pytest

_HAS_PLATINUM = importlib.util.find_spec('platinum') is not None
collect_ignore = (
    []
    if _HAS_PLATINUM
    else ['test_backend.py', 'test_conformance.py', 'test_platinum_live.py', 'test_platinum_sandbox.py']
)

# Live sandboxes are deleted by the tests that create them, and `_sweep_live_sandboxes` deletes what their cleanup
# missed once the session ends. CI stops a live job after 15 minutes, before either can run, so each one also stops
# after this many idle minutes: a sandbox left behind that way bills only its disk until someone deletes it.
LIVE_AUTO_STOP_MINUTES = 5

if TYPE_CHECKING or _HAS_PLATINUM:  # pragma: no branch - the SDK-installed and slim jobs take opposite branches
    import platinum
    from platinum.aio import AsyncPlatinum, AsyncSandbox

    from pydantic_ai.workspaces import WorkspaceRef
    from pydantic_ai_harness.platinum_sandbox import PlatinumSandbox

    from .fake_platinum import FakePlatinum

_SWEEP_REPORT = pytest.StashKey[str]()


def _live_tier_enabled() -> bool:
    return os.getenv('PYDANTIC_AI_HARNESS_PLATINUM_LIVE') == '1' and bool(os.getenv('PT_TOKEN'))


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Gate the live tier, before any live fixture can create a sandbox.

    It runs only with `PYDANTIC_AI_HARNESS_PLATINUM_LIVE=1` and a non-empty `PT_TOKEN`. Otherwise it skips, unless
    `PLATINUM_REQUIRE_LIVE=1` (CI's live job sets it), where a missing key fails instead: a live job that tested
    nothing must not pass as a silent skip.
    """
    if item.get_closest_marker('platinum_live') is None:
        return
    if _live_tier_enabled():  # pragma: no cover - live job
        return
    reason = 'requires PYDANTIC_AI_HARNESS_PLATINUM_LIVE=1 and a non-empty PT_TOKEN'
    if os.getenv('PLATINUM_REQUIRE_LIVE') == '1':  # pragma: no cover - live job
        pytest.fail(f'PLATINUM_REQUIRE_LIVE=1 but the live tier cannot run: it {reason}.')
    pytest.skip(reason)


@dataclass
class LiveSandboxes:  # pragma: no cover - live tier runs without coverage
    """The sandboxes a live session's creates made, until a delete removes them.

    `sweep` deletes what the tests' own cleanup missed, by the ids the creates returned: other live runs share the
    Platinum organization, so a name prefix would also match their sandboxes.
    """

    pending: set[str] = field(default_factory=set[str])
    recorded: int = 0

    def watch(self, monkeypatch: pytest.MonkeyPatch, client: AsyncPlatinum) -> None:
        """Record what `sandboxes.create` returns on any client, and forget each sandbox `sandboxes.delete` deletes."""
        resource = type(client.sandboxes)
        create, delete = resource.create, resource.delete

        async def created(namespace: object, **kwargs: object) -> AsyncSandbox:
            sandbox = await create(namespace, **kwargs)  # pyright: ignore[reportArgumentType]
            self.pending.add(sandbox.id)
            self.recorded += 1
            return sandbox

        async def deleted(namespace: object, sandbox_id: str) -> None:
            await delete(namespace, sandbox_id)  # pyright: ignore[reportArgumentType]
            self.pending.discard(sandbox_id)

        monkeypatch.setattr(resource, 'create', created)
        monkeypatch.setattr(resource, 'delete', deleted)

    async def sweep(self) -> None:
        async with platinum.AsyncPlatinum() as client:
            for sandbox_id in sorted(self.pending):
                try:
                    await client.sandboxes.delete(sandbox_id)
                except platinum.NotFoundError:
                    pass
                self.pending.discard(sandbox_id)


@pytest.fixture(scope='session', autouse=True)
def _sweep_live_sandboxes(pytestconfig: pytest.Config) -> Iterator[None]:
    """In a live session, delete each sandbox its creates made that no delete removed, and report what it did."""
    if _live_tier_enabled():  # pragma: no cover - live job
        live = LiveSandboxes()
        with pytest.MonkeyPatch.context() as monkeypatch:
            # Any client reaches the namespace class whose methods are watched; this one sends nothing.
            probe = platinum.AsyncPlatinum(token='unused', api_url='https://unused.invalid')
            live.watch(monkeypatch, probe)
            yield
            missed = sorted(live.pending)
            try:
                # A thread of its own: the session's event loop may still be set.
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(anyio.run, live.sweep).result()
                    pool.submit(anyio.run, probe.aclose).result()
            finally:
                pytestconfig.stash[_SWEEP_REPORT] = (
                    f'Platinum live sweep: the session created {live.recorded} sandbox(es); '
                    f'deleted {len(missed) - len(live.pending)} the tests left: {missed}; '
                    f'could not delete {len(live.pending)}: {sorted(live.pending)}'
                )
    else:
        yield


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter, config: pytest.Config) -> None:
    """Print what `_sweep_live_sandboxes` did after the test results, so a live run's log shows it."""
    if (report := config.stash.get(_SWEEP_REPORT, None)) is not None:  # pragma: no cover - live job
        terminalreporter.write_line(report)


async def delete_live_sandbox(ref: WorkspaceRef) -> None:  # pragma: no cover - live tier runs without coverage
    """Delete a live test's sandbox, or leave it to `_sweep_live_sandboxes` when Platinum fails the delete."""
    try:
        await PlatinumSandbox().destroy(ref)
    except platinum.PlatinumError as error:
        logging.getLogger(__name__).warning(
            'Platinum did not delete sandbox %s, so the session sweep retries it: %r', ref.id, error
        )


def _refuse_real_client(**_: object) -> AsyncPlatinum:  # pragma: no cover - tripwire, hit only by a misbehaving test
    raise AssertionError(
        'A platinum_sandbox unit test built a real `AsyncPlatinum`. Use the `fake_platinum` fixture, '
        'or mark the test `platinum_live`.'
    )


@pytest.fixture(autouse=True)
def _no_real_platinum(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep every test here except the opt-in live tier off a real control plane.

    A unit test that forgets the `fake_platinum` fixture would otherwise reach the API with a developer's `PT_TOKEN`
    and create real billed sandboxes.
    """
    if 'platinum_live' in request.keywords:  # pragma: no cover - live tier runs without coverage
        yield
        return
    monkeypatch.delenv('PT_TOKEN', raising=False)
    monkeypatch.delenv('PT_API_URL', raising=False)
    monkeypatch.delenv('PLATINUM_API_URL', raising=False)
    monkeypatch.setattr(platinum, 'AsyncPlatinum', _refuse_real_client)
    yield


if _HAS_PLATINUM:  # pragma: no branch - the fixture cannot be defined without its SDK-backed fake

    @pytest.fixture
    async def fake_platinum(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AsyncIterator[FakePlatinum]:
        """Serve the backend's own clients from the fake, and return its control surface.

        The backend's scratch files (stop claims, upload parts, listings) go under `tmp_path` instead of the host's `/tmp`.
        """
        fake = FakePlatinum()
        monkeypatch.setattr(platinum, 'AsyncPlatinum', fake.client)
        scratch = tmp_path / 'scratch'
        scratch.mkdir()
        monkeypatch.setattr('pydantic_ai_harness.platinum_sandbox._backend._SCRATCH_DIR', str(scratch))
        # A read that hasn't answered this soon is probed for a FIFO; the host answers at once otherwise.
        monkeypatch.setattr('pydantic_ai_harness.platinum_sandbox._backend._FIFO_PROBE_AFTER', 0.1)
        yield fake
        await fake.reap()

    @pytest.fixture
    def work(tmp_path: Path) -> Path:
        """The host directory the fake's sandboxes work in, resolved so `pwd -P` reports it as it is."""
        return (tmp_path / 'work').resolve()
