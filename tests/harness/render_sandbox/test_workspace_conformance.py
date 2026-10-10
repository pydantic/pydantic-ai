"""Run the released public WorkspaceBackendSuite without editing its tests.

Offline checks verify the resource recorder. Live execution is deliberately opt-in:

    RENDER_SANDBOX_LIVE=1 uv run --no-sync python -m pytest -q -rs \\
        tests/harness/render_sandbox/test_workspace_conformance.py::TestLiveRenderSandboxBackend

The host needs RENDER_API_KEY and RENDER_WORKSPACE_ID. Do not use pytest-xdist:
ordinary rules share one sandbox; fresh factories supply the concurrent-acquisition
and destructive rules. All returned creation IDs are recorded immediately in
.cache/render-sandboxes/workspace-conformance-<UTC>-<pid>.json and audited during module teardown.
RENDER_SANDBOX_EVIDENCE can select an explicit JSON destination. A module-scoped AnyIO runner keeps the SDK connection pool on one event loop.

The suite itself may skip its optional native-realpath rule for this
backend and its permission rule when the sandbox runs as root. No command,
filesystem, cancellation, reattachment, or background-child rule is disabled.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from ...conftest import try_import

with try_import() as imports_successful:
    from render import RenderAsync
    from render.experimental.sandbox import Sandbox

    from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
    from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite

    from .workspace_live_helpers import ResourceLedger

pytestmark = pytest.mark.skipif(not imports_successful(), reason='render not installed')

if TYPE_CHECKING:
    from pydantic_ai_harness.render_sandbox import RenderSandboxBackend


@pytest.fixture(scope='module')
def anyio_backend() -> str:
    # Render's public asynchronous client and this adapter use asyncio.
    return 'asyncio'


@pytest.fixture(scope='module')
async def live_resources(
    anyio_backend: str, request: pytest.FixtureRequest
) -> AsyncIterator[tuple[RenderAsync, ResourceLedger]]:
    if os.environ.get('RENDER_SANDBOX_LIVE') != '1':
        pytest.skip('live Render conformance requires RENDER_SANDBOX_LIVE=1')
    # The public suite is AnyIO-native. Its module-scoped async fixture retains
    # one runner for SDK connections, tests, temporary cleanup, and final audit.
    # pytest-asyncio auto mode would also claim these tests and split the loops.
    assert anyio_backend == 'asyncio'
    missing = [name for name in ('RENDER_API_KEY', 'RENDER_WORKSPACE_ID') if not os.getenv(name)]
    if missing:
        pytest.fail('live conformance requested but host variables are missing: ' + ', '.join(missing))
    if os.getenv('PYTEST_XDIST_WORKER'):
        pytest.fail('run live conformance sequentially so sandboxes are shared and slots stay bounded')
    stamp = datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')
    default = (
        Path(__file__).resolve().parents[3]
        / '.cache'
        / 'render-sandboxes'
        / f'workspace-conformance-{stamp}-{os.getpid()}.json'
    )
    ledger = ResourceLedger(Path(os.getenv('RENDER_SANDBOX_EVIDENCE', str(default))))
    client = RenderAsync(token=os.environ['RENDER_API_KEY'], owner_id=os.environ['RENDER_WORKSPACE_ID'])
    control = client.experimental.sandboxes
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(control, 'create', ledger.recording_create(control.create))
        try:
            yield client, ledger
        finally:
            # Independent public API cleanup still works if backend teardown is broken.
            try:
                await ledger.cleanup_and_audit(control)
            finally:
                # This fixture owns the SDK client and outlives all workspace calls.
                await client.experimental.sandboxes.client.get_async_httpx_client().aclose()


@pytest.fixture(scope='module')
def shared_backend(live_resources: tuple[RenderAsync, ResourceLedger]) -> RenderSandboxBackend:
    from pydantic_ai_harness.render_sandbox import RenderSandboxBackend, RenderSandboxConfig

    client, _ = live_resources
    # The full suite shares one sandbox across many operations. Teardown terminates
    # it immediately; the longer lifetime is a backstop for an interrupted host.
    return RenderSandboxBackend(client=client, config=RenderSandboxConfig(lifetime_seconds=3600))


@pytest.fixture
async def temporary_backends(
    live_resources: tuple[RenderAsync, ResourceLedger],
) -> AsyncIterator[list[RenderSandboxBackend]]:
    # Release fresh environments after their individual test, keeping at most the
    # shared sandbox and one temporary sandbox alive during a sequential run.
    client, ledger = live_resources
    backends: list[RenderSandboxBackend] = []
    try:
        yield backends
    finally:
        failed: list[str] = []
        for backend in backends:
            if backend.ref is not None:
                try:
                    await ledger.terminate_owned(client.experimental.sandboxes, backend.ref.id)
                except RuntimeError:
                    failed.append(backend.ref.id)
        if failed:
            pytest.fail('fresh sandbox cleanup failed: ' + ', '.join(failed))


@pytest.mark.skipif(
    os.environ.get('RENDER_SANDBOX_LIVE') != '1',
    reason='live Render conformance requires RENDER_SANDBOX_LIVE=1',
)
class TestLiveRenderSandboxBackend(WorkspaceBackendSuite):
    @pytest.fixture
    def backend(self, shared_backend: RenderSandboxBackend) -> WorkspaceBackend:
        return shared_backend

    @pytest.fixture
    def fresh_backend(
        self,
        live_resources: tuple[RenderAsync, ResourceLedger],
        temporary_backends: list[RenderSandboxBackend],
    ) -> Callable[[], WorkspaceBackend]:
        from pydantic_ai_harness.render_sandbox import RenderSandboxBackend, RenderSandboxConfig

        client, _ = live_resources

        def fresh() -> WorkspaceBackend:
            backend = RenderSandboxBackend(client=client, config=RenderSandboxConfig(lifetime_seconds=3600))
            temporary_backends.append(backend)
            return backend

        return fresh

    @pytest.fixture
    def destructive_backend(self, fresh_backend: Callable[[], WorkspaceBackend]) -> Callable[[], WorkspaceBackend]:
        return fresh_backend

    @pytest.fixture
    def attach_backend(
        self, live_resources: tuple[RenderAsync, ResourceLedger]
    ) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        from pydantic_ai_harness.render_sandbox import RenderSandboxBackend, RenderSandboxConfig

        client, _ = live_resources
        return lambda ref: RenderSandboxBackend(
            client=client, ref=ref, config=RenderSandboxConfig(lifetime_seconds=3600)
        )

    @pytest.fixture
    def destroy_environment(
        self, live_resources: tuple[RenderAsync, ResourceLedger]
    ) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        client, ledger = live_resources

        async def destroy(backend: WorkspaceBackend) -> None:
            assert backend.ref is not None
            await ledger.terminate_owned(client.experimental.sandboxes, backend.ref.id)

        return destroy


def _snapshot(sandbox_id: str, status: str = 'running') -> Sandbox:
    return Sandbox(
        sandbox_id,
        status,
        'starter',
        'deny-all',
        'oregon',
        3600,
        datetime.now(UTC),
        datetime.now(UTC) if status == 'terminated' else None,
    )


class _FakeControl:
    """Only the public terminate/from_id surface, with no HTTP or credentials."""

    def __init__(self, statuses: dict[str, str]) -> None:
        self.statuses = statuses
        self.calls: list[str] = []
        self.fail_terminate: set[str] = set()
        self.stuck: set[str] = set()

    async def terminate(self, sandbox_id: str, *, owner_id: str | None = None) -> None:
        self.calls.append(sandbox_id)
        if sandbox_id not in self.stuck:
            self.statuses[sandbox_id] = 'terminated'
        if sandbox_id in self.fail_terminate:
            raise RuntimeError('credential-shaped exception text must never enter the ledger')

    async def from_id(self, sandbox_id: str, *, owner_id: str | None = None) -> Sandbox:
        return _snapshot(sandbox_id, self.statuses[sandbox_id])


async def test_creation_is_persisted_before_returning_to_backend(tmp_path: Path) -> None:
    path = tmp_path / 'resources.json'
    ledger = ResourceLedger(path)

    async def create(*, token: str) -> Sandbox:
        return _snapshot('sbx-recorded')

    created = await ledger.recording_create(create)(token='do-not-serialize-me')
    data = json.loads(path.read_text())
    assert data['resources'][0]['id'] == created.id
    assert not data['audit_complete']
    assert 'do-not-serialize-me' not in path.read_text()


async def test_cleanup_audits_every_id_after_uncertain_termination(tmp_path: Path) -> None:
    ledger = ResourceLedger(tmp_path / 'resources.json')
    for sandbox_id in ('sbx-first', 'sbx-second'):
        ledger.record_created(_snapshot(sandbox_id))
    control = _FakeControl({sandbox_id: 'running' for sandbox_id in ledger.records})
    control.fail_terminate.add('sbx-first')
    await ledger.cleanup_and_audit(control)
    assert control.calls == ['sbx-first', 'sbx-second']
    assert ledger.audit_complete
    assert all(record.terminal for record in ledger.records.values())
    assert 'credential-shaped' not in ledger.path.read_text()
    assert ledger.records['sbx-first'].errors == ['RuntimeError']


async def test_cleanup_failure_retains_ids_and_still_cleans_other_resources(tmp_path: Path) -> None:
    ledger = ResourceLedger(tmp_path / 'resources.json')
    for sandbox_id in ('sbx-stuck', 'sbx-clean'):
        ledger.record_created(_snapshot(sandbox_id))
    control = _FakeControl({sandbox_id: 'running' for sandbox_id in ledger.records})
    control.stuck.add('sbx-stuck')
    with pytest.raises(RuntimeError, match='sbx-stuck'):
        await ledger.cleanup_and_audit(control, timeout=0.01)
    assert control.calls == ['sbx-stuck', 'sbx-clean']
    data = json.loads(ledger.path.read_text())
    assert not data['audit_complete']
    assert {record['id'] for record in data['resources']} == {'sbx-stuck', 'sbx-clean'}
    assert ledger.records['sbx-clean'].terminal


async def test_cleanup_refuses_an_unowned_sandbox(tmp_path: Path) -> None:
    ledger = ResourceLedger(tmp_path / 'resources.json')
    control = _FakeControl({'sbx-external': 'running'})
    with pytest.raises(ValueError, match='not created by this test run'):
        await ledger.terminate_owned(control, 'sbx-external')
    assert control.calls == []
