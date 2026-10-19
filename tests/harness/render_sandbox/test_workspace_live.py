"""Run the released public WorkspaceBackendSuite without editing its tests.

Live execution is deliberately opt-in:

    RENDER_SANDBOX_LIVE=1 uv run --no-sync python -m pytest -q -rs \\
        tests/harness/render_sandbox/test_workspace_live.py::TestLiveRenderSandboxBackend

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

import os
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from ...conftest import try_import

with try_import() as imports_successful:
    from render import RenderAsync

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
    from pydantic_ai_harness.render_sandbox import RenderSandboxBackend

    client, _ = live_resources
    # The full suite shares one sandbox across many operations. Teardown terminates
    # it immediately; the default one-hour lifetime is a backstop for an interrupted host.
    return RenderSandboxBackend(client=client)


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
        from pydantic_ai_harness.render_sandbox import RenderSandboxBackend

        client, _ = live_resources

        def fresh() -> WorkspaceBackend:
            backend = RenderSandboxBackend(client=client)
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
        from pydantic_ai_harness.render_sandbox import RenderSandboxBackend

        client, _ = live_resources
        return lambda ref: RenderSandboxBackend(client=client, ref=ref)

    @pytest.fixture
    def destroy_environment(
        self, live_resources: tuple[RenderAsync, ResourceLedger]
    ) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        client, ledger = live_resources

        async def destroy(backend: WorkspaceBackend) -> None:
            assert backend.ref is not None
            await ledger.terminate_owned(client.experimental.sandboxes, backend.ref.id)

        return destroy
