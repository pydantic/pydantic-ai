"""`KubernetesSandbox` and the backend behaviour the conformance suite does not reach."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import anyio
import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import (
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)

from ...conftest import try_import

with try_import() as imports_successful:
    import grpc
    import grpc.aio
    from k8s_agent_sandbox.exceptions import SandboxClaimFailedError, SandboxNotFoundError
    from k8s_agent_sandbox.models import (
        SandboxdInClusterConnectionConfig,
        SandboxDirectConnectionConfig,
        SandboxdPodTunnelConnectionConfig,
    )

    from pydantic_ai_harness import KubernetesSandbox, KubernetesSandboxBackend

    from .fake_sandboxd import FakeCluster, FakeSandboxd

pytestmark = pytest.mark.skipif(not imports_successful(), reason='pydantic-ai-harness[kubernetes] not installed')

asyncio_only = pytest.mark.parametrize('anyio_backend', ['asyncio'])


def _sandboxd(cluster: FakeCluster, ref: WorkspaceRef | None) -> FakeSandboxd:
    assert ref is not None
    namespace, _, claim_name = ref.id.partition('/')
    return cluster.claims[namespace, claim_name]


def test_refs_name_a_namespace_and_a_claim() -> None:
    assert KubernetesSandboxBackend(ref=WorkspaceRef(provider='kubernetes', id='agents/claim')).ref is not None
    for bad in ('claim', '/claim', 'agents/', 'a/b/c'):
        with pytest.raises(ValueError, match='<namespace>/<claim name>'):
            KubernetesSandboxBackend(ref=WorkspaceRef(provider='kubernetes', id=bad))
    with pytest.raises(ValueError, match="expected 'kubernetes'"):
        KubernetesSandboxBackend(ref=WorkspaceRef(provider='modal', id='sb-1'))


@pytest.mark.parametrize('module', ['k8s_agent_sandbox', 'grpc'])
def test_missing_kubernetes_extra_has_an_install_hint(module: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, module, None)
    for name in [name for name in sys.modules if name.startswith('pydantic_ai_harness.kubernetes_sandbox')]:
        monkeypatch.delitem(sys.modules, name)
    with pytest.raises(ImportError, match=r'^Install `pydantic-ai-harness\[kubernetes\]` to use KubernetesSandbox\.$'):
        importlib.import_module('pydantic_ai_harness.kubernetes_sandbox')


def test_backend_arguments_are_checked_without_io() -> None:
    with pytest.raises(ValueError, match='pass `warm_pool`'):
        KubernetesSandboxBackend()
    with pytest.raises(TypeError, match='env keys and values must be strings'):
        KubernetesSandboxBackend(warm_pool='pool', env={'A': 1})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match='working_dir must be an absolute'):
        KubernetesSandboxBackend(warm_pool='pool', working_dir='relative')
    with pytest.raises(ValueError, match="connection must be one of 'port-forward'"):
        KubernetesSandboxBackend(warm_pool='pool', connection='gateway')  # type: ignore[arg-type]
    # The SDK's other configs reach its legacy runtime, not sandboxd.
    with pytest.raises(ValueError, match='got SandboxDirectConnectionConfig: commands run through sandboxd'):
        KubernetesSandboxBackend(warm_pool='pool', connection=SandboxDirectConnectionConfig(api_url='http://router'))  # type: ignore[arg-type]


def test_agent_spec_schema_publishes_the_spec_expressible_fields() -> None:
    """The SDK connection configs in `connection`'s annotation must not erase the whole schema entry."""
    definitions = AgentSpec.model_json_schema_with_capabilities([KubernetesSandbox])['$defs']
    params = definitions['spec_params_KubernetesSandbox']
    assert sorted(params['properties']) == [
        'connection',
        'description',
        'env',
        'id',
        'namespace',
        'sandbox_timeout',
        'warm_pool',
        'working_dir',
    ]
    assert params['required'] == ['warm_pool']


def test_from_spec_forwards_the_spec_fields() -> None:
    spec_fields: dict[str, Any] = {
        'id': 'kubernetes',
        'description': 'a sandbox',
        'warm_pool': 'python-pool',
        'namespace': 'agents',
        'connection': 'in-cluster-service',
        'sandbox_timeout': 600,
        'working_dir': '/work',
        'env': {'A': '1'},
    }
    assert KubernetesSandbox[None].from_spec(**spec_fields) == KubernetesSandbox(**spec_fields)


@asyncio_only
async def test_the_sdk_client_matches_the_connection(cluster: FakeCluster) -> None:
    other_ports = SandboxdPodTunnelConnectionConfig(grpc_port=9191, rest_port=8181)
    for connection in ('port-forward', 'in-cluster-pod-ip', other_ports):
        backend = KubernetesSandboxBackend(warm_pool='pool', connection=connection)
        await backend.run(['true'])
        await backend.aclose()
    tunnel, in_cluster, configured = cluster.clients
    assert isinstance(tunnel.connection_config, SandboxdPodTunnelConnectionConfig)
    assert isinstance(in_cluster.connection_config, SandboxdInClusterConnectionConfig)
    assert in_cluster.connection_config.mode == 'in-cluster-pod-ip'
    assert configured.connection_config is other_ports
    # The SDK would otherwise delete its claims when the interpreter exits.
    assert not tunnel.cleanup and not in_cluster.cleanup and not configured.cleanup


@pytest.mark.parametrize('anyio_backend', ['trio'])
async def test_trio_is_refused() -> None:
    backend = KubernetesSandboxBackend(warm_pool='pool')
    with pytest.raises(UserError, match='needs asyncio'):
        await backend.run(['true'])
    with pytest.raises(UserError, match='needs asyncio'):
        await backend.get_sandbox()
    with pytest.raises(UserError, match='needs asyncio'):
        await KubernetesSandbox(warm_pool='pool').destroy(WorkspaceRef(provider='kubernetes', id='agents/claim'))


@asyncio_only
async def test_a_claim_uses_the_pool_namespace_and_timeout(
    cluster: FakeCluster, caplog: pytest.LogCaptureFixture
) -> None:
    backend = KubernetesSandboxBackend(warm_pool='pool', namespace='agents', sandbox_timeout=60)
    with caplog.at_level('INFO', logger='pydantic_ai_harness.kubernetes_sandbox._backend'):
        await backend.run(['true'])
    assert backend.ref is not None and backend.ref.id.startswith('agents/sandbox-claim-')
    [(pool, namespace, kwargs)] = cluster.created
    assert (pool, namespace, kwargs['shutdown_after_seconds']) == ('pool', 'agents', 60)
    # Logged when claimed, so a claim whose run ends before its ref is stored can still be found.
    assert caplog.messages == [f'Claimed Kubernetes sandbox {backend.ref.id}']
    sandbox = await backend.get_sandbox()
    assert sandbox.claim_name == backend.ref.id.split('/')[1]
    await backend.aclose()
    await backend.aclose()  # Nothing left to close.


@asyncio_only
async def test_env_is_layered(cluster: FakeCluster) -> None:
    backend = KubernetesSandboxBackend(warm_pool='pool', env={'A': 'base', 'B': 'base'})
    assert (await backend.run('printf "$A $B"', shell=True, env={'B': 'call'})).stdout == 'base call'
    with pytest.raises(TypeError, match='env keys and values must be strings'):
        await backend.run(['true'], env={'B': 1})  # type: ignore[dict-item]
    await backend.aclose()


@asyncio_only
@pytest.mark.parametrize('working_dir', [None, '/tmp'])
async def test_each_argument_keeps_its_own_size_limit(cluster: FakeCluster, working_dir: str | None) -> None:
    """Two 100 KB arguments with quotes in them run as given.

    Linux limits each `execve` string to 128 KiB. Quoted into one `/bin/sh -c` script, these two
    would pass that limit together, and each `'` would take five characters.
    """
    backend = KubernetesSandboxBackend(warm_pool='pool', working_dir=working_dir)
    first, second = "'a" * 50_000, '"b' * 50_000
    result = await backend.run(['sh', '-c', 'printf %s "$1" "$2"', 'sh', first, second])
    assert result.exit_code == 0 and result.stdout == first + second
    # One argument past the limit fails, as it would in an argv.
    with pytest.raises(RuntimeError, match='argument list too long'):
        await backend.run(['printf', '%s', 'x' * 200_000])
    await backend.aclose()


@asyncio_only
async def test_the_argument_variables_do_not_reach_the_command(cluster: FakeCluster) -> None:
    backend = KubernetesSandboxBackend(warm_pool='pool')
    result = await backend.run(['sh', '-c', 'env | grep -c PYDANTIC_AI_HARNESS_ARG_ || true'])
    assert result.stdout == '0\n'
    await backend.aclose()


@asyncio_only
async def test_aclose_in_a_cancelled_scope_still_closes(cluster: FakeCluster) -> None:
    backend = KubernetesSandboxBackend(warm_pool='pool')
    await backend.run(['true'])
    # As in a `finally:` after the caller was cancelled.
    with anyio.CancelScope() as scope:
        scope.cancel()
        await backend.aclose()
    [client] = cluster.clients
    assert client.closed


@asyncio_only
async def test_working_dir_is_reported_as_the_sandbox_resolves_it(cluster: FakeCluster) -> None:
    probe = KubernetesSandboxBackend(warm_pool='pool')
    root = await probe.working_dir()
    assert (await probe.run(['mkdir', f'{root}/sub'])).exit_code == 0
    assert (await probe.run(['ln', '-s', f'{root}/sub', f'{root}/link'])).exit_code == 0
    via_link = KubernetesSandboxBackend(ref=probe.ref, working_dir=f'{root}/link')
    assert await via_link.working_dir() == f'{root}/sub'
    for each in (probe, via_link):
        await each.aclose()


@asyncio_only
async def test_commands_start_in_working_dir(cluster: FakeCluster) -> None:
    probe = KubernetesSandboxBackend(warm_pool='pool')
    root = await probe.working_dir()
    assert (await probe.run(['mkdir', f'{root}/sub'])).exit_code == 0
    backend = KubernetesSandboxBackend(ref=probe.ref, working_dir=f'{root}/sub')
    assert (await backend.run(['pwd', '-P'])).stdout == f'{root}/sub\n'
    assert (await backend.run(['missing-program'])).exit_code == 127
    gone = KubernetesSandboxBackend(ref=probe.ref, working_dir=f'{root}/gone')
    with pytest.raises(WorkspaceError, match=f"working_dir '{root}/gone' does not exist"):
        await gone.run(['true'])
    for each in (probe, backend, gone):
        await each.aclose()


@asyncio_only
async def test_an_unreadable_working_dir_is_unavailable(cluster: FakeCluster) -> None:
    backend = KubernetesSandboxBackend(warm_pool='pool')
    await backend.run(['true'])
    _sandboxd(cluster, backend.ref).root.rmdir()
    with pytest.raises(WorkspaceUnavailableError, match='working directory'):
        await backend.working_dir()
    await backend.aclose()


@asyncio_only
async def test_an_unknown_warm_pool_or_failed_claim_is_unavailable(cluster: FakeCluster) -> None:
    with pytest.raises(WorkspaceUnavailableError, match="warm pool 'missing'"):
        await KubernetesSandboxBackend(warm_pool='missing').run(['true'])
    cluster.create_error = SandboxClaimFailedError('SandboxClaim failed: PodFailed')
    backend = KubernetesSandboxBackend(warm_pool='pool')
    with pytest.raises(WorkspaceUnavailableError, match='PodFailed'):
        await backend.run(['true'])
    assert backend.ref is None
    clients = len(cluster.clients)
    # The next operation claims again, with the same client.
    cluster.create_error = None
    assert (await backend.run(['true'])).exit_code == 0
    assert backend.ref is not None and len(cluster.clients) == clients
    await backend.aclose()


@asyncio_only
async def test_a_claim_cancelled_while_it_is_made_is_deleted(cluster: FakeCluster) -> None:
    cluster.create_gate = anyio.Event()
    backend = KubernetesSandboxBackend(warm_pool='pool', namespace='agents')
    async with anyio.create_task_group() as group:
        group.start_soon(backend.run, ['true'])
        await cluster.create_started.wait()
        group.cancel_scope.cancel()
    assert cluster.rolled_back and not cluster.claims
    assert backend.ref is None
    await backend.aclose()


async def _wait_for_file(path: Path) -> None:
    with anyio.fail_after(30):
        while not path.exists():
            await anyio.sleep(0.01)


@asyncio_only
async def test_aclose_waits_for_commands_in_flight(cluster: FakeCluster) -> None:
    backend = KubernetesSandboxBackend(warm_pool='pool')
    root = Path(await backend.working_dir())
    results: list[str] = []

    async def wait_for_go() -> None:
        results.append(
            (await backend.run('touch started; until [ -e go ]; do sleep 0.01; done; echo done', shell=True)).stdout
        )

    async with anyio.create_task_group() as group:
        group.start_soon(wait_for_go)
        await _wait_for_file(root / 'started')
        await backend.aclose()
        assert not cluster.clients[0].closed  # Closing now would cancel the command.
        (root / 'go').touch()
    assert results == ['done\n'] and cluster.clients[0].closed


@asyncio_only
async def test_a_deferred_aclose_closes_only_the_connection_in_use(cluster: FakeCluster) -> None:
    """After the deferred close, later operations share one connection again until the next `aclose()`."""
    backend = KubernetesSandboxBackend(warm_pool='pool')
    root = Path(await backend.working_dir())

    async def wait_for_go() -> None:
        await backend.run('touch started; until [ -e go ]; do sleep 0.01; done', shell=True)

    async with anyio.create_task_group() as group:
        group.start_soon(wait_for_go)
        await _wait_for_file(root / 'started')
        await backend.aclose()
        (root / 'go').touch()
    assert cluster.clients[0].closed

    await backend.run('true', shell=True)
    await backend.run('true', shell=True)
    assert len(cluster.clients) == 2 and not cluster.clients[1].closed
    await backend.aclose()
    assert cluster.clients[1].closed


@asyncio_only
async def test_a_closed_connection_mid_command_is_a_connection_error(cluster: FakeCluster) -> None:
    backend = KubernetesSandboxBackend(warm_pool='pool')
    root = Path(await backend.working_dir())
    errors: list[Exception] = []

    async def run_until_closed() -> None:
        try:
            await backend.run('touch started; sleep 30', shell=True)
        except Exception as error:
            errors.append(error)

    async with anyio.create_task_group() as group:
        group.start_soon(run_until_closed)
        await _wait_for_file(root / 'started')
        # As when another command's lost connection drops the channel they share.
        [sandbox] = cluster.clients[0].sandboxes
        assert sandbox.connector.channel is not None
        await sandbox.connector.channel.close()
    # Not a cancellation, which a task group would take as its own and drop without a trace.
    [error] = errors
    assert isinstance(error, ConnectionError) and 'closed while the command was running' in str(error)
    await backend.aclose()


@asyncio_only
async def test_a_failed_lookup_of_an_existing_claim_propagates(cluster: FakeCluster) -> None:
    probe = KubernetesSandboxBackend(warm_pool='pool')
    await probe.run(['true'])
    assert probe.ref is not None
    cluster.lookup_error = True
    # Not `WorkspaceUnavailableError`: the claim still exists, so a retry may attach to it.
    with pytest.raises(SandboxNotFoundError):
        await KubernetesSandboxBackend(ref=probe.ref).run(['true'])
    await probe.aclose()


@asyncio_only
async def test_a_dropped_connection_is_reopened(cluster: FakeCluster) -> None:
    backend = KubernetesSandboxBackend(warm_pool='pool')
    await backend.run(['true'])
    cluster.drop_connections()
    cluster.reconnect_fails = True
    # The SDK re-raises gRPC errors as `RuntimeError`, keeping the original as the cause.
    with pytest.raises(RuntimeError, match='UNAVAILABLE') as caught:
        await backend.run(['true'])
    cause = caught.value.__cause__
    assert isinstance(cause, grpc.aio.AioRpcError) and cause.code() == grpc.StatusCode.UNAVAILABLE
    # The SDK drops the unreachable connection from its cache.
    assert len(cluster.invalidated) == 1
    cluster.reconnect_fails = False
    assert (await backend.run(['true'])).exit_code == 0
    # Any other gRPC error leaves the connection cached.
    _sandboxd(cluster, backend.ref).start_error = grpc.StatusCode.INTERNAL
    with pytest.raises(RuntimeError, match='INTERNAL'):
        await backend.run(['true'])
    assert len(cluster.invalidated) == 1
    await backend.aclose()


@asyncio_only
async def test_a_lost_connection_to_a_live_sandbox_propagates(
    cluster: FakeCluster, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = KubernetesSandboxBackend(warm_pool='pool')
    await backend.run(['true'])
    assert backend.ref is not None
    cluster.channel_error = ConnectionError('port-forward dropped')
    # The Sandbox still exists, so the error is transient and left for durable engines to retry.
    with pytest.raises(ConnectionError):
        await backend.run(['true'])
    # An API server that cannot be asked, or does not answer, leaves the original error too, and says so.
    cluster.sandbox_lookup_error = RuntimeError('API server unreachable')
    with caplog.at_level('WARNING', logger='pydantic_ai_harness.kubernetes_sandbox._backend'):
        with pytest.raises(ConnectionError):
            await backend.run(['true'])
        cluster.sandbox_lookup_error = None
        cluster.sandbox_lookup_gate = anyio.Event()
        monkeypatch.setattr('pydantic_ai_harness.kubernetes_sandbox._backend._GONE_CHECK_TIMEOUT', 0.1)
        with pytest.raises(ConnectionError):
            await backend.run(['true'])
    assert [record.getMessage() for record in caplog.records] == [
        f'Could not ask the API server whether Kubernetes sandbox {backend.ref.id} still exists',
        f'The API server did not say within 0.1 seconds whether Kubernetes sandbox {backend.ref.id} exists',
    ]
    await backend.aclose()


@asyncio_only
@pytest.mark.parametrize('negative', [False, True], ids=['128+n', 'sandboxd-before-1.0.7'])
async def test_a_signal_exit_in_a_live_sandbox_keeps_its_exit_code(cluster: FakeCluster, negative: bool) -> None:
    backend = KubernetesSandboxBackend(warm_pool='pool')
    await backend.run(['true'])
    _sandboxd(cluster, backend.ref).negative_signal_exits = negative
    assert (await backend.run(['sh', '-c', 'kill -TERM $$'])).exit_code == (-1 if negative else 143)
    await backend.aclose()


@asyncio_only
@pytest.mark.parametrize('negative', [False, True], ids=['128+n', 'sandboxd-before-1.0.7'])
async def test_a_signal_exit_in_a_deleted_sandbox_is_unavailable(cluster: FakeCluster, negative: bool) -> None:
    backend = KubernetesSandboxBackend(warm_pool='pool')
    await backend.run(['true'])
    assert backend.ref is not None
    sandboxd = _sandboxd(cluster, backend.ref)
    sandboxd.negative_signal_exits = negative
    # The API server no longer has the Sandbox, while its sandboxd still answers until the pod stops.
    namespace, claim_name = backend.ref.id.split('/')
    del cluster.claims[namespace, claim_name]
    with pytest.raises(WorkspaceUnavailableError, match='is gone: its Sandbox was deleted'):
        await backend.run(['sh', '-c', 'kill -TERM $$'])
    await backend.aclose()
    await sandboxd.stop()


@asyncio_only
async def test_a_program_that_cannot_run_exits_126(cluster: FakeCluster) -> None:
    backend = KubernetesSandboxBackend(warm_pool='pool')
    root = await backend.working_dir()
    assert (await backend.run(['touch', f'{root}/not-executable'])).exit_code == 0
    assert (await backend.run([f'{root}/not-executable'])).exit_code == 126
    await backend.aclose()


@asyncio_only
async def test_a_stream_without_an_exit_event_is_an_error(cluster: FakeCluster) -> None:
    backend = KubernetesSandboxBackend(warm_pool='pool')
    await backend.run(['true'])
    _sandboxd(cluster, backend.ref).send_exit = False
    with pytest.raises(RuntimeError, match='ended without an Exit event'):
        await backend.run(['true'])
    await backend.aclose()


@asyncio_only
async def test_a_timeout_keeps_the_output_so_far(cluster: FakeCluster) -> None:
    backend = KubernetesSandboxBackend(warm_pool='pool')
    with pytest.raises(WorkspaceTimeoutError) as info:
        await backend.run('echo before; sleep 30', shell=True, timeout=0.1)
    assert info.value.stdout == 'before\n'
    await backend.aclose()


@asyncio_only
async def test_output_past_the_limit_stops_the_command(cluster: FakeCluster, monkeypatch: pytest.MonkeyPatch) -> None:
    # A small cap proves the same behavior without streaming 10 MiB through the fake.
    monkeypatch.setattr('pydantic_ai_harness.kubernetes_sandbox._backend._MAX_OUTPUT_BYTES', 4096)
    monkeypatch.setattr('pydantic_ai_harness.kubernetes_sandbox._backend._OUTPUT_PREVIEW_BYTES', 1024)
    backend = KubernetesSandboxBackend(warm_pool='pool')
    with pytest.raises(WorkspaceOutputLimitError, match='exceeded the 10 MiB limit') as caught:
        await backend.run('head -c 5000 /dev/zero | tr "\\0" x; sleep 30', shell=True)
    assert caught.value.limit == 4096
    assert caught.value.stdout == 'x' * 1024 and caught.value.stderr == ''
    await backend.aclose()


@asyncio_only
async def test_destroying_a_missing_claim_returns_quietly(cluster: FakeCluster) -> None:
    capability = KubernetesSandbox(warm_pool='pool')
    await capability.destroy(WorkspaceRef(provider='kubernetes', id='agents/gone'))


@asyncio_only
async def test_a_failed_destroy_raises(cluster: FakeCluster) -> None:
    backend = KubernetesSandboxBackend(warm_pool='pool')
    await backend.run(['true'])
    assert backend.ref is not None
    cluster.delete_error = RuntimeError('403 Forbidden')
    with pytest.raises(RuntimeError, match='Forbidden'):
        await KubernetesSandbox(warm_pool='pool').destroy(backend.ref)
    await backend.aclose()


def test_capability_arguments_are_checked() -> None:
    with pytest.raises(UserError, match='defer_loading'):
        KubernetesSandbox(warm_pool='pool', defer_loading=True)
    with pytest.raises(UserError, match='sandbox_timeout'):
        KubernetesSandbox(warm_pool='pool', sandbox_timeout=0)
    with pytest.raises(UserError, match='working_dir'):
        KubernetesSandbox(warm_pool='pool', working_dir='relative')
    with pytest.raises(UserError, match="connection must be one of 'port-forward'"):
        KubernetesSandbox(warm_pool='pool', connection='gateway')  # type: ignore[arg-type]
    assert KubernetesSandbox(warm_pool='pool', sandbox_timeout=None).sandbox_timeout is None


def _ctx() -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage(), run_id='run-1')


def test_get_workspace_only_takes_kubernetes_refs() -> None:
    capability = KubernetesSandbox(warm_pool='pool', namespace='agents')
    ctx = _ctx()
    assert capability.get_workspace(ctx, ref=WorkspaceRef(provider='modal', id='sb')) is None
    backend = capability.get_workspace(ctx, ref=None)
    assert isinstance(backend, KubernetesSandboxBackend) and backend.ref is None
    attached = capability.backend(WorkspaceRef(provider='kubernetes', id='agents/claim'))
    assert type(attached) is KubernetesSandboxBackend and attached.ref is not None


@asyncio_only
async def test_a_run_keeps_its_connection_open_and_closes_it_at_the_end(cluster: FakeCluster) -> None:
    capability = KubernetesSandbox(warm_pool='pool', namespace='agents')
    agent = Agent(TestModel(), capabilities=[capability])

    @agent.tool
    async def write_note(ctx: RunContext[object]) -> str:
        await ctx.workspace.run('echo hi > note.txt', shell=True)
        return (await ctx.workspace.run(['cat', 'note.txt'])).stdout

    result = await agent.run('write a note')
    # One client for the whole run, closed when the run ended; the sandbox keeps running.
    [client] = cluster.clients
    assert client.closed and result.output and 'hi' in result.output
    ref = result.workspace.ref
    assert ref is not None
    await capability.destroy(ref)


@asyncio_only
async def test_a_cancelled_run_still_closes_its_connection(cluster: FakeCluster) -> None:
    agent = Agent(TestModel(), capabilities=[KubernetesSandbox(warm_pool='pool')])
    used = anyio.Event()

    @agent.tool
    async def hang(ctx: RunContext[object]) -> None:
        await ctx.workspace.run(['true'])
        used.set()
        await anyio.sleep_forever()

    async with anyio.create_task_group() as group:
        group.start_soon(agent.run, 'hang')
        await used.wait()
        group.cancel_scope.cancel()
    [client] = cluster.clients
    assert client.closed


@asyncio_only
async def test_a_backend_outside_a_run_closes_after_each_operation(cluster: FakeCluster) -> None:
    ctx = _ctx()
    backend = KubernetesSandbox(warm_pool='pool').get_workspace(ctx, ref=None)
    assert isinstance(backend, KubernetesSandboxBackend)
    await backend.run(['true'])
    await backend.run(['true'])
    # Nothing would call `aclose()` (Temporal builds one per activity), so each operation closed its own client.
    assert [client.closed for client in cluster.clients] == [True, True]


@asyncio_only
async def test_a_subagent_does_not_close_its_parents_workspace(cluster: FakeCluster) -> None:
    capability = KubernetesSandbox(warm_pool='pool')
    child = Agent(TestModel(), capabilities=[capability])

    @child.tool
    async def touch(ctx: RunContext[object]) -> None:
        await ctx.workspace.run(['touch', 'from-child'])

    parent = Agent(TestModel(), capabilities=[capability])

    @parent.tool
    async def delegate(ctx: RunContext[object]) -> str:
        await ctx.workspace.run(['true'])
        await child.run('child', workspace=ctx.workspace)
        return (await ctx.workspace.run(['ls'])).stdout

    await parent.run('go')
    # The child used the parent's connection and left it open; the parent's run closed it.
    [client] = cluster.clients
    assert client.closed
