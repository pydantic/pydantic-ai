from __future__ import annotations

import asyncio
import json
import signal
from collections.abc import AsyncIterator
from pathlib import Path

import anyio
import httpx
import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import (
    Workspace,
    WorkspaceError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from sprites import AsyncSprite
from sprites.exceptions import (
    APIError,
    AuthenticationError,
    FileNotFoundError_,
    FilesystemError,
    NetworkError,
    NotFoundError,
    SpriteError,
)
from websockets.datastructures import Headers
from websockets.exceptions import InvalidMessage, InvalidStatus
from websockets.http11 import Response

from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.sprites_sandbox import SpritesSandbox, SpritesSandboxBackend

from .conftest import live_token
from .fake_sprites import SpriteTransport

pytestmark = pytest.mark.anyio


def context(conversation: str = 'chat') -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage(), conversation_id=conversation, run_id='run')


def _caused_by(error: Exception, cause: Exception) -> Exception:
    error.__cause__ = cause
    return error


def _handshake(status: int) -> InvalidStatus:
    return InvalidStatus(Response(status, 'status', Headers()))


class TestSpritesSandbox:
    def test_env_values_are_not_represented(self) -> None:
        secret = 'secret-environment-value-123'
        capability = SpritesSandbox[None](env={'API_KEY': secret})
        backend = capability.get_workspace(context(), ref=None)
        assert backend is not None
        for value in (repr(capability), str(capability), repr(backend), str(backend)):
            assert secret not in value

    async def test_construction_is_lazy_and_first_use_is_shared(self, transport: SpriteTransport) -> None:
        backend = SpritesSandbox[None]().get_workspace(context(), ref=None)
        assert isinstance(backend, SpritesSandboxBackend)
        assert transport.clients == []
        first, second = await asyncio.gather(backend.get_sandbox(), backend.get_sandbox())
        assert first is second
        assert transport.created == [first.name]
        assert backend.ref == WorkspaceRef(provider='sprites', id=first.name)

    @pytest.mark.parametrize(
        'kwargs',
        [
            {'working_dir': 'relative'},
            {'defer_loading': True},
        ],
    )
    def test_invalid_configuration_fails_at_construction(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(UserError, match=f'^{next(iter(kwargs))} must be'):
            SpritesSandbox[None](**kwargs)  # pyright: ignore[reportArgumentType]

    async def test_cancelled_creation_still_names_the_sprite_and_a_retry_attaches(
        self, transport: SpriteTransport
    ) -> None:
        backend = SpritesSandboxBackend()
        transport.release_create = asyncio.Event()

        async def acquire() -> AsyncSprite:
            return await backend.get_sandbox()

        task = asyncio.create_task(acquire())
        await transport.create_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        transport.release_create.set()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert backend.ref == WorkspaceRef(provider='sprites', id=transport.created[0])
        assert (await backend.get_sandbox()).name == transport.created[0]
        assert len(transport.created) == 1

    def test_foreign_reference_is_declined_and_backend_rejects_it(self) -> None:
        assert SpritesSandbox[None]().get_workspace(context(), ref=WorkspaceRef(provider='other', id='x')) is None
        with pytest.raises(ValueError, match="expected 'sprites'"):
            SpritesSandboxBackend(ref=WorkspaceRef(provider='other', id='x'))

    async def test_destroy_ref_uses_id_without_attaching(self, transport: SpriteTransport) -> None:
        provider = SpritesSandbox[None]()
        transport.names.add('target')
        ref = WorkspaceRef(provider='sprites', id='target')
        backend = provider.backend(ref)
        assert isinstance(backend, SpritesSandboxBackend)
        assert backend.ref == ref
        await provider.destroy(ref)
        assert 'target' not in transport.names
        assert transport.execs == []
        assert transport.close_calls == 1
        with pytest.raises(ValueError, match='provider'):
            await provider.destroy(WorkspaceRef(provider='other', id='target'))

    async def test_native_handle_conflict_and_identity(self, transport: SpriteTransport) -> None:
        seed = SpritesSandboxBackend()
        native = await seed.get_sandbox()
        backend = SpritesSandboxBackend(sandbox=native)
        assert await backend.get_sandbox() is native
        assert backend.ref == WorkspaceRef(provider='sprites', id=native.name)
        with pytest.raises(ValueError, match='either `sandbox` or `ref`'):
            SpritesSandboxBackend(sandbox=native, ref=backend.ref)

    async def test_agent_without_workspace_use_does_not_create(self, transport: SpriteTransport) -> None:
        result = await Agent(TestModel(custom_output_text='done'), capabilities=[SpritesSandbox()]).run('go')
        assert result.output == 'done'
        assert transport.created == []

    async def test_run_end_closes_the_owned_client_and_the_result_reattaches(self, transport: SpriteTransport) -> None:
        agent = Agent(TestModel(), capabilities=[SpritesSandbox()])

        @agent.tool
        async def write(ctx: RunContext[object]) -> str:
            await ctx.workspace.write_bytes('result.bin', b'\x00\xff\n')
            return 'written'

        result = await agent.run('write')
        assert transport.close_calls == 1

        assert await result.workspace.read_bytes('result.bin') == b'\x00\xff\n'
        assert len(transport.clients) == 2
        assert len(transport.names) == 1

    async def test_cancelled_run_still_closes_the_owned_client(self, transport: SpriteTransport) -> None:
        agent = Agent(TestModel(), capabilities=[SpritesSandbox()])
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

        assert transport.close_calls == 1

    async def test_run_end_leaves_caller_clients_and_backends_open(self, transport: SpriteTransport) -> None:
        client = transport.client('test-token')
        agent = Agent(TestModel(), capabilities=[SpritesSandbox(client=client)])

        @agent.tool
        async def touch(ctx: RunContext[object]) -> str:
            await ctx.workspace.write_text('touched', '')
            return 'touched'

        await agent.run('touch')
        explicit = SpritesSandboxBackend()
        await agent.run('touch', workspace=explicit)

        assert transport.close_calls == 0
        assert len(transport.clients) == 2

    async def test_a_subagent_run_leaves_the_parent_runs_backend_open(self, transport: SpriteTransport) -> None:
        child = Agent(TestModel(), capabilities=[SpritesSandbox()])

        @child.tool
        async def child_echo(ctx: RunContext[object]) -> str:
            return (await ctx.workspace.run(['echo', 'child'])).stdout

        parent = Agent(TestModel(), capabilities=[SpritesSandbox()])

        @parent.tool
        async def delegate(ctx: RunContext[object]) -> int:
            await ctx.workspace.run(['true'])
            await child.run('go', workspace=ctx.workspace)
            return transport.close_calls

        result = await parent.run('go')

        assert '"delegate":0' in result.output
        assert transport.close_calls == 1

    async def test_coder_tools_run_in_the_sprite(self, transport: SpriteTransport) -> None:
        async def model(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
            returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
            if not returns:
                yield {0: DeltaToolCall('write_file', json.dumps({'path': 'notes.txt', 'content': 'hello'}))}
            elif len(returns) == 1:
                yield {0: DeltaToolCall('shell', json.dumps({'command': 'cat notes.txt'}))}
            else:
                yield str(returns[-1].content)

        # Coder's members stream, so the scripted model is a stream function.
        agent = Agent(FunctionModel(stream_function=model), capabilities=[SpritesSandbox(), Coder()])
        result = await agent.run('go')

        assert result.output.startswith('hello\n')
        assert '"exit_code": 0' in result.output
        assert (transport.root / 'notes.txt').read_text() == 'hello'

    async def test_missing_reference_does_not_recreate(self, transport: SpriteTransport) -> None:
        backend = SpritesSandboxBackend(ref=WorkspaceRef(provider='sprites', id='missing'))
        with pytest.raises(WorkspaceUnavailableError, match="Sprite 'missing' no longer exists"):
            await backend.get_sandbox()
        assert transport.created == []

    @pytest.mark.parametrize(
        'stage,error,expected',
        [
            ('acquire', AuthenticationError('bad token'), WorkspaceUnavailableError),
            ('acquire', NotFoundError('gone'), WorkspaceUnavailableError),
            ('acquire', SpriteError('Failed get sprite (status 400)'), WorkspaceError),
            ('acquire', SpriteError('Failed get sprite (status 429): rate limited'), None),
            ('acquire', SpriteError('Failed get sprite (status 503): unavailable'), None),
            ('acquire', NetworkError('reset'), None),
            ('connect', _handshake(401), WorkspaceUnavailableError),
            ('connect', _handshake(404), WorkspaceUnavailableError),
            ('connect', _handshake(429), None),
            ('connect', _handshake(500), None),
            ('connect', ConnectionResetError('reset'), None),
        ],
    )
    async def test_errors_are_mapped_or_propagate(
        self,
        transport: SpriteTransport,
        stage: str,
        error: Exception,
        expected: type[WorkspaceError] | None,
    ) -> None:
        transport.names.add('target')
        backend = SpritesSandboxBackend(ref=WorkspaceRef(provider='sprites', id='target'))
        if stage == 'acquire':
            transport.get_error = error
        else:
            transport.connect_error = error

        with pytest.raises(Exception) as caught:
            await backend.run(['true'])

        if expected is not None:
            assert type(caught.value) is expected
        cause = caught.value if expected is None else caught.value.__cause__
        if isinstance(error, InvalidStatus):
            # The SDK turns a failed exec handshake into an `APIError` with its status.
            assert isinstance(cause, APIError)
            assert cause.status_code == error.response.status_code
        else:
            assert cause is error

    @pytest.mark.parametrize(
        'error,expected',
        [
            (SpriteError('Failed create sprite (status 400): unknown runtime'), WorkspaceUnavailableError),
            (NotFoundError('Resource not found for create sprite'), WorkspaceUnavailableError),
            (AuthenticationError('bad token'), WorkspaceUnavailableError),
            (SpriteError('Failed create sprite (status 429): rate limited'), None),
            (SpriteError('Failed create sprite (status 502): bad gateway'), None),
            (NetworkError('reset'), None),
        ],
    )
    async def test_refused_creation_is_unavailable(
        self, transport: SpriteTransport, error: Exception, expected: type[WorkspaceError] | None
    ) -> None:
        transport.create_error = error
        with pytest.raises(Exception) as caught:
            await SpritesSandboxBackend(runtime='nope').run(['true'])

        if expected is None:
            assert caught.value is error
        else:
            assert type(caught.value) is expected
            assert caught.value.__cause__ is error
            if not isinstance(error, AuthenticationError):
                assert str(caught.value) == f'Could not start Sprites sandbox: {error}'

    async def test_lost_create_reply_recovers_same_sprite(self, transport: SpriteTransport) -> None:
        backend = SpritesSandboxBackend()
        transport.create_error_after_commit = NetworkError('lost reply')
        sprite = await backend.get_sandbox()
        assert backend.ref == WorkspaceRef(provider='sprites', id=sprite.name)
        assert transport.created == [sprite.name]
        assert await backend.get_sandbox() is sprite

    async def test_lost_create_reply_before_lookup_visible_preserves_cleanup_ref(
        self, transport: SpriteTransport
    ) -> None:
        backend = SpritesSandboxBackend()
        transport.create_error_after_commit = NetworkError('lost reply')
        transport.get_error_once = NotFoundError('not visible yet')
        with pytest.raises(NetworkError, match='lost reply'):
            await backend.get_sandbox()
        [name] = transport.created
        assert backend.ref == WorkspaceRef(provider='sprites', id=name)
        assert (await backend.get_sandbox()).name == name
        assert transport.created == [name]

    async def test_auth_error_reports_safe_reason_without_token(self, transport: SpriteTransport) -> None:
        secret = 'secret-sprite-token-123'
        transport.get_error = AuthenticationError(f'credential expired: {secret}')
        backend = SpritesSandboxBackend(ref=WorkspaceRef(provider='sprites', id='target'))
        with pytest.raises(WorkspaceUnavailableError) as caught:
            await backend.get_sandbox()
        assert 'Credential expired' in str(caught.value)
        assert secret not in str(caught.value)

    async def test_missing_token(self, transport: SpriteTransport, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('SPRITE_TOKEN')
        with pytest.raises(WorkspaceUnavailableError, match='SPRITE_TOKEN'):
            await SpritesSandboxBackend().get_sandbox()

    @pytest.mark.parametrize('attach', [False, True], ids=['create', 'attach'])
    async def test_stalled_acquisition_propagates_as_a_transport_timeout(
        self, transport: SpriteTransport, monkeypatch: pytest.MonkeyPatch, attach: bool
    ) -> None:
        monkeypatch.setattr('pydantic_ai_harness.sprites_sandbox._backend._ACQUIRE_TIMEOUT', 0.05)
        transport.names.add('remote')
        transport.release_create = asyncio.Event()
        transport.release_get = asyncio.Event()
        backend = SpritesSandboxBackend(ref=WorkspaceRef(provider='sprites', id='remote') if attach else None)

        with pytest.raises(TimeoutError, match='control plane may be unreachable') as caught:
            await backend.run(['true'], timeout=30)

        assert not isinstance(caught.value, WorkspaceError)
        assert ('connection' if attach else 'creation') in str(caught.value)
        assert transport.execs == []

    async def test_run_deadline_starts_once_the_sprite_is_acquired(self, transport: SpriteTransport) -> None:
        transport.names.add('remote')
        transport.release_get = asyncio.Event()
        backend = SpritesSandboxBackend(ref=WorkspaceRef(provider='sprites', id='remote'))
        # Attaching outlasts the timeout; the command itself fits in it comfortably.
        task = asyncio.create_task(backend.run(['echo', 'ok'], timeout=1))
        await asyncio.sleep(1.1)
        transport.release_get.set()

        assert (await task).stdout == 'ok\n'

    async def test_failed_acquisition_keeps_the_client_for_a_retry(self, transport: SpriteTransport) -> None:
        transport.get_error = SpriteError('lookup failed')
        transport.names.add('remote')
        backend = SpritesSandboxBackend(ref=WorkspaceRef(provider='sprites', id='remote'))
        with pytest.raises(WorkspaceError):
            await backend.get_sandbox()
        transport.get_error = None
        assert (await backend.get_sandbox()).name == 'remote'
        assert (len(transport.clients), transport.close_calls) == (1, 0)

    async def test_owned_client_reads_sprite_token(
        self, transport: SpriteTransport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('SPRITE_TOKEN', 'sentinel')
        native = await SpritesSandboxBackend().get_sandbox()
        assert native.client.token == 'sentinel'

    async def test_aclose_leaves_an_injected_client_open(self, transport: SpriteTransport) -> None:
        injected = SpritesSandboxBackend(client=transport.client('test-token'))
        await injected.get_sandbox()
        await injected.aclose()
        assert transport.close_calls == 0

    async def test_failed_close_is_logged_and_retried_by_the_next_aclose(
        self, transport: SpriteTransport, caplog: pytest.LogCaptureFixture
    ) -> None:
        owned = SpritesSandboxBackend()
        await owned.get_sandbox()
        transport.close_error = RuntimeError('close failed')
        await owned.aclose()
        assert 'Could not close Sprites SDK client' in caplog.text
        await owned.aclose()
        await owned.aclose()
        assert transport.close_calls == 2

    async def test_cancelled_aclose_finishes_the_close_first(self, transport: SpriteTransport) -> None:
        owned = SpritesSandboxBackend()
        await owned.get_sandbox()
        transport.release_close = asyncio.Event()
        task = asyncio.create_task(owned.aclose())
        await transport.close_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        transport.release_close.set()

        with pytest.raises(asyncio.CancelledError):
            await task
        await owned.aclose()
        assert transport.close_calls == 1

    async def test_aclose_waits_for_an_in_flight_creation(self, transport: SpriteTransport) -> None:
        owned = SpritesSandboxBackend()
        transport.release_create = asyncio.Event()
        creating = asyncio.create_task(owned.get_sandbox())
        await transport.create_started.wait()
        closing = asyncio.create_task(owned.aclose())
        await asyncio.sleep(0)
        assert transport.close_calls == 0
        transport.release_create.set()
        await creating
        await closing

        assert transport.close_calls == 1
        assert (await owned.get_sandbox()).client is transport.clients[1]

    async def test_cancelled_command_finishes_closing_its_connection(self, transport: SpriteTransport) -> None:
        backend = SpritesSandboxBackend()
        await backend.get_sandbox()
        transport.release_exec_close = asyncio.Event()
        task = asyncio.create_task(backend.run(['true']))
        await transport.exec_close_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        transport.release_exec_close.set()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert transport.exec_closes == 1

    async def test_deleted_sprite_is_unavailable_to_attach_commands_and_files(self, transport: SpriteTransport) -> None:
        owner = SpritesSandboxBackend()
        native = await owner.get_sandbox()
        assert owner.ref is not None
        await native.delete()

        with pytest.raises(WorkspaceUnavailableError):
            await SpritesSandboxBackend(ref=owner.ref).working_dir()
        with pytest.raises(WorkspaceUnavailableError, match=native.name):
            await owner.run(['true'])
        with pytest.raises(WorkspaceUnavailableError):
            await Workspace(owner).read_bytes('/tmp/anything')
        with pytest.raises(WorkspaceUnavailableError):
            await Workspace(owner).write_bytes('/tmp/anything', b'x')

    @pytest.mark.parametrize('failure', ['error', 'hang'])
    async def test_close_failure_after_exit_returns_the_result_and_aborts_the_socket(
        self,
        transport: SpriteTransport,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        failure: str,
    ) -> None:
        backend = SpritesSandboxBackend()
        await backend.get_sandbox()
        if failure == 'error':
            transport.exec_close_error = RuntimeError('close failed')
        else:
            transport.exec_close_hang = True
            monkeypatch.setattr('pydantic_ai_harness.sprites_sandbox._backend._CLOSE_TIMEOUT', 0.01)

        result = await backend.run(['echo', 'ok'])

        assert (result.exit_code, result.stdout) == (0, 'ok\n')
        assert transport.aborted == 1
        assert 'Could not close a Sprite exec connection' in caplog.text

    async def test_deleted_mid_command_is_unavailable(self, transport: SpriteTransport) -> None:
        backend = SpritesSandboxBackend()
        sprite = await backend.get_sandbox()
        transport.exit_override = 137
        transport.delete_on_exit = True
        with pytest.raises(WorkspaceUnavailableError, match=sprite.name):
            await backend.run(['true'])

    async def test_sigkill_with_live_sprite_returns_exit(self, transport: SpriteTransport) -> None:
        backend = SpritesSandboxBackend()
        transport.exit_override = 137
        assert (await backend.run(['true'])).exit_code == 137

    async def test_sigkill_with_failing_liveness_check_returns_exit(self, transport: SpriteTransport) -> None:
        backend = SpritesSandboxBackend()
        await backend.get_sandbox()
        transport.exit_override = 137
        transport.get_error = NetworkError('control plane unavailable')
        assert (await backend.run(['true'])).exit_code == 137

    async def test_exec_handshake_transport_failure_is_network_error(self, transport: SpriteTransport) -> None:
        transport.connect_error = InvalidMessage('bad handshake')
        with pytest.raises(NetworkError, match='handshake'):
            await SpritesSandboxBackend().run(['true'])
        assert transport.execs == []

    async def test_exec_handshake_retries_when_command_cannot_have_started(self, transport: SpriteTransport) -> None:
        transport.connect_error_once = TimeoutError('connect stalled')
        result = await SpritesSandboxBackend().run(['echo', 'ok'])
        assert result.stdout == 'ok\n'
        assert len(transport.execs) == 1

    async def test_lost_exit_after_side_effect_is_not_retryable(self, transport: SpriteTransport) -> None:
        backend = SpritesSandboxBackend()
        await backend.get_sandbox()
        transport.connection_dropped = True
        target = transport.root / 'executions'
        with pytest.raises(WorkspaceUnavailableError, match='command may have run'):
            await backend.run(['sh', '-c', f'echo executed >> {target}'])
        assert target.read_text() == 'executed\n'

    async def test_exec_socket_carries_the_command_and_asks_the_sprite_to_end_it_on_disconnect(
        self, transport: SpriteTransport
    ) -> None:
        result = await SpritesSandboxBackend(working_dir=str(transport.root)).run(['echo', 'a b'])
        assert (result.exit_code, result.stdout) == (0, 'a b\n')
        check, socket = transport.execs
        assert check.query['cmd'][-3:] == ['test', '-d', str(transport.root)]
        # The wrapper ends both streams with a marker; the fake conformance suite checks separation.
        assert socket.query['cmd'][:2] == ['sh', '-c']
        assert socket.query['cmd'][5:] == ['echo', 'a b']
        assert socket.query['dir'] == [str(transport.root)]
        # Without it a non-TTY command outlives a closed socket by 10 seconds.
        assert socket.query['max_run_after_disconnect'] == ['1s']
        assert (socket.query['stdin'], socket.query['tty']) == (['true'], ['false'])

    async def test_argv_shell_environment_and_nonzero_exit(self, transport: SpriteTransport) -> None:
        backend = SpritesSandbox[None](env={'BASE': 'base', 'LAYERED': 'base'}).get_workspace(context(), ref=None)
        assert isinstance(backend, SpritesSandboxBackend)

        result = await backend.run(['/bin/echo', 'a; echo injected'])
        assert result.stdout == 'a; echo injected\n'
        result = await backend.run(
            'printf "$0 $BASE $LAYERED"; printf error >&2; exit 124',
            shell=True,
            env={'LAYERED': 'command'},
        )
        assert (result.exit_code, result.stdout, result.stderr) == (124, '/bin/sh base command', 'error')

    async def test_env_is_added_to_the_sprite_environment(
        self, transport: SpriteTransport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The fake runs commands in this process's environment, standing in for the Sprite's own.
        monkeypatch.setenv('SPRITE_OWN', 'kept')
        result = await SpritesSandboxBackend().run(['env'], env={'-i': 'option-like'})
        assert {'SPRITE_OWN=kept', '-i=option-like'} <= set(result.stdout.splitlines())

    @pytest.mark.parametrize(
        'command,env',
        [
            (['true'], {'A=B': 'x'}),
            (['true'], {'': 'x'}),
            (['true'], {'A': 'x\0'}),
        ],
    )
    async def test_env_that_env_cannot_express_is_rejected(
        self, transport: SpriteTransport, command: list[str], env: dict[str, str]
    ) -> None:
        with pytest.raises(ValueError):
            await SpritesSandboxBackend().run(command, env=env)
        assert transport.execs == []

    async def test_resolved_working_directory_preserves_spaces(self, transport: SpriteTransport) -> None:
        target = transport.root / ' directory '
        target.mkdir()
        link = transport.root / 'link'
        link.symlink_to(target)
        backend = SpritesSandboxBackend(working_dir=str(link))
        assert await backend.working_dir() == str(target.resolve())
        assert await backend.working_dir() == str(target.resolve())

    async def test_unexpected_working_directory_output(self, transport: SpriteTransport) -> None:
        backend = SpritesSandboxBackend()
        transport.exit_override = 1
        with pytest.raises(WorkspaceError, match='working directory'):
            await backend.working_dir()

    async def test_a_file_name_with_a_newline(self, transport: SpriteTransport) -> None:
        sandbox = Workspace(SpritesSandboxBackend())
        await sandbox.write_bytes('folder/a\nb', b'data')
        assert await sandbox.read_bytes('folder/a\nb') == b'data'
        assert [entry.name for entry in await sandbox.list_dir('folder')] == ['a\nb']

    async def test_output_printed_before_the_stream_attaches_is_kept(self, transport: SpriteTransport) -> None:
        """The Sprite starts a command before the client's stream attaches and drops what it printed until then."""
        backend = SpritesSandboxBackend()
        await backend.get_sandbox()
        transport.release_stdin_eof = asyncio.Event()
        task = asyncio.create_task(backend.run(['seq', '1', '20000']))
        await transport.exec_started.wait()
        await anyio.sleep(0.2)  # Long enough for an ungated `seq` to finish before the stream attaches.
        transport.release_stdin_eof.set()
        assert (await task).stdout == ''.join(f'{i}\n' for i in range(1, 20001))

    async def test_large_writes_go_through_the_filesystem_api(self, transport: SpriteTransport) -> None:
        """A command carries its argv in the exec URL, which the Sprite refuses above about 40 KB."""
        sandbox = Workspace(SpritesSandboxBackend())
        with pytest.raises(WorkspaceError, match='status 414'):
            await sandbox.run(['printf', 'x' * 50_000])
        data = bytes(range(256)) * 4096
        await sandbox.write_bytes('nested/big.bin', data)
        assert await sandbox.read_bytes('nested/big.bin') == data
        await sandbox.write_text('tool.sh', '#!/bin/sh\n')
        await sandbox.run(['chmod', '755', 'tool.sh'])
        await sandbox.write_text('tool.sh', '#!/bin/sh\necho hi\n')
        assert (await sandbox.run(['./tool.sh'])).stdout == 'hi\n'
        with pytest.raises(IsADirectoryError):
            await sandbox.write_bytes('nested', b'x')
        with pytest.raises(NotADirectoryError):
            await sandbox.write_bytes('tool.sh/child', b'x')

    @pytest.mark.parametrize(
        ('failure', 'expected'),
        [
            (FileNotFoundError_('write', '/f'), WorkspaceError),
            (FilesystemError('permission', 'write', '/f'), WorkspaceError),
            # The SDK raises a transport failure as a `FilesystemError` from the `httpx` error.
            (_caused_by(FilesystemError('connect failed', 'write', '/f'), httpx.ConnectError('down')), FilesystemError),
        ],
    )
    async def test_filesystem_api_errors_are_mapped_or_propagate(
        self, transport: SpriteTransport, monkeypatch: pytest.MonkeyPatch, failure: Exception, expected: type[Exception]
    ) -> None:
        async def fail(*args: object) -> None:
            raise failure

        monkeypatch.setattr(transport, 'fs_write', fail)
        with pytest.raises(expected) as caught:
            await Workspace(SpritesSandboxBackend()).write_bytes('/f', b'x')
        assert type(caught.value) is expected

    async def test_deadline_closes_the_socket_and_preserves_partial_output(self, transport: SpriteTransport) -> None:
        backend = SpritesSandboxBackend()
        await backend.get_sandbox()
        with pytest.raises(WorkspaceTimeoutError, match='Command timed out after 0.3 seconds') as caught:
            await backend.run('printf ready; exec sleep 5', shell=True, timeout=0.3)
        assert caught.value.stdout == 'ready'
        assert transport.execs[0].process.wait(timeout=1) == -signal.SIGKILL

    async def test_timeout_keeps_partial_stderr_and_removes_capture(self, transport: SpriteTransport) -> None:
        backend = SpritesSandboxBackend()
        await backend.get_sandbox()
        with pytest.raises(WorkspaceTimeoutError) as caught:
            await backend.run('printf ready >&2; exec sleep 5', shell=True, timeout=0.3)
        assert caught.value.stderr == 'ready'
        assert not Path(transport.execs[0].query['cmd'][4]).exists()

    async def test_cancellation_closes_the_socket(self, transport: SpriteTransport) -> None:
        backend = SpritesSandboxBackend()
        await backend.get_sandbox()
        async with anyio.create_task_group() as group:
            group.start_soon(backend.run, ['sleep', '5'])
            await transport.exec_started.wait()
            group.cancel_scope.cancel()
        assert transport.execs[0].process.wait(timeout=1) == -signal.SIGKILL

    async def test_a_stopped_command_leaves_no_stderr_file(self, transport: SpriteTransport, tmp_path: Path) -> None:
        running = tmp_path / 'running'
        backend = SpritesSandboxBackend()
        await backend.get_sandbox()
        async with anyio.create_task_group() as group:
            group.start_soon(backend.run, ['sh', '-c', f'echo secret >&2; : > {running}; exec sleep 30'])
            while not running.exists():
                await anyio.sleep(0.01)
            group.cancel_scope.cancel()
        socket = transport.execs[0]
        assert socket.process.wait(timeout=5) == -signal.SIGKILL
        # Unlike an unlinked mktemp file, the named capture must be removed during cancellation.
        assert not Path(socket.query['cmd'][4]).exists()
        assert (await backend.run('echo oops >&2', shell=True)).stderr == 'oops\n'

    @pytest.mark.parametrize('timeout', [0, -1, float('inf')])
    async def test_invalid_timeout(self, transport: SpriteTransport, timeout: float) -> None:
        with pytest.raises(ValueError, match='timeout'):
            await SpritesSandboxBackend().run(['true'], timeout=timeout)

    def test_relative_working_dir_is_rejected(self) -> None:
        with pytest.raises(ValueError, match='absolute'):
            SpritesSandboxBackend(working_dir='relative')


def test_sprite_timeout_stop_guidance() -> None:
    root = Path(__file__).resolve().parents[2]
    for page in (root / 'docs/sprites-sandbox.md', root / 'pydantic_ai_harness/sprites_sandbox/README.md'):
        content = page.read_text()
        assert 'What a timeout stops' in content
        assert 'process group' in content


def test_sprite_ripgrep_install_guidance() -> None:
    root = Path(__file__).resolve().parents[2]
    for page in (root / 'docs/sprites-sandbox.md', root / 'pydantic_ai_harness/sprites_sandbox/README.md'):
        content = page.read_text()
        assert 'not preinstalled' in content
        assert 'apt-get install ripgrep' in content
        assert 'ref' in content


def test_sprite_default_user_and_relative_path_guidance() -> None:
    root = Path(__file__).resolve().parents[2]
    for page in (root / 'docs/sprites-sandbox.md', root / 'pydantic_ai_harness/sprites_sandbox/README.md'):
        content = page.read_text()
        assert '/home/sprite' in content
        assert 'relative paths' in content


def test_background_process_docs_explain_sprite_pause() -> None:
    root = Path(__file__).resolve().parents[2]
    for page in (root / 'docs/sprites-sandbox.md', root / 'pydantic_ai_harness/sprites_sandbox/README.md'):
        content = page.read_text()
        assert 'plain `&`' in content
        assert 'Sprites service' in content


@pytest.mark.parametrize(
    'env,outcome',
    [
        ({'PYDANTIC_AI_HARNESS_SPRITES_LIVE': '1', 'SPRITE_TOKEN': 'token'}, 'run'),
        ({'PYDANTIC_AI_HARNESS_SPRITES_LIVE': '1', 'SPRITE_TOKEN': ''}, 'skip'),
        ({'SPRITE_TOKEN': 'token'}, 'skip'),
        ({'PYDANTIC_AI_HARNESS_SPRITES_LIVE': '1', 'SPRITES_REQUIRE_LIVE': '1'}, 'fail'),
    ],
)
def test_live_tier_gate(monkeypatch: pytest.MonkeyPatch, env: dict[str, str], outcome: str) -> None:
    for name in ('PYDANTIC_AI_HARNESS_SPRITES_LIVE', 'SPRITE_TOKEN', 'SPRITES_REQUIRE_LIVE'):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    if outcome == 'run':
        assert live_token() == 'token'
    else:
        with pytest.raises(pytest.skip.Exception if outcome == 'skip' else pytest.fail.Exception):
            live_token()
