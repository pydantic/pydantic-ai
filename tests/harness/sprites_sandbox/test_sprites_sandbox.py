from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import signal
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import anyio
import httpx
import pytest
from sprites import AsyncSprite
from sprites.exceptions import (
    APIError,
    AuthenticationError,
    FileNotFoundError_,
    FilesystemError,
    NetworkError,
    NotFoundError,
    PermissionError_,
    SpriteError,
)
from sprites.websocket import WSCommand
from websockets.datastructures import Headers
from websockets.exceptions import InvalidMessage, InvalidStatus
from websockets.http11 import Response

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import (
    CommandResult,
    SupportsCommands,
    Workspace,
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.sprites_sandbox import SpritesSandbox, SpritesSandboxBackend, _backend

from .conftest import live_token
from .fake_sprites import SpriteTransport


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

    def test_root_package_exports_the_capability_and_backend(self) -> None:
        import pydantic_ai_harness

        assert pydantic_ai_harness.SpritesSandbox is SpritesSandbox
        assert pydantic_ai_harness.SpritesSandboxBackend is SpritesSandboxBackend
        assert {'SpritesSandbox', 'SpritesSandboxBackend'} <= set(pydantic_ai_harness.__all__)

    async def test_construction_is_lazy_and_first_use_is_shared(self, transport: SpriteTransport) -> None:
        backend = SpritesSandbox[None]().get_workspace(context(), ref=None)
        assert isinstance(backend, SpritesSandboxBackend)
        assert transport.clients == []
        first, second = await asyncio.gather(backend.get_sandbox(), backend.get_sandbox())
        assert first is second
        assert transport.created == [first.name]
        assert backend.ref == WorkspaceRef(provider='sprites', id=first.name)

    async def test_creation_logs_the_sprite_id(
        self, transport: SpriteTransport, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level('INFO', logger='pydantic_ai_harness.sprites_sandbox._backend')
        backend = SpritesSandboxBackend()
        await backend.run(['true'])
        # Attaching creates nothing, so it logs nothing.
        await SpritesSandboxBackend(ref=backend.ref).run(['true'])
        assert [record.getMessage() for record in caplog.records] == [f'Created Sprite {transport.created[0]}']

    @pytest.mark.parametrize(
        ('kwargs', 'message'),
        [
            ({'working_dir': 'relative'}, "working_dir must be an absolute POSIX path or None, got 'relative'."),
            (
                {'defer_loading': True},
                '`SpritesSandbox` does not support `defer_loading=True`: '
                'the workspace is selected before deferred capabilities load.',
            ),
        ],
    )
    def test_invalid_configuration_fails_at_construction(self, kwargs: dict[str, object], message: str) -> None:
        with pytest.raises(UserError) as caught:
            SpritesSandbox[None](**kwargs)  # pyright: ignore[reportArgumentType]
        assert str(caught.value) == message

    def test_missing_sprites_extra_fails_at_import_with_install_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The package refuses to import without `sprites`, so no later call can hit a bare `ModuleNotFoundError`."""
        monkeypatch.setitem(sys.modules, 'sprites', None)
        for name in [name for name in sys.modules if name.startswith('pydantic_ai_harness.sprites_sandbox')]:
            monkeypatch.delitem(sys.modules, name)
        with pytest.raises(ImportError, match=r'^Install `pydantic-ai-harness\[sprites\]` to use SpritesSandbox\.$'):
            importlib.import_module('pydantic_ai_harness.sprites_sandbox')

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
        with pytest.raises(ValueError, match=r"^unsupported workspace provider 'other'; expected 'sprites'$"):
            await provider.destroy(WorkspaceRef(provider='other', id='target'))

    async def test_destroy_is_idempotent_and_maps_rejected_credentials(self, transport: SpriteTransport) -> None:
        provider = SpritesSandbox[None]()
        transport.names.add('gone')
        ref = WorkspaceRef(provider='sprites', id='gone')
        await provider.destroy(ref)
        await provider.destroy(ref)
        await provider.destroy(WorkspaceRef(provider='sprites', id='never-existed'))
        transport.destroy_error = AuthenticationError('credential expired: secret-sprite-token-123')
        with pytest.raises(WorkspaceUnavailableError) as caught:
            await provider.destroy(ref)
        assert str(caught.value) == (
            'Credential expired. Sprites rejected the credentials. '
            'Set SPRITE_TOKEN, or pass a configured `AsyncSpritesClient` as `client=`.'
        )
        transport.destroy_error = SpriteError('Failed destroy sprite (status 503): busy')
        with pytest.raises(SpriteError):
            await provider.destroy(ref)

    async def test_destroy_without_a_token_is_unavailable(
        self, transport: SpriteTransport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv('SPRITE_TOKEN')
        with pytest.raises(WorkspaceUnavailableError, match=r'^No Sprites credentials found\. Set SPRITE_TOKEN'):
            await SpritesSandbox[None]().destroy(WorkspaceRef(provider='sprites', id='target'))
        assert transport.clients == []

    async def test_destroy_with_a_caller_client_leaves_it_open(self, transport: SpriteTransport) -> None:
        client = transport.client('test-token')
        transport.names.add('target')
        await SpritesSandbox[None](client=client).destroy(WorkspaceRef(provider='sprites', id='target'))
        assert 'target' not in transport.names
        assert transport.close_calls == 0

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
            await ctx.workspace.run(['true'])
            return 'written'

        result = await agent.run('write')
        # The run keeps one client across its operations and closes it at the end.
        assert (len(transport.clients), transport.close_calls) == (1, 1)

        # After the run nothing would close the backend, so each operation closes its own client.
        assert await result.workspace.read_bytes('result.bin') == b'\x00\xff\n'
        assert (len(transport.clients), transport.close_calls) == (2, 2)
        assert len(transport.names) == 1

    async def test_backend_used_outside_a_run_closes_its_client_after_each_operation(
        self, transport: SpriteTransport
    ) -> None:
        # What a Temporal activity does: rebuild the backend from the ref, use it, and drop it.
        backend = SpritesSandbox[None]().get_workspace(context(), ref=None)
        assert isinstance(backend, SpritesSandboxBackend)
        note = str(transport.root / 'note.txt')
        await backend.write_bytes(note, b'hi')
        assert (len(transport.clients), transport.close_calls) == (1, 1)

        attached = SpritesSandbox[None](working_dir=str(transport.root)).get_workspace(context(), ref=backend.ref)
        assert isinstance(attached, SpritesSandboxBackend)
        # One operation, however many commands it runs (here the working-directory check and the
        # command), and concurrent operations share one client, closed once the last of them ends.
        first, second = await asyncio.gather(attached.run(['cat', 'note.txt']), attached.exists(note))
        assert (first.stdout, second) == ('hi', True)
        assert (len(transport.clients), transport.close_calls) == (2, 2)

    async def test_operations_on_one_backend_look_the_sprite_up_once(self, transport: SpriteTransport) -> None:
        owner = SpritesSandboxBackend()
        native = await owner.get_sandbox()
        # What a Temporal activity does: rebuild the backend from the ref and run several operations.
        backend = SpritesSandbox[None]().get_workspace(context(), ref=owner.ref)
        assert isinstance(backend, SpritesSandboxBackend)
        note = str(transport.root / 'note.txt')
        await backend.write_bytes(note, b'hi')
        assert await backend.read_bytes(note) == b'hi'
        assert (await backend.run(['true'])).exit_code == 0
        # Each operation closed its own client, and none looked up the Sprite this process just created.
        assert (len(transport.clients), transport.close_calls, transport.gets) == (4, 3, 0)

        # The public handle is always a fetched one.
        assert (await backend.get_sandbox()).name == native.name
        assert transport.gets == 1

        await native.delete()
        with pytest.raises(WorkspaceUnavailableError):
            await backend.run(['true'])
        # A Sprite reported unavailable is looked up again before the next operation.
        with pytest.raises(WorkspaceUnavailableError, match='no longer exists'):
            await backend.exists(note)
        assert transport.gets == 2

    async def test_backends_for_one_sprite_share_its_lookup(self, transport: SpriteTransport) -> None:
        transport.names.add('shared')
        ref = WorkspaceRef(provider='sprites', id='shared')
        capability = SpritesSandbox[None](working_dir=str(transport.root))
        # What two Temporal activities do: each builds its own backend from the ref.
        for _ in range(2):
            backend = capability.get_workspace(context(), ref=ref)
            assert isinstance(backend, SpritesSandboxBackend)
            assert await backend.working_dir() == str(transport.root.resolve())
            assert (await backend.run(['true'])).exit_code == 0
        # One lookup, then each backend's own `pwd -P` and command.
        assert (transport.gets, len(transport.execs)) == (1, 4)
        # Other credentials may not see the Sprite, so they look it up themselves.
        other = SpritesSandboxBackend(client=transport.client('other-token'), ref=ref)
        assert (await other.run(['true'])).exit_code == 0
        assert transport.gets == 2
        # Deleted outside this process: the cached lookup must not hide it from `working_dir()`.
        transport.names.discard('shared')
        backend = capability.get_workspace(context(), ref=ref)
        assert isinstance(backend, SpritesSandboxBackend)
        with pytest.raises(WorkspaceUnavailableError):
            await backend.working_dir()

    @pytest.mark.parametrize('end', ['destroyed', 'unavailable'])
    async def test_a_sprite_destroyed_or_unavailable_is_looked_up_again(
        self, transport: SpriteTransport, end: str
    ) -> None:
        transport.names.add('ending')
        ref = WorkspaceRef(provider='sprites', id='ending')
        capability = SpritesSandbox[None]()
        backend = capability.get_workspace(context(), ref=ref)
        assert isinstance(backend, SpritesSandboxBackend)
        await backend.run(['true'])
        assert transport.gets == 1
        if end == 'destroyed':
            await capability.destroy(ref)
        else:
            transport.names.discard('ending')
            # Found gone by an operation of a backend that trusted the cache.
            with pytest.raises(WorkspaceUnavailableError):
                await capability.backend(ref).run(['true'])
            assert transport.gets == 1
        # Recreated under the same name, as only a fresh lookup would notice.
        transport.names.add('ending')
        await capability.backend(ref).run(['true'])
        assert transport.gets == 2

    async def test_an_expired_lookup_is_repeated(self, transport: SpriteTransport) -> None:
        now = [0.0]
        _backend._lookups.clock = lambda: now[0]  # pyright: ignore[reportPrivateUsage]
        transport.names.add('aging')
        ref = WorkspaceRef(provider='sprites', id='aging')
        await SpritesSandboxBackend(ref=ref).working_dir()
        now[0] = 59.0
        await SpritesSandboxBackend(ref=ref).working_dir()
        # Each backend runs its own `pwd -P`; only the lookup is shared.
        assert (transport.gets, len(transport.execs)) == (1, 2)
        now[0] = 61.0
        await SpritesSandboxBackend(ref=ref).working_dir()
        assert (transport.gets, len(transport.execs)) == (2, 3)

    async def test_a_failed_lookup_is_not_cached(self, transport: SpriteTransport) -> None:
        transport.names.add('flaky')
        ref = WorkspaceRef(provider='sprites', id='flaky')
        transport.get_error_once = NetworkError('reset')
        with pytest.raises(NetworkError):
            await SpritesSandboxBackend(ref=ref).run(['true'])
        await SpritesSandboxBackend(ref=ref).run(['true'])
        assert transport.gets == 2

    async def test_an_operation_during_the_last_close_opens_its_own_client(self, transport: SpriteTransport) -> None:
        backend = SpritesSandbox[None]().get_workspace(context(), ref=None)
        assert isinstance(backend, SpritesSandboxBackend)
        note = str(transport.root / 'note.txt')
        transport.release_close = release = asyncio.Event()
        found: list[bool] = []

        async def check() -> None:
            found.append(await backend.exists(note))

        async with anyio.create_task_group() as tg:
            tg.start_soon(backend.write_bytes, note, b'hi')
            await transport.close_started.wait()
            # The first client is still closing, so this operation must open another.
            tg.start_soon(check)
            try:
                with anyio.fail_after(10):
                    while len(transport.clients) < 2:
                        await anyio.sleep(0)
            finally:
                # Closes are shielded, so the task group can only finish once they are released.
                release.set()

        assert found == [True]
        assert (len(transport.clients), transport.close_calls) == (2, 2)

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
        # A 404 while the created Sprite becomes visible is a transport failure, not a missing Sprite.
        transport.get_error_once = NotFoundError('still not visible')
        with pytest.raises(NetworkError, match='may still be becoming visible'):
            await backend.get_sandbox()
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
        with pytest.raises(WorkspaceUnavailableError) as caught:
            await SpritesSandboxBackend().get_sandbox()
        assert str(caught.value) == (
            'No Sprites credentials found. Set SPRITE_TOKEN, or pass a configured `AsyncSpritesClient` as `client=`.'
        )

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

    async def test_run_deadline_starts_once_the_sprite_is_acquired(
        self, transport: SpriteTransport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transport.names.add('remote')
        acquired_at: list[float] = []
        deadlines: list[float] = []
        get, connect = transport.get, transport.connect

        async def observed_get(*args: Any, **kwargs: Any) -> Any:
            # Held until the clock has moved on, so a deadline taken before acquisition lands earlier.
            await anyio.sleep(0.001)
            acquired_at.append(anyio.current_time())
            return await get(*args, **kwargs)

        async def observed_connect(*args: Any, **kwargs: Any) -> Any:
            deadlines.append(anyio.current_effective_deadline())
            return await connect(*args, **kwargs)

        monkeypatch.setattr(transport, 'get', observed_get)
        monkeypatch.setattr('sprites.websocket.connect', observed_connect)
        backend = SpritesSandboxBackend(ref=WorkspaceRef(provider='sprites', id='remote'))

        assert (await backend.run(['echo', 'ok'], timeout=60)).stdout == 'ok\n'
        assert deadlines[-1] >= acquired_at[0] + 60

    async def test_a_created_sprite_gets_its_working_directory(self, transport: SpriteTransport) -> None:
        directory = transport.root / 'new' / 'nested'
        backend = SpritesSandboxBackend(working_dir=str(directory))
        await backend.get_sandbox()
        assert directory.is_dir()
        assert (await backend.run(['pwd'])).stdout == f'{directory}\n'

    async def test_a_working_directory_that_cannot_be_made_is_retried(self, transport: SpriteTransport) -> None:
        (transport.root / 'blocker').write_text('')
        directory = transport.root / 'blocker' / 'work'
        backend = SpritesSandboxBackend(working_dir=str(directory))
        with pytest.raises(
            WorkspaceError,
            match=r"^Could not create working_dir '.*/blocker/work' in Sprite '.*': `mkdir -p` exited 1: mkdir: ",
        ):
            await backend.run(['true'])
        (transport.root / 'blocker').unlink()
        assert (await backend.run(['pwd'])).stdout == f'{directory}\n'
        assert len(transport.created) == 1

    async def test_an_attached_sprite_without_the_working_directory_fails_the_command(
        self, transport: SpriteTransport
    ) -> None:
        transport.names.add('remote')
        directory = transport.root / 'missing'
        backend = SpritesSandboxBackend(ref=WorkspaceRef(provider='sprites', id='remote'), working_dir=str(directory))
        with pytest.raises(WorkspaceError) as caught:
            await backend.run(['touch', 'ran'])
        assert type(caught.value) is WorkspaceError
        assert str(caught.value) == (
            f"working_dir '{directory}' does not exist in Sprite remote. "
            'Create it there, or pass a working_dir that exists.'
        )
        assert not directory.exists()
        assert not (transport.root / 'ran').exists()
        # Confirmed from `/` only after the command failed, not before every command.
        failed, check = transport.execs
        assert failed.query['cmd'][5:] == ['touch', 'ran']
        assert (check.query['cmd'][-3:], check.query['dir']) == (['test', '-d', str(directory)], ['/'])

    async def test_the_working_directory_check_stays_within_the_command_timeout(
        self, transport: SpriteTransport
    ) -> None:
        transport.names.add('remote')
        directory = transport.root / 'missing'
        backend = SpritesSandboxBackend(ref=WorkspaceRef(provider='sprites', id='remote'), working_dir=str(directory))
        transport.release_stdin_eof = asyncio.Event()
        caught: list[WorkspaceTimeoutError] = []

        async def run() -> None:
            with pytest.raises(WorkspaceTimeoutError) as error:
                await backend.run(['true'], timeout=0.5)
            caught.append(error.value)

        with anyio.fail_after(120):  # Hang guard only.
            async with anyio.create_task_group() as group:
                group.start_soon(run)
                await transport.exec_started.wait()
                # The command fails in the missing directory; the `test -d` that follows stalls.
                transport.exec_latency = 45
                transport.release_stdin_eof.set()
        assert str(caught[0]) == 'Command timed out after 0.5 seconds'
        assert caught[0].stdout == f'chdir to `{directory}`: No such file or directory\n'

    async def test_a_working_directory_found_by_the_check_leaves_the_failure_to_the_command(
        self, transport: SpriteTransport
    ) -> None:
        transport.names.add('remote')
        directory = transport.root / 'late'
        backend = SpritesSandboxBackend(ref=WorkspaceRef(provider='sprites', id='remote'), working_dir=str(directory))
        transport.release_stdin_eof = asyncio.Event()
        results: list[CommandResult] = []

        async def run() -> None:
            results.append(await backend.run(['touch', 'ran']))

        with anyio.fail_after(30):  # Hang guard only.
            async with anyio.create_task_group() as group:
                group.start_soon(run)
                await transport.exec_started.wait()
                # The command already failed to enter the directory, which exists by the time `test -d` runs.
                directory.mkdir()
                transport.release_stdin_eof.set()
        assert (results[0].exit_code, results[0].stdout) == (1, f'chdir to `{directory}`: No such file or directory\n')
        _, check = transport.execs
        assert check.query['cmd'][-3:] == ['test', '-d', str(directory)]

    async def test_a_command_in_an_existing_working_directory_is_one_exec(self, transport: SpriteTransport) -> None:
        """No `test -d` runs before or after a command, whether it succeeds or fails."""
        transport.names.add('remote')
        backend = SpritesSandboxBackend(
            ref=WorkspaceRef(provider='sprites', id='remote'), working_dir=str(transport.root)
        )
        assert (await backend.run(['pwd'])).stdout == f'{transport.root}\n'
        assert len(transport.execs) == 1
        result = await backend.run('echo "chdir to nowhere"; exit 1', shell=True)
        assert (result.exit_code, result.stdout) == (1, 'chdir to nowhere\n')
        assert len(transport.execs) == 2

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

    async def test_failed_close_is_logged(self, transport: SpriteTransport, caplog: pytest.LogCaptureFixture) -> None:
        owned = SpritesSandboxBackend()
        await owned.get_sandbox()
        transport.close_error = RuntimeError('close failed')
        await owned.aclose()
        assert 'Could not close Sprites SDK client' in caplog.text
        await owned.aclose()
        assert transport.close_calls == 1

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
        # Three attempts for the command; with no socket opened there is no stderr capture to read back.
        assert (transport.connects, transport.execs) == (3, [])

    async def test_timeout_during_a_stalled_handshake_reads_no_capture(self, transport: SpriteTransport) -> None:
        backend = SpritesSandboxBackend()
        await backend.get_sandbox()
        transport.exec_latency = 5
        with pytest.raises(WorkspaceTimeoutError):
            await backend.run(['true'], timeout=0.3)
        assert (transport.connects, transport.execs) == (1, [])

    async def test_exec_handshake_retries_when_command_cannot_have_started(self, transport: SpriteTransport) -> None:
        backend = SpritesSandboxBackend()
        await backend.get_sandbox()
        transport.connect_errors = [TimeoutError('connect stalled'), InvalidMessage('bad handshake')]
        result = await backend.run(['echo', 'ok'])
        assert result.stdout == 'ok\n'
        assert (transport.connects, len(transport.execs)) == (3, 1)

    async def test_a_handshake_failure_after_the_socket_opened_is_not_retried(
        self, transport: SpriteTransport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # In sprites-py 0.7 nothing in `start` after the socket opens raises these errors; the patch
        # stands in for a handshake step that fails once the command may have started.
        start = WSCommand.start

        async def start_then_fail(command: WSCommand) -> None:
            await start(command)
            raise TimeoutError('session setup stalled')

        backend = SpritesSandboxBackend()
        await backend.get_sandbox()
        monkeypatch.setattr(WSCommand, 'start', start_then_fail)
        with pytest.raises(WorkspaceUnavailableError, match='command may have run'):
            await backend.run(['true'])
        # One handshake for the command and one to read its stderr capture back; neither is retried.
        assert transport.connects == 2

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
        make, socket = transport.execs
        # The backend created the Sprite, so it made `working_dir` first, from `/`.
        assert make.query['cmd'][-4:] == ['mkdir', '-p', '--', str(transport.root)]
        assert make.query['dir'] == ['/']
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
            (PermissionError_('write', '/f'), PermissionError),
            (FilesystemError('refused', 'write', '/f'), WorkspaceError),
            # The SDK raises a transport failure as a `FilesystemError` from the `httpx` error.
            (_caused_by(FilesystemError('connect failed', 'write', '/f'), httpx.ConnectError('down')), FilesystemError),
        ],
    )
    async def test_filesystem_api_errors_are_mapped_or_propagate(
        self, transport: SpriteTransport, monkeypatch: pytest.MonkeyPatch, failure: Exception, expected: type[Exception]
    ) -> None:
        def fail(*args: object) -> None:
            raise failure

        monkeypatch.setattr(transport, 'fs_write', fail)
        with pytest.raises(expected) as caught:
            await Workspace(SpritesSandboxBackend()).write_bytes('/f', b'x')
        assert type(caught.value) is expected

    async def test_deadline_closes_the_socket_and_preserves_partial_output(self, transport: SpriteTransport) -> None:
        backend = SpritesSandboxBackend()
        await backend.get_sandbox()
        with pytest.raises(WorkspaceTimeoutError, match=r'Command timed out after 0.3 seconds') as caught:
            await backend.run('printf ready; exec sleep 5', shell=True, timeout=0.3)
        assert caught.value.stdout == 'ready'
        assert transport.execs[0].process.wait(timeout=1) == -signal.SIGKILL

    async def test_output_past_the_cap_stops_the_command(self, transport: SpriteTransport) -> None:
        backend = SpritesSandboxBackend()
        await backend.get_sandbox()
        # `yes` never ends by itself: only the cap stops it. The bound is a hang guard.
        with anyio.fail_after(60), pytest.raises(WorkspaceOutputLimitError) as caught:
            await backend.run(['yes'])
        assert caught.value.limit == 10 * 1024 * 1024
        assert caught.value.stdout == 'y\n' * (32 * 1024)
        assert transport.execs[0].process.wait(timeout=1) == -signal.SIGKILL
        assert not Path(transport.execs[0].query['cmd'][4]).exists()

    async def test_timeout_keeps_partial_stderr_and_removes_capture(self, transport: SpriteTransport) -> None:
        backend = SpritesSandboxBackend()
        await backend.get_sandbox()
        with pytest.raises(WorkspaceTimeoutError) as caught:
            await backend.run('printf ready >&2; exec sleep 5', shell=True, timeout=0.3)
        assert caught.value.stderr == 'ready'
        assert not Path(transport.execs[0].query['cmd'][4]).exists()

    async def test_timeout_reads_stderr_back_over_a_slow_exec(self, transport: SpriteTransport) -> None:
        backend = SpritesSandboxBackend()
        await backend.get_sandbox()
        caught: list[WorkspaceTimeoutError] = []

        async def run() -> None:
            with pytest.raises(WorkspaceTimeoutError) as error:
                await backend.run('printf ready >&2; exec sleep 5', shell=True, timeout=0.5)
            caught.append(error.value)

        async with anyio.create_task_group() as group:
            group.start_soon(run)
            await transport.exec_started.wait()
            # Only the exec that reads the capture back pays the handshake time of a live Sprite.
            transport.exec_latency = 1.5
        assert caught[0].stderr == 'ready'
        assert not Path(transport.execs[0].query['cmd'][4]).exists()

    async def test_timeout_keeps_the_output_when_the_stderr_capture_cannot_be_read(
        self, transport: SpriteTransport, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = SpritesSandboxBackend()
        await backend.get_sandbox()
        caught: list[WorkspaceTimeoutError] = []

        async def run() -> None:
            with pytest.raises(WorkspaceTimeoutError) as error:
                await backend.run('printf ready; printf lost >&2; exec sleep 5', shell=True, timeout=0.5)
            caught.append(error.value)

        async with anyio.create_task_group() as group:
            group.start_soon(run)
            await transport.exec_started.wait()
            # The exec that would read the stderr capture back cannot connect.
            transport.connect_error = InvalidMessage('bad handshake')
        assert (caught[0].stdout, caught[0].stderr) == ('ready', '')
        assert 'Could not retrieve Sprite stderr capture' in caplog.text

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

    def test_run_signature_matches_the_protocol(self) -> None:
        assert inspect.signature(SpritesSandboxBackend.run) == inspect.signature(SupportsCommands.run)

    @pytest.mark.parametrize('env', [{'PORT': 3000}, {1: 'one'}, {'TOKEN': None}])
    def test_non_string_env_fails_at_construction(self, env: dict[object, object]) -> None:
        with pytest.raises(TypeError, match=r'^env keys and values must be strings$'):
            SpritesSandboxBackend(env=env)  # pyright: ignore[reportArgumentType]

    def test_relative_working_dir_is_rejected(self) -> None:
        with pytest.raises(ValueError, match='absolute'):
            SpritesSandboxBackend(working_dir='relative')


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


@pytest.mark.parametrize('anyio_backend', ['trio'])
async def test_trio_is_refused_before_any_sprites_call(transport: SpriteTransport) -> None:
    """The Sprites SDK runs its calls on asyncio tasks, so Trio gets a clear `UserError` instead."""
    message = r'^Sprites needs the asyncio event loop: the Sprites SDK runs its calls on asyncio tasks\.$'
    backend = SpritesSandbox[None]().get_workspace(context(), ref=None)
    assert isinstance(backend, SpritesSandboxBackend)
    with pytest.raises(UserError, match=message):
        await backend.run(['true'])
    with pytest.raises(UserError, match=message):
        await backend.read_bytes('/file')
    with pytest.raises(UserError, match=message):
        await backend.write_bytes('/file', b'')
    with pytest.raises(UserError, match=message):
        await backend.get_sandbox()
    with pytest.raises(UserError, match=message):
        await SpritesSandbox[None]().destroy(WorkspaceRef(provider='sprites', id='target'))
    await backend.aclose()
    assert transport.clients == []
    assert transport.created == []
