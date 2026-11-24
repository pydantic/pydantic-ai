from __future__ import annotations

import os
from pathlib import Path

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import (
    Workspace,
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness.runta_sandbox import RuntaSandbox, RuntaSandboxBackend

from .fake_runta import FakeRunta

pytestmark = [pytest.mark.anyio, pytest.mark.skipif(os.name != 'posix', reason='needs a POSIX host')]


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def context() -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage(), run_id='run')


def ref_of(runtime_id: str) -> WorkspaceRef:
    return WorkspaceRef(provider='runta', id=runtime_id)


class TestRuntaSandbox:
    async def test_run_creates_one_runtime_and_keeps_it(self, runta: FakeRunta):
        capability = RuntaSandbox[None](image='clean', vcpus=2, memory_mib=2048)
        backend = capability.get_workspace(context(), ref=None)
        assert isinstance(backend, RuntaSandboxBackend) and backend.ref is None

        assert (await backend.run('echo out; echo err >&2; exit 3', shell=True)).exit_code == 3
        assert (await backend.run(['true'])).exit_code == 0

        (runtime,) = runta.runtimes.values()
        assert backend.ref == ref_of(runtime.id)
        assert runtime.name.startswith('pydantic-ai-')
        assert (runtime.image, runtime.resources) == ('clean', {'requests': {'vcpus': 2, 'memory_mib': 2048}})
        assert [request for request in runta.requests if request[0] == 'POST'] == [('POST', '/v2/runtimes')]

    async def test_agent_run_leaves_the_runtime_for_the_next_run(self, runta: FakeRunta):
        agent = Agent(TestModel(), deps_type=type(None), capabilities=[RuntaSandbox[None]()])

        @agent.tool
        async def write(ctx: RunContext[None]) -> str:
            assert ctx.workspace is not None
            await ctx.workspace.write_bytes('result.bin', b'\x00\xff\n')
            return 'written'

        result = await agent.run('write')
        assert result.workspace is not None and result.workspace.ref is not None
        assert len(runta.runtimes) == 1

        attached = RuntaSandbox[None]().backend(result.workspace.ref)
        assert (await attached.run(['cat', 'result.bin'])).stdout == '\x00�\n'
        assert len(runta.runtimes) == 1

    async def test_a_prepared_backend_and_the_message_history_keep_one_runtime(self, runta: FakeRunta, tmp_path: Path):
        project = str(tmp_path / 'project')
        agent = Agent(TestModel(), deps_type=type(None), capabilities=[RuntaSandbox[None](working_dir=project)])
        seen: list[str] = []

        @agent.tool
        async def read(ctx: RunContext[None]) -> str:
            assert ctx.workspace is not None
            seen.append(await ctx.workspace.read_text('TASK.md'))
            return 'read'

        backend = RuntaSandboxBackend(working_dir=project)
        await Workspace(backend).write_text('TASK.md', 'Add a flag.\n')
        result = await agent.run('read', workspace=backend)
        followup = await agent.run('read', message_history=result.all_messages())

        assert seen == ['Add a flag.\n']
        assert followup.workspace is not None and followup.workspace.ref == backend.ref
        assert await followup.workspace.read_text('TASK.md') == 'Add a flag.\n'
        assert len(runta.runtimes) == 1

        assert backend.ref is not None
        await RuntaSandbox[None]().destroy(backend.ref)
        assert runta.runtimes == {}

    async def test_commands_get_the_sandbox_env_and_the_working_dir(self, runta: FakeRunta, tmp_path: Path):
        directory = str(tmp_path / 'work' / 'dir')
        backend = RuntaSandboxBackend(working_dir=directory, env={'BASE': 'sandbox', 'odd-name': 'kept'})

        result = await backend.run(
            'printf "%s %s\\n" "$BASE" "$CALL"; env | grep odd-name', shell=True, env={'CALL': 'c'}
        )

        assert result.stdout == 'sandbox c\nodd-name=kept\n'
        assert await backend.working_dir() == os.path.realpath(directory)
        assert await backend.working_dir() == os.path.realpath(directory)
        with pytest.raises(ValueError, match='illegal environment variable'):
            await backend.run(['true'], env={'a=b': 'x'})

    async def test_a_working_dir_that_cannot_be_created_or_is_missing_fails(self, runta: FakeRunta, tmp_path: Path):
        (tmp_path / 'file').write_text('x')
        blocked = RuntaSandboxBackend(working_dir=str(tmp_path / 'file' / 'under'))
        with pytest.raises(WorkspaceError, match='Could not create working_dir'):
            await blocked.run(['true'])

        runtime = runta.create()
        attached = RuntaSandboxBackend(ref=ref_of(runtime.id), working_dir=str(tmp_path / 'missing'))
        with pytest.raises(WorkspaceError, match='does not exist in Runta runtime'):
            await attached.run(['true'])

        nowhere = RuntaSandboxBackend(ref=ref_of(runtime.id), env={'PATH': '/nonexistent'})
        with pytest.raises(WorkspaceError, match='Could not determine the working directory'):
            await nowhere.working_dir()

    @pytest.mark.parametrize(
        ('status', 'action'), [('paused', 'resume'), ('suspended', 'resume'), ('shutdown', 'start')]
    )
    async def test_attaching_starts_a_runtime_that_is_not_running(self, runta: FakeRunta, status: str, action: str):
        runtime = runta.create(status=status)

        assert (await RuntaSandboxBackend(ref=ref_of(runtime.id)).run(['true'])).exit_code == 0

        assert runtime.status == 'running'
        assert ('POST', f'/v2/runtimes/{runtime.id}/{action}') in runta.requests

    async def test_a_command_wakes_a_runtime_paused_after_attaching(self, runta: FakeRunta):
        backend = RuntaSandboxBackend()
        await backend.run(['true'])
        (runtime,) = runta.runtimes.values()
        runtime.status = 'paused'

        assert (await backend.run(['echo', 'awake'])).stdout == 'awake\n'
        assert runta.commands.count(runta.commands[-1]) == 1

    async def test_a_runtime_paused_before_its_working_dir_is_made_is_woken(self, runta: FakeRunta, tmp_path: Path):
        runta.pause_before_next_exec = True
        backend = RuntaSandboxBackend(working_dir=str(tmp_path / 'work'))

        assert (await backend.run(['true'])).exit_code == 0
        assert (tmp_path / 'work').is_dir()

    async def test_a_runtime_that_never_runs_fails_after_the_ready_timeout(
        self, runta: FakeRunta, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr('pydantic_ai_harness.runta_sandbox._backend._READY_TIMEOUT', 0.05)
        stuck = runta.create(status='pausing')

        with pytest.raises(WorkspaceError, match='is still pausing after'):
            await RuntaSandboxBackend(ref=ref_of(stuck.id)).run(['true'])

    async def test_a_refused_command_is_not_sent_again_to_a_running_runtime(self, runta: FakeRunta):
        backend = RuntaSandboxBackend()
        await backend.run(['true'])
        sent = len(runta.commands)

        runta.refuse_next = [503]
        with pytest.raises(Exception, match='refused the exec connection with HTTP 503'):
            await backend.run(['true'])
        # The lookup that would tell why the handshake was refused fails too: the refusal is what is reported.
        runta.refuse_next = [503]
        runta.fail_next = [500]
        with pytest.raises(Exception, match='refused the exec connection with HTTP 503'):
            await backend.run(['true'])
        assert len(runta.commands) == sent

    async def test_a_runtime_that_is_gone_is_unavailable(self, runta: FakeRunta):
        with pytest.raises(WorkspaceUnavailableError, match='no longer exists'):
            await RuntaSandboxBackend(ref=ref_of('00000000-0000-4000-8000-000000000000')).run(['true'])

        crashed = runta.create(status='crashed')
        with pytest.raises(WorkspaceUnavailableError, match='is crashed'):
            await RuntaSandboxBackend(ref=ref_of(crashed.id)).run(['true'])

        backend = RuntaSandboxBackend()
        await backend.run(['true'])
        assert backend.ref is not None
        del runta.runtimes[backend.ref.id]
        with pytest.raises(WorkspaceUnavailableError, match='no longer exists'):
            await backend.run(['true'])

        # Deleted while it was being woken.
        paused = runta.create(status='paused')
        runta.delete_on_wake = True
        with pytest.raises(WorkspaceUnavailableError, match='no longer exists'):
            await RuntaSandboxBackend(ref=ref_of(paused.id)).run(['true'])

    async def test_rejected_credentials_and_api_failures(self, runta: FakeRunta, monkeypatch: pytest.MonkeyPatch):
        runtime = runta.create()
        with pytest.raises(WorkspaceUnavailableError, match='rejected the API key'):
            await RuntaSandboxBackend(ref=ref_of(runtime.id), token='wrong').run(['true'])

        runta.fail_next = [500]
        with pytest.raises(Exception, match='Runta API answered 500'):
            await RuntaSandboxBackend(ref=ref_of(runtime.id)).run(['true'])

        runta.malformed_next = True
        with pytest.raises(Exception, match='unexpected shape'):
            await RuntaSandboxBackend(ref=ref_of(runtime.id)).run(['true'])

        monkeypatch.delenv('RUNTA_TOKEN')
        with pytest.raises(UserError, match='RUNTA_TOKEN'):
            await RuntaSandboxBackend().run(['true'])

    async def test_a_lost_exec_connection_is_unavailable(self, runta: FakeRunta):
        backend = RuntaSandboxBackend()
        await backend.run(['true'])

        runta.drop_exit = True
        with pytest.raises(WorkspaceUnavailableError, match='the command may have run'):
            await backend.run(['true'])
        runta.drop_exit, runta.end_without_exit = False, True
        with pytest.raises(WorkspaceUnavailableError, match='the command may have run'):
            await backend.run(['true'])
        for malformed in ('not json', '{"type": "stdout", "data_base64": "not base64"}'):
            runta.frames_next = [malformed]
            with pytest.raises(WorkspaceUnavailableError, match='the command may have run'):
                await backend.run(['true'])
        # The lookup that would tell whether the runtime is gone fails: the lost connection is what is reported.
        runta.fail_next = [500]
        with pytest.raises(WorkspaceUnavailableError, match='the command may have run'):
            await backend.run(['true'])

    @pytest.mark.parametrize(('fate', 'message'), [('crashed', 'is crashed'), ('deleted', 'no longer exists')])
    async def test_a_runtime_lost_during_a_command_is_unavailable(self, runta: FakeRunta, fate: str, message: str):
        backend = RuntaSandboxBackend()
        await backend.run(['true'])

        runta.drop_exit, runta.lost_runtime_status = True, fate
        with pytest.raises(WorkspaceUnavailableError, match=message):
            await backend.run(['true'])

    async def test_a_runtime_deleted_during_a_command_is_unavailable_when_runta_reports_an_error(
        self, runta: FakeRunta
    ):
        backend = RuntaSandboxBackend()
        await backend.run(['true'])

        runta.drop_exit, runta.lost_runtime_status, runta.lost_as_error = True, 'deleted', True
        with pytest.raises(WorkspaceUnavailableError, match='no longer exists'):
            await backend.run(['true'])

    async def test_a_command_runta_cannot_run_reports_its_reason(self, runta: FakeRunta):
        backend = RuntaSandboxBackend()
        runta.frames_next = ['{"type": "error", "message": "no such runtime process"}']
        with pytest.raises(Exception, match='Runta could not run the command: no such runtime process'):
            await backend.run(['true'])

    async def test_output_over_the_limit_keeps_a_preview(self, runta: FakeRunta, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr('pydantic_ai_harness.runta_sandbox._backend._MAX_OUTPUT_BYTES', 1000)
        monkeypatch.setattr('pydantic_ai_harness.runta_sandbox._backend._PREVIEW_BYTES', 10)

        with pytest.raises(WorkspaceOutputLimitError) as raised:
            await RuntaSandboxBackend().run('head -c 5000 /dev/zero | tr "\\0" a', shell=True)

        assert (raised.value.limit, raised.value.stdout) == (1000, 'a' * 10)

    async def test_a_timeout_stops_the_command_and_keeps_its_output(self, runta: FakeRunta):
        with pytest.raises(WorkspaceTimeoutError) as raised:
            await RuntaSandboxBackend().run('echo started; sleep 30', shell=True, timeout=0.5)

        assert raised.value.stdout == 'started\n'
        assert all(socket.closed for socket in runta.sockets)

    async def test_destroy_deletes_the_runtime(self, runta: FakeRunta):
        capability = RuntaSandbox[None]()
        runtime = runta.create()

        await capability.destroy(ref_of(runtime.id))
        assert runta.runtimes == {}
        await capability.destroy(ref_of(runtime.id))

        other = runta.create()
        runta.fail_next = [500]
        with pytest.raises(Exception, match='Runta API answered 500'):
            await capability.destroy(ref_of(other.id))
        with pytest.raises(ValueError, match='unsupported workspace provider'):
            await capability.destroy(WorkspaceRef(provider='other', id='x'))

    def test_capability_configuration(self):
        capability = RuntaSandbox[None].from_spec(image='clean', vcpus=2, working_dir='/work', env={'A': 'b'})
        assert (capability.image, capability.vcpus, capability.working_dir) == ('clean', 2, '/work')
        assert 'token' not in repr(RuntaSandbox[None](token='secret-key'))
        assert capability.get_workspace(context(), ref=WorkspaceRef(provider='other', id='x')) is None

        with pytest.raises(UserError, match='vcpus must be an integer'):
            RuntaSandbox[None](vcpus=0)
        with pytest.raises(UserError, match='working_dir must be an absolute POSIX path'):
            RuntaSandbox[None](working_dir='relative')
        with pytest.raises(ValueError, match='unsupported workspace provider'):
            RuntaSandboxBackend(ref=WorkspaceRef(provider='other', id='x'))
