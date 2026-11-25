"""Protocol and SDK-boundary tests for Platinum sandboxes, over the fake Platinum API."""

from __future__ import annotations

import errno
import logging
import os
import re
from pathlib import Path

import anyio
import httpx
import platinum
import pytest
from platinum.aio import AsyncPlatinum

from pydantic_ai.exceptions import UserError
from pydantic_ai.workspaces import (
    SupportsCommands,
    SupportsFilesystem,
    SupportsRealpath,
    Workspace,
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness.platinum_sandbox import PlatinumSandbox, PlatinumSandboxBackend
from tests._timeouts import READINESS_WAIT_TIMEOUT

from .fake_platinum import FakePlatinum, Outcome, Request, is_str_keyed_dict, wait_until_gone

pytestmark = [
    pytest.mark.anyio(backends=['asyncio']),
    pytest.mark.skipif(os.name != 'posix', reason='needs a POSIX host'),
]

_MODULE = 'pydantic_ai_harness.platinum_sandbox._backend'
_NAME = re.compile(r'pydantic-ai-[0-9a-f]{12}')


def _ref(sandbox_id: str) -> WorkspaceRef:
    return WorkspaceRef(provider='platinum', id=sandbox_id)


def _attached(fake: FakePlatinum, work: Path, *, state: str = 'running') -> PlatinumSandboxBackend:
    """A backend attached to a sandbox that exists already, working in `work`."""
    return PlatinumSandboxBackend(ref=_ref(fake.add(state=state)), working_dir=str(work))


def _answer(status: int, code: str, message: str = '', **extra: object) -> httpx.Response:
    return FakePlatinum.error(status, code, message, **extra)


def _created_name(request: Request) -> str:
    """The name a create request asks for."""
    assert is_str_keyed_dict(request.body)
    name = request.body.get('name')
    assert isinstance(name, str)
    return name


def _launched(argv: list[str]) -> list[str]:
    """The command an exec or process start runs, without the backend's launch script around it."""
    assert argv[:2] == ['sh', '-c'], argv
    return argv[6:]


class TestConstruction:
    @pytest.mark.parametrize(
        ('ref', 'message'),
        [
            (
                WorkspaceRef(provider='daytona', id='x'),
                "^unsupported workspace provider 'daytona'; expected 'platinum'$",
            ),
            (WorkspaceRef(provider='platinum', id='../sbx'), "^invalid Platinum workspace ref id '../sbx'$"),
            (WorkspaceRef(provider='platinum', id=''), "^invalid Platinum workspace ref id ''$"),
        ],
    )
    def test_an_unusable_ref_is_refused(self, ref: WorkspaceRef, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            PlatinumSandboxBackend(ref=ref)

    def test_invalid_settings_are_refused(self) -> None:
        with pytest.raises(TypeError, match='env keys and values must be strings'):
            PlatinumSandboxBackend(env={'PORT': 3000})  # pyright: ignore[reportArgumentType]
        with pytest.raises(ValueError, match='env keys must be non-empty without `=`'):
            PlatinumSandboxBackend(env={'A=B': 'c'})
        with pytest.raises(TypeError, match=r"^`secrets` takes a list of secret names, such as \['search'\]"):
            PlatinumSandboxBackend(secrets='search')
        with pytest.raises(ValueError, match='working_dir must be an absolute workspace path'):
            PlatinumSandboxBackend(working_dir='repo')

    async def test_a_sandbox_and_a_ref_are_exclusive(self, fake_platinum: FakePlatinum) -> None:
        sandbox = await fake_platinum.client().sandboxes.connect(fake_platinum.add())
        with pytest.raises(ValueError, match='pass either `sandbox` or `ref`, not both'):
            PlatinumSandboxBackend(sandbox=sandbox, ref=_ref(sandbox.id))

    def test_building_one_does_no_io(self, fake_platinum: FakePlatinum) -> None:
        backend = PlatinumSandboxBackend()
        assert isinstance(backend, SupportsCommands | SupportsFilesystem | SupportsRealpath)
        assert backend.ref is None
        assert fake_platinum.requests == []


class TestCreate:
    async def test_the_options_shape_the_new_sandbox(self, fake_platinum: FakePlatinum, work: Path) -> None:
        backend = PlatinumSandboxBackend(
            template='agent-tools',
            cpu=2,
            ram_mb=2048,
            disk_gb=10,
            auto_stop_minutes=15,
            allow_internet_access=False,
            secrets=['search-api'],
            env={'MODE': 'test'},
            working_dir=str(work),
        )
        assert await backend.working_dir() == str(work)
        [create] = fake_platinum.calls('POST', '/v1/sandboxes')
        assert is_str_keyed_dict(create.body)
        name = create.body.pop('name')
        assert isinstance(name, str) and _NAME.fullmatch(name)
        assert create.body == {
            'template': 'agent-tools',
            'cpu': 2,
            'ram_mb': 2048,
            'disk_gb': 10,
            'env': {'MODE': 'test'},
            'auto_stop_minutes': 15,
            'allow_internet_access': False,
            'secrets': ['search-api'],
        }
        assert create.query['wait_for_state'] == ['running']
        assert backend.ref == _ref(next(iter(fake_platinum.sandboxes)))

    async def test_defaults_leave_the_platform_settings_alone(self, fake_platinum: FakePlatinum, work: Path) -> None:
        await PlatinumSandboxBackend(working_dir=str(work)).working_dir()
        [create] = fake_platinum.calls('POST', '/v1/sandboxes')
        assert is_str_keyed_dict(create.body) and set(create.body) == {'name'}

    async def test_the_new_sandbox_is_logged(
        self, fake_platinum: FakePlatinum, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger=_MODULE):
            sandbox = await PlatinumSandboxBackend().get_sandbox()
        assert f'Created Platinum sandbox {sandbox.id} (pydantic-ai-' in caplog.text

    async def test_a_create_whose_reply_was_lost_finds_its_sandbox_by_name(self, fake_platinum: FakePlatinum) -> None:
        async def lose_the_reply(request: Request) -> httpx.Response | None:
            if request.method == 'POST' and request.path == '/v1/sandboxes':
                fake_platinum.add(name=_created_name(request))
                raise httpx.ReadTimeout('the reply was lost')
            return None

        fake_platinum.add(name='someone-else')
        fake_platinum.route_hook = lose_the_reply
        sandbox = await PlatinumSandboxBackend().get_sandbox()
        assert [s.id for s in fake_platinum.sandboxes.values() if s.name and _NAME.fullmatch(s.name)] == [sandbox.id]

    async def test_a_retried_create_finds_the_sandbox_an_earlier_attempt_made(
        self, fake_platinum: FakePlatinum
    ) -> None:
        async def lose_the_reply_and_the_listing(request: Request) -> httpx.Response | None:
            if request.method == 'POST':
                fake_platinum.add(name=_created_name(request))
                raise httpx.RemoteProtocolError('the connection dropped')
            return _answer(503, 'unavailable')  # the listing that would have found it

        backend = PlatinumSandboxBackend()
        fake_platinum.route_hook = lose_the_reply_and_the_listing
        with pytest.raises(platinum.PlatinumConnectionError):
            await backend.get_sandbox()
        assert backend.ref is None
        fake_platinum.route_hook = None
        sandbox = await backend.get_sandbox()  # the same name: `name_taken`, so it is looked up
        assert len(fake_platinum.sandboxes) == 1 and backend.ref == _ref(sandbox.id)

    async def test_a_lost_reply_with_no_sandbox_behind_it_is_transient(self, fake_platinum: FakePlatinum) -> None:
        async def lose_the_reply(request: Request) -> httpx.Response | None:
            if request.method == 'POST' and request.path == '/v1/sandboxes':
                fake_platinum.sandboxes[fake_platinum.add(name=_created_name(request))].state = 'deleted'  # not reused
                raise httpx.ReadTimeout('the reply was lost')
            return None

        fake_platinum.add(name='someone-else')
        fake_platinum.route_hook = lose_the_reply
        with pytest.raises(platinum.PlatinumTimeoutError):
            await PlatinumSandboxBackend().get_sandbox()

    async def test_a_sandbox_that_failed_to_start_is_recorded_and_unavailable(
        self, fake_platinum: FakePlatinum
    ) -> None:
        async def failing_start(request: Request) -> httpx.Response | None:
            if request.method == 'POST' and request.path == '/v1/sandboxes':
                sandbox = fake_platinum.sandboxes[fake_platinum.add(state='failed-start')]
                return httpx.Response(201, json={'id': sandbox.id, 'state': 'provisioning'})
            return None

        fake_platinum.route_hook = failing_start
        backend = PlatinumSandboxBackend()
        with pytest.raises(
            WorkspaceUnavailableError, match=r'failed to start .* Delete it with `PlatinumSandbox\(\)\.destroy\(ref\)`'
        ):
            await backend.get_sandbox()
        assert backend.ref == _ref(next(iter(fake_platinum.sandboxes)))

    async def test_a_sandbox_slow_to_start_is_recorded_and_attached_next_time(
        self, fake_platinum: FakePlatinum, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(f'{_MODULE}._START_TIMEOUT', 0.001)

        async def slow_start(request: Request) -> httpx.Response | None:
            if request.method == 'POST' and request.path == '/v1/sandboxes':
                sandbox = fake_platinum.sandboxes[fake_platinum.add(state='provisioning')]
                return httpx.Response(201, json={'id': sandbox.id, 'state': 'provisioning'})
            return None

        fake_platinum.route_hook = slow_start
        backend = PlatinumSandboxBackend()
        with pytest.raises(TimeoutError, match='was not running after'):
            await backend.get_sandbox()
        assert backend.ref is not None
        fake_platinum.sandboxes[backend.ref.id].state = 'running'
        assert (await backend.get_sandbox()).id == backend.ref.id

    @pytest.mark.parametrize(
        ('response', 'expected', 'message'),
        [
            (_answer(400, 'invalid_template', 'unknown template'), WorkspaceUnavailableError, 'unknown template'),
            (_answer(402, 'insufficient_credits'), WorkspaceUnavailableError, 'insufficient_credits'),
            (_answer(401, 'unauthorized'), WorkspaceUnavailableError, 'Platinum refused the API key'),
            (_answer(503, 'no_capacity'), platinum.ServerError, 'no_capacity'),
            (_answer(429, 'rate_limited'), platinum.RateLimitError, 'rate_limited'),
        ],
    )
    async def test_a_refused_create_ends_the_run_and_a_transient_one_propagates(
        self,
        fake_platinum: FakePlatinum,
        response: httpx.Response,
        expected: type[Exception],
        message: str,
    ) -> None:
        async def refuse(request: Request) -> httpx.Response | None:
            return response if request.method == 'POST' else None

        fake_platinum.route_hook = refuse
        with pytest.raises(expected, match=message):
            await PlatinumSandboxBackend().get_sandbox()

    async def test_a_missing_api_key_is_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(platinum, 'AsyncPlatinum', AsyncPlatinum)
        with pytest.raises(WorkspaceUnavailableError, match=r'No Platinum API key found. Set PT_TOKEN'):
            await PlatinumSandboxBackend().get_sandbox()

    async def test_a_cancelled_create_still_records_its_sandbox(self, fake_platinum: FakePlatinum) -> None:
        entered, release = anyio.Event(), anyio.Event()

        async def hold(request: Request) -> httpx.Response | None:
            entered.set()  # the create, the only request
            await release.wait()
            return None

        fake_platinum.route_hook = hold
        backend = PlatinumSandboxBackend()
        async with anyio.create_task_group() as group:
            group.start_soon(backend.get_sandbox)
            await entered.wait()
            group.cancel_scope.cancel()
            release.set()
        assert backend.ref == _ref(next(iter(fake_platinum.sandboxes)))


class TestAttach:
    @pytest.mark.parametrize('state', ['stopped', 'archived'])
    async def test_attaching_starts_a_stopped_sandbox(
        self, fake_platinum: FakePlatinum, work: Path, state: str
    ) -> None:
        backend = _attached(fake_platinum, work, state=state)
        assert await backend.working_dir() == str(work)
        assert len(fake_platinum.calls('POST', '/start')) == 1

    async def test_attaching_waits_for_a_starting_sandbox(self, fake_platinum: FakePlatinum, work: Path) -> None:
        backend = _attached(fake_platinum, work)
        fake_platinum.sandboxes[backend.ref.id].next_states = ['starting', 'starting']  # pyright: ignore[reportOptionalMemberAccess]
        await backend.get_sandbox()
        assert fake_platinum.calls('POST', '/start') == []

    async def test_a_starting_sandbox_that_settles_stopped_is_started(
        self, fake_platinum: FakePlatinum, work: Path
    ) -> None:
        backend = _attached(fake_platinum, work, state='stopped')
        fake_platinum.sandboxes[backend.ref.id].next_states = ['resuming', 'resuming']  # pyright: ignore[reportOptionalMemberAccess]
        await backend.get_sandbox()
        assert len(fake_platinum.calls('POST', '/start')) == 1

    async def test_a_starting_sandbox_that_fails_is_unavailable(self, fake_platinum: FakePlatinum) -> None:
        sandbox = fake_platinum.sandboxes[fake_platinum.add(state='failed-start')]
        sandbox.next_states = ['provisioning', 'provisioning']
        with pytest.raises(WorkspaceUnavailableError, match='it is failed-start'):
            await PlatinumSandboxBackend(ref=_ref(sandbox.id)).get_sandbox()

    async def test_a_deleted_sandbox_is_unavailable(self, fake_platinum: FakePlatinum) -> None:
        with pytest.raises(WorkspaceUnavailableError, match=r"'sbx_gone' is no longer available: it was deleted"):
            await PlatinumSandboxBackend(ref=_ref('sbx_gone')).get_sandbox()

    @pytest.mark.parametrize('state', ['failed-start', 'lost'])
    async def test_a_sandbox_that_cannot_run_again_is_unavailable(
        self, fake_platinum: FakePlatinum, state: str
    ) -> None:
        sandbox_id = fake_platinum.add(state=state)
        with pytest.raises(WorkspaceUnavailableError, match=f'is no longer available: it is {state}'):
            await PlatinumSandboxBackend(ref=_ref(sandbox_id)).get_sandbox()

    @pytest.mark.parametrize(
        ('response', 'expected'),
        [(_answer(403, 'forbidden'), WorkspaceUnavailableError), (_answer(500, 'internal'), platinum.ServerError)],
    )
    async def test_a_refused_attach_ends_the_run_and_a_transient_one_propagates(
        self, fake_platinum: FakePlatinum, response: httpx.Response, expected: type[Exception]
    ) -> None:
        async def refuse(request: Request) -> httpx.Response | None:
            return response

        sandbox_id = fake_platinum.add()
        fake_platinum.route_hook = refuse
        with pytest.raises(expected):
            await PlatinumSandboxBackend(ref=_ref(sandbox_id)).get_sandbox()

    async def test_a_start_that_fails_is_unavailable(self, fake_platinum: FakePlatinum) -> None:
        sandbox = fake_platinum.sandboxes[fake_platinum.add(state='stopped')]

        async def fail_the_start(request: Request) -> httpx.Response | None:
            if request.path.endswith('/start'):
                sandbox.state = 'failed-start'
                return httpx.Response(200, json={'id': sandbox.id, 'state': 'starting'})
            return None

        fake_platinum.route_hook = fail_the_start
        with pytest.raises(WorkspaceUnavailableError, match='it is failed-start'):
            await PlatinumSandboxBackend(ref=_ref(sandbox.id)).get_sandbox()

    async def test_a_start_that_takes_too_long_is_transient(
        self, fake_platinum: FakePlatinum, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(f'{_MODULE}._START_TIMEOUT', 0.001)
        sandbox = fake_platinum.sandboxes[fake_platinum.add(state='stopped')]

        async def slow_start(request: Request) -> httpx.Response | None:
            if request.path.endswith('/start'):
                sandbox.state = 'starting'
                return httpx.Response(200, json={'id': sandbox.id, 'state': 'starting'})
            return None

        fake_platinum.route_hook = slow_start
        with pytest.raises(TimeoutError, match='was not running after'):
            await PlatinumSandboxBackend(ref=_ref(sandbox.id)).get_sandbox()

    async def test_a_refused_start_ends_the_run(self, fake_platinum: FakePlatinum) -> None:
        async def refuse(request: Request) -> httpx.Response | None:
            return _answer(402, 'insufficient_credits') if request.path.endswith('/start') else None

        sandbox_id = fake_platinum.add(state='stopped')
        fake_platinum.route_hook = refuse
        with pytest.raises(WorkspaceUnavailableError, match='Could not start Platinum sandbox'):
            await PlatinumSandboxBackend(ref=_ref(sandbox_id)).get_sandbox()

    async def test_a_sandbox_handle_is_used_as_given(self, fake_platinum: FakePlatinum, work: Path) -> None:
        client = fake_platinum.client()
        sandbox = await client.sandboxes.connect(fake_platinum.add())
        backend = PlatinumSandboxBackend(sandbox=sandbox, working_dir=str(work))
        assert backend.ref == _ref(sandbox.id)
        assert await backend.get_sandbox() is sandbox
        assert (await backend.run(['pwd', '-P'], timeout=10)).stdout == f'{work}\n'
        await backend.aclose()
        assert client not in fake_platinum.closed


class TestOperations:
    async def test_a_sandbox_stopped_meanwhile_is_started_and_the_operation_runs_once_more(
        self, fake_platinum: FakePlatinum, work: Path
    ) -> None:
        backend = _attached(fake_platinum, work)
        await backend.get_sandbox()
        assert backend.ref is not None
        fake_platinum.sandboxes[backend.ref.id].state = 'stopped'
        assert (await backend.run(['echo', 'again'], timeout=10)).stdout == 'again\n'
        assert len(fake_platinum.calls('POST', '/start')) == 1
        assert len(fake_platinum.calls('POST', '/exec')) == 2

    async def test_a_conflict_other_than_a_stop_is_a_workspace_error(
        self, fake_platinum: FakePlatinum, work: Path
    ) -> None:
        async def conflict(request: Request) -> httpx.Response | None:
            return _answer(409, 'operation_in_progress') if request.path.endswith('/exec') else None

        fake_platinum.route_hook = conflict
        with pytest.raises(WorkspaceError, match='operation_in_progress'):
            await _attached(fake_platinum, work).run(['true'], timeout=10)

    async def test_a_sandbox_deleted_meanwhile_is_unavailable(self, fake_platinum: FakePlatinum, work: Path) -> None:
        backend = _attached(fake_platinum, work)
        await backend.get_sandbox()
        assert backend.ref is not None
        fake_platinum.sandboxes[backend.ref.id].state = 'deleted'
        for operation in (backend.read_bytes(str(work / 'f')), backend.run(['true'], timeout=10)):
            with pytest.raises(WorkspaceUnavailableError, match="Pass `workspace='new'`"):
                await operation

    @pytest.mark.parametrize(
        ('lookup', 'expected'),
        [
            (_answer(404, 'sandbox_not_found'), WorkspaceUnavailableError),
            (httpx.Response(200, json={'state': 'lost'}), WorkspaceUnavailableError),
            (httpx.Response(200, json={'state': 'running'}), platinum.ServerError),
            (_answer(500, 'internal'), platinum.ServerError),
        ],
    )
    async def test_a_server_error_is_unavailable_only_when_the_sandbox_is_gone(
        self, fake_platinum: FakePlatinum, work: Path, lookup: httpx.Response, expected: type[Exception]
    ) -> None:
        backend = _attached(fake_platinum, work)
        await backend.get_sandbox()

        async def fail(request: Request) -> httpx.Response | None:
            if request.path.endswith('/files'):
                return _answer(502, 'bad_gateway')
            return lookup if request.method == 'GET' else None

        fake_platinum.route_hook = fail
        with pytest.raises(expected):
            await backend.read_bytes(str(work / 'f'))

    async def test_rejected_credentials_end_the_run(self, fake_platinum: FakePlatinum, work: Path) -> None:
        backend = _attached(fake_platinum, work)
        await backend.get_sandbox()

        async def reject(request: Request) -> httpx.Response | None:
            return _answer(401, 'unauthorized')

        fake_platinum.route_hook = reject
        with pytest.raises(WorkspaceUnavailableError, match=r'Credentials rejected. Platinum refused the API key'):
            await backend.stat(str(work))


class TestRun:
    async def test_a_short_command_is_one_exec_with_the_merged_env(
        self, fake_platinum: FakePlatinum, work: Path
    ) -> None:
        backend = PlatinumSandboxBackend(ref=_ref(fake_platinum.add()), working_dir=str(work), env={'A': '1', 'B': '1'})
        result = await backend.run(
            'printf "%s %s" "$A" "$B"; pwd -P >&2; exit 3', shell=True, env={'B': '2'}, timeout=80
        )
        assert (result.exit_code, result.stdout, result.stderr) == (3, '1 2', f'{work}\n')
        [command] = fake_platinum.commands()
        assert _launched(command)[:4] == ['env', '--', 'A=1', 'B=2']
        assert fake_platinum.calls('POST', '/processes') == []

    async def test_a_long_or_unbounded_command_is_a_managed_process(
        self, fake_platinum: FakePlatinum, work: Path
    ) -> None:
        backend = PlatinumSandboxBackend(ref=_ref(fake_platinum.add()), working_dir=str(work), env={'A': '1'})
        for timeout in (80.5, None):
            result = await backend.run(['sh', '-c', 'printf "$A"; printf err >&2'], timeout=timeout)
            assert (result.exit_code, result.stdout, result.stderr) == (0, '1', 'err')
        first, second = (request.body for request in fake_platinum.calls('POST', '/processes'))
        assert is_str_keyed_dict(first) and is_str_keyed_dict(second)
        assert (first['env'], first['timeout_ms'], 'timeout_ms' in second) == ({'A': '1'}, 80500, False)
        assert fake_platinum.calls('POST', '/exec') == []

    async def test_a_command_a_signal_ended_exits_128_plus_the_signal(
        self, fake_platinum: FakePlatinum, work: Path
    ) -> None:
        backend = _attached(fake_platinum, work)
        assert (await backend.run(['sh', '-c', 'kill -TERM $$'], timeout=10)).exit_code == 143
        assert (await backend.run(['sh', '-c', 'kill -TERM $$'])).exit_code == 143

    async def test_a_working_dir_that_cannot_be_created_fails_the_command(
        self, fake_platinum: FakePlatinum, work: Path
    ) -> None:
        work.mkdir()
        (work / 'file').write_text('x')
        backend = PlatinumSandboxBackend(ref=_ref(fake_platinum.add()), working_dir=str(work / 'file' / 'dir'))
        with pytest.raises(WorkspaceError, match='cannot be entered or created in the Platinum sandbox'):
            await backend.run(['true'], timeout=10)

    async def test_without_a_working_dir_commands_start_in_the_template_s(self, fake_platinum: FakePlatinum) -> None:
        backend = PlatinumSandboxBackend(ref=_ref(fake_platinum.add()), working_dir=None)
        assert await backend.working_dir() == os.path.realpath(os.getcwd())

    async def test_a_working_dir_pwd_cannot_print_is_an_error(self, fake_platinum: FakePlatinum, work: Path) -> None:
        fake_platinum.exec_hook = lambda argv, timeout_ms: Outcome(exit_code=1, stderr=b'no pwd')
        with pytest.raises(WorkspaceError, match='`pwd -P` exited 1: no pwd'):
            await _attached(fake_platinum, work).working_dir()

    async def test_an_exec_timeout_has_the_output_so_far_and_frees_its_claim(
        self, fake_platinum: FakePlatinum, work: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(WorkspaceTimeoutError, match=r'^Command timed out after 0\.2 seconds$') as raised:
            await _attached(fake_platinum, work).run(['sh', '-c', 'echo before; sleep 30'], timeout=0.2)
        assert raised.value.stdout == 'before\n'
        assert len(fake_platinum.calls('POST', '/exec')) == 2  # the command, then its stop
        assert list((tmp_path / 'scratch').iterdir()) == []

    async def test_an_exec_the_sandbox_could_not_start_is_an_error(
        self, fake_platinum: FakePlatinum, work: Path
    ) -> None:
        refusal = {'result': {'ok': False, 'exit_code': -1, 'error': 'fork/exec /bin/sh: no such file'}}
        fake_platinum.exec_hook = lambda argv, timeout_ms: httpx.Response(200, json={**refusal, 'error': 'fork/exec'})
        with pytest.raises(WorkspaceError, match='could not start the command: fork/exec'):
            await _attached(fake_platinum, work).run(['true'], timeout=10)

    async def test_exec_output_over_the_limit_raises(self, fake_platinum: FakePlatinum, work: Path) -> None:
        fake_platinum.max_output_bytes = 10
        with pytest.raises(WorkspaceOutputLimitError, match='exceeded the 16 MiB limit') as raised:
            await _attached(fake_platinum, work).run(['sh', '-c', 'printf 0123456789ab'], timeout=10)
        assert raised.value.limit == 16 * 1024 * 1024

    async def test_an_exec_the_sandbox_did_not_answer_is_stopped_and_times_out(
        self, fake_platinum: FakePlatinum, work: Path
    ) -> None:
        def lose(argv: list[str], timeout_ms: int | None) -> Exception | None:
            return None if 'kill -9' in ' '.join(argv) else httpx.ReadTimeout('no answer')

        backend = _attached(fake_platinum, work)
        await backend.get_sandbox()
        fake_platinum.exec_hook = lose
        with pytest.raises(WorkspaceTimeoutError, match='the sandbox did not answer'):
            await backend.run(['sleep', '1'], timeout=10)
        assert len(fake_platinum.calls('POST', '/exec')) == 2

    @pytest.mark.parametrize(
        ('failure', 'stopped'),
        [(httpx.ConnectError('refused'), False), (httpx.ReadError('reset'), True)],
    )
    async def test_a_transport_failure_stops_a_command_that_may_have_started(
        self, fake_platinum: FakePlatinum, work: Path, failure: Exception, stopped: bool
    ) -> None:
        backend = _attached(fake_platinum, work)
        await backend.get_sandbox()
        fake_platinum.exec_hook = lambda argv, timeout_ms: None if 'kill -9' in ' '.join(argv) else failure
        with pytest.raises(platinum.PlatinumConnectionError):
            await backend.run(['true'], timeout=10)
        assert len(fake_platinum.calls('POST', '/exec')) == (2 if stopped else 1)

    async def test_a_cancelled_exec_is_stopped(self, fake_platinum: FakePlatinum, work: Path) -> None:
        backend = _attached(fake_platinum, work)
        pid_file = work / 'pid'
        async with anyio.create_task_group() as group:
            group.start_soon(
                lambda: backend.run(['sh', '-c', 'echo $$ > "$1"; exec sleep 60', 'sh', str(pid_file)], timeout=60)
            )
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                # real-wait: the command writes its pid on the host, which nothing announces.
                while not pid_file.exists() or not pid_file.read_text():
                    await anyio.sleep(0.01)
            group.cancel_scope.cancel()
        await wait_until_gone(pid_file)

    async def test_a_managed_process_timeout_has_the_output_so_far(
        self, fake_platinum: FakePlatinum, work: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(f'{_MODULE}._EXEC_MAX_TIMEOUT', 0.1)
        with pytest.raises(WorkspaceTimeoutError, match=r'^Command timed out after 0\.2 seconds$') as raised:
            await _attached(fake_platinum, work).run(['sh', '-c', 'echo before; sleep 30'], timeout=0.2)
        assert raised.value.stdout == 'before\n'

    async def test_a_managed_process_past_its_client_deadline_is_killed(
        self, fake_platinum: FakePlatinum, work: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(f'{_MODULE}._EXEC_MAX_TIMEOUT', 0.001)
        monkeypatch.setattr(f'{_MODULE}._PROCESS_GRACE', 0)
        never = anyio.Event()

        async def hang(request: Request) -> httpx.Response | None:
            if request.path.endswith('/stream'):
                await never.wait()
            return None

        fake_platinum.route_hook = hang
        with pytest.raises(WorkspaceTimeoutError, match=r'Command timed out after 0.01 seconds'):
            await _attached(fake_platinum, work).run(['true'], timeout=0.01)
        assert len(fake_platinum.calls('DELETE', '')) == 1

    async def test_managed_process_output_over_the_limit_kills_it(
        self, fake_platinum: FakePlatinum, work: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(f'{_MODULE}._MAX_OUTPUT_BYTES', 10)
        monkeypatch.setattr(f'{_MODULE}._OUTPUT_PREVIEW_CHARS', 4)
        with pytest.raises(WorkspaceOutputLimitError) as raised:
            await _attached(fake_platinum, work).run(['sh', '-c', 'printf 0123456789ab'])
        assert (raised.value.stdout, raised.value.limit) == ('0123', 10)
        assert len(fake_platinum.calls('DELETE', '')) == 1

    async def test_managed_process_output_the_sandbox_dropped_is_not_a_success(
        self, fake_platinum: FakePlatinum, work: Path
    ) -> None:
        fake_platinum.exec_hook = lambda argv, timeout_ms: Outcome(exit_code=0, stdout=b'tail', dropped=1000)
        with pytest.raises(WorkspaceOutputLimitError, match='1000 bytes of the command output were lost') as raised:
            await _attached(fake_platinum, work).run(['yes'])
        assert (raised.value.stdout, raised.value.limit) == ('tail', 256 * 1024)


class TestFiles:
    async def test_a_large_write_goes_in_parts_sized_to_the_stated_limit(
        self, fake_platinum: FakePlatinum, work: Path, tmp_path: Path
    ) -> None:
        fake_platinum.max_write_bytes = 1000
        backend = _attached(fake_platinum, work)
        data = os.urandom(2500)
        await backend.write_bytes(str(work / 'big.bin'), data)
        assert (work / 'big.bin').read_bytes() == data
        assert len(fake_platinum.calls('PUT')) == 4  # the refused whole, then three parts
        assert list((tmp_path / 'scratch').iterdir()) == []

    async def test_upload_parts_shrink_to_a_lower_stated_limit(
        self, fake_platinum: FakePlatinum, work: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(f'{_MODULE}._UPLOAD_PART_BYTES', 1500)
        fake_platinum.max_write_bytes = 1000
        data = os.urandom(2500)
        await _attached(fake_platinum, work).write_bytes(str(work / 'big.bin'), data)
        assert (work / 'big.bin').read_bytes() == data

    async def test_a_refused_write_without_a_stated_limit_is_an_error(
        self, fake_platinum: FakePlatinum, work: Path
    ) -> None:
        async def edge(request: Request) -> httpx.Response | None:
            return (
                httpx.Response(413, text='<html>Request Entity Too Large</html>') if request.method == 'PUT' else None
            )

        fake_platinum.route_hook = edge
        with pytest.raises(WorkspaceError, match='413'):
            await _attached(fake_platinum, work).write_bytes(str(work / 'f'), b'x')

    @pytest.mark.parametrize(
        ('setup', 'target', 'expected'),
        [('mkdir -p "$1/dir"', 'dir', IsADirectoryError), (': > "$1/file"', 'file/child', NotADirectoryError)],
    )
    async def test_joining_parts_fails_like_a_write(
        self, fake_platinum: FakePlatinum, work: Path, setup: str, target: str, expected: type[OSError]
    ) -> None:
        fake_platinum.max_write_bytes = 10
        backend = _attached(fake_platinum, work)
        await backend.run(['sh', '-c', setup, 'sh', str(work)], timeout=10)
        with pytest.raises(expected):
            await backend.write_bytes(str(work / target), b'x' * 25)

    async def test_a_join_that_fails_otherwise_is_an_error(self, fake_platinum: FakePlatinum, work: Path) -> None:
        def fail_the_join(argv: list[str], timeout_ms: int | None) -> Outcome | None:
            return Outcome(exit_code=1, stderr=b'no space') if 'part-*' in ' '.join(argv) else None

        fake_platinum.max_write_bytes = 10
        fake_platinum.exec_hook = fail_the_join
        with pytest.raises(WorkspaceError, match=r'Joining the uploaded parts of .* failed: no space'):
            await _attached(fake_platinum, work).write_bytes(str(work / 'f'), b'x' * 25)

    async def test_a_part_refused_without_a_stated_limit_fails_the_write(
        self, fake_platinum: FakePlatinum, work: Path
    ) -> None:
        async def refuse_parts(request: Request) -> httpx.Response | None:
            return _answer(413, 'payload_too_large') if request.query.get('path', [''])[0].count('part-') else None

        fake_platinum.max_write_bytes = 10
        fake_platinum.route_hook = refuse_parts
        with pytest.raises(WorkspaceError, match='payload_too_large'):
            await _attached(fake_platinum, work).write_bytes(str(work / 'f'), b'x' * 25)

    async def test_a_file_too_large_for_one_read_is_read_in_parts(
        self, fake_platinum: FakePlatinum, work: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(f'{_MODULE}._READ_PART_MIB', 1)
        fake_platinum.max_read_bytes = 1024 * 1024
        work.mkdir()
        data = os.urandom(2 * 1024 * 1024 + 100)
        (work / 'big.bin').write_bytes(data)
        assert await _attached(fake_platinum, work).read_bytes(str(work / 'big.bin')) == data
        assert list((tmp_path / 'scratch').iterdir()) == []

    async def test_a_part_that_cannot_be_copied_fails_the_read(self, fake_platinum: FakePlatinum, work: Path) -> None:
        fake_platinum.max_read_bytes = 1
        fake_platinum.exec_hook = lambda argv, timeout_ms: (
            Outcome(exit_code=1, stderr=b'dd: failed') if 'dd' in ' '.join(argv) else None
        )
        work.mkdir()
        (work / 'big.bin').write_bytes(b'xx')
        with pytest.raises(WorkspaceError, match=r'Could not read .* in parts: dd: failed'):
            await _attached(fake_platinum, work).read_bytes(str(work / 'big.bin'))

    async def test_scratch_files_left_behind_are_logged(
        self, fake_platinum: FakePlatinum, work: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def keep_scratch(request: Request) -> httpx.Response | None:
            path = request.query.get('path', [''])[0]
            return _answer(500, 'internal') if request.method == 'DELETE' and '.pydantic-ai-upload-' in path else None

        fake_platinum.max_write_bytes = 10
        fake_platinum.route_hook = keep_scratch
        with caplog.at_level(logging.DEBUG, logger=_MODULE):
            await _attached(fake_platinum, work).write_bytes(str(work / 'f'), b'x' * 25)
        assert 'removing' in caplog.text and 'failed' in caplog.text

    async def test_a_slow_read_whose_fifo_probe_fails_is_still_read(
        self, fake_platinum: FakePlatinum, work: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(f'{_MODULE}._FIFO_PROBE_AFTER', 0)
        probed = anyio.Event()

        def fail_the_probe(argv: list[str], timeout_ms: int | None) -> Exception:
            assert argv[-3:-1] == ['test', '-p']
            probed.set()
            return httpx.ConnectError('refused')

        async def slow_read(request: Request) -> httpx.Response | None:
            if request.method == 'GET' and request.path.endswith('/files'):
                await probed.wait()
            return None

        work.mkdir()
        (work / 'f').write_bytes(b'data')
        backend = _attached(fake_platinum, work)
        await backend.get_sandbox()
        fake_platinum.exec_hook, fake_platinum.route_hook = fail_the_probe, slow_read
        assert await backend.read_bytes(str(work / 'f')) == b'data'

    @pytest.mark.parametrize(
        ('response', 'expected', 'number'),
        [
            (_answer(507, 'disk_full', 'write {path}: no space left on device'), OSError, errno.ENOSPC),
            (_answer(409, 'directory_not_empty'), OSError, errno.ENOTEMPTY),
            (_answer(409, 'conflict', 'mkdir /x: file exists'), FileExistsError, errno.EEXIST),
            (_answer(500, 'internal', 'open {path}: permission denied'), PermissionError, errno.EACCES),
            (_answer(404, 'not_found', 'open {path}: permission denied'), PermissionError, errno.EACCES),
        ],
    )
    async def test_path_failures_are_the_builtin_errors(
        self, fake_platinum: FakePlatinum, work: Path, response: httpx.Response, expected: type[OSError], number: int
    ) -> None:
        path = str(work / 'f')
        body = response.json()
        body['error'] = body['error'].format(path=path)

        async def fail(request: Request) -> httpx.Response | None:
            return httpx.Response(response.status_code, json=body) if request.method == 'PUT' else None

        fake_platinum.route_hook = fail
        with pytest.raises(expected) as raised:
            await _attached(fake_platinum, work).write_bytes(path, b'x')
        assert raised.value.errno == number

    async def test_a_bad_path_request_is_a_value_error(self, fake_platinum: FakePlatinum, work: Path) -> None:
        async def refuse(request: Request) -> httpx.Response | None:
            return _answer(400, 'bad_request', 'path is too long') if request.path.endswith('/stat') else None

        fake_platinum.route_hook = refuse
        with pytest.raises(ValueError, match='path is too long'):
            await _attached(fake_platinum, work).stat(str(work))

    async def test_a_server_error_without_the_path_is_transient(self, fake_platinum: FakePlatinum, work: Path) -> None:
        async def fail(request: Request) -> httpx.Response | None:
            return _answer(500, 'internal', 'the host agent restarted') if request.method == 'PUT' else None

        fake_platinum.route_hook = fail
        with pytest.raises(platinum.ServerError):
            await _attached(fake_platinum, work).write_bytes(str(work / 'f'), b'x')

    @pytest.mark.parametrize('path', ['relative', '/with\0nul'])
    async def test_paths_must_be_absolute_without_nul(self, fake_platinum: FakePlatinum, path: str) -> None:
        backend = PlatinumSandboxBackend(ref=_ref(fake_platinum.add()))
        with pytest.raises(ValueError):
            await backend.read_bytes(path)
        with pytest.raises(ValueError):
            await backend.realpath('/with\0nul')

    async def test_paths_are_normalized_before_the_api_sees_them(self, fake_platinum: FakePlatinum, work: Path) -> None:
        backend = _attached(fake_platinum, work)
        await backend.write_bytes(f'/{work}/./sub/../f', b'x')
        assert (work / 'f').read_bytes() == b'x'
        entry = await backend.stat(f'//{work}/f')
        assert (entry.path, entry.size) == (str(work / 'f'), 1)

    async def test_a_listing_over_the_reply_limit_is_read_from_a_file(
        self, fake_platinum: FakePlatinum, work: Path, tmp_path: Path
    ) -> None:
        fake_platinum.max_list_entries = 1
        backend = _attached(fake_platinum, work)
        await backend.write_bytes(str(work / 'b'), b'x')
        await backend.make_dir(str(work / 'a'))
        entries = await backend.list_dir(str(work))
        assert [(e.name, e.path, e.is_dir, e.size) for e in entries] == [
            ('a', str(work / 'a'), True, None),
            ('b', str(work / 'b'), False, None),
        ]
        assert list((tmp_path / 'scratch').iterdir()) == []

    @pytest.mark.parametrize(
        ('name', 'expected'),
        [('missing', FileNotFoundError), ('file', NotADirectoryError)],
    )
    async def test_a_large_listing_of_what_is_not_a_directory_fails_like_one(
        self, fake_platinum: FakePlatinum, work: Path, name: str, expected: type[OSError]
    ) -> None:
        async def too_large(request: Request) -> httpx.Response | None:
            return _answer(413, 'list_too_large') if request.path.endswith('/list') else None

        backend = _attached(fake_platinum, work)
        await backend.write_bytes(str(work / 'file'), b'x')
        fake_platinum.route_hook = too_large
        with pytest.raises(expected):
            await backend.list_dir(str(work / name))

    async def test_a_large_listing_that_fails_otherwise_is_an_error(
        self, fake_platinum: FakePlatinum, work: Path
    ) -> None:
        async def too_large(request: Request) -> httpx.Response | None:
            return _answer(413, 'list_too_large') if request.path.endswith('/list') else None

        backend = _attached(fake_platinum, work)
        await backend.make_dir(str(work))
        fake_platinum.route_hook = too_large
        fake_platinum.exec_hook = lambda argv, timeout_ms: (
            Outcome(exit_code=2, stderr=b'find: oops') if 'find -H' in ' '.join(argv) else None
        )
        with pytest.raises(WorkspaceError, match=r'Listing .* failed: find: oops'):
            await backend.list_dir(str(work))

    async def test_listings_follow_symlinks_and_skip_nameless_entries(
        self, fake_platinum: FakePlatinum, work: Path
    ) -> None:
        backend = _attached(fake_platinum, work)
        await backend.write_bytes(str(work / 'target'), b'four')
        await backend.run(
            ['sh', '-c', 'ln -s target "$1/link"; ln -s missing "$1/dangling"', 'sh', str(work)], timeout=10
        )
        entries = {e.name: (e.is_dir, e.size) for e in await backend.list_dir(str(work))}
        assert entries == {'dangling': (False, None), 'link': (False, 4), 'target': (False, 4)}

        async def nameless(request: Request) -> httpx.Response:
            assert request.path.endswith('/list')
            return httpx.Response(
                200, json={'ok': True, 'entries': [{'name': ''}, {'name': 'x', 'size': 1, 'is_dir': False}]}
            )

        fake_platinum.route_hook = nameless
        assert [e.name for e in await backend.list_dir(str(work))] == ['x']

    async def test_a_failed_listing_is_mapped(self, fake_platinum: FakePlatinum, work: Path) -> None:
        with pytest.raises(FileNotFoundError):
            await _attached(fake_platinum, work).list_dir(str(work / 'missing'))

    @pytest.mark.parametrize(
        ('answer', 'expected'), [('not_a_directory', NotADirectoryError), ('conflict', WorkspaceError)]
    )
    async def test_a_make_dir_refused_for_an_existing_directory_keeps_the_refusal(
        self, fake_platinum: FakePlatinum, work: Path, answer: str, expected: type[Exception]
    ) -> None:
        async def refuse(request: Request) -> httpx.Response | None:
            return _answer(409, answer, 'cannot mkdir') if request.path.endswith('/mkdir') else None

        work.mkdir()
        fake_platinum.route_hook = refuse
        with pytest.raises(expected):
            await _attached(fake_platinum, work).make_dir(str(work))

    async def test_a_refused_remove_is_mapped(self, fake_platinum: FakePlatinum, work: Path) -> None:
        async def refuse(request: Request) -> httpx.Response | None:
            return _answer(409, 'busy') if request.method == 'DELETE' else None

        backend = _attached(fake_platinum, work)
        await backend.write_bytes(str(work / 'f'), b'x')
        fake_platinum.route_hook = refuse
        with pytest.raises(WorkspaceError, match='busy'):
            await backend.remove(str(work / 'f'))

    async def test_exists_is_false_for_a_symlink_loop_and_raises_for_other_failures(
        self, fake_platinum: FakePlatinum, work: Path
    ) -> None:
        backend = _attached(fake_platinum, work)
        await backend.run(['sh', '-c', 'ln -s loop "$1/loop"', 'sh', str(work)], timeout=10)
        assert not await backend.exists(str(work / 'loop'))

        async def refuse(request: Request) -> httpx.Response | None:
            return (
                _answer(404, 'not_found', f'stat {work}: permission denied') if request.path.endswith('/stat') else None
            )

        fake_platinum.route_hook = refuse
        with pytest.raises(PermissionError):
            await backend.exists(str(work))


class TestClientLifecycle:
    async def test_aclose_closes_the_backend_s_own_client_and_a_later_operation_reattaches(
        self, fake_platinum: FakePlatinum, work: Path
    ) -> None:
        backend = PlatinumSandboxBackend(working_dir=str(work))
        await backend.working_dir()
        await backend.aclose()
        assert fake_platinum.closed == fake_platinum.clients
        await backend.aclose()  # nothing left to close
        assert (await backend.run(['true'], timeout=10)).exit_code == 0
        assert len(fake_platinum.clients) == 2 and len(fake_platinum.sandboxes) == 1

    async def test_two_closes_at_once_close_the_client_once(self, fake_platinum: FakePlatinum, work: Path) -> None:
        backend = _attached(fake_platinum, work)
        await backend.get_sandbox()
        async with anyio.create_task_group() as group:
            group.start_soon(backend.aclose)
            group.start_soon(backend.aclose)
        assert fake_platinum.closed == fake_platinum.clients

    async def test_a_client_you_pass_is_left_open(self, fake_platinum: FakePlatinum, work: Path) -> None:
        client = fake_platinum.client()
        backend = PlatinumSandboxBackend(client=client, working_dir=str(work))
        await backend.working_dir()
        await backend.aclose()
        assert client not in fake_platinum.closed

    async def test_closing_waits_for_the_operation_in_flight(self, fake_platinum: FakePlatinum, work: Path) -> None:
        entered, release = anyio.Event(), anyio.Event()

        async def hold(request: Request) -> httpx.Response | None:
            assert request.path.endswith('/stat')
            entered.set()
            await release.wait()
            return None

        backend = _attached(fake_platinum, work)
        await backend.make_dir(str(work))
        fake_platinum.route_hook = hold
        async with anyio.create_task_group() as group:
            group.start_soon(backend.stat, str(work))
            await entered.wait()
            await backend.aclose()
            assert fake_platinum.closed == []
            release.set()
        assert fake_platinum.closed == fake_platinum.clients

    async def test_a_client_that_fails_to_close_is_logged(
        self, fake_platinum: FakePlatinum, work: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _attached(fake_platinum, work)
        await backend.get_sandbox()

        client = fake_platinum.clients[0]
        close = client.aclose

        async def broken() -> None:
            monkeypatch.setattr(client, 'aclose', close)  # so the fake's teardown can close it
            raise RuntimeError('socket gone')

        monkeypatch.setattr(client, 'aclose', broken)
        with caplog.at_level(logging.WARNING, logger=_MODULE):
            await backend.aclose()
        assert 'Could not close the Platinum API client' in caplog.text


@pytest.mark.parametrize('anyio_backend', ['trio'])
async def test_trio_is_refused_before_any_platinum_call() -> None:
    """The SDK runs its requests on asyncio, so Trio gets a clear `UserError` instead."""
    message = 'Platinum needs the asyncio event loop'
    backend = PlatinumSandboxBackend(ref=_ref('sbx_1'))
    with pytest.raises(UserError, match=message):
        await backend.run(['true'])
    with pytest.raises(UserError, match=message):
        await backend.get_sandbox()
    with pytest.raises(UserError, match=message):
        await PlatinumSandbox().destroy(_ref('sbx_1'))
    await backend.aclose()  # a no-op off asyncio


class TestDestroy:
    async def test_destroy_deletes_without_starting_and_closes_its_client(self, fake_platinum: FakePlatinum) -> None:
        sandbox_id = fake_platinum.add(state='stopped')
        await PlatinumSandbox().destroy(_ref(sandbox_id))
        assert fake_platinum.sandboxes[sandbox_id].state == 'deleted'
        assert [(r.method, r.path) for r in fake_platinum.requests] == [('DELETE', f'/v1/sandboxes/{sandbox_id}')]
        assert fake_platinum.closed == fake_platinum.clients

    async def test_destroying_a_missing_sandbox_returns_quietly(self, fake_platinum: FakePlatinum) -> None:
        client = fake_platinum.client()
        await PlatinumSandbox(client=client).destroy(_ref('sbx_gone'))
        assert fake_platinum.closed == []

    async def test_destroy_with_rejected_credentials_is_unavailable(self, fake_platinum: FakePlatinum) -> None:
        async def reject(request: Request) -> httpx.Response | None:
            return _answer(403, 'insufficient_scope')

        fake_platinum.route_hook = reject
        with pytest.raises(WorkspaceUnavailableError, match='Platinum refused the API key'):
            await PlatinumSandbox().destroy(_ref(fake_platinum.add()))

    async def test_destroy_refuses_another_provider_s_ref(self, fake_platinum: FakePlatinum) -> None:
        with pytest.raises(ValueError, match="expected 'platinum'"):
            await PlatinumSandbox().destroy(WorkspaceRef(provider='e2b', id='sbx'))


async def test_a_workspace_over_the_backend_reaches_the_sandbox(fake_platinum: FakePlatinum, work: Path) -> None:
    workspace = Workspace(_attached(fake_platinum, work))
    await workspace.write_text('notes/todo.txt', 'ship it\n')
    assert await workspace.read_text(str(work / 'notes' / 'todo.txt')) == 'ship it\n'
    assert (await workspace.run('cat notes/todo.txt', shell=True)).stdout == 'ship it\n'
