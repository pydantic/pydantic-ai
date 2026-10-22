"""Protocol and API-boundary tests for Boat sandboxes, over the fake Boat API.

They are not VCR tests: what they pin is how the backend reacts to answers Boat gives only on a bad
day (lost replies, refusals, a sandbox stopped or deleted mid-run, retry signals), which a recording
of a working session does not hold. The fake serves Boat's real envelopes and runs real commands.
"""

from __future__ import annotations

import asyncio
import errno
import logging
import os
import pickle
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import anyio
import httpx
import pytest

from pydantic_ai.workspaces import (
    CommandResult,
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness.boat_sandbox import BoatAPIError, BoatSandbox, BoatSandboxBackend

# The factory the backend opens its own clients with; it is not exported, and a test needs it to
# check the client that reaches Boat.
from pydantic_ai_harness.boat_sandbox._api import open_client

from .fake_boat import FakeBoat, FakeCommand, FakeSandbox, boat_error, boat_ok

pytestmark = pytest.mark.anyio(backends=['asyncio'])

_BACKEND = 'pydantic_ai_harness.boat_sandbox._backend'
_AUTH = 'Boat rejected the credentials. Set BOAT_API_KEY, or pass a configured `httpx.AsyncClient` as `client=`.'
_FRESH = "Pass `workspace='new'` to start a fresh sandbox."


async def started(backend: BoatSandboxBackend) -> BoatSandboxBackend:
    """Resolve `backend` now: building one does no I/O, so a test that asserts on creating or attaching touches it first."""
    await backend.working_dir()
    return backend


def only(fake: FakeBoat) -> FakeSandbox:
    [sandbox] = fake.sandboxes.values()
    return sandbox


def only_command(fake: FakeBoat) -> FakeCommand:
    [command] = only(fake).commands.values()
    return command


def unmarked(command: object) -> str:
    assert isinstance(command, str)
    return re.sub(r'pydantic-ai-[0-9a-f]{32}', 'MARK', command)


def routes(fake: FakeBoat) -> list[str]:
    return [call.route for call in fake.calls]


async def until(condition: Callable[[], bool]) -> None:
    with anyio.fail_after(10):
        while not condition():
            await anyio.sleep(0.001)


def status(*, running: bool, exit_code: int | None = None, stdout: str = '', stderr: str = '') -> httpx.Response:
    """Boat's answer to a check of a detached command."""
    return boat_ok(
        'command.status',
        success=True,
        processId=1,
        status='running' if running else 'exited',
        running=running,
        known=True,
        exitCode=exit_code,
        stdout=stdout,
        stderr=stderr,
        stdoutTruncated=False,
        stderrTruncated=False,
    )


def finished(
    *, exit_code: int = 0, stdout: str = '', stderr: str = '', timed_out: bool = False, truncated: bool = False
) -> httpx.Response:
    """Boat's answer to a synchronous command."""
    return boat_ok(
        'command.finished',
        success=exit_code == 0,
        exitCode=exit_code,
        stdout=stdout,
        stderr=stderr,
        stdoutTruncated=truncated,
        stderrTruncated=False,
        timedOut=timed_out,
    )


def ip_limited(*, retry_after: float) -> httpx.Response:
    """Boat's answer once one IP address sent more than 1,000 requests in a minute, with the wait in the body."""
    body = boat_error(429, 'IP_RATE_LIMIT_EXCEEDED', 'Too many requests.').json()
    return httpx.Response(429, json={**body, 'retryAfter': retry_after})


def record_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The delays slept from now on, each returned at once instead of waited out."""
    delays: list[float] = []
    sleep = anyio.sleep

    async def record(delay: float) -> None:
        delays.append(delay)
        await sleep(0)

    monkeypatch.setattr(anyio, 'sleep', record)
    return delays


def without_repeats(monkeypatch: pytest.MonkeyPatch) -> None:
    """Close the window in which a call that is safe to repeat is sent again, so its first failure surfaces."""
    monkeypatch.setattr('pydantic_ai_harness.boat_sandbox._api._RETRY_WINDOW', 0)


def advance_clock(monkeypatch: pytest.MonkeyPatch) -> Callable[[float], None]:
    """A function that moves the clock the backend reads forward by some seconds, without waiting them out."""
    offset = 0.0
    now = anyio.current_time

    def advance(seconds: float) -> None:
        nonlocal offset
        offset += seconds

    monkeypatch.setattr(anyio, 'current_time', lambda: now() + offset)
    return advance


def unavailable(sandbox_id: str) -> str:
    return f'The Boat sandbox {sandbox_id!r} no longer exists (it was deleted). {_FRESH}'


def failed(sandbox_id: str) -> str:
    return f"The Boat sandbox {sandbox_id!r} is in the 'error' state and cannot be used. {_FRESH}"


def stopped(sandbox_id: str) -> str:
    return f'The Boat sandbox {sandbox_id!r} was stopped; a later run that attaches to it resumes it. {_FRESH}'


@pytest.mark.parametrize(
    ('build', 'error', 'message'),
    [
        pytest.param(
            lambda: BoatSandboxBackend(ref=WorkspaceRef(provider='daytona', id='x')),
            ValueError,
            r"^unsupported workspace provider 'daytona'; expected 'boat'$",
            id='foreign-ref',
        ),
        pytest.param(
            lambda: BoatSandboxBackend(ref=WorkspaceRef(provider='boat', id='  ')),
            ValueError,
            r'^Boat workspace ref id cannot be empty$',
            id='blank-ref',
        ),
        pytest.param(
            lambda: BoatSandboxBackend(working_dir='repo'),
            ValueError,
            r"^working_dir must be an absolute workspace path or None, got 'repo'\.$",
            id='relative-working-dir',
        ),
        pytest.param(
            lambda: BoatSandboxBackend(env={'PORT': 3000}),  # pyright: ignore[reportArgumentType]
            TypeError,
            r'^env keys and values must be strings$',
            id='non-string-env',
        ),
    ],
)
def test_invalid_settings_fail_at_construction(
    build: Callable[[], object], error: type[Exception], message: str
) -> None:
    # No fake: the autouse tripwire fails the test if construction reached for a Boat client.
    with pytest.raises(error, match=message):
        build()


class TestCreation:
    @pytest.mark.parametrize(
        ('build', 'body'),
        [
            pytest.param(BoatSandboxBackend, {'noEnv': True}, id='defaults'),
            pytest.param(
                lambda: BoatSandboxBackend(machine_type='small', snapshot='base', auto_stop=900),
                {'noEnv': True, 'type': 'small', 'from': 'base', 'ttlSeconds': 900},
                id='settings',
            ),
            # Boat reads `0` as its default of an hour; `null` turns the auto-stop off.
            pytest.param(
                lambda: BoatSandboxBackend(auto_stop=0), {'noEnv': True, 'ttlSeconds': None}, id='no-auto-stop'
            ),
        ],
    )
    async def test_creation_sends_the_settings_and_keeps_the_account_credentials_out(
        self, fake_boat: FakeBoat, build: Callable[[], BoatSandboxBackend], body: dict[str, object]
    ) -> None:
        fake_boat.snapshots.add('base')
        backend = await started(build())
        assert fake_boat.create_bodies == [body]
        [create] = [call.request for call in fake_boat.calls if call.route == 'create']
        assert re.fullmatch(r'pydantic-ai-[0-9a-f]{32}', create.headers['idempotency-key'])
        assert backend.ref == WorkspaceRef(provider='boat', id=only(fake_boat).id)

    async def test_a_new_sandbox_gets_its_working_dir_and_an_attached_one_does_not(
        self, fake_boat: FakeBoat, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        directory = tmp_path.resolve() / 'project' / 'nested'
        with caplog.at_level(logging.INFO, logger=_BACKEND):
            backend = await started(BoatSandboxBackend(working_dir=str(directory)))
        sandbox = only(fake_boat)
        # Logged at creation, so a sandbox whose run ended before its ref was stored can be found.
        assert caplog.messages == [f'Created Boat sandbox {sandbox.id}']
        assert await backend.working_dir() == str(directory)
        mkdir = f'exec sh -c \': MARK; "$@" </dev/null; exit $?\' sh mkdir -p -- {directory}'
        made = [body for body in fake_boat.bodies('run') if 'mkdir' in str(body['command'])]
        assert [{**body, 'command': unmarked(body['command'])} for body in made] == [
            {'command': mkdir, 'timeoutSeconds': 30}
        ]

        attached = await started(BoatSandboxBackend(ref=backend.ref, working_dir=str(directory)))
        assert await attached.working_dir() == str(directory)
        assert [body for body in fake_boat.bodies('run') if 'mkdir' in str(body['command'])] == made
        assert len(fake_boat.create_bodies) == 1

    async def test_a_failed_working_dir_is_retried_by_the_next_operation(
        self, fake_boat: FakeBoat, tmp_path: Path
    ) -> None:
        blocker = tmp_path.resolve() / 'file'
        blocker.write_bytes(b'')
        directory = blocker / 'project'
        backend = BoatSandboxBackend(working_dir=str(directory))
        with pytest.raises(
            WorkspaceError,
            match=rf'^Could not create working_dir {re.escape(repr(str(directory)))} in Boat sandbox bx_\w+: '
            r'mkdir: .*Not a directory$',
        ):
            await backend.run(['true'])
        # The sandbox is recorded, so the caller can clean it up, and the command never started.
        assert backend.ref == WorkspaceRef(provider='boat', id=only(fake_boat).id)
        assert fake_boat.bodies('start') == []
        blocker.unlink()
        assert await backend.working_dir() == str(directory)
        assert len([body for body in fake_boat.bodies('run') if 'mkdir' in str(body['command'])]) == 2
        assert len(fake_boat.create_bodies) == 1

    async def test_a_lost_creation_reply_recovers_the_accepted_sandbox(self, fake_boat: FakeBoat) -> None:
        fake_boat.lost_replies['create'] = 1
        backend = await started(BoatSandboxBackend())
        assert backend.ref == WorkspaceRef(provider='boat', id=only(fake_boat).id)
        keys = [call.request.headers['idempotency-key'] for call in fake_boat.calls if call.route == 'create']
        assert len(keys) == 2 and len(set(keys)) == 1

    async def test_a_creation_still_in_progress_elsewhere_is_waited_for(self, fake_boat: FakeBoat) -> None:
        message = 'A request with this Idempotency-Key is still in progress.'
        fake_boat.errors['create'] = [boat_error(409, 'idempotency_in_progress', message)]
        await started(BoatSandboxBackend())
        assert routes(fake_boat).count('create') == 2
        assert len(fake_boat.sandboxes) == 1

    async def test_a_retried_creation_finds_the_sandbox_a_lost_reply_left(self, fake_boat: FakeBoat) -> None:
        # The sandbox is created but its reply is lost, and the repeat that would recover it is refused.
        fake_boat.errors['create'] = [None, boat_error(503, 'no_ready_machine', 'No machine is ready.')]
        fake_boat.lost_replies['create'] = 1
        backend = BoatSandboxBackend()
        with pytest.raises(BoatAPIError) as exc_info:
            await backend.working_dir()
        assert (exc_info.value.status, exc_info.value.code) == (503, 'no_ready_machine')
        assert backend.ref is None
        # The next operation repeats the creation under the same key instead of starting a second sandbox.
        await backend.working_dir()
        assert backend.ref == WorkspaceRef(provider='boat', id=only(fake_boat).id)

    @pytest.mark.parametrize(
        ('answer', 'message'),
        [
            pytest.param((401, 'unauthorized', 'unauthorized'), f'Credentials rejected. {_AUTH}', id='unauthorized'),
            pytest.param((403, 'forbidden', 'forbidden'), f'Credentials rejected. {_AUTH}', id='forbidden'),
            pytest.param(
                (403, 'trial_machine_class_not_allowed', 'Trial accounts start small machines only.'),
                'Could not start Boat sandbox: Trial accounts start small machines only.',
                id='plan',
            ),
            pytest.param(
                (402, 'billing_required', 'Add a payment method to start sandboxes.'),
                'Could not start Boat sandbox: Add a payment method to start sandboxes.',
                id='billing',
            ),
            # A 429 that no timed retry clears: the trial ends only by paying.
            pytest.param(
                (429, 'trial_compute_limit_reached', 'The trial used its 25 hours.'),
                'Could not start Boat sandbox: The trial used its 25 hours.',
                id='trial-hours',
            ),
            pytest.param(
                (409, 'idempotency_key_reused', 'This Idempotency-Key was already used with a different request body.'),
                'Could not start Boat sandbox: Boat API returned HTTP 409 (idempotency_key_reused): '
                'This Idempotency-Key was already used with a different request body.',
                id='key-reused',
            ),
        ],
    )
    async def test_a_refused_creation_is_unavailable(
        self, fake_boat: FakeBoat, answer: tuple[int, str, str], message: str
    ) -> None:
        fake_boat.errors['create'] = [boat_error(*answer)]
        with pytest.raises(WorkspaceUnavailableError) as exc_info:
            await BoatSandboxBackend().run(['true'])
        assert str(exc_info.value) == message
        assert isinstance(exc_info.value.__cause__, BoatAPIError)
        assert not fake_boat.sandboxes

    @pytest.mark.parametrize(
        ('build', 'message'),
        [
            pytest.param(
                lambda: BoatSandboxBackend(snapshot='nope'),
                'Could not start Boat sandbox: Boat API returned HTTP 404 (snapshot_not_found): '
                'No named snapshot "nope". Run boat snapshots to list yours.',
                id='unknown-snapshot',
            ),
            pytest.param(
                # A size outside `MachineType`, as an untyped caller can pass.
                lambda: BoatSandboxBackend(machine_type='huge'),  # pyright: ignore[reportArgumentType]
                'Could not start Boat sandbox: Boat API returned HTTP 400 (invalid_type): '
                'type must be one of small, default, large, xlarge',
                id='unknown-machine-type',
            ),
        ],
    )
    async def test_settings_boat_refuses_are_unavailable(
        self, fake_boat: FakeBoat, build: Callable[[], BoatSandboxBackend], message: str
    ) -> None:
        # No retry can fix them, so the run ends rather than being retried by a durable engine.
        with pytest.raises(WorkspaceUnavailableError) as exc_info:
            await build().run(['true'])
        assert str(exc_info.value) == message
        assert not fake_boat.sandboxes

    @pytest.mark.parametrize(
        ('answer', 'status', 'code'),
        [
            pytest.param(lambda: boat_error(429, 'rate_limited', 'Too many starts.'), 429, 'rate_limited', id='429'),
            pytest.param(lambda: boat_error(503, 'out_of_capacity', 'No capacity.'), 503, 'out_of_capacity', id='503'),
            pytest.param(
                lambda: httpx.Response(502, text='<html>Bad Gateway</html>'), 502, 'invalid_response', id='html'
            ),
            pytest.param(lambda: httpx.Response(202, json={'ok': True}), 202, 'invalid_response', id='malformed-202'),
        ],
    )
    async def test_a_transient_creation_failure_propagates_unchanged(
        self, fake_boat: FakeBoat, answer: Callable[[], httpx.Response], status: int, code: str
    ) -> None:
        fake_boat.errors['create'] = [answer()]
        with pytest.raises(BoatAPIError) as exc_info:
            await BoatSandboxBackend().run(['true'])
        assert (exc_info.value.status, exc_info.value.code) == (status, code)
        if code == 'invalid_response':
            assert (
                str(exc_info.value) == f'Boat API returned HTTP {status} (invalid_response): Unexpected Boat response'
            )

    async def test_a_transient_failure_survives_pickling(self, fake_boat: FakeBoat) -> None:
        # Durable engines and process pools pickle the errors an operation raises.
        body = boat_error(503, 'out_of_capacity', 'No capacity.', retryable=True).json()
        fake_boat.errors['create'] = [httpx.Response(503, json={**body, 'retryAfterSeconds': 120})]
        with pytest.raises(BoatAPIError) as exc_info:
            await BoatSandboxBackend().run(['true'])
        error = exc_info.value
        restored = pickle.loads(pickle.dumps(error))
        assert type(restored) is BoatAPIError
        assert (restored.status, restored.code, restored.message, restored.retryable, restored.retry_after) == (
            503,
            'out_of_capacity',
            'No capacity.',
            True,
            120,
        )
        assert str(restored) == str(error) == 'Boat API returned HTTP 503 (out_of_capacity): No capacity.'

    async def test_rejected_credentials_do_not_expose_the_key(self, fake_boat: FakeBoat) -> None:
        fake_boat.errors['create'] = [boat_error(401, 'unauthorized', 'expired token secret-value')]
        with pytest.raises(WorkspaceUnavailableError) as exc_info:
            await BoatSandboxBackend().working_dir()
        assert str(exc_info.value) == f'Credential expired. {_AUTH}'
        # A key Boat no longer accepts: the request the backend built carries it, the error does not.
        fake_boat.api_key = 'rotated-key'
        with pytest.raises(WorkspaceUnavailableError) as exc_info:
            await BoatSandboxBackend().working_dir()
        assert str(exc_info.value) == f'Credentials rejected. {_AUTH}'
        assert 'test-key' not in str(exc_info.value)

    @pytest.mark.parametrize('key', [None, '  \n'])
    async def test_missing_api_key_is_unavailable(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch, key: str | None
    ) -> None:
        if key is None:
            monkeypatch.delenv('BOAT_API_KEY')
        else:
            monkeypatch.setenv('BOAT_API_KEY', key)
        with pytest.raises(WorkspaceUnavailableError) as exc_info:
            await BoatSandboxBackend().run(['true'])
        assert str(exc_info.value) == (
            'No Boat credentials found. Set BOAT_API_KEY, or pass a configured `httpx.AsyncClient` as `client=`.'
        )
        assert fake_boat.opened_clients == 0

    async def test_the_client_opened_from_boat_api_key_authenticates_against_boats_api(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The real factory, with the fake answering at the transport: its URL, header and timeout are what reach Boat."""
        opened: list[httpx.AsyncClient] = []

        def open_and_keep(api_key: str) -> httpx.AsyncClient:
            opened.append(client := open_client(api_key))
            return client

        async def answer(transport: httpx.AsyncHTTPTransport, request: httpx.Request) -> httpx.Response:
            return await fake_boat.handle(request)

        monkeypatch.setattr('pydantic_ai_harness.boat_sandbox._api.open_client', open_and_keep)
        monkeypatch.setattr(httpx.AsyncHTTPTransport, 'handle_async_request', answer)
        monkeypatch.setenv('BOAT_API_KEY', f' {fake_boat.api_key}\n')
        backend = BoatSandboxBackend()
        # The fake refuses any request that is not to Boat's API URL with `Bearer test-key`.
        assert (await backend.run(['echo', 'hi'])).stdout == 'hi\n'
        [client] = opened
        assert (str(client.base_url), client.timeout) == ('https://boat.dev/api/v1/', httpx.Timeout(30))
        await backend.aclose()
        assert client.is_closed

    async def test_a_client_without_boats_url_fails_at_once(self) -> None:
        # The request never leaves the client, so nothing was created: no retry, and no wait for the creation bound.
        async with httpx.AsyncClient() as client:
            backend = BoatSandboxBackend(client=client)
            with anyio.fail_after(5), pytest.raises(httpx.UnsupportedProtocol):
                await backend.run(['true'])
        assert backend.ref is None

    async def test_a_create_that_never_answers_is_transient(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(f'{_BACKEND}._CREATE_TIMEOUT', 0.05)
        fake_boat.gates['create'] = asyncio.Event()
        backend = BoatSandboxBackend()
        with pytest.raises(TimeoutError) as exc_info:
            await backend.run(['true'])
        # The builtin, not a command timeout: a durable engine retries it.
        assert type(exc_info.value) is TimeoutError
        assert str(exc_info.value) == (
            'Boat sandbox creation did not complete within 0.05s; the Boat control plane may be unreachable.'
        )
        assert backend.ref is None


class TestCancellationDuringCreation:
    async def test_native_cancellation_during_committed_creation_keeps_ref(self, fake_boat: FakeBoat) -> None:
        gate = fake_boat.gates['create'] = asyncio.Event()
        backend = BoatSandboxBackend()
        task = asyncio.create_task(backend.working_dir())
        await fake_boat.reached['create'].wait()
        task.cancel()
        task.cancel()
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert backend.ref == WorkspaceRef(provider='boat', id=only(fake_boat).id)

    async def test_a_creation_failure_after_cancellation_is_not_reported_to_the_loop(self, fake_boat: FakeBoat) -> None:
        # Creation keeps going after the caller is cancelled, then Boat refuses it.
        gate = fake_boat.gates['create'] = asyncio.Event()
        fake_boat.errors['create'] = [boat_error(400, 'invalid_type', 'type must be one of small, default')]
        backend = BoatSandboxBackend()
        # What reaches the loop's exception handler, such as a failure reported as never retrieved.
        loop = asyncio.get_running_loop()
        errors: list[dict[str, Any]] = []
        previous = loop.get_exception_handler()
        loop.set_exception_handler(lambda _, context: errors.append(context))
        try:
            task = asyncio.create_task(backend.working_dir())
            await fake_boat.reached['create'].wait()
            task.cancel()
            await asyncio.sleep(0)
            gate.set()
            # The caller's cancellation wins; the refusal is retrieved, not reported.
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            loop.set_exception_handler(previous)
        assert errors == []

    async def test_cancellation_during_creation_keeps_the_created_sandbox(
        self, fake_boat: FakeBoat, tmp_path: Path
    ) -> None:
        directory = tmp_path.resolve() / 'work'
        gate = fake_boat.gates['create'] = asyncio.Event()
        backend = BoatSandboxBackend(working_dir=str(directory))
        with anyio.CancelScope() as scope:

            async def cancel_once_created() -> None:
                await fake_boat.reached['create'].wait()
                scope.cancel()
                gate.set()

            async with anyio.create_task_group() as tasks:
                tasks.start_soon(cancel_once_created)
                await backend.run(['true'])
        assert scope.cancelled_caught
        assert backend.ref == WorkspaceRef(provider='boat', id=only(fake_boat).id)
        # The cancelled call neither starts the command nor makes `working_dir`; the next operation does.
        assert fake_boat.bodies('start') == []
        assert not directory.exists()
        assert await backend.working_dir() == str(directory)
        assert len(fake_boat.create_bodies) == 1

    async def test_the_command_timeout_starts_once_the_sandbox_is_acquired(self, fake_boat: FakeBoat) -> None:
        gate = fake_boat.gates['create'] = asyncio.Event()
        # A command Boat finishes at once, so only acquiring the sandbox outlasts the timeout.
        fake_boat.errors['start'] = [boat_ok('command.started', processId=4242)]
        fake_boat.errors['status'] = [status(running=False, exit_code=0, stdout='ready\n')]

        async def release() -> None:
            await anyio.sleep(0.1)
            gate.set()

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(release)
            result = await BoatSandboxBackend().run(['echo', 'ready'], timeout=0.05)
            assert result == CommandResult(exit_code=0, stdout='ready\n', stderr='')


class TestAttach:
    async def test_attaching_resumes_a_stopped_sandbox(self, fake_boat: FakeBoat) -> None:
        existing = fake_boat.sandbox('archived')
        backend = BoatSandboxBackend(ref=WorkspaceRef(provider='boat', id=existing.id))
        assert (await backend.run(['echo', 'resumed'])).stdout == 'resumed\n'
        assert existing.resume_calls == 1
        assert fake_boat.create_bodies == []

    async def test_operations_wait_until_a_starting_sandbox_is_ready(self, fake_boat: FakeBoat) -> None:
        # File calls sent to a starting sandbox can report an existing file as missing, so none is sent early.
        fake_boat.boot_polls = 2
        existing = fake_boat.sandbox('provisioning')
        backend = BoatSandboxBackend(ref=WorkspaceRef(provider='boat', id=existing.id))
        assert await backend.exists(f'{existing.home}/missing') is False
        # The first operation on an attached sandbox extends its auto-stop.
        assert routes(fake_boat)[:5] == ['get', 'get', 'get', 'update', 'run']

    @pytest.mark.parametrize('operation', ['run', 'read_bytes', 'exists', 'working_dir'])
    async def test_a_missing_sandbox_remains_terminal(
        self, fake_boat: FakeBoat, tmp_path: Path, operation: str
    ) -> None:
        backend = BoatSandboxBackend(ref=WorkspaceRef(provider='boat', id='bx_missing'))
        calls: dict[str, Callable[[], Awaitable[object]]] = {
            'run': lambda: backend.run(['true']),
            'read_bytes': lambda: backend.read_bytes(f'{tmp_path}/note'),
            'exists': lambda: backend.exists(f'{tmp_path}/note'),
            'working_dir': backend.working_dir,
        }
        for _ in range(2):
            with pytest.raises(WorkspaceUnavailableError) as exc_info:
                await calls[operation]()
            assert str(exc_info.value) == (
                "The Boat sandbox 'bx_missing' was not found (it was deleted, or never existed in this Boat "
                f'account). {_FRESH}'
            )
        # A dead reference is reported, not replaced with a fresh sandbox.
        assert backend.ref == WorkspaceRef(provider='boat', id='bx_missing')
        assert fake_boat.create_bodies == []

    @pytest.mark.parametrize('state', ['error', 'cancelled'])
    async def test_a_failed_sandbox_is_unavailable(self, fake_boat: FakeBoat, state: str) -> None:
        existing = fake_boat.sandbox(state)
        with pytest.raises(WorkspaceUnavailableError) as exc_info:
            await BoatSandboxBackend(ref=WorkspaceRef(provider='boat', id=existing.id)).run(['true'])
        assert str(exc_info.value) == (
            f'The Boat sandbox {existing.id!r} is in the {state!r} state and cannot be used. {_FRESH}'
        )

    async def test_a_resumed_sandbox_is_checked_again_after_a_pause(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(f'{_BACKEND}._READY_POLL', 0.5)
        existing = fake_boat.sandbox('archived')
        delays = record_sleeps(monkeypatch)
        await BoatSandboxBackend(ref=WorkspaceRef(provider='boat', id=existing.id)).working_dir()
        assert routes(fake_boat) == ['get', 'resume', 'get', 'run']
        assert delays == [0.5]

    @pytest.mark.parametrize('code', ['sandbox_resuming', 'sandbox_stopping', 'sandbox_starting'])
    async def test_a_resume_refused_while_the_sandbox_changes_state_is_sent_again(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch, code: str
    ) -> None:
        monkeypatch.setattr(f'{_BACKEND}._READY_POLL', 0.5)
        existing = fake_boat.sandbox('archived')
        fake_boat.errors['resume'] = [boat_error(409, code, 'The sandbox is changing state.')]
        delays = record_sleeps(monkeypatch)
        backend = BoatSandboxBackend(ref=WorkspaceRef(provider='boat', id=existing.id))
        assert (await backend.run(['echo', 'resumed'])).stdout == 'resumed\n'
        assert routes(fake_boat)[:5] == ['get', 'resume', 'get', 'resume', 'get']
        assert delays[:2] == [0.5, 0.5]
        assert existing.resume_calls == 1

    @pytest.mark.parametrize(
        ('code', 'message'),
        [
            ('snapshot_data_expired', 'The sandbox stayed stopped for 14 days.'),
            ('resume_failed', 'The snapshot of this sandbox has expired.'),
        ],
    )
    async def test_a_sandbox_boat_cannot_resume_is_unavailable(
        self, fake_boat: FakeBoat, code: str, message: str
    ) -> None:
        existing = fake_boat.sandbox('archived')
        fake_boat.errors['resume'] = [boat_error(409, code, message)]
        with pytest.raises(WorkspaceUnavailableError) as exc_info:
            await BoatSandboxBackend(ref=WorkspaceRef(provider='boat', id=existing.id)).run(['true'])
        assert str(exc_info.value) == f'Could not resume Boat sandbox {existing.id!r}: {message}'
        assert routes(fake_boat) == ['get', 'resume']

    async def test_attach_errors_are_translated_or_propagate(self, fake_boat: FakeBoat) -> None:
        existing = fake_boat.sandbox()
        ref = WorkspaceRef(provider='boat', id=existing.id)
        offline = httpx.ConnectError('offline')
        fake_boat.errors['get'] = [
            boat_error(400, 'invalid_request', 'Bad request.'),
            boat_error(401, 'unauthorized', 'unauthorized'),
            boat_error(429, 'rate_limited', 'Too many requests.'),
            offline,
        ]
        with pytest.raises(WorkspaceError) as refused:
            await BoatSandboxBackend(ref=ref).working_dir()
        assert type(refused.value) is WorkspaceError
        assert str(refused.value) == (
            f'Could not attach to Boat sandbox {existing.id!r}: Boat API returned HTTP 400 (invalid_request): Bad request.'
        )
        with pytest.raises(WorkspaceUnavailableError, match=re.escape(_AUTH)):
            await BoatSandboxBackend(ref=ref).working_dir()
        with pytest.raises(BoatAPIError) as limited:
            await BoatSandboxBackend(ref=ref).working_dir()
        assert limited.value.status == 429
        with pytest.raises(httpx.ConnectError) as raised:
            await BoatSandboxBackend(ref=ref).working_dir()
        assert raised.value is offline

    @pytest.mark.parametrize(
        'answer',
        [
            pytest.param(lambda: httpx.Response(200, json={'ok': True, 'type': 'sandbox.info'}), id='malformed'),
            pytest.param(lambda: httpx.Response(502, text='<html>Bad Gateway</html>'), id='not-json'),
        ],
    )
    async def test_an_answer_that_is_not_boats_envelope_is_transient(
        self, fake_boat: FakeBoat, answer: Callable[[], httpx.Response]
    ) -> None:
        existing = fake_boat.sandbox()
        backend = BoatSandboxBackend(ref=WorkspaceRef(provider='boat', id=existing.id))
        fake_boat.errors['get'] = [answer()]
        with pytest.raises(BoatAPIError, match=r'^Boat API returned HTTP \d+ \(invalid_response\): Unexpected Boat'):
            await backend.working_dir()
        assert await backend.working_dir() == str(existing.home)

    async def test_an_attach_that_never_answers_is_transient(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        existing = fake_boat.sandbox()
        monkeypatch.setattr(f'{_BACKEND}._CREATE_TIMEOUT', 0.05)
        fake_boat.gates['get'] = asyncio.Event()
        with pytest.raises(TimeoutError) as exc_info:
            await BoatSandboxBackend(ref=WorkspaceRef(provider='boat', id=existing.id)).working_dir()
        assert str(exc_info.value) == f'Connecting to Boat sandbox {existing.id!r} did not complete within 0.05s.'


class TestAutoStop:
    """Boat stops a sandbox once its auto-stop runs out from creation or resume, whatever runs in it."""

    # At most once a minute, or once per half of a shorter auto-stop.
    @pytest.mark.parametrize(('auto_stop', 'interval'), [(900, 60), (100, 50)])
    async def test_an_operation_extends_the_auto_stop_at_most_once_an_interval(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch, auto_stop: int, interval: int
    ) -> None:
        advance = advance_clock(monkeypatch)
        backend = await started(BoatSandboxBackend(auto_stop=auto_stop))
        path = f'{only(fake_boat).home}/note'
        advance(interval - 1)
        assert await backend.exists(path) is False
        assert fake_boat.bodies('update') == []
        advance(1)
        assert await backend.exists(path) is False
        # A full auto-stop from now, which the next operations do not repeat until the interval has passed again.
        assert fake_boat.bodies('update') == [{'ttlSeconds': auto_stop}]
        advance(interval - 1)
        assert await backend.exists(path) is False
        assert fake_boat.bodies('update') == [{'ttlSeconds': auto_stop}]
        advance(1)
        assert await backend.exists(path) is False
        assert fake_boat.bodies('update') == [{'ttlSeconds': auto_stop}, {'ttlSeconds': auto_stop}]

    @pytest.mark.parametrize(('state', 'updates'), [('idle', [{'ttlSeconds': 3600}]), ('archived', [])])
    async def test_attaching_extends_a_running_sandbox_and_a_resume_sets_it_again(
        self, fake_boat: FakeBoat, state: str, updates: list[dict[str, object]]
    ) -> None:
        # When a running sandbox's deadline was set is unknown, so its first operation sets it.
        existing = fake_boat.sandbox(state)
        backend = BoatSandboxBackend(ref=WorkspaceRef(provider='boat', id=existing.id))
        for _ in range(2):
            assert await backend.exists(f'{existing.home}/note') is False
        assert fake_boat.bodies('update') == updates

    async def test_a_sandbox_without_an_auto_stop_is_not_extended(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        advance = advance_clock(monkeypatch)
        backend = await started(BoatSandboxBackend(auto_stop=0))
        advance(30 * 24 * 3600)
        assert await backend.exists(f'{only(fake_boat).home}/note') is False
        assert 'update' not in routes(fake_boat)

    async def test_a_long_command_extends_the_auto_stop_between_checks(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        advance = advance_clock(monkeypatch)
        backend = await started(BoatSandboxBackend(auto_stop=900))
        fake_boat.errors['status'] = [status(running=True), status(running=False, exit_code=0)]
        sleep = anyio.sleep

        async def pass_time(delay: float) -> None:
            advance(450)
            await sleep(0)

        monkeypatch.setattr(anyio, 'sleep', pass_time)
        assert (await backend.run(['true'])).exit_code == 0
        assert routes(fake_boat)[-4:] == ['start', 'status', 'update', 'status']

    async def test_a_failed_extension_does_not_fail_the_operation(
        self, fake_boat: FakeBoat, caplog: pytest.LogCaptureFixture
    ) -> None:
        existing = fake_boat.sandbox()
        forbidden = boat_error(403, 'api_key_action_forbidden', 'The action is not on the key.')
        fake_boat.errors['update'] = [forbidden, httpx.ConnectError('offline')]
        backend = BoatSandboxBackend(ref=WorkspaceRef(provider='boat', id=existing.id))
        with caplog.at_level(logging.DEBUG, logger=_BACKEND):
            # Each operation tries again until one succeeds; only the first failure in a row is a warning.
            for _ in range(4):
                assert await backend.exists(f'{existing.home}/note') is False
        assert routes(fake_boat).count('update') == 3
        assert [(record.levelno, record.getMessage()) for record in caplog.records] == [
            (
                logging.WARNING,
                f'Could not extend the auto-stop of Boat sandbox {existing.id}: BoatAPIError('
                "'Boat API returned HTTP 403 (api_key_action_forbidden): The action is not on the key.')",
            ),
            (
                logging.DEBUG,
                f"Could not extend the auto-stop of Boat sandbox {existing.id}: ConnectError('offline')",
            ),
        ]

    @pytest.mark.parametrize('change', ['deleted', 'stopped'])
    async def test_an_extension_that_finds_the_sandbox_gone_is_unavailable(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch, change: str
    ) -> None:
        advance = advance_clock(monkeypatch)
        backend = await started(BoatSandboxBackend())
        assert backend.ref is not None
        sandbox_id = backend.ref.id
        if change == 'deleted':
            await BoatSandbox().destroy(backend.ref)
            expected = unavailable(sandbox_id)
        else:
            fake_boat.errors['update'] = [boat_error(409, 'machine_not_running', 'Sandbox machine is not running.')]
            expected = stopped(sandbox_id)
        advance(1800)
        with pytest.raises(WorkspaceUnavailableError) as exc_info:
            await backend.run(['true'])
        assert str(exc_info.value) == expected
        assert routes(fake_boat)[-1] == 'update'


class TestRetries:
    @pytest.mark.parametrize(
        ('route', 'answer'),
        [
            pytest.param('run', (409, 'sandbox_starting', 'Sandbox is starting.'), id='starting'),
            pytest.param('run', (502, 'sandbox_direct_failed', 'sandbox_restoring'), id='restoring'),
            pytest.param('read', (502, 'sandbox_direct_failed', 'Machine did not answer.'), id='retryable-read'),
        ],
    )
    async def test_answers_boat_documents_as_not_taking_effect_are_repeated(
        self, fake_boat: FakeBoat, route: str, answer: tuple[int, str, str]
    ) -> None:
        backend = await started(BoatSandboxBackend())
        path = f'{only(fake_boat).home}/note'
        await backend.write_bytes(path, b'kept')
        before = routes(fake_boat).count(route)
        fake_boat.errors[route] = [boat_error(*answer, retryable=True)]
        if route == 'run':
            assert await backend.exists(path) is True
        else:
            assert await backend.read_bytes(path) == b'kept'
        assert routes(fake_boat).count(route) == before + 2

    async def test_a_command_that_may_have_run_is_not_repeated(self, fake_boat: FakeBoat) -> None:
        # Retryable, but neither "starting" nor "restoring": the command may have run, so it is not sent again.
        backend = await started(BoatSandboxBackend())
        fake_boat.errors['run'] = [boat_error(502, 'sandbox_direct_failed', 'Machine did not answer.', retryable=True)]
        with pytest.raises(BoatAPIError) as exc_info:
            await backend.exists(f'{only(fake_boat).home}/note')
        assert (exc_info.value.status, exc_info.value.retryable) == (502, True)
        assert routes(fake_boat).count('run') == 2

    async def test_a_sandbox_still_starting_after_the_retry_window_is_a_workspace_error(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr('pydantic_ai_harness.boat_sandbox._api._RETRY_WINDOW', 0)
        fake_boat.errors['run'] = [boat_error(409, 'sandbox_starting', 'Sandbox is starting.')]
        with pytest.raises(WorkspaceError) as exc_info:
            await BoatSandboxBackend().working_dir()
        assert str(exc_info.value) == (
            'Could not run a command in the sandbox: Boat API returned HTTP 409 (sandbox_starting): Sandbox is starting.'
        )
        assert routes(fake_boat).count('run') == 1


class TestCommands:
    @pytest.mark.parametrize(
        ('case', 'error'),
        [
            ('string-without-shell', TypeError),
            ('argv-with-shell', TypeError),
            ('empty-argv', TypeError),
            ('zero-timeout', ValueError),
            ('blank-env-name', ValueError),
        ],
    )
    async def test_invalid_arguments_do_not_create_a_sandbox(
        self, fake_boat: FakeBoat, case: str, error: type[Exception]
    ) -> None:
        backend = BoatSandboxBackend()
        calls: dict[str, Callable[[], Awaitable[object]]] = {
            'string-without-shell': lambda: backend.run('echo hi'),
            'argv-with-shell': lambda: backend.run(['true'], shell=True),
            'empty-argv': lambda: backend.run([]),
            'zero-timeout': lambda: backend.run(['true'], timeout=0),
            'blank-env-name': lambda: backend.run(['true'], env={'': 'x'}),
        }
        with pytest.raises(error):
            await calls[case]()
        assert fake_boat.opened_clients == 0

    async def test_commands_start_detached_with_their_env_inside_the_wrapper(self, fake_boat: FakeBoat) -> None:
        # `env` runs inside the wrapper's `sh`; outside it, a dash `sh` would drop `A.B` before the command ran.
        backend = await started(BoatSandboxBackend(env={'A': '1', 'B': '2'}))
        result = await backend.run(['sh', '-c', 'printf "%s %s" "$A" "$B"'], env={'B': '3', 'A.B': 'x'})
        assert result == CommandResult(exit_code=0, stdout='1 3', stderr='')
        result = await backend.run('printf hello | tr a-z A-Z', shell=True)
        assert result.stdout == 'HELLO'
        assert [{**body, 'command': unmarked(body['command'])} for body in fake_boat.bodies('start')] == [
            {
                'command': 'exec sh -c \': MARK; "$@" </dev/null; exit $?\' sh env -- A=1 B=3 A.B=x '
                'sh -c \'printf "%s %s" "$A" "$B"\'',
                'detached': True,
            },
            {
                'command': 'exec sh -c \': MARK; "$@" </dev/null; exit $?\' sh env -- A=1 B=2 '
                "/bin/sh -c 'printf hello | tr a-z A-Z'",
                'detached': True,
            },
        ]

    async def test_a_command_killed_by_a_signal_reports_its_status(self, fake_boat: FakeBoat) -> None:
        # The wrapper's `sh` stays the command's parent and exits with the status the shell reports.
        backend = await started(BoatSandboxBackend())
        assert (await backend.run(['sh', '-c', 'kill -KILL $$'])).exit_code == 137

    async def test_env_names_that_are_not_shell_identifiers_reach_the_command(self, fake_boat: FakeBoat) -> None:
        backend = BoatSandboxBackend(env={'A.B': 'sandbox'})
        result = await backend.run(['env'], env={'C-D': 'command'})
        assert {'A.B=sandbox', 'C-D=command'} <= set(result.stdout.splitlines())

    async def test_commands_start_in_the_working_dir_and_a_missing_one_is_a_workspace_error(
        self, fake_boat: FakeBoat
    ) -> None:
        owner = await started(BoatSandboxBackend())
        sandbox = only(fake_boat)
        (sandbox.home / 'sub').mkdir()
        attached = BoatSandboxBackend(ref=owner.ref, working_dir=f'{sandbox.home}/sub')
        assert (await attached.run(['pwd', '-P'])).stdout == f'{sandbox.home}/sub\n'

        # An attached sandbox must already have the configured `working_dir`.
        missing = f'{sandbox.home}/missing'
        attached = BoatSandboxBackend(ref=owner.ref, working_dir=missing)
        message = (
            f'working_dir {missing!r} does not exist in Boat sandbox {sandbox.id}. '
            'Create it there, or pass a working_dir that exists.'
        )
        with pytest.raises(WorkspaceError) as exc_info:
            await attached.run(['true'])
        assert str(exc_info.value) == message
        with pytest.raises(WorkspaceError) as exc_info:
            await attached.working_dir()
        assert str(exc_info.value) == message
        # The command had exited, so nothing was stopped; and a command's own exit 125 is its result.
        assert 'kill' not in routes(fake_boat)
        assert await owner.run(['sh', '-c', 'echo out; exit 125']) == CommandResult(
            exit_code=125, stdout='out\n', stderr=''
        )

    async def test_output_over_the_limit_raises_with_a_preview(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Boat keeps the last 8 MiB of each stream; the same rule at a small size.
        monkeypatch.setattr(f'{_BACKEND}._MAX_OUTPUT_BYTES', 8192)
        monkeypatch.setattr(f'{_BACKEND}._OUTPUT_PREVIEW_BYTES', 64)
        backend = await started(BoatSandboxBackend())
        script = (
            'printf b; head -c 10000 /dev/zero | tr "\\0" a; '
            'printf x >&2; for i in $(seq 40); do printf "\\303\\251" >&2; done'
        )
        with pytest.raises(WorkspaceOutputLimitError) as exc_info:
            await backend.run(['sh', '-c', script])
        error = exc_info.value
        # The preview is the beginning of each stream, read from offset 0, not the tail Boat kept. It
        # is cut at 64 bytes, not characters, and the two-byte character split at the cut is dropped.
        assert (error.limit, error.stdout, error.stderr) == (8192, 'b' + 'a' * 63, 'x' + '\u00e9' * 31)
        assert [dict(call.request.url.params) for call in fake_boat.calls if call.route == 'status'][-2:] == [
            {'tailBytes': '8192'},
            {'stdoutOffset': '0', 'stderrOffset': '0'},
        ]
        assert 'kill' not in routes(fake_boat)

    async def test_the_timeout_carries_partial_output_and_stops_the_command(self, fake_boat: FakeBoat) -> None:
        backend = await started(BoatSandboxBackend())
        fake_boat.errors['status'] = [
            status(running=True, stdout='partial out', stderr='partial err') for _ in range(100)
        ]
        with anyio.fail_after(10), pytest.raises(WorkspaceTimeoutError) as exc_info:
            await backend.run(['sleep', '30'], timeout=0.05)
        assert (exc_info.value.stdout, exc_info.value.stderr) == ('partial out', 'partial err')
        assert str(exc_info.value) == 'Command timed out after 0.05 seconds'
        # Each check reads only the tail a timeout reports.
        assert {call.request.url.params['tailBytes'] for call in fake_boat.calls if call.route == 'status'} == {'4096'}
        command = only_command(fake_boat)
        assert command.stopped and command.process.poll() is not None

    @pytest.mark.parametrize('answered', [pytest.param(True, id='answered'), pytest.param(False, id='cut-off')])
    async def test_a_stop_waits_for_boats_answer_until_the_kill_timeout(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, answered: bool
    ) -> None:
        # Boat answers a stop once the command exits, about 5.4 seconds for one that ignores SIGTERM: longer
        # than the 2 seconds a stop gets by default. The same rule at a small size.
        monkeypatch.setattr(f'{_BACKEND}._KILL_TIMEOUT', 0.5 if answered else 0.05)
        backend = await started(BoatSandboxBackend())
        release = fake_boat.gates['kill'] = asyncio.Event()

        async def answer_late() -> None:
            await fake_boat.reached['kill'].wait()
            await anyio.sleep(0.1)
            if answered:
                release.set()

        # Well under the default grace, so a stop bounded by it instead would fail the test.
        with anyio.fail_after(1.5):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(answer_late)
                with pytest.raises(WorkspaceTimeoutError):
                    await backend.run(['sleep', '30'], timeout=0.05)
        assert only_command(fake_boat).stopped is answered
        assert ('it may still be running' in caplog.text) is not answered

    async def test_checks_of_a_running_command_back_off(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Boat allows 1,000 API requests a minute from one IP address; checks every 0.2 seconds would
        # use a fifth of that for each running command.
        monkeypatch.setattr(f'{_BACKEND}._POLL_INTERVAL', 0.2)
        monkeypatch.setattr(f'{_BACKEND}._MAX_POLL_INTERVAL', 2.0)
        backend = await started(BoatSandboxBackend())
        fake_boat.errors['status'] = [*(status(running=True) for _ in range(6)), status(running=False, exit_code=0)]
        delays = record_sleeps(monkeypatch)
        assert (await backend.run(['true'])).exit_code == 0
        assert delays == [0.2, 0.4, 0.8, 1.6, 2.0, 2.0]

    async def test_a_check_refused_by_boats_request_limit_waits_as_told(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = await started(BoatSandboxBackend())
        fake_boat.errors['status'] = [ip_limited(retry_after=3)]
        delays = record_sleeps(monkeypatch)
        assert await backend.run(['echo', 'done']) == CommandResult(exit_code=0, stdout='done\n', stderr='')
        # Boat sends no `Retry-After` header; the wait is in the body.
        assert delays[0] == 3
        assert routes(fake_boat).count('status') >= 2
        assert 'kill' not in routes(fake_boat)

    @pytest.mark.parametrize(
        ('answer', 'logged'),
        [
            pytest.param((500, 'internal_error', 'Internal error.'), True, id='failed'),
            # Boat no longer knows the command, or its sandbox is gone: nothing is left running.
            pytest.param(
                (400, 'sandbox_direct_failed', 'Process 1 was not started by a detached command.'), False, id='gone'
            ),
            pytest.param((404, 'not_found', 'not_found'), False, id='sandbox-gone'),
        ],
    )
    async def test_a_failed_stop_is_logged_and_the_timeout_still_raises(
        self,
        fake_boat: FakeBoat,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        answer: tuple[int, str, str],
        logged: bool,
    ) -> None:
        without_repeats(monkeypatch)
        backend = await started(BoatSandboxBackend())
        fake_boat.errors['kill'] = [boat_error(*answer)]
        with anyio.fail_after(10), pytest.raises(WorkspaceTimeoutError):
            await backend.run(['sleep', '30'], timeout=0.05)
        assert routes(fake_boat).count('kill') == 1
        assert ('it may still be running' in caplog.text) is logged

    async def test_a_command_boat_refuses_to_start_is_a_workspace_error(self, fake_boat: FakeBoat) -> None:
        backend = await started(BoatSandboxBackend())
        with pytest.raises(WorkspaceError) as exc_info:
            await backend.run(['echo', 'x' * 140_000])
        assert type(exc_info.value) is WorkspaceError
        assert str(exc_info.value) == (
            'Could not start the command: Boat API returned HTTP 400 (sandbox_direct_failed): '
            "E2BIG: argument list too long, posix_spawn 'systemd-run'"
        )
        # Nothing started, so nothing is looked up or stopped.
        assert routes(fake_boat)[-1] == 'start'

    @pytest.mark.parametrize(
        ('failure', 'expected'),
        [
            # Boat documents that a refused start did not run.
            pytest.param(
                lambda: boat_error(429, 'rate_limited', 'Too many commands.'), BoatAPIError, id='rate-limited'
            ),
            # The request never reached Boat.
            pytest.param(lambda: httpx.ConnectError('connection refused'), httpx.ConnectError, id='not-sent'),
        ],
    )
    async def test_a_start_that_did_not_run_propagates_unchanged(
        self,
        fake_boat: FakeBoat,
        caplog: pytest.LogCaptureFixture,
        failure: Callable[[], httpx.Response | Exception],
        expected: type[Exception],
    ) -> None:
        # So a durable engine may retry it.
        backend = await started(BoatSandboxBackend())
        fake_boat.errors['start'] = [failure()]
        with pytest.raises(expected):
            await backend.run(['true'])
        # Nothing to look for or stop.
        assert routes(fake_boat)[-2:] == ['start', 'get']
        assert caplog.records == []

    async def test_a_command_whose_start_was_not_confirmed_is_found_by_its_marker_and_stopped(
        self, fake_boat: FakeBoat
    ) -> None:
        backend = await started(BoatSandboxBackend())
        fake_boat.lost_replies['start'] = 1
        with pytest.raises(WorkspaceError) as exc_info:
            await backend.run(['sleep', '30'])
        # Not transient: a retry could run the command a second time.
        assert type(exc_info.value) is WorkspaceError
        assert str(exc_info.value) == 'Boat did not confirm that the command started, so it may be running: reply lost'
        command = only_command(fake_boat)
        assert command.stopped and command.process.poll() is not None
        [kill] = [call.request for call in fake_boat.calls if call.route == 'kill']
        assert kill.url.path.endswith(f'/commands/{command.process.pid}')

    async def test_an_unconfirmed_start_that_never_ran_stops_nothing(self, fake_boat: FakeBoat) -> None:
        backend = await started(BoatSandboxBackend())
        fake_boat.errors['start'] = [
            boat_error(502, 'sandbox_direct_failed', 'Machine did not answer.', retryable=False)
        ]
        with pytest.raises(WorkspaceError, match=r'^Boat did not confirm that the command started'):
            await backend.run(['sleep', '30'])
        # The process list is searched for the command, which is not there.
        assert routes(fake_boat)[-3:] == ['start', 'get', 'run']
        assert only(fake_boat).commands == {}

    async def test_a_command_boat_lost_track_of_is_unknown_and_stopped(self, fake_boat: FakeBoat) -> None:
        # Boat forgets a command's exit when the sandbox's agent restarts; the command may still run.
        backend = await started(BoatSandboxBackend())
        fake_boat.errors['status'] = [
            boat_ok('command.status', status='lost', known=False, running=False, exitCode=None, stdout='', stderr='')
        ]
        with pytest.raises(WorkspaceError) as exc_info:
            await backend.run(['sleep', '30'])
        assert type(exc_info.value) is WorkspaceError
        assert str(exc_info.value) == 'Boat lost track of the command; its result is unknown.'
        assert only_command(fake_boat).stopped

    async def test_stopping_a_command_boat_forgot_is_quiet(
        self, fake_boat: FakeBoat, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = await started(BoatSandboxBackend())
        # Boat confirms the start, then knows nothing of the process: its checks and its stop say so.
        fake_boat.errors['start'] = [boat_ok('command.started', processId=999_999)]
        with pytest.raises(WorkspaceError, match=r'^Boat lost track of the command; its result is unknown\.$'):
            await backend.run(['true'])
        assert routes(fake_boat)[-4:] == ['start', 'status', 'get', 'kill']
        assert caplog.records == []

    async def test_a_command_lost_with_its_sandbox_is_unavailable(self, fake_boat: FakeBoat) -> None:
        backend = await started(BoatSandboxBackend())
        sandbox_id = only(fake_boat).id
        fake_boat.errors['status'] = [status(running=False, exit_code=None)]
        fake_boat.errors['get'] = [boat_error(404, 'not_found', 'not_found')]
        with pytest.raises(WorkspaceUnavailableError) as exc_info:
            await backend.run(['true'])
        assert str(exc_info.value) == unavailable(sandbox_id)

    @pytest.mark.parametrize(
        ('failure', 'delay'),
        [
            pytest.param(lambda: httpx.ConnectError('offline'), 0, id='transport'),
            pytest.param(lambda: boat_error(500, 'internal_error', 'Internal error.'), 0, id='500'),
            pytest.param(lambda: httpx.Response(502, text='<html>Bad Gateway</html>'), 0, id='502'),
            pytest.param(
                lambda: httpx.Response(
                    503, json={**boat_error(503, 'no_ready_machine', 'No machine.').json(), 'retryAfterSeconds': 5}
                ),
                5,
                id='503',
            ),
            pytest.param(lambda: boat_error(504, 'gateway_timeout', 'Gateway timeout.'), 0, id='504'),
        ],
    )
    async def test_a_failed_check_of_a_running_command_is_sent_again(
        self,
        fake_boat: FakeBoat,
        monkeypatch: pytest.MonkeyPatch,
        failure: Callable[[], httpx.Response | Exception],
        delay: float,
    ) -> None:
        # A long command is checked many times; one failed check must not end it.
        backend = await started(BoatSandboxBackend())
        fake_boat.errors['status'] = [failure()]
        delays = record_sleeps(monkeypatch)
        assert await backend.run(['echo', 'done']) == CommandResult(exit_code=0, stdout='done\n', stderr='')
        # A 503 says how long to wait in its body.
        assert delays[0] == delay
        assert 'kill' not in routes(fake_boat)

    @pytest.mark.parametrize(
        'failure',
        [
            pytest.param(lambda: httpx.ConnectError('offline'), id='transport'),
            pytest.param(lambda: boat_error(500, 'internal_error', 'Internal error.'), id='server'),
        ],
    )
    async def test_an_unreadable_result_is_a_workspace_error_and_the_command_is_stopped(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch, failure: Callable[[], httpx.Response | Exception]
    ) -> None:
        # Once the window for sending the check again has run out.
        without_repeats(monkeypatch)
        backend = await started(BoatSandboxBackend())
        fake_boat.errors['status'] = [failure()]
        with pytest.raises(
            WorkspaceError, match=r'^Could not read the result of a command that may still be running: '
        ):
            await backend.run(['sleep', '30'])
        assert only_command(fake_boat).stopped

    async def test_aclose_waits_for_overlapping_runs(self, fake_boat: FakeBoat) -> None:
        backend = await started(BoatSandboxBackend())
        gate = fake_boat.gates['status'] = asyncio.Event()
        runs = [asyncio.create_task(backend.run(['echo', str(index)])) for index in range(2)]
        await until(lambda: routes(fake_boat).count('status') == 2)
        closing = asyncio.create_task(backend.aclose())
        await anyio.wait_all_tasks_blocked()
        assert fake_boat.closed_clients == 0
        gate.set()
        with anyio.fail_after(10):
            results = await asyncio.gather(*runs)
            await closing
        assert [result.stdout for result in results] == ['0\n', '1\n']
        assert fake_boat.closed_clients == 1

    async def test_a_cancelled_command_keeps_the_client_open_until_it_is_stopped(self, fake_boat: FakeBoat) -> None:
        backend = await started(BoatSandboxBackend())
        release = fake_boat.gates['kill'] = asyncio.Event()
        task = asyncio.create_task(backend.run(['sleep', '30']))
        with anyio.fail_after(10):
            await fake_boat.reached['status'].wait()
            task.cancel()
            await fake_boat.reached['kill'].wait()
            closing = asyncio.create_task(backend.aclose())
            await anyio.wait_all_tasks_blocked()
            assert fake_boat.closed_clients == 0
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            await closing
        assert only_command(fake_boat).stopped
        assert fake_boat.closed_clients == 1

    async def test_a_stop_a_repeated_cancellation_leaves_running_keeps_the_client_open(
        self, fake_boat: FakeBoat
    ) -> None:
        backend = await started(BoatSandboxBackend())
        release = fake_boat.gates['kill'] = asyncio.Event()
        task = asyncio.create_task(backend.run(['sleep', '30']))
        with anyio.fail_after(10):
            await fake_boat.reached['status'].wait()
            task.cancel()
            await fake_boat.reached['kill'].wait()
            # The second cancellation ends the run's wait for the stop, which keeps going.
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await backend.aclose()
            assert fake_boat.closed_clients == 0
            release.set()
            await until(lambda: fake_boat.closed_clients == 1)
        assert only_command(fake_boat).stopped


class TestErrorTranslation:
    @pytest.mark.parametrize(
        ('answer', 'expected', 'message'),
        [
            pytest.param(
                (401, 'unauthorized', 'unauthorized'),
                WorkspaceUnavailableError,
                f'Credentials rejected. {_AUTH}',
                id='unauthorized',
            ),
            pytest.param(
                (403, 'org_suspended', 'This organization is suspended.'),
                WorkspaceUnavailableError,
                'Could not access {path!r} in the sandbox: This organization is suspended.',
                id='suspended',
            ),
            pytest.param(
                (409, 'conflict', 'Busy.'),
                WorkspaceError,
                'Could not access {path!r} in the sandbox: Boat API returned HTTP 409 (conflict): Busy.',
                id='other-4xx',
            ),
            # A POSIX error Boat's message does not name is not a path failure the protocol knows.
            pytest.param(
                (400, 'sandbox_direct_failed', 'EIO: i/o error, read'),
                WorkspaceError,
                'Could not access {path!r} in the sandbox: Boat API returned HTTP 400 (sandbox_direct_failed): '
                'EIO: i/o error, read',
                id='unknown-errno',
            ),
            pytest.param(
                (400, 'sandbox_direct_failed', 'ENOTDIR: not a directory, open'),
                NotADirectoryError,
                "[Errno 20] Not a directory in the Boat sandbox: '{path}'",
                id='errno',
            ),
            pytest.param((429, 'rate_limited', 'Too many requests.'), BoatAPIError, None, id='rate-limited'),
            pytest.param((503, 'no_ready_machine', 'No machine is ready.'), BoatAPIError, None, id='server'),
        ],
    )
    async def test_file_call_failures_are_translated(
        self,
        fake_boat: FakeBoat,
        monkeypatch: pytest.MonkeyPatch,
        answer: tuple[int, str, str],
        expected: type[Exception],
        message: str | None,
    ) -> None:
        without_repeats(monkeypatch)
        backend = await started(BoatSandboxBackend())
        path = f'{only(fake_boat).home}/note'
        fake_boat.errors['read'] = [boat_error(*answer)]
        with pytest.raises(expected) as exc_info:
            await backend.read_bytes(path)
        assert type(exc_info.value) is expected
        if message is None:
            # Transient: once the read is no longer sent again, it propagates unchanged if a lookup finds the
            # sandbox still running.
            assert isinstance(exc_info.value, BoatAPIError) and exc_info.value.status == answer[0]
            assert routes(fake_boat)[-2:] == ['read', 'get']
        else:
            assert str(exc_info.value) == message.format(path=path)

    @pytest.mark.parametrize(
        ('change', 'expected'),
        [
            pytest.param('deleted', unavailable, id='deleted'),
            pytest.param('archived', stopped, id='stopped'),
            pytest.param('error', failed, id='failed'),
            # The lookup finds it running, or cannot tell: the failure itself is what the caller sees.
            pytest.param('none', None, id='running'),
            pytest.param('lookup-offline', None, id='lookup-offline'),
            pytest.param('lookup-failed', None, id='lookup-failed'),
        ],
    )
    async def test_a_transient_failure_ends_the_run_when_the_sandbox_is_gone_or_stopped(
        self,
        fake_boat: FakeBoat,
        monkeypatch: pytest.MonkeyPatch,
        change: str,
        expected: Callable[[str], str] | None,
    ) -> None:
        without_repeats(monkeypatch)
        backend = await started(BoatSandboxBackend())
        sandbox = only(fake_boat)
        fake_boat.errors['read'] = [boat_error(503, 'no_ready_machine', 'No machine is ready.')]
        if change == 'deleted':
            await BoatSandbox().destroy(WorkspaceRef(provider='boat', id=sandbox.id))
        elif change == 'lookup-offline':
            fake_boat.errors['get'] = [httpx.ConnectError('offline')]
        elif change == 'lookup-failed':
            fake_boat.errors['get'] = [boat_error(500, 'internal_error', 'Internal error.')]
        elif change != 'none':
            sandbox.state = change
        with pytest.raises(Exception) as exc_info:
            await backend.read_bytes(f'{sandbox.home}/note')
        if expected is None:
            assert isinstance(exc_info.value, BoatAPIError) and exc_info.value.status == 503
        else:
            assert type(exc_info.value) is WorkspaceUnavailableError
            assert str(exc_info.value) == expected(sandbox.id)
            assert isinstance(exc_info.value.__cause__, BoatAPIError)

    @pytest.mark.parametrize(
        ('state', 'operation'),
        [
            # While Boat archives it, the machine still answers file calls but refuses commands.
            pytest.param('archiving', 'run', id='archiving'),
            pytest.param('archived', 'run', id='archived-run'),
            pytest.param('archived', 'read_bytes', id='archived-read'),
        ],
    )
    async def test_a_sandbox_stopped_under_a_held_backend_is_unavailable_until_attached_again(
        self, fake_boat: FakeBoat, state: str, operation: str
    ) -> None:
        backend = await started(BoatSandboxBackend())
        sandbox = only(fake_boat)
        path = f'{sandbox.home}/kept'
        await backend.write_bytes(path, b'kept')
        sandbox.state = state
        with pytest.raises(WorkspaceUnavailableError) as exc_info:
            await (backend.run(['true']) if operation == 'run' else backend.read_bytes(path))
        assert str(exc_info.value) == stopped(sandbox.id)
        # Ending the run is the held backend's answer; attaching again resumes it.
        assert sandbox.resume_calls == 0
        assert await BoatSandboxBackend(ref=backend.ref).read_bytes(path) == b'kept'
        assert sandbox.resume_calls == 1

    @pytest.mark.parametrize(
        ('answer', 'expected', 'message'),
        [
            pytest.param(
                lambda: finished(exit_code=137, timed_out=True),
                WorkspaceError,
                'An internal command did not finish within 30s in Boat sandbox {id}.',
                id='timed-out',
            ),
            pytest.param(
                lambda: finished(stdout='x', truncated=True),
                WorkspaceOutputLimitError,
                'Boat command output exceeded the 8 MiB limit per stream; '
                'redirect it to a file and read part of it instead',
                id='truncated',
            ),
            pytest.param(
                lambda: finished(exit_code=1, stderr='pwd: boom'),
                WorkspaceError,
                'Could not determine the working directory of Boat sandbox {id}: pwd: boom',
                id='failed',
            ),
            pytest.param(
                lambda: finished(stdout='relative\n'),
                WorkspaceError,
                'Could not determine the working directory of Boat sandbox {id}: ',
                id='relative',
            ),
        ],
    )
    async def test_internal_command_failures_are_workspace_errors(
        self, fake_boat: FakeBoat, answer: Callable[[], httpx.Response], expected: type[Exception], message: str
    ) -> None:
        backend = await started(BoatSandboxBackend())
        sandbox_id = only(fake_boat).id
        await backend.aclose()
        # The next operation probes the working directory again, and gets this answer.
        fake_boat.errors['run'] = [answer()]
        with pytest.raises(expected) as exc_info:
            await backend.working_dir()
        assert type(exc_info.value) is expected
        assert str(exc_info.value) == message.format(id=sandbox_id)


class TestDeletedSandbox:
    """A deleted sandbox must never look like a missing file."""

    @pytest.mark.parametrize(
        'operation',
        ['read_bytes', 'write_bytes', 'stat', 'list_dir', 'make_dir', 'remove', 'exists', 'realpath', 'run'],
    )
    async def test_operations_on_a_deleted_sandbox_are_unavailable(self, fake_boat: FakeBoat, operation: str) -> None:
        backend = await started(BoatSandboxBackend())
        sandbox = only(fake_boat)
        path = f'{sandbox.home}/note'
        calls: dict[str, Callable[[], Awaitable[object]]] = {
            'read_bytes': lambda: backend.read_bytes(path),
            'write_bytes': lambda: backend.write_bytes(path, b'x'),
            'stat': lambda: backend.stat(path),
            'list_dir': lambda: backend.list_dir(str(sandbox.home)),
            'make_dir': lambda: backend.make_dir(path),
            'remove': lambda: backend.remove(path),
            'exists': lambda: backend.exists(path),
            'realpath': lambda: backend.realpath(path),
            'run': lambda: backend.run(['true']),
        }
        assert backend.ref is not None
        await BoatSandbox().destroy(backend.ref)
        with pytest.raises(WorkspaceUnavailableError) as exc_info:
            await calls[operation]()
        assert str(exc_info.value) == unavailable(sandbox.id)

    async def test_the_working_dir_is_cached_and_reverified_after_aclose(self, fake_boat: FakeBoat) -> None:
        backend = BoatSandboxBackend()
        first = await backend.working_dir()
        assert await backend.working_dir() == first
        assert routes(fake_boat).count('run') == 1
        await backend.aclose()
        assert backend.ref is not None
        await BoatSandbox().destroy(backend.ref)
        with pytest.raises(WorkspaceUnavailableError, match='was not found'):
            await backend.working_dir()


class TestFiles:
    @pytest.mark.parametrize('through', ['path', 'symlink'])
    async def test_paths_outside_the_file_api_roots_go_through_commands(
        self, fake_boat: FakeBoat, through: str
    ) -> None:
        backend = await started(BoatSandboxBackend())
        sandbox = only(fake_boat)
        # Beyond the sandbox's home and `/tmp`, which are all Boat's file API reaches.
        outside = sandbox.home.parent / 'outside'
        outside.mkdir()
        if through == 'symlink':
            (sandbox.home / 'link').symlink_to(outside)
            path = f'{sandbox.home}/link/nested/data.bin'
        else:
            path = f'{outside}/nested/data.bin'
        data = bytes(range(256))
        await backend.write_bytes(path, data)
        assert (outside / 'nested' / 'data.bin').read_bytes() == data
        assert await backend.read_bytes(path) == data
        # Boat's file API was asked first, and refused.
        assert (routes(fake_boat).count('write'), routes(fake_boat).count('read')) == (1, 1)

    @pytest.mark.parametrize(
        ('answer', 'expected'),
        [
            # Boat signs the URL without checking the path, so the machine answers for a file gone meanwhile.
            pytest.param(None, FileNotFoundError, id='gone'),
            pytest.param(lambda: httpx.Response(502, text='Bad Gateway'), BoatAPIError, id='not-json'),
        ],
    )
    async def test_a_failed_download_of_a_large_file(
        self, fake_boat: FakeBoat, answer: Callable[[], httpx.Response] | None, expected: type[Exception]
    ) -> None:
        backend = await started(BoatSandboxBackend())
        path = f'{only(fake_boat).home}/big.bin'
        fake_boat.errors['read'] = [
            boat_error(400, 'sandbox_direct_failed', 'File is too large for read_file (6000000 bytes > 5242880).')
        ]
        if answer is not None:
            fake_boat.errors['transfer'] = [answer()]
        with pytest.raises(expected) as exc_info:
            await backend.read_bytes(path)
        if isinstance(exc_info.value, FileNotFoundError):
            assert (exc_info.value.errno, exc_info.value.filename) == (errno.ENOENT, path)
        else:
            assert isinstance(exc_info.value, BoatAPIError)
            assert (exc_info.value.status, exc_info.value.code) == (502, 'invalid_response')
        assert [route for route in routes(fake_boat) if route in ('download_url', 'transfer')] == [
            'download_url',
            'transfer',
        ]

    async def test_a_sandbox_stopped_during_a_download_is_unavailable(self, fake_boat: FakeBoat) -> None:
        # The signed URL points at the sandbox's machine, which stops answering once Boat stops it.
        backend = await started(BoatSandboxBackend())
        sandbox = only(fake_boat)
        fake_boat.errors['read'] = [
            boat_error(400, 'sandbox_direct_failed', 'File is too large for read_file (6000000 bytes > 5242880).')
        ]
        release = fake_boat.gates['transfer'] = asyncio.Event()
        read = asyncio.create_task(backend.read_bytes(f'{sandbox.home}/big.bin'))
        with anyio.fail_after(10):
            await fake_boat.reached['transfer'].wait()
            sandbox.state = 'archived'
            release.set()
            with pytest.raises(WorkspaceUnavailableError) as exc_info:
                await read
        assert str(exc_info.value) == stopped(sandbox.id)
        assert isinstance(exc_info.value.__cause__, httpx.ConnectError)

    @pytest.mark.parametrize(
        'target',
        [
            'inside',
            # Boat refuses to sign a URL for a path outside its file API's roots.
            'outside',
            # A write below a file fails the same way whichever way it goes.
            'below-a-file',
            # The machine refuses it as `not_a_file`, which the backend checks like an inline refusal.
            'onto-a-directory',
            'stopped',
        ],
    )
    async def test_a_large_write_goes_through_a_signed_upload(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch, target: str
    ) -> None:
        # The same rule as Boat's 5 MiB inline limit, at a small size.
        monkeypatch.setattr(f'{_BACKEND}._MAX_INLINE_FILE_BYTES', 16)
        backend = await started(BoatSandboxBackend())
        sandbox = only(fake_boat)
        data = bytes(range(100))
        (sandbox.home / 'file').write_bytes(b'')
        (sandbox.home / 'dir').mkdir()
        (sandbox.home.parent / 'outside').mkdir()
        paths = {
            'inside': sandbox.home / 'new' / 'big.bin',
            'outside': sandbox.home.parent / 'outside' / 'big.bin',
            'below-a-file': sandbox.home / 'file' / 'big.bin',
            'onto-a-directory': sandbox.home / 'dir',
            'stopped': sandbox.home / 'big.bin',
        }
        path = paths[target]
        if target == 'stopped':
            sandbox.state = 'archived'
        if target in ('inside', 'outside'):
            await backend.write_bytes(str(path), data)
            assert path.read_bytes() == data
            # Never inline, and the API key never reaches the machine the URL points at (the fake checks it).
            assert 'write' not in routes(fake_boat)
            assert ('transfer' in routes(fake_boat)) is (target == 'inside')
        else:
            expected = {
                'below-a-file': NotADirectoryError,
                'onto-a-directory': IsADirectoryError,
                'stopped': WorkspaceUnavailableError,
            }[target]
            with pytest.raises(expected):
                await backend.write_bytes(str(path), data)

    @pytest.mark.parametrize('target', ['fifo', 'regular-file'])
    async def test_a_write_that_times_out_is_refused_on_a_fifo(
        self, fake_boat: FakeBoat, monkeypatch: pytest.MonkeyPatch, target: str
    ) -> None:
        # Boat answers a write to a FIFO only once something reads it, so the request times out; the
        # path is checked once the write is no longer sent again, rather than before every write.
        without_repeats(monkeypatch)
        backend = await started(BoatSandboxBackend())
        path = only(fake_boat).home / target
        if target == 'fifo':
            os.mkfifo(path)
        else:
            path.write_bytes(b'')
            fake_boat.errors['write'] = [httpx.ReadTimeout('timed out')]
        with pytest.raises(OSError if target == 'fifo' else httpx.ReadTimeout) as exc_info:
            await backend.write_bytes(str(path), b'data')
        if target == 'fifo':
            assert type(exc_info.value) is OSError
            assert str(exc_info.value) == f'Not a regular file in the Boat sandbox: {str(path)!r}'
            assert isinstance(exc_info.value.__cause__, httpx.ReadTimeout)

    async def test_a_write_through_a_dangling_symlink_creates_its_target(self, fake_boat: FakeBoat) -> None:
        # Boat's file API refuses a dangling link as if it led outside its roots; the shell writes through it.
        backend = await started(BoatSandboxBackend())
        home = only(fake_boat).home
        (home / 'link').symlink_to(home / 'target.txt')
        await backend.write_bytes(f'{home}/link', b'data')
        assert (home / 'link').is_symlink()
        assert (home / 'target.txt').read_bytes() == b'data'

    async def test_a_path_boat_calls_not_a_file_that_is_a_regular_file_when_checked(self, fake_boat: FakeBoat) -> None:
        # Replaced between Boat's answer and the check: reported, rather than read as if nothing happened.
        backend = await started(BoatSandboxBackend())
        path = f'{only(fake_boat).home}/note'
        await backend.write_bytes(path, b'x')
        fake_boat.errors['read'] = [boat_error(400, 'sandbox_direct_failed', 'Path is not a file.')]
        with pytest.raises(OSError) as exc_info:
            await backend.read_bytes(path)
        assert type(exc_info.value) is OSError
        assert str(exc_info.value) == f'Not a regular file in the Boat sandbox: {path!r}'

    @pytest.mark.parametrize(
        'operation', ['read_bytes', 'write_bytes', 'stat', 'list_dir', 'make_dir', 'remove', 'exists', 'realpath']
    )
    async def test_a_newline_in_a_path_is_refused_before_acquisition(self, fake_boat: FakeBoat, operation: str) -> None:
        backend = BoatSandboxBackend()
        path = '/home/user/bad\nname'
        calls: dict[str, Callable[[], Awaitable[object]]] = {
            'read_bytes': lambda: backend.read_bytes(path),
            'write_bytes': lambda: backend.write_bytes(path, b'x'),
            'stat': lambda: backend.stat(path),
            'list_dir': lambda: backend.list_dir(path),
            'make_dir': lambda: backend.make_dir(path),
            'remove': lambda: backend.remove(path),
            'exists': lambda: backend.exists(path),
            'realpath': lambda: backend.realpath(path),
        }
        with pytest.raises(ValueError, match=r'^Boat file paths cannot contain a newline or carriage return$'):
            await calls[operation]()
        assert fake_boat.opened_clients == 0


class TestClients:
    async def test_a_supplied_client_is_used_and_never_closed(self, fake_boat: FakeBoat) -> None:
        client = fake_boat.supplied_client()
        backend = BoatSandboxBackend(client=client)
        assert (await backend.run(['echo', 'mine'])).stdout == 'mine\n'
        await backend.aclose()
        assert not client.is_closed
        assert (await backend.run(['true'])).exit_code == 0
        assert (fake_boat.opened_clients, fake_boat.closed_clients) == (0, 0)

    async def test_aclose_closes_the_owned_client_and_later_use_reattaches(self, fake_boat: FakeBoat) -> None:
        backend = await started(BoatSandboxBackend())
        path = f'{only(fake_boat).home}/note'
        await backend.write_bytes(path, b'kept')
        await backend.aclose()
        assert (fake_boat.opened_clients, fake_boat.closed_clients) == (1, 1)
        assert await backend.read_bytes(path) == b'kept'
        assert fake_boat.opened_clients == 2
        assert len(fake_boat.create_bodies) == 1

    async def test_aclose_waits_for_an_in_flight_file_operation(self, fake_boat: FakeBoat) -> None:
        backend = await started(BoatSandboxBackend())
        path = f'{only(fake_boat).home}/note'
        await backend.write_bytes(path, b'kept')
        release = fake_boat.gates['read'] = asyncio.Event()
        read = asyncio.create_task(backend.read_bytes(path))
        with anyio.fail_after(10):
            await fake_boat.reached['read'].wait()
            close = asyncio.create_task(backend.aclose())
            await anyio.wait_all_tasks_blocked()
            assert fake_boat.closed_clients == 0
            release.set()
            assert await read == b'kept'
            await close
        assert fake_boat.closed_clients == 1

    async def test_concurrent_acloses_close_the_client_once(
        self, fake_boat: FakeBoat, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = await started(BoatSandboxBackend())
        await asyncio.gather(backend.aclose(), backend.aclose())
        assert fake_boat.closed_clients == 1
        assert [record for record in caplog.records if record.levelno >= logging.WARNING] == []

    @pytest.mark.parametrize(
        ('failure', 'logged'),
        [
            pytest.param('error', "Could not close the Boat API client: ConnectError('offline')", id='error'),
            pytest.param('hang', 'Closing the Boat API client did not complete within 0.05s.', id='hang'),
        ],
    )
    async def test_a_failed_close_is_logged(
        self,
        fake_boat: FakeBoat,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        failure: str,
        logged: str,
    ) -> None:
        monkeypatch.setattr(f'{_BACKEND}._TEARDOWN_TIMEOUT', 0.05)
        backend = await started(BoatSandboxBackend())
        if failure == 'error':
            fake_boat.close_errors = iter([httpx.ConnectError('offline')])
        else:
            fake_boat.close_gate = asyncio.Event()
        with anyio.fail_after(10):
            await backend.aclose()
        assert logged in caplog.messages
        fake_boat.close_gate = None
        # The next operation opens a client of its own and attaches by `ref`.
        await backend.realpath('.')
        assert fake_boat.opened_clients == 2

    async def test_repeated_cancellation_finishes_owned_client_close(self, fake_boat: FakeBoat) -> None:
        backend = await started(BoatSandboxBackend())
        gate = fake_boat.close_gate = asyncio.Event()
        task = asyncio.create_task(backend.aclose())
        await anyio.wait_all_tasks_blocked()
        task.cancel()
        task.cancel()
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert fake_boat.closed_clients == 1
