"""Tests for the workspace wrapper that mounts `S3Mount`s before the first operation, against a fake root runner.

Unit tests rather than integration tests: the real runner is a sandbox (Docker here, see
`tests/harness/docker_sandbox/test_mounts_live.py`), and these pin what the wrapper sends it and when.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import posixpath
import re
import threading
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import anyio
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st
from inline_snapshot import snapshot

import pydantic_ai_harness._mounts as mounts_package
from pydantic_ai import RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import CommandResult, Workspace, WrapperWorkspace
from pydantic_ai_harness import MountError, S3Mount
from pydantic_ai_harness._mounts import mounted
from tests.workspace_fakes import FakeWorkspace

NOW = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
CTX = RunContext[None](deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=0)
SECRETS = ('AKIA-the-key-id', 'the-secret-key', 'the-session-token')


@dataclass
class Clock:
    now: datetime = NOW


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    clock = Clock()
    monkeypatch.setattr('pydantic_ai_harness._mounts._now', lambda: clock.now)
    return clock


class FakeRoot:
    """Records each root command and answers with the next scripted stdout."""

    def __init__(self, *replies: str, wait: anyio.Event | None = None) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[list[str], bytes]] = []
        self.wait = wait
        self.started = anyio.Event()
        self.finished = 0

    async def __call__(self, argv: Sequence[str], stdin: bytes) -> CommandResult:
        self.calls.append((list(argv), stdin))
        self.started.set()
        if self.wait is not None:
            await self.wait.wait()
        self.finished += 1
        stdout = self.replies.pop(0) if self.replies else '{"recovered": []}\n'
        return CommandResult(exit_code=0, stdout=stdout, stderr='')

    def requests(self) -> list[dict[str, Any]]:
        return [json.loads(stdin) for _, stdin in self.calls]


def s3(
    path: str = 'data',
    url: str = 's3://bucket/reports/',
    *,
    key: str = SECRETS[0],
    expires_in: timedelta = timedelta(hours=1),
    **kwargs: Any,
) -> S3Mount:
    return S3Mount(
        path=path,
        url=url,
        access_key_id=key,
        secret_access_key=SECRETS[1],
        session_token=SECRETS[2],
        expires_at=NOW + expires_in,
        **kwargs,
    )


def wrap(root: FakeRoot, mounts: Sequence[S3Mount] | Callable[[], Sequence[S3Mount]]) -> Workspace:
    """Mount `mounts`: a list as it is, a function as a `mounts=` function that ignores the run context."""
    backend = mounted(FakeWorkspace('box'), root, (lambda _: mounts()) if callable(mounts) else mounts, CTX)
    assert isinstance(backend, Workspace)
    return backend


async def test_the_first_operation_mounts_and_later_ones_do_not() -> None:
    root = FakeRoot()
    workspace = wrap(root, lambda: [s3(endpoint='https://minio.example', region='eu-1', read_only=True)])

    await workspace.write_bytes('data/a.txt', b'a')
    await workspace.read_bytes('data/a.txt')

    assert len(root.calls) == 1
    argv, stdin = root.calls[0]
    assert argv[:2] == ['python3', '-c']
    assert argv[2] == (Path(mounts_package.__file__).parent / '_helper.py').read_text(encoding='utf-8')
    assert stdin.endswith(b'\n') and stdin.count(b'\n') == 1
    assert root.requests() == snapshot(
        [
            {
                'working_dir': '/workspace',
                'mounts': [
                    {
                        'path': '/workspace/data',
                        'fingerprint': 'a89e9054fca5c54b',
                        'remote': {
                            'type': 's3',
                            'provider': 'Other',
                            'endpoint': 'https://minio.example',
                            'region': 'eu-1',
                        },
                        'root': 'bucket/reports/',
                        'read_only': True,
                        'keys': {
                            'access_key_id': 'AKIA-the-key-id',
                            'secret_access_key': 'the-secret-key',
                            'session_token': 'the-session-token',
                        },
                        'expires_at': 1794315600.0,
                    }
                ],
            }
        ]
    )


async def test_keys_go_only_on_stdin() -> None:
    root = FakeRoot()
    workspace = wrap(root, lambda: [s3()])

    await workspace.exists('data')

    argv, stdin = root.calls[0]
    for secret in SECRETS:
        assert secret.encode() in stdin
        assert not any(secret in arg for arg in argv)


async def test_aws_buckets_name_no_endpoint_and_a_missing_token_is_sent_empty() -> None:
    root = FakeRoot()
    mount = S3Mount(
        path='data', url='s3://bucket', access_key_id='id', secret_access_key='key', expires_at=NOW + timedelta(hours=1)
    )
    workspace = wrap(root, lambda: [mount])

    await workspace.exists('data')

    (wanted,) = root.requests()[0]['mounts']
    assert (wanted['remote'], wanted['root'], wanted['keys']['session_token']) == snapshot(
        ({'type': 's3', 'provider': 'AWS'}, 'bucket', '')
    )


async def test_the_fingerprint_names_what_the_folder_shows_not_the_keys() -> None:
    first, second, other = FakeRoot(), FakeRoot(), FakeRoot()
    await wrap(first, lambda: [s3()]).exists('data')
    await wrap(second, lambda: [s3(key='another-key')]).exists('data')
    await wrap(other, lambda: [s3(read_only=True)]).exists('data')

    fingerprint = [root.requests()[0]['mounts'][0]['fingerprint'] for root in (first, second, other)]
    assert fingerprint[0] == fingerprint[1] != fingerprint[2]


OPERATIONS: dict[str, Callable[[Workspace], Awaitable[object]]] = {
    'run': lambda ws: ws.run(['true']),
    'read_bytes': lambda ws: ws.read_bytes('data/a'),
    'write_bytes': lambda ws: ws.write_bytes('data/a', b''),
    'stat': lambda ws: ws.stat('.'),
    'list_dir': lambda ws: ws.list_dir('.'),
    'make_dir': lambda ws: ws.make_dir('data'),
    'remove': lambda ws: ws.remove('data/a'),
    'exists': lambda ws: ws.exists('data'),
    'realpath': lambda ws: ws.realpath('data'),
    'read_text': lambda ws: ws.read_text('data/a'),
    'write_text': lambda ws: ws.write_text('data/a', ''),
}


@pytest.mark.parametrize('operation', list(OPERATIONS.values()), ids=list(OPERATIONS))
async def test_every_operation_that_touches_files_mounts_first(
    operation: Callable[[Workspace], Awaitable[object]],
) -> None:
    root = FakeRoot()
    backend = FakeWorkspace('box', {'/workspace/data/a': b''})
    workspace = mounted(backend, root, [s3()], CTX)
    assert isinstance(workspace, Workspace)

    await operation(workspace)

    assert len(root.calls) == 1


_NOT_GATED = {
    'working_dir': 'the gate asks it where the mounts go',
    'resolve': 'joins paths as text',
    'read_text': 'reads through the gated `read_bytes`',
    'write_text': 'writes through the gated `write_bytes`',
    'authority_of': 'reads the filesystem through the gated `realpath`',
}


def test_every_workspace_coroutine_is_gated_or_explained() -> None:
    coroutines = {
        name
        for name, member in inspect.getmembers(Workspace)
        if inspect.iscoroutinefunction(member) and not name.startswith('_')
    }
    wrapper = vars(type(mounted(FakeWorkspace('box'), FakeRoot(), [s3()], CTX)))
    gated = {name for name in wrapper if inspect.iscoroutinefunction(wrapper[name]) and not name.startswith('_')}
    assert coroutines == gated | set(_NOT_GATED)
    assert not gated & set(_NOT_GATED)


async def test_a_workspace_is_wrapped_as_it_is() -> None:
    inner = Workspace(FakeWorkspace('box'))
    workspace = mounted(inner, FakeRoot(), [s3()], CTX)

    assert isinstance(workspace, WrapperWorkspace)
    assert workspace.wrapped is inner


def test_an_empty_list_leaves_the_backend_as_it_is() -> None:
    backend = FakeWorkspace('box')

    assert mounted(backend, FakeRoot(), [], CTX) is backend


async def test_a_function_is_first_called_by_the_first_operation_in_a_worker_thread() -> None:
    loop_thread = threading.get_ident()
    threads: list[int] = []

    def mounts(ctx: RunContext[None]) -> list[S3Mount]:
        assert ctx is CTX
        threads.append(threading.get_ident())
        return [s3()]

    root = FakeRoot()
    workspace = mounted(FakeWorkspace('box'), root, mounts, CTX)
    assert threads == []

    assert isinstance(workspace, Workspace)
    await workspace.exists('data')
    assert len(threads) == 1 and threads[0] != loop_thread
    assert len(root.calls) == 1


async def test_a_function_that_returns_no_mounts_mounts_nothing_and_is_not_called_again() -> None:
    calls: list[None] = []

    def mounts(_: RunContext[None]) -> list[S3Mount]:
        calls.append(None)
        return []

    root = FakeRoot()
    workspace = mounted(FakeWorkspace('box'), root, mounts, CTX)
    assert isinstance(workspace, Workspace)

    await workspace.exists('data')
    await workspace.exists('data')

    assert root.calls == [] and len(calls) == 1


async def test_a_function_returning_overlapping_paths_is_refused_at_the_first_operation() -> None:
    workspace = wrap(FakeRoot(), lambda: [s3('data'), s3('data/inner', 's3://other/')])

    with pytest.raises(UserError, match="The mount paths 'data' and 'data/inner' overlap"):
        await workspace.exists('data')


def test_a_folder_beside_another_with_a_longer_name_is_not_inside_it() -> None:
    assert isinstance(
        mounted(FakeWorkspace('box'), FakeRoot(), [s3('data'), s3('database', 's3://other/')], CTX), WrapperWorkspace
    )


@pytest.mark.parametrize(('first', 'second'), [('data', 'data'), ('data', 'data/inner')])
def test_overlapping_paths_are_refused(first: str, second: str) -> None:
    with pytest.raises(UserError, match=f"The mount paths '{first}' and '{second}' overlap"):
        mounted(FakeWorkspace('box'), FakeRoot(), [s3(first), s3(second, 's3://other/')], CTX)


def test_overlapping_paths_with_a_folder_sorted_between_them_are_refused() -> None:
    mounts = [s3('data'), s3('data-other', 's3://other/'), s3('data/inner', 's3://third/')]

    with pytest.raises(UserError, match="The mount paths 'data' and 'data/inner' overlap"):
        mounted(FakeWorkspace('box'), FakeRoot(), mounts, CTX)


async def test_fresh_keys_are_fetched_before_the_old_ones_expire(clock: Clock) -> None:
    root = FakeRoot()
    keys = iter(['first', 'second'])
    workspace = wrap(root, lambda: [s3(key=next(keys), expires_in=clock.now - NOW + timedelta(hours=1))])

    await workspace.exists('data')
    clock.now = NOW + timedelta(minutes=54)
    await workspace.exists('data')
    clock.now = NOW + timedelta(minutes=55)
    await workspace.exists('data')

    assert [request['mounts'][0]['keys']['access_key_id'] for request in root.requests()] == ['first', 'second']


async def test_keys_shorter_lived_than_the_margin_are_used_until_they_expire(clock: Clock) -> None:
    root = FakeRoot()
    workspace = wrap(root, lambda: [s3(expires_in=timedelta(minutes=3))])

    await workspace.exists('data')
    clock.now = NOW + timedelta(minutes=2)
    await workspace.exists('data')

    assert len(root.calls) == 1


async def test_a_list_whose_keys_expired_is_refused(clock: Clock) -> None:
    root = FakeRoot()
    workspace = wrap(root, [s3()])
    await workspace.exists('data')

    clock.now = NOW + timedelta(hours=1)
    with pytest.raises(
        MountError, match=re.escape("The keys for mount 'data' have expired. Pass `mounts=` as a function")
    ):
        await workspace.exists('data')
    assert len(root.calls) == 1


CHANGES: dict[str, Callable[[], list[S3Mount]]] = {
    'another-bucket': lambda: [s3(url='s3://other/')],
    'one-more': lambda: [s3(), s3('more', 's3://other/')],
    'repeated': lambda: [s3(), s3()],
    'none': lambda: [],
}


@pytest.mark.parametrize('changed', list(CHANGES.values()), ids=list(CHANGES))
async def test_mounts_that_change_between_calls_are_refused(changed: Callable[[], list[S3Mount]], clock: Clock) -> None:
    calls = iter([[s3()], changed()])
    workspace = wrap(FakeRoot(), lambda: next(calls))
    await workspace.exists('data')

    clock.now = NOW + timedelta(minutes=55)
    with pytest.raises(UserError, match=re.escape('`mounts=` must return the same mounts each time, with fresh keys.')):
        await workspace.exists('data')


async def test_the_helpers_error_is_raised_without_the_keys() -> None:
    message = 'data already shows another bucket in this sandbox. Mount at another folder, or destroy the sandbox.'
    root = FakeRoot(f'starting\n{json.dumps({"error": "conflict", "message": message})}\n')
    workspace = wrap(root, lambda: [s3()])

    with pytest.raises(MountError) as raised:
        await workspace.run(['true'])

    assert str(raised.value) == message


async def test_output_that_is_not_the_helpers_is_quoted_with_the_keys_blanked() -> None:
    root = FakeRoot(f'Traceback (most recent call last):\nValueError: {SECRETS[1]}\n')
    workspace = wrap(root, lambda: [s3()])

    with pytest.raises(MountError) as raised:
        await workspace.run(['true'])

    assert str(raised.value) == snapshot(
        'Could not mount storage in the sandbox: Traceback (most recent call last):\nValueError: ***'
    )


@pytest.mark.parametrize('reply', ['{"recovered": null}', '{"error": "unheard-of", "message": "no"}'])
async def test_a_reply_not_in_the_helpers_shape_is_a_mount_error(reply: str) -> None:
    workspace = wrap(FakeRoot(f'{reply}\n'), [s3()])

    with pytest.raises(MountError, match='Could not mount storage in the sandbox'):
        await workspace.exists('data')


async def test_a_key_the_helper_quotes_escaped_is_blanked_too() -> None:
    key = 'AKIA"quoted\\slash'
    reply = json.dumps({'error': 'rclone', 'message': f'rclone rejected {key}'})
    workspace = wrap(FakeRoot(f'{reply}\n'), lambda: [s3(key=key)])

    with pytest.raises(MountError) as raised:
        await workspace.run(['true'])

    assert str(raised.value) == snapshot('rclone rejected ***')


async def test_a_sandbox_that_never_runs_the_helper_is_timed_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('pydantic_ai_harness._mounts._CALL_SECONDS', 0)
    root = FakeRoot(wait=anyio.Event())
    workspace = wrap(root, lambda: [s3()])

    with pytest.raises(
        MountError, match=re.escape('The sandbox did not run the mount helper within 0 seconds. Try again.')
    ):
        await workspace.exists('data')


async def test_an_operation_after_a_failed_mount_tries_again() -> None:
    root = FakeRoot('not json', '{"recovered": []}')
    workspace = wrap(root, lambda: [s3()])

    with pytest.raises(MountError):
        await workspace.exists('data')
    await workspace.exists('data')

    assert len(root.calls) == 2


async def test_files_moved_aside_are_reported() -> None:
    root = FakeRoot('{"recovered": ["/workspace/.pydantic-ai-recovered/20261110T120000Z-abc123-data"]}')
    workspace = wrap(root, lambda: [s3()])

    with pytest.warns(
        UserWarning, match='were moved to /workspace/.pydantic-ai-recovered/20261110T120000Z-abc123-data'
    ):
        await workspace.exists('data')


async def test_concurrent_first_operations_share_one_mount() -> None:
    release = anyio.Event()
    root = FakeRoot(wait=release)
    workspace = wrap(root, lambda: [s3()])

    async with anyio.create_task_group() as tg:
        tg.start_soon(workspace.exists, 'data')
        tg.start_soon(workspace.exists, 'data')
        await root.started.wait()
        release.set()

    assert len(root.calls) == 1


async def test_a_cancelled_operation_lets_the_mount_finish() -> None:
    release = anyio.Event()
    root = FakeRoot(wait=release)
    workspace = wrap(root, lambda: [s3()])

    async with anyio.create_task_group() as tg:
        tg.start_soon(workspace.exists, 'data')
        await root.started.wait()
        tg.cancel_scope.cancel()
        release.set()

    assert root.finished == 1
    # The mount it finished counts: the next operation doesn't mount again.
    await workspace.exists('data')
    assert len(root.calls) == 1


def test_the_durable_policy_names_the_mounts_but_not_their_keys() -> None:
    def policy(key: str) -> tuple[object, ...]:
        workspace = mounted(FakeWorkspace('box'), FakeRoot(), [s3(key=key, region='eu-1')], CTX)
        assert isinstance(workspace, Workspace)
        return workspace.durable_policy()

    assert policy('first') == policy('second') == snapshot((('data', 's3://bucket/reports/', None, 'eu-1', False),))
    assert not any(secret in repr(policy('first')) for secret in SECRETS)


def test_the_durable_policy_of_a_function_is_the_function() -> None:
    def mounts(_: RunContext[None]) -> list[S3Mount]:
        raise AssertionError('not called to build the policy')  # pragma: no cover

    def other(_: RunContext[None]) -> list[S3Mount]:
        raise AssertionError('not called to build the policy')  # pragma: no cover

    def policy(function: Callable[[RunContext[None]], list[S3Mount]]) -> tuple[object, ...]:
        workspace = mounted(FakeWorkspace('box'), FakeRoot(), function, CTX)
        assert isinstance(workspace, Workspace)
        return workspace.durable_policy()

    assert policy(mounts) == policy(mounts) == (mounts,)
    assert policy(mounts) != policy(other)


def test_the_repr_leaves_out_the_keys() -> None:
    assert repr(s3()) == snapshot(
        "S3Mount(path='data', url='s3://bucket/reports/', expires_at=datetime.datetime(2026, 11, 10, 13, 0, tzinfo=datetime.timezone.utc), endpoint=None, region=None, read_only=False)"
    )


@pytest.mark.parametrize(
    ('changes', 'message'),
    [
        ({'path': '/data'}, "The mount path '/data' must be a folder in the working directory"),
        ({'path': ''}, "The mount path '' must be a folder"),
        ({'path': '.'}, "The mount path '.' must be a folder"),
        ({'path': 'data/'}, "The mount path 'data/' must be a folder"),
        ({'path': 'a//b'}, "The mount path 'a//b' must be a folder"),
        ({'path': 'a/../b'}, "The mount path 'a/../b' must be a folder"),
        ({'path': '..'}, "The mount path '..' leads out of the working directory"),
        ({'path': '../data'}, "The mount path '../data' leads out of the working directory"),
        ({'url': 'gs://bucket/'}, "The mount URL 'gs://bucket/' must look like 's3://bucket/'"),
        ({'url': 's3://'}, "The bucket name '' in 's3://' is not valid"),
        ({'url': 's3://Bucket/'}, "The bucket name 'Bucket' in 's3://Bucket/' is not valid"),
        ({'url': 's3://ab/'}, "The bucket name 'ab'"),
        ({'url': 's3://bucket//x/'}, "The prefix in 's3://bucket//x/' must not hold empty, '.' or '..' parts"),
        ({'url': 's3://bucket/a//b/'}, 'must not hold empty'),
        ({'url': 's3://bucket/a/../b'}, 'must not hold empty'),
        ({'url': 's3://bucket/./b'}, 'must not hold empty'),
        ({'expires_at': datetime(2026, 11, 10, 13)}, "The `expires_at` of mount 'data' has no time zone"),
        ({'expires_at': NOW}, "The keys for mount 'data' have already expired. Pass keys that expire later."),
    ],
)
def test_invalid_mounts_are_refused(changes: dict[str, Any], message: str) -> None:
    fields: dict[str, Any] = {
        'path': 'data',
        'url': 's3://bucket/',
        'access_key_id': 'id',
        'secret_access_key': 'key',
        'expires_at': NOW + timedelta(hours=1),
    } | changes
    with pytest.raises(UserError, match=re.escape(message)):
        S3Mount(**fields)


@given(st.text(alphabet='ab./-_\x00 ', max_size=12))
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_any_accepted_path_is_a_folder_inside_the_working_directory(path: str) -> None:
    try:
        mount = S3Mount(
            path=path, url='s3://bucket/', access_key_id='', secret_access_key='', expires_at=NOW + timedelta(hours=1)
        )
    except UserError:
        return
    joined = posixpath.normpath(posixpath.join('/workspace', mount.path))
    assert joined.startswith('/workspace/') and joined == posixpath.join('/workspace', mount.path)


@given(st.from_regex(r's3://[a-z./_-]{0,8}(/[a-z./]{0,8})?', fullmatch=True) | st.text(max_size=20))
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_any_accepted_url_names_a_bucket_and_a_clean_prefix(url: str) -> None:
    try:
        mount = S3Mount(
            path='data', url=url, access_key_id='', secret_access_key='', expires_at=NOW + timedelta(hours=1)
        )
    except UserError:
        return
    helper = FakeRoot()
    await wrap(helper, [mount]).exists('data')
    root: str = helper.requests()[0]['mounts'][0]['root']
    bucket, *prefix = root.removesuffix('/').split('/')
    assert url.startswith(f's3://{bucket}')
    assert all(part not in ('', '.', '..') for part in prefix)
    # A prefix is sent as a folder, so rclone never takes it for an object of the same name.
    assert root.endswith('/') == bool(prefix)


@given(st.text())
@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_any_output_mounts_or_raises_a_mount_error(stdout: str) -> None:
    workspace = wrap(FakeRoot(stdout), [s3()])
    with contextlib.suppress(MountError):
        await workspace.exists('data')
