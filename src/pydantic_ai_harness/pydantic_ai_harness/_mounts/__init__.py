"""Mounting S3 buckets into sandboxes with rclone, shared by every sandbox capability that offers `mounts=`.

A sandbox backend gives one primitive of its own, `RunAsRoot`: run an argv as root with one line on stdin, kept out of
logs. The line carries the request, keys included; the argv runs `_helper.py`'s source, which does the mounting.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import posixpath
import re
import warnings
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from functools import cached_property
from pathlib import Path
from typing import assert_never

import anyio
import anyio.to_thread
from pydantic import TypeAdapter, ValidationError
from typing_extensions import TypedDict

from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import (
    CommandResult,
    FileEntry,
    Workspace,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceError,
    WrapperWorkspace,
)
from pydantic_ai_harness._mounts._helper import DEADLINE_SECONDS, FailureCode, Keys, Request, Wanted

__all__ = ('MountError', 'S3Mount')

RunAsRoot = Callable[[Sequence[str], bytes], Awaitable[CommandResult]]
"""Run an argv as root in the sandbox with `stdin` (one line) as its input, which must not reach any log."""

_HELPER = (Path(__file__).parent / '_helper.py').read_text(encoding='utf-8')

_REFRESH_MARGIN = timedelta(minutes=5)
"""How long before the keys expire the next operation gets fresh ones, so a long command doesn't outlive them."""

_CALL_SECONDS = DEADLINE_SECONDS + 60
"""How long one helper call may take: its own deadline, plus the sandbox starting it."""

_ERROR_TAIL = 500
"""How much of the helper's output a `MountError` quotes when the helper didn't answer."""


class _Reply(TypedDict, total=False):
    """What the helper prints: the folders strays were moved to, or why it refused."""

    recovered: list[str]
    error: FailureCode
    message: str


_REPLY = TypeAdapter(_Reply)

_URL = re.compile(r's3://(?P<bucket>[^/]*)(?:/(?P<prefix>.*))?', re.DOTALL)
_BUCKET = re.compile(r'[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]')
"""S3's bucket rules, plus underscores, which some S3-compatible stores allow."""


class MountError(WorkspaceError):
    """The sandbox refused a mount or could not make it. The message says what to change.

    A mistake in an `S3Mount` or in what `mounts=` returns raises `UserError` instead.
    """


def _now() -> datetime:
    """The current time; a function so tests can move the clock."""
    return datetime.now(UTC)


@dataclass(frozen=True, kw_only=True)
class S3Mount:
    """A folder in the sandbox that shows an S3 (or S3-compatible) bucket. Shell commands and file tools see the same files."""

    path: str
    """The folder, relative to the sandbox's working directory, like `'data'`. It is created if missing."""
    url: str
    """The bucket, and optionally a prefix in it to show instead of the whole bucket: `'s3://bucket/'` or `'s3://bucket/prefix/'`."""
    access_key_id: str = field(repr=False, compare=False)
    """The access key ID, kept out of the `repr` along with the other keys."""
    secret_access_key: str = field(repr=False, compare=False)
    """The secret access key that goes with `access_key_id`."""
    expires_at: datetime | None = field(default=None, compare=False)
    """When the keys expire, so a function passed as `mounts=` is called again for fresh ones before then.

    `None` means long-lived keys, such as most R2 and MinIO keys, used until they are revoked. Code in the sandbox can
    read the keys, so prefer short-lived ones scoped to the bucket.
    """
    session_token: str | None = field(default=None, repr=False, compare=False)
    """The session token that comes with temporary keys, such as those from AWS STS."""
    endpoint: str | None = None
    """The endpoint of an S3-compatible store, such as Cloudflare R2 or MinIO. `None` means AWS S3."""
    region: str | None = None
    """The store's region, passed to rclone as written. Cloudflare R2 takes `'auto'`."""
    read_only: bool = False
    """Whether the folder refuses writes."""

    def __post_init__(self) -> None:
        if posixpath.isabs(self.path) or self.path in ('', '.') or posixpath.normpath(self.path) != self.path:
            raise UserError(
                f'The mount path {self.path!r} must be a folder in the working directory, written like '
                "'data' or 'data/reports'."
            )
        if self.path == '..' or self.path.startswith('../'):
            raise UserError(f'The mount path {self.path!r} leads out of the working directory. Mount inside it.')
        _root(self.url)
        if self.expires_at is None:
            return
        if self.expires_at.utcoffset() is None:
            raise UserError(
                f'The `expires_at` of mount {self.path!r} has no time zone. Pass an aware datetime, '
                'like `datetime.now(UTC) + timedelta(hours=1)`.'
            )
        if self.expires_at <= _now():
            raise UserError(f'The keys for mount {self.path!r} have already expired. Pass keys that expire later.')


def _root(url: str) -> str:
    """`bucket/prefix/`, what rclone mounts, from `s3://bucket/prefix/`."""
    found = _URL.fullmatch(url)
    if found is None:
        raise UserError(f"The mount URL {url!r} must look like 's3://bucket/' or 's3://bucket/prefix/'.")
    bucket = found['bucket']
    if not _BUCKET.fullmatch(bucket):
        raise UserError(
            f'The bucket name {bucket!r} in {url!r} is not valid: use 3 to 63 lowercase letters, digits, '
            'dots and hyphens, starting and ending with a letter or digit.'
        )
    prefix = (found['prefix'] or '').removesuffix('/')
    if not prefix:
        return bucket
    if any(part in ('', '.', '..') for part in prefix.split('/')):
        raise UserError(
            f"The prefix in {url!r} must not hold empty, '.' or '..' parts. Write it like 's3://bucket/reports/2026/'."
        )
    # The trailing slash tells rclone it is a folder, even beside an object of the same name.
    return f'{bucket}/{prefix}/'


def _settings(mount: S3Mount) -> tuple[dict[str, str], str, Keys]:
    """The rclone remote's parameters, the root it mounts, and its keys. The only code that reads a mount's keys."""
    match mount:
        case S3Mount():
            remote = {'type': 's3', 'provider': 'AWS' if mount.endpoint is None else 'Other'}
            if mount.endpoint is not None:
                remote['endpoint'] = mount.endpoint
            if mount.region is not None:
                remote['region'] = mount.region
            keys: Keys = {
                'access_key_id': mount.access_key_id,
                'secret_access_key': mount.secret_access_key,
                # Always sent, so fresh keys without a token clear the old one.
                'session_token': mount.session_token or '',
            }
            return remote, _root(mount.url), keys
    assert_never(mount)


def _request(mounts: Sequence[S3Mount], working_dir: str) -> tuple[Request, datetime | None]:
    """The helper's request, and when its keys expire first; `None` when none of them expire."""
    wanted: list[Wanted] = []
    for mount in mounts:
        remote, root, keys = _settings(mount)
        path = posixpath.join(working_dir, mount.path)
        # What the folder shows, never the keys: the helper compares it to tell a refresh from another bucket.
        identity = json.dumps([remote, root, mount.read_only, path], sort_keys=True)
        wanted.append(
            {
                'path': path,
                'fingerprint': hashlib.sha256(identity.encode()).hexdigest()[:16],
                'remote': remote,
                'root': root,
                'read_only': mount.read_only,
                'keys': keys,
                'expires_at': None if mount.expires_at is None else mount.expires_at.timestamp(),
            }
        )
    expiries = [mount.expires_at for mount in mounts if mount.expires_at is not None]
    return {'working_dir': working_dir, 'mounts': wanted}, min(expiries, default=None)


def _reply(result: CommandResult) -> _Reply:
    """The helper's answer: the last line that starts with `{`, since `sudo` or a login shell may print around it."""
    line = next((line for line in reversed(result.stdout.splitlines()) if line.startswith('{')), None)
    answer: _Reply | None = None
    if line is not None:
        try:
            answer = _REPLY.validate_json(line)
        except ValidationError:
            pass
    if answer is None:
        output = (result.stderr or result.stdout).strip()[-_ERROR_TAIL:]
        raise MountError(f'Could not mount storage in the sandbox: {output}')
    if 'error' in answer:
        raise MountError(answer.get('message', answer['error']))
    return answer


def _check_paths(mounts: Sequence[S3Mount]) -> None:
    # Every pair, not neighbours: `data-other` sorts between `data` and `data/inner`.
    for path, after in itertools.combinations(sorted(mount.path for mount in mounts), 2):
        if after == path or after.startswith(f'{path}/'):
            raise UserError(
                f'The mount paths {path!r} and {after!r} overlap. Give each mount its own folder, neither inside another.'
            )


def mounted(
    backend: WorkspaceBackend,
    run_as_root: RunAsRoot,
    mounts: Sequence[S3Mount]
    | Callable[[RunContext[AgentDepsT]], Sequence[S3Mount]]
    | Callable[[RunContext[AgentDepsT]], Awaitable[Sequence[S3Mount]]],
    ctx: RunContext[AgentDepsT],
) -> WorkspaceBackend:
    """`backend`, mounting a sandbox capability's `mounts=` before its first command or file operation.

    A function is not called here, since building a workspace does no I/O and the function may ask a service such as
    AWS STS for keys: the first operation calls it, a sync one in a worker thread. An empty list leaves `backend` as
    it is.
    """
    if callable(mounts):
        # The function stands for the mounts it will return, so a run-level capability with another one is told apart.
        return _Mounted(_as_workspace(backend), run_as_root, lambda: mounts(ctx), policy=(mounts,), refreshes=True)
    listed = tuple(mounts)
    if not listed:
        return backend
    _check_paths(listed)
    policy = sorted((mount.path, mount.url, mount.endpoint, mount.region, mount.read_only) for mount in listed)
    return _Mounted(_as_workspace(backend), run_as_root, lambda: listed, policy=tuple(policy), refreshes=False)


def _as_workspace(backend: WorkspaceBackend) -> Workspace:
    return backend if isinstance(backend, Workspace) else Workspace(backend)


def _shown(mounts: Sequence[S3Mount]) -> frozenset[S3Mount]:
    """What the folders show: `S3Mount` equality leaves out the keys and their expiry."""
    return frozenset(mounts)


class _Mounted(WrapperWorkspace):
    """Mounts the buckets before the first operation that can touch files, and gives them fresh keys before they expire."""

    def __init__(
        self,
        wrapped: Workspace,
        run_as_root: RunAsRoot,
        mounts: Callable[[], Sequence[S3Mount] | Awaitable[Sequence[S3Mount]]],
        *,
        policy: tuple[object, ...],
        refreshes: bool,
    ) -> None:
        super().__init__(wrapped)
        self._run_as_root = run_as_root
        self._mounts = mounts
        self._policy = policy
        self._refreshes = refreshes
        """Whether `mounts` can return fresh keys; a list's keys are used until they expire, then left to fail."""
        self._first: frozenset[S3Mount] | None = None
        """What the first call returned, which every later call must match."""
        self._remount_at: datetime | None = None
        """When the next operation mounts again to refresh the keys; `None` before the first."""

    @cached_property
    def _lock(self) -> anyio.Lock:
        # Made on first use, inside the event loop that uses it.
        return anyio.Lock()

    def durable_policy(self) -> tuple[object, ...]:
        """What the folders show for a list, never the keys; the function itself for a function."""
        return (*super().durable_policy(), *self._policy)

    def _due(self) -> bool:
        return self._remount_at is None or _now() >= self._remount_at

    async def _ensure(self) -> None:
        if not self._due():
            return
        async with self._lock:
            # Another operation may have mounted while this one waited for the lock.
            if not self._due():
                return
            # In a worker thread: a sync function may ask a service such as AWS STS for keys. An async one only makes its
            # coroutine there, which runs here.
            returned = await anyio.to_thread.run_sync(self._mounts)
            fresh = tuple(await returned if isinstance(returned, Awaitable) else returned)
            if self._first is None:
                _check_paths(fresh)
            elif len(fresh) != len(self._first) or _shown(fresh) != self._first:
                raise UserError('`mounts=` must return the same mounts each time, with fresh keys.')
            now = _now()
            if not fresh:
                # A function that returned no mounts: nothing to mount, and nothing to refresh.
                self._first = frozenset()
                self._remount_at = datetime.max.replace(tzinfo=UTC)
                return
            for mount in fresh:
                if mount.expires_at is not None and mount.expires_at <= now:
                    raise MountError(
                        f'The keys for mount {mount.path!r} have expired. '
                        'Pass `mounts=` as a function that returns fresh keys.'
                    )
            request, expires_at = _request(fresh, await self.working_dir())
            reply = await self._call(request)
            self._first = _shown(fresh)
            for folder in reply.get('recovered', []):
                warnings.warn(
                    f'Files written into a mount folder while the bucket was not mounted there were moved to {folder}.',
                    stacklevel=2,
                )
            if not self._refreshes or expires_at is None:
                # Nothing could give a list fresh keys: once they expire, only commands that use the folder fail.
                # Keys that don't expire need none.
                self._remount_at = datetime.max.replace(tzinfo=UTC)
                return
            # Keys too short-lived for the margin are used until they expire, not refreshed on every operation.
            refresh_at = expires_at - _REFRESH_MARGIN
            self._remount_at = refresh_at if refresh_at > now else expires_at

    async def _call(self, request: Request) -> _Reply:
        # Each key as written, and as written inside JSON, where quotes, backslashes and control characters are escaped;
        # longest first, so a form inside a longer one doesn't leave the rest of the longer one showing.
        forms = {
            form
            for wanted in request['mounts']
            for value in (
                wanted['keys']['access_key_id'],
                wanted['keys']['secret_access_key'],
                wanted['keys']['session_token'],
            )
            if value
            for form in (value, json.dumps(value)[1:-1])
        }
        secrets = sorted(forms, key=len, reverse=True)
        # A cancelled caller waits for the helper, which has a deadline of its own: stopped halfway, it could leave a
        # mount half made, or keep the lock that every later call waits for.
        # The helper's own deadline starts only once it runs, so a sandbox that never starts it is timed out here.
        result: CommandResult | None = None
        with anyio.CancelScope(shield=True), anyio.move_on_after(_CALL_SECONDS):
            # `-I`: root's Python reads no `PYTHON*` variable, user site-packages or working directory, any of which
            # the agent's user could point at a `sitecustomize.py` of its own. The helper needs only the standard library.
            result = await self._run_as_root(['python3', '-I', '-c', _HELPER], f'{json.dumps(request)}\n'.encode())
        if result is None:
            raise MountError(f'The sandbox did not run the mount helper within {_CALL_SECONDS} seconds. Try again.')
        # Blanked out in both spellings, so an error message doesn't quote a key as the helper or rclone wrote it.
        stdout, stderr = result.stdout, result.stderr
        for secret in secrets:
            stdout, stderr = stdout.replace(secret, '***'), stderr.replace(secret, '***')
        return _reply(replace(result, stdout=stdout, stderr=stderr))

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        await self._ensure()
        return await super().run(command, shell=shell, env=env, timeout=timeout)

    async def read_bytes(self, path: str) -> bytes:
        await self._ensure()
        return await super().read_bytes(path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        await self._ensure()
        await super().write_bytes(path, data)

    async def stat(self, path: str) -> FileEntry:
        await self._ensure()
        return await super().stat(path)

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        await self._ensure()
        return await super().list_dir(path)

    async def make_dir(self, path: str) -> None:
        await self._ensure()
        await super().make_dir(path)

    async def remove(self, path: str) -> None:
        await self._ensure()
        await super().remove(path)

    async def exists(self, path: str) -> bool:
        await self._ensure()
        return await super().exists(path)

    async def realpath(self, path: str) -> str:
        await self._ensure()
        return await super().realpath(path)
