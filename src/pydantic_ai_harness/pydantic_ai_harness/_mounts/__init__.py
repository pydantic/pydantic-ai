"""Mounting cloud storage into sandboxes with rclone, shared by every sandbox backend.

A backend implements `SupportsMounts` with a `Mounter` and one primitive of its own: running a command as root with one
line on stdin, kept out of logs. The line carries the keys; the command is `_helper.py`'s source.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import uuid
import warnings
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import anyio
import anyio.to_thread

from pydantic_ai.workspaces import CommandResult, Mount, MountError

__all__ = ('Mounter', 'RunAsRoot', 'check_mount')

RunAsRoot = Callable[[Sequence[str], bytes], Awaitable[CommandResult]]
"""Run an argv as root in the sandbox with `stdin` (one line) as its input, which must not reach any log."""

_HELPER = (Path(__file__).parent / '_helper.py').read_text()

_ONLY_IF_MOUNTED = (
    """grep -q ' - fuse.rclone pai-' /proc/self/mountinfo || { head -n 1 > /dev/null; echo '{"recovered": []}'; exit 0; }; """
    'exec python3 -c "$1"'
)
"""With nothing to mount, the helper only has mounts of an earlier run to remove: skip it, and its need for Python,
when there are none. The request line is still read, as some sandboxes deliver it only to a command that does."""

_LEASE = timedelta(minutes=15)
"""How long a run holds the sandbox's mounts after it last ensured them, so a run that crashed frees them in time."""

_RELEASE_SECONDS = 30
"""How long the end of a run waits to release its mounts; the lease ends by itself if this gives up."""

_ERROR_TAIL = 500
"""How much of the helper's output a `MountError` quotes when the helper didn't answer."""


@runtime_checkable
class _Mountable(Protocol):
    """A filesystem rclone can mount: `S3Filesystem` and `AzureFilesystem`."""

    url: str
    credentials: Any
    share_long_lived_credentials: bool

    def _rclone_remote(self) -> tuple[dict[str, str], str]: ...

    def _rclone_keys(self, keys: Mapping[str, Any]) -> dict[str, str]: ...


def check_mount(mount: Mount) -> None:
    """Refuse a mount whose filesystem rclone can't mount without putting long-lived keys in the sandbox. No I/O."""
    filesystem = mount.filesystem
    name = type(filesystem).__name__
    if not isinstance(filesystem, _Mountable):
        raise MountError(
            f'{name} cannot be mounted for commands; an `S3Filesystem` or `AzureFilesystem` can. '
            'Leave out `commands=True` and file tools still reach the folder.'
        )
    credentials = filesystem.credentials
    if credentials is None:
        raise MountError(
            f'{name} finds its keys in the environment, so they cannot be handed to the sandbox. '
            'Pass `credentials=`: keys that expire, or a function returning them.'
        )
    if not callable(credentials):
        _keys(filesystem, credentials)


class Mounter:
    """A sandbox backend's rclone mounts: which run holds them, and whether an earlier run may have left some."""

    def __init__(self, run_as_root: RunAsRoot, *, attached: bool) -> None:
        self._run_as_root = run_as_root
        self.owner = uuid.uuid4().hex
        """The run holding the mounts; the capability that built the backend sets its run's id."""
        # A sandbox this backend creates starts with nothing mounted; one it attaches to may have an earlier run's.
        self._leftovers_possible = attached
        self._leased = False

    def needed(self, mounts: Sequence[Mount]) -> bool:
        """Whether `ensure` has anything to do, so a sandbox isn't started or asked for nothing."""
        return bool(mounts) or self._leftovers_possible

    async def ensure(self, mounts: Sequence[Mount], working_dir: str) -> datetime | None:
        """Mount `mounts` (the whole set), removing others, and hold them for this run. Returns when to call again."""
        requests: list[dict[str, Any]] = []
        keys: dict[str, dict[str, str]] = {}
        expiries: list[datetime] = []
        for mount in mounts:
            filesystem = mount.filesystem
            assert isinstance(filesystem, _Mountable)  # `check_mount` admitted only these.
            path = posixpath.join(working_dir, mount.path)
            remote, root = filesystem._rclone_remote()  # pyright: ignore[reportPrivateUsage]
            identity = json.dumps([remote, root, mount.read_only, path], sort_keys=True)
            fingerprint = hashlib.sha256(identity.encode()).hexdigest()[:16]
            credentials = filesystem.credentials
            # A credentials function may call out (to STS, say), so it runs off the event loop.
            fetched = await anyio.to_thread.run_sync(credentials) if callable(credentials) else credentials
            fresh = cast('Mapping[str, Any]', fetched)
            keys[fingerprint] = _keys(filesystem, fresh)
            expires_at: datetime | None = fresh.get('expires_at')
            requests.append(
                {
                    'path': path,
                    'fingerprint': fingerprint,
                    'remote': remote,
                    'root': root,
                    'read_only': mount.read_only,
                    'keys_expire': None if expires_at is None else expires_at.timestamp(),
                }
            )
            # Only a function can give fresher keys; given keys are used until they expire.
            if callable(credentials) and fresh.get('expires_at') is not None:
                expiries.append(fresh['expires_at'])
        # The lease is a duration, so the sandbox's clock decides when it ends.
        lease_until = datetime.now(UTC) + _LEASE
        reply = await self._call(
            {
                'mounts': requests,
                'keys': keys,
                'working_dir': working_dir,
                'owner': self.owner,
                'lease_seconds': _LEASE.total_seconds(),
            },
            only_if_mounted=not mounts,
        )
        for folder in reply['recovered']:
            warnings.warn(
                f'Files written into a mount folder while the bucket was not mounted there were moved to {folder}.',
                stacklevel=2,
            )
        self._leftovers_possible = False
        self._leased = bool(mounts)
        return min([*expiries, lease_until]) if mounts else None

    async def release(self) -> None:
        """Let other runs mount their own filesystems in the sandbox, leaving this run's mounted until then."""
        if not self._leased:
            return
        self._leased = False
        # The mounts stay in the sandbox, for the next run to keep or remove.
        self._leftovers_possible = True
        # Best effort and bounded, and never the reason a run fails: if the sandbox can't be reached, the lease ends
        # by itself.
        with anyio.move_on_after(_RELEASE_SECONDS, shield=True):
            try:
                await self._call({'release': self.owner})
            except Exception:
                pass

    async def _call(self, request: dict[str, Any], *, only_if_mounted: bool = False) -> dict[str, Any]:
        argv = ['sh', '-c', _ONLY_IF_MOUNTED, 'sh', _HELPER] if only_if_mounted else ['python3', '-c', _HELPER]
        # A cancelled caller waits for the helper, which has a deadline of its own: stopped halfway, it could leave a
        # mount half made, or keep the lock that every later call waits for.
        result: CommandResult | None = None
        with anyio.CancelScope(shield=True):
            result = await self._run_as_root(argv, f'{json.dumps(request)}\n'.encode())
        assert result is not None  # A shielded scope never swallows the return.
        # The last JSON line is the helper's; `sudo` and login shells may print around it.
        line = next((line for line in reversed(result.stdout.splitlines()) if line.startswith('{')), None)
        if line is None:
            output = (result.stderr or result.stdout).strip()[-_ERROR_TAIL:]
            raise MountError(f'Could not mount storage in the sandbox: {output}')
        reply: dict[str, Any] = json.loads(line)
        if 'error' in reply:
            raise MountError(reply['message'])
        return reply


def _keys(filesystem: _Mountable, keys: Mapping[str, Any]) -> dict[str, str]:
    """The keys in rclone's terms, refused when commands shouldn't be handed them."""
    # Imported here: it needs the `cloud-storage` extra, which only a filesystem that got this far has.
    from pydantic_ai_harness.cloud_storage._filesystem import check_expiry

    try:
        check_expiry(keys)
        converted = filesystem._rclone_keys(keys)  # pyright: ignore[reportPrivateUsage]
    except ValueError as error:
        raise MountError(f'The keys for {filesystem.url} cannot be used: {error}') from error
    expires_at = keys.get('expires_at')
    if expires_at is not None and expires_at <= datetime.now(UTC):
        raise MountError(
            f'The keys for {filesystem.url} have expired. Pass keys that expire later, or a function returning fresh ones.'
        )
    if expires_at is None and not filesystem.share_long_lived_credentials:
        raise MountError(
            f'The keys for {filesystem.url} do not expire, so they are not handed to the sandbox, where commands '
            'could read them. Use keys that expire, or set `share_long_lived_credentials=True`.'
        )
    return converted
