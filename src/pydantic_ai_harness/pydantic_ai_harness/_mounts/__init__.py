"""Mounting buckets into sandboxes with rclone, shared by every sandbox backend.

A backend implements `SupportsMounts` with these two functions and one primitive of its own: running a command as
root with one line on stdin, kept out of logs. The line carries the keys; the command is `_helper.py`'s source.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import sys
import warnings
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

import anyio
import anyio.to_thread

from pydantic_ai.workspaces import CommandResult, Mount, MountError

if TYPE_CHECKING:
    from pydantic_ai_harness.bucket import Bucket

__all__ = ('RunAsRoot', 'can_mount', 'ensure_mounts')

RunAsRoot = Callable[[Sequence[str], bytes], Awaitable[CommandResult]]
"""Run an argv as root in the sandbox with `stdin` (one line) as its input, which must not reach any log."""

_HELPER = (Path(__file__).parent / '_helper.py').read_text()

_ONLY_IF_MOUNTED = (
    """grep -q ' - fuse.rclone pai-' /proc/self/mountinfo || { head -n 1 > /dev/null; echo '{"recovered": []}'; exit 0; }; """
    'exec python3 -c "$1"'
)
"""With nothing to mount, the helper only has mounts of an earlier configuration to remove: skip it, and its need for
Python, when there are none. The request line is still read, as some sandboxes deliver it only to a command that does."""

_ERROR_TAIL = 500
"""How much of the helper's output a `MountError` quotes when the helper didn't answer."""


def can_mount(mount: Mount) -> bool:
    """Whether rclone can mount `mount`'s storage without putting long-lived keys in the sandbox. No I/O."""
    storage = _bucket(mount)
    if storage is None:
        return False
    # S3 and Azure, whose mounts are tested; Google Cloud Storage buckets are reachable through file tools.
    url = urlsplit(storage.url)
    if url.scheme not in ('s3', 'az') and not (url.scheme == 'https' and url.netloc.endswith('.blob.core.windows.net')):
        return False
    credentials = storage.credentials
    if credentials is None:
        # Keys found in this process's environment can't be read back out to hand on.
        return False
    if callable(credentials):
        return True
    kind = 'access_key_id' if url.scheme == 's3' else 'access_key'
    expiring = credentials.get('expires_at') is not None
    return kind in credentials and (expiring or storage.share_long_lived_credentials)


async def ensure_mounts(run_as_root: RunAsRoot, mounts: Sequence[Mount], working_dir: str) -> datetime | None:
    """Mount `mounts` (the whole set) with rclone, repairing lost ones and removing ones no longer wanted.

    Returns when the earliest keys expire, so they are refreshed before then.
    """
    requests: list[dict[str, Any]] = []
    keys: dict[str, dict[str, str]] = {}
    expiries: list[datetime] = []
    for mount in mounts:
        # `can_mount` admitted only buckets.
        bucket = cast('Bucket', mount.storage)
        path = posixpath.join(working_dir, mount.path)
        remote, root = _remote(bucket)
        identity = json.dumps([remote, root, mount.read_only, path], sort_keys=True)
        fingerprint = hashlib.sha256(identity.encode()).hexdigest()[:16]
        requests.append(
            {'path': path, 'fingerprint': fingerprint, 'remote': remote, 'root': root, 'read_only': mount.read_only}
        )
        credentials = await _credentials(bucket)
        keys[fingerprint] = _keys(bucket, credentials)
        # Only a function can give fresher keys; given keys are used until they expire.
        if callable(bucket.credentials) and credentials.get('expires_at') is not None:
            expiries.append(credentials['expires_at'])
    request = json.dumps({'mounts': requests, 'keys': keys, 'working_dir': working_dir})
    argv = ['python3', '-c', _HELPER] if mounts else ['sh', '-c', _ONLY_IF_MOUNTED, 'sh', _HELPER]
    result = await _run_to_the_end(run_as_root, argv, f'{request}\n'.encode())
    # The last JSON line is the helper's; `sudo` and login shells may print around it.
    line = next((line for line in reversed(result.stdout.splitlines()) if line.startswith('{')), None)
    if line is None:
        output = (result.stderr or result.stdout).strip()[-_ERROR_TAIL:]
        raise MountError(f'Could not mount storage in the sandbox: {output}')
    reply = json.loads(line)
    if 'error' in reply:
        raise MountError(f'Could not mount storage in the sandbox: {reply["message"]}')
    for folder in reply['recovered']:
        warnings.warn(
            f'Files written into a mount folder while nothing was mounted there were moved to {folder}.', stacklevel=2
        )
    return min(expiries, default=None)


async def _run_to_the_end(run_as_root: RunAsRoot, argv: Sequence[str], stdin: bytes) -> CommandResult:  # pyright: ignore[reportReturnType]
    # A cancelled caller waits for the helper, which has a deadline of its own: stopped halfway, it could leave a mount
    # half made, or keep the lock that every later call waits for. (A shielded scope never swallows the return, which
    # pyright can't tell.)
    with anyio.CancelScope(shield=True):
        return await run_as_root(argv, stdin)


def _bucket(mount: Mount) -> Bucket | None:
    # Not imported here: it needs the `bucket` extra, which a sandbox without bucket mounts doesn't. Until something
    # imports it, no storage can be a `Bucket`.
    module = sys.modules.get('pydantic_ai_harness.bucket._bucket')
    storage = mount.storage
    return cast('Bucket', storage) if module is not None and isinstance(storage, module.Bucket) else None


async def _credentials(bucket: Bucket) -> Mapping[str, Any]:
    credentials = bucket.credentials
    # A credentials function may call out (to STS, say), so it runs off the event loop.
    keys: Mapping[str, Any] = await anyio.to_thread.run_sync(credentials) if callable(credentials) else credentials  # type: ignore[arg-type]
    expires_at = keys.get('expires_at')
    if expires_at is not None and expires_at <= datetime.now(UTC):
        raise MountError(f'The keys for {bucket.url} have expired.')
    if expires_at is None and not bucket.share_long_lived_credentials:
        raise MountError(
            f'The keys for {bucket.url} do not expire, so they are not handed to the sandbox. Use keys that expire, '
            'or set `share_long_lived_credentials=True` on the bucket.'
        )
    return keys


def _remote(bucket: Bucket) -> tuple[dict[str, str], str]:
    """The bucket's rclone settings, without keys, and the path of its folder in rclone's terms."""
    url = urlsplit(bucket.url)
    folder = url.path.strip('/')
    if url.scheme == 's3':
        remote = {'type': 's3', 'provider': 'Other' if bucket.endpoint else 'AWS', 'env_auth': 'false'}
        if bucket.endpoint:
            remote['endpoint'] = bucket.endpoint
        if bucket.region:
            remote['region'] = bucket.region
        return remote, posixpath.join(url.netloc, folder)
    # Azure: `az://<container>/<folder>` with `account_name` in options, or `https://<account>.blob.../<container>/...`.
    if url.scheme == 'az':
        account, container = bucket.options['account_name'], url.netloc
    else:
        account = url.netloc.split('.', 1)[0]
        container, _, folder = folder.partition('/')
    remote = {'type': 'azureblob', 'account': account}
    if bucket.endpoint:
        remote['endpoint'] = bucket.endpoint
    return remote, posixpath.join(container, folder)


def _keys(bucket: Bucket, keys: Mapping[str, Any]) -> dict[str, str]:
    """The keys in rclone's terms."""
    if 'access_key_id' in keys:
        return {
            'access_key_id': keys['access_key_id'],
            'secret_access_key': keys['secret_access_key'],
            'session_token': keys.get('token') or '',
        }
    if 'access_key' in keys:
        return {'key': keys['access_key']}
    raise MountError(f'Mounting {bucket.url} needs an access key: shell mounts take S3 keys or an Azure account key.')
