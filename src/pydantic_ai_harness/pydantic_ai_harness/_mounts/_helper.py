"""Mounts buckets with rclone inside a sandbox. Runs there as root, as `python3 -c <this file>`.

Standard library only, since it runs on the sandbox's Python. The request, keys included, is one JSON line on stdin,
so keys never reach a command line, a file or a log: they go from here to the rclone daemon over a socket only root
can open, and live in its memory. It prints one JSON line: `{"recovered": [...]}` or `{"error": ..., "message": ...}`.

The request names every mount the sandbox should have. Each mount's kernel mount source is `pai-<fingerprint>`, so
this can tell its own mounts, and which configuration each is, from anything else mounted.
"""

from __future__ import annotations

import fcntl
import hashlib
import http.client
import io
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
import zipfile
from typing import Any

STATE = '/dev/shm/pydantic-ai-mounts'
"""Daemon socket and lock. In memory, so a restarted sandbox starts clean, as its kernel mounts do."""
CACHE = '/var/cache/pydantic-ai-mounts'
"""rclone's write cache, on disk, so uploads still queued when the sandbox stops are sent after it restarts."""
SOCKET = f'{STATE}/rc.sock'
RCLONE = '/usr/local/bin/rclone'
RCLONE_VERSION = 'v1.75.2'
RCLONE_SHA256 = {
    'x86_64': '349ac8fba6ff65d6247043f1750cdcb518ec5d500ef91463a10d37c0ccdf3702',
    'aarch64': '7e1e8d69654941b7b7df84ee74c5f7fb09cce7d5496947fb7bf16b55ff427d10',
}
"""From https://downloads.rclone.org/v1.75.2/SHA256SUMS."""
RECOVERED = '.pydantic-ai-recovered'
DIR_CACHE_NANOSECONDS = 10 * 1_000_000_000
"""How long a listing is reused before rclone asks the bucket again, so changes made elsewhere show within seconds."""
UPLOAD_WAIT_SECONDS = 300
"""How long an unmount waits for queued uploads before giving up and leaving the mount in place."""
DOWNLOAD_SECONDS = 120
"""How long the rclone download (about 30 MB) may take."""
APT_SECONDS = 300
"""How long each apt step installing fuse3 may take."""
DAEMON_START_SECONDS = 10
"""How long rclone's daemon may take to answer after it is started."""
PROBE_SECONDS = 10
"""How long a mount may take to answer a lookup before it counts as dead."""
DEADLINE_SECONDS = 1800
"""The most one call may take, all steps together, so a caller that waits for it is never left waiting for ever."""


class MountFailed(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class _UnixConnection(http.client.HTTPConnection):
    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX)
        self.sock.settimeout(self.timeout)
        self.sock.connect(SOCKET)


def rc(command: str, body: dict[str, Any] | None = None, timeout: float = 60) -> dict[str, Any]:
    connection = _UnixConnection('localhost', timeout=timeout)
    connection.request('POST', f'/{command}', json.dumps(body or {}), {'Content-Type': 'application/json'})
    response = connection.getresponse()
    reply = json.loads(response.read() or b'{}')
    if response.status != 200:
        # rclone echoes the request, keys included, in its errors: report only its own message.
        raise MountFailed('rclone', f'rclone {command} failed: {reply.get("error", response.status)}')
    return reply


def install_rclone() -> str:
    """The image's rclone, or the pinned build, downloaded on first use."""
    found = shutil.which('rclone')
    if found is not None:
        return found
    if os.access(RCLONE, os.X_OK):
        return RCLONE
    arch = platform.machine()
    if arch not in RCLONE_SHA256:
        raise MountFailed('rclone', f'no rclone build for {arch}; install rclone in the sandbox image')
    name = f'rclone-{RCLONE_VERSION}-linux-{"amd64" if arch == "x86_64" else "arm64"}'
    try:
        # The download site refuses Python's default User-Agent.
        url = f'https://downloads.rclone.org/{RCLONE_VERSION}/{name}.zip'
        request = urllib.request.Request(url, headers={'User-Agent': 'pydantic-ai-harness'})
        with urllib.request.urlopen(request, timeout=DOWNLOAD_SECONDS) as reply:
            archive = reply.read()
    except OSError as error:
        raise MountFailed(
            'rclone', f'could not download rclone ({error}); install rclone in the sandbox image'
        ) from None
    if hashlib.sha256(archive).hexdigest() != RCLONE_SHA256[arch]:
        raise MountFailed('rclone', 'the downloaded rclone does not match its published checksum')
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        binary = zipped.read(f'{name}/rclone')
    with open(f'{RCLONE}.tmp', 'wb') as file:
        file.write(binary)
    os.chmod(f'{RCLONE}.tmp', 0o755)
    os.replace(f'{RCLONE}.tmp', RCLONE)
    return RCLONE


def install_fuse() -> None:
    """`fusermount3`, from the image or installed with apt."""
    if shutil.which('fusermount3'):
        return
    if shutil.which('apt-get'):
        env = {**os.environ, 'DEBIAN_FRONTEND': 'noninteractive'}
        for command in (['apt-get', 'update', '-q'], ['apt-get', 'install', '-y', '-q', 'fuse3']):
            subprocess.run(command, capture_output=True, env=env, timeout=APT_SECONDS)
    if not shutil.which('fusermount3'):
        raise MountFailed('fuse', 'the sandbox has no `fusermount3`; install fuse3 in its image')


def start_daemon(rclone: str) -> None:
    try:
        rc('rc/noop', timeout=5)
        return
    except (OSError, MountFailed):
        pass
    # Mounts of a daemon that died are dead too; they are cleared below, mount by mount.
    if os.path.exists(SOCKET):
        os.unlink(SOCKET)
    os.makedirs(CACHE, mode=0o700, exist_ok=True)
    subprocess.Popen(
        [rclone, 'rcd', '--rc-addr', f'unix://{SOCKET}', '--rc-no-auth', '--config', '', '--cache-dir', CACHE],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=open(f'{STATE}/rcd.log', 'ab'),  # The daemon keeps it open.
        start_new_session=True,
    )
    deadline = time.monotonic() + DAEMON_START_SECONDS
    while time.monotonic() < deadline:
        try:
            rc('rc/noop', timeout=5)
            return
        except (OSError, MountFailed):
            time.sleep(0.05)
    raise MountFailed('rclone', f'the rclone daemon did not start; see {STATE}/rcd.log')


def kernel_mounts() -> dict[str, str]:
    """Mount point to mount source, for every mount in the kernel's table (the last one wins on a stacked path)."""
    mounts: dict[str, str] = {}
    with open('/proc/self/mountinfo', 'rb') as table:
        for line in table:
            fields, _, rest = line.partition(b' - ')
            mounts[_unescape(fields.split()[4])] = _unescape(rest.split()[1])
    return mounts


def _unescape(field: bytes) -> str:
    """Undo the kernel's escapes: it writes a space, tab, newline or backslash in a path as three octal digits."""
    return re.sub(rb'\\([0-7]{3})', lambda match: bytes([int(match[1], 8)]), field).decode(errors='surrogateescape')


def alive(path: str) -> bool:
    """Whether the mount at `path` answers; a lookup of a missing name goes to the daemon, a `stat` of the root may not."""
    probe = os.path.join(path, f'.pydantic-ai-probe-{os.getpid()}')
    try:
        result = subprocess.run(['stat', probe], capture_output=True, text=True, timeout=PROBE_SECONDS)
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0 or 'No such file' in result.stderr


def served() -> dict[str, str]:
    """Mount point to the rclone file system it shows, for the mounts the running daemon serves."""
    try:
        return {mount['MountPoint']: mount['Fs'] for mount in rc('mount/listmounts')['mountPoints']}
    except (OSError, MountFailed):
        return {}


def unmount(path: str) -> None:
    """Send queued uploads, then unmount; a dead mount is detached so a new one can take its place."""
    mounted = served()
    if path in mounted:
        for _ in range(UPLOAD_WAIT_SECONDS * 10):
            stats: dict[str, int] = rc('vfs/stats', {'fs': mounted[path]})['diskCache']
            if not stats['uploadsQueued'] and not stats['uploadsInProgress']:
                break
            time.sleep(0.1)
        else:
            raise MountFailed('busy', f'{path} still has files uploading; try again once they are sent')
        rc('mount/unmount', {'mountPoint': path})
    else:
        # The daemon that served it is gone, so nothing is left to send.
        subprocess.run(['fusermount3', '-uz', path], capture_output=True)
        subprocess.run(['umount', '-l', path], capture_output=True)
    if kernel_mounts().get(path, '').startswith('pai-'):
        raise MountFailed('stuck', f'{path} could not be unmounted; restart the sandbox to clear it')


def make_folder(path: str, owner: os.stat_result) -> None:
    """Create `path` and its missing parents, owned by the agent's user."""
    if os.path.lexists(path) and not os.path.isdir(path):
        raise MountFailed('conflict', f'{path} is a file, not a folder to mount on')
    missing: list[str] = []
    while not os.path.isdir(path):
        missing.append(path)
        path = os.path.dirname(path)
    for folder in reversed(missing):
        os.mkdir(folder)
        os.chown(folder, owner.st_uid, owner.st_gid)


def recover(path: str) -> str | None:
    """Move aside what was written into the bare folder while nothing was mounted there, so the mount doesn't hide it."""
    entries = os.listdir(path)
    if not entries:
        return None
    parent, name = os.path.split(path)
    stamp = f'{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}-{uuid.uuid4().hex[:6]}'
    destination = os.path.join(parent, RECOVERED, f'{stamp}-{name}')
    os.makedirs(destination)
    for entry in entries:
        shutil.move(os.path.join(path, entry), os.path.join(destination, entry))
    return destination


def mount(request: dict[str, Any], keys: dict[str, str], owner: os.stat_result) -> None:
    path, fingerprint = str(request['path']), str(request['fingerprint'])
    remote = f'pai-{fingerprint}'
    parameters: dict[str, str] = {**request['remote'], **keys}
    rc('config/delete', {'name': remote})
    rc(
        'config/create',
        {'name': remote, 'type': parameters.pop('type'), 'parameters': parameters, 'opt': {'obscure': True}},
    )
    rc(
        'mount/mount',
        {
            'fs': f'{remote}:{request["root"]}',
            'mountPoint': path,
            'mountOpt': {'AllowOther': True, 'DeviceName': remote},
            'vfsOpt': {
                # The write cache is what makes appends, in-place edits and renames work on object storage.
                'CacheMode': 'writes',
                'DirCacheTime': DIR_CACHE_NANOSECONDS,
                'Links': True,
                'ReadOnly': bool(request['read_only']),
                'UID': owner.st_uid,
                'GID': owner.st_gid,
                'FilePerms': 0o755,
                'DirPerms': 0o755,
            },
        },
    )
    if not alive(path) or kernel_mounts().get(path) != remote:
        raise MountFailed('mount', f'{path} did not answer right after it was mounted')


def ensure(request: dict[str, Any]) -> dict[str, Any]:
    wanted: dict[str, dict[str, Any]] = {mount['path']: mount for mount in request['mounts']}
    keys: dict[str, dict[str, str]] = request['keys']
    # The request is the whole set: a mount of an earlier configuration, or at a path no longer wanted, goes.
    for path, source in kernel_mounts().items():
        if source.startswith('pai-') and (path not in wanted or source != f'pai-{wanted[path]["fingerprint"]}'):
            unmount(path)
    if not wanted:
        return {'recovered': []}
    install_fuse()
    start_daemon(install_rclone())
    # Mount folders belong to the agent's user, who owns the working directory.
    owner = os.stat(request['working_dir'])
    recovered: list[str] = []
    for path, wanted_mount in wanted.items():
        fingerprint = str(wanted_mount['fingerprint'])
        if os.path.realpath(path) != path:
            # A symlink would put the mount, and the moving aside of strays, somewhere else.
            raise MountFailed('conflict', f'{path} leads through a symlink; mount folders must be real folders')
        source = kernel_mounts().get(path)
        if source is not None and not source.startswith('pai-') and os.path.ismount(path):
            raise MountFailed('conflict', f'{path} already has something else mounted on it')
        if source == f'pai-{fingerprint}':
            fs = served().get(path)
            if fs is not None and alive(path):
                if wanted_mount['remote']['type'] == 's3':
                    # Fresh keys for the mount already there, so it outlives the ones it was mounted with. Azure
                    # mounts use an account key, which doesn't expire.
                    rc('backend/command', {'command': 'set', 'fs': fs, 'opt': keys[fingerprint]})
                continue
            unmount(path)
        make_folder(path, owner)
        moved = recover(path)
        if moved is not None:
            recovered.append(moved)
        mount(wanted_mount, keys[fingerprint], owner)
    return {'recovered': recovered}


def _out_of_time(signum: int, frame: object) -> None:
    raise MountFailed('timeout', f'mounting took longer than {DEADLINE_SECONDS} seconds')


def main() -> None:
    signal.signal(signal.SIGALRM, _out_of_time)
    signal.alarm(DEADLINE_SECONDS)
    request = json.loads(sys.stdin.readline())
    os.makedirs(STATE, mode=0o700, exist_ok=True)
    with open(f'{STATE}/lock', 'w', encoding='utf-8') as lock:
        # Two runs ensuring at once would mount twice on one path.
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            result = ensure(request)
        except MountFailed as error:
            result = {'error': error.code, 'message': str(error)}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
