"""Mounts S3 buckets with rclone inside a sandbox. Runs there as root, as `python3 -c <this file>`.

Standard library only, since it runs on the sandbox's Python. The request, keys included, is one JSON line on stdin,
so keys never reach a command line, a file or a log: they go from here to the rclone daemon over a socket only root
can open, and live in its memory. It prints one JSON line: `{"recovered": [...]}` or `{"error": ..., "message": ...}`.

Each mount's kernel mount source is `pai-<fingerprint>`, so this can tell its own mounts, and which bucket each shows,
from anything else mounted. It only changes the paths it is asked for: a mount at any other path stays as it is.
"""

from __future__ import annotations

import contextlib
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
import stat
import subprocess
import sys
import time
import urllib.request
import uuid
import zipfile
from typing import Any, Literal

STATE = '/dev/shm/pydantic-ai-mounts'
"""Daemon socket, lock and each mount's key expiry. In memory, so a restarted sandbox starts clean, as its kernel
mounts do."""
CACHE = '/var/cache/pydantic-ai-mounts'
"""rclone's write cache, on disk, so uploads still queued when the sandbox stops are sent after it restarts."""
SOCKET = f'{STATE}/rc.sock'
MARKER = '.pydantic-ai-mount'
"""Written into a folder before mounting on it: the mount hides it, so seeing it means the mount was lost."""
MOUNTINFO = '/proc/self/mountinfo'
ROOT_PATH = '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
"""Where root finds `rclone`, `fusermount3` and `apt-get`: folders root owns, never one the image's `PATH` adds ahead of
them, which the agent's user could write to."""
RCLONE = '/usr/local/lib/pydantic-ai-mounts/rclone'
"""Where the pinned build goes: off `PATH`, so it never replaces or shadows an rclone the image has."""
RCLONE_VERSION = 'v1.75.2'
RCLONE_MIN = (1, 75, 2)
"""The oldest rclone used from `PATH`. This uses the `backend/command` `set`, `mount/listmounts` and `vfs/stats`
remote control commands, verified on the pinned build only, so that is the floor."""
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
"""How long a mount may take to answer a lookup before it counts as dead, and `rclone version` to answer."""
DEADLINE_SECONDS = 1800
"""The most one call may take, all steps together, so a caller that waits for it is never left waiting for ever."""


RCLONE_VERSION_OUTPUT = re.compile(r'rclone v(\d+)\.(\d+)\.(\d+)')
KERNEL_ESCAPE = re.compile(rb'\\([0-7]{3})')
"""How the kernel writes a space, tab, newline or backslash in a mount table path: as three octal digits."""


FailureCode = Literal['busy', 'conflict', 'fuse', 'mount', 'rclone', 'stuck', 'timeout']
"""Why a call failed, as the reply's `error` names it."""


class MountFailed(Exception):
    def __init__(self, code: FailureCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class _UnixConnection(http.client.HTTPConnection):
    def connect(self) -> None:  # pragma: no cover
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
        raise MountFailed(
            'rclone', f'rclone {command} failed: {reply.get("error", response.status)}. See {STATE}/rcd.log.'
        )
    return reply


def new_enough(rclone: str) -> bool:
    """Whether `rclone` is at least `RCLONE_MIN`."""
    try:
        result = subprocess.run([rclone, 'version'], capture_output=True, text=True, timeout=PROBE_SECONDS)
    except (OSError, subprocess.TimeoutExpired):
        return False
    found = RCLONE_VERSION_OUTPUT.match(result.stdout)
    return found is not None and tuple(int(part) for part in found.groups()) >= RCLONE_MIN


def install_rclone() -> str:
    """The image's rclone if it is new enough, else the pinned build, downloaded on first use."""
    found = shutil.which('rclone')
    if found is not None and new_enough(found):
        return found
    if os.access(RCLONE, os.X_OK):
        return RCLONE
    arch = platform.machine()
    if arch not in RCLONE_SHA256:
        raise MountFailed('rclone', f'There is no rclone build for {arch}. Install rclone in the sandbox image.')
    name = f'rclone-{RCLONE_VERSION}-linux-{"amd64" if arch == "x86_64" else "arm64"}'
    try:
        # The download site refuses Python's default User-Agent.
        url = f'https://downloads.rclone.org/{RCLONE_VERSION}/{name}.zip'
        request = urllib.request.Request(url, headers={'User-Agent': 'pydantic-ai-harness'})
        with urllib.request.urlopen(request, timeout=DOWNLOAD_SECONDS) as reply:
            archive = reply.read()
    except (OSError, http.client.HTTPException) as error:
        raise MountFailed(
            'rclone', f'Could not download rclone ({error}). Install rclone in the sandbox image.'
        ) from None
    if hashlib.sha256(archive).hexdigest() != RCLONE_SHA256[arch]:
        raise MountFailed(
            'rclone',
            'The downloaded rclone does not match its published checksum. Install rclone in the sandbox image.',
        )
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        binary = zipped.read(f'{name}/rclone')
    os.makedirs(os.path.dirname(RCLONE), exist_ok=True)
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
        raise MountFailed('fuse', 'The sandbox has no `fusermount3`. Install fuse3 in its image.')


def start_daemon(rclone: str) -> None:
    try:
        rc('rc/noop', timeout=5)
        return
    except (OSError, MountFailed):
        pass
    # Mounts of a daemon that died are dead too; `place` replaces each one it is asked for.
    with contextlib.suppress(FileNotFoundError):
        os.unlink(SOCKET)
    os.makedirs(CACHE, mode=0o700, exist_ok=True)
    # rclone reads any `RCLONE_*` variable as a flag: one like `RCLONE_LOG_LEVEL=DEBUG` would log the keys.
    env = {name: value for name, value in os.environ.items() if not name.startswith('RCLONE_')}
    with open(f'{STATE}/rcd.log', 'ab') as log:
        subprocess.Popen(
            [rclone, 'rcd', '--rc-addr', f'unix://{SOCKET}', '--rc-no-auth', '--config', '', '--cache-dir', CACHE],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=log,
            env=env,
            start_new_session=True,
        )
    deadline = time.monotonic() + DAEMON_START_SECONDS
    while time.monotonic() < deadline:
        try:
            rc('rc/noop', timeout=5)
            return
        except (OSError, MountFailed):
            time.sleep(0.05)
    raise MountFailed('rclone', f'The rclone daemon did not start. See {STATE}/rcd.log in the sandbox.')


def kernel_mounts() -> dict[str, str]:
    """Mount point to mount source, for every mount in the kernel's table (the last one wins on a stacked path)."""
    mounts: dict[str, str] = {}
    with open(MOUNTINFO, 'rb') as table:
        for line in table:
            fields, _, rest = line.partition(b' - ')
            mounts[_unescape(fields.split()[4])] = _unescape(rest.split()[1])
    return mounts


def _unescape(field: bytes) -> str:
    """Undo the kernel's escapes: it writes a space, tab, newline or backslash in a path as three octal digits."""
    return KERNEL_ESCAPE.sub(lambda match: bytes([int(match[1], 8)]), field).decode(errors='surrogateescape')


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
            raise MountFailed('busy', f'{path} still has files uploading. Try again once they are sent.')
        rc('mount/unmount', {'mountPoint': path})
    else:
        # The daemon that served it is gone, so nothing is left to send.
        subprocess.run(['fusermount3', '-uz', path], capture_output=True)
        subprocess.run(['umount', '-l', path], capture_output=True)
    if kernel_mounts().get(path, '').startswith('pai-'):
        raise MountFailed('stuck', f'{path} could not be unmounted. Restart the sandbox to clear it.')
    # Unmounted on purpose, so the marker no longer means a lost mount.
    with contextlib.suppress(FileNotFoundError):
        os.remove(os.path.join(path, MARKER))


def forget_lost() -> None:
    """Drop the daemon's record of mounts the kernel no longer has, as after `fusermount3 -u`.

    rclone refuses to mount at a path it still lists, so without this such a folder could not be mounted again.
    """
    kernel = kernel_mounts()
    for path in served():
        if path not in kernel:
            rc('mount/unmount', {'mountPoint': path})


def make_folder(path: str, owner: os.stat_result) -> None:
    """Create `path` and its missing parents, owned by the agent's user."""
    if os.path.lexists(path) and not os.path.isdir(path):
        raise MountFailed('conflict', f'{path} is a file. Mount at a folder.')
    missing: list[str] = []
    while not os.path.isdir(path):
        missing.append(path)
        path = os.path.dirname(path)
    for folder in reversed(missing):
        os.mkdir(folder)
        os.chown(folder, owner.st_uid, owner.st_gid, follow_symlinks=False)


def prepare(path: str, owner: os.stat_result) -> tuple[bool, str | None]:
    """Make the bare folder ready to mount on.

    Returns whether the marker was written now, and where files written there while it wasn't mounted went. Only a
    folder this mounted on before (it holds the marker) has such files; a folder with other files is refused, rather
    than its files being moved away.
    """
    entries = os.listdir(path)
    if MARKER not in entries:
        if entries:
            raise MountFailed('conflict', f'{path} already holds files. Mount at a new or empty folder.')
        # The agent's user owns the folder: never follow a link it put there since the listing.
        try:
            os.close(os.open(os.path.join(path, MARKER), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644))
        except FileExistsError:
            raise MountFailed('conflict', f'{path} already holds files. Mount at a new or empty folder.') from None
        return True, None
    strays = [entry for entry in entries if entry != MARKER]
    if not strays:
        return False, None
    parent, name = os.path.split(path)
    recovered = os.path.join(parent, RECOVERED)
    staged = f'{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}-{uuid.uuid4().hex[:6]}-{name}'
    # The agent's user owns these folders and could swap one for a link while root works in it, so every move goes
    # through descriptors opened without following links, into a folder only root can change until the moves are done.
    folder = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    with contextlib.suppress(FileExistsError):
        os.mkdir(recovered, 0o755)
    try:
        recovered_fd = os.open(recovered, folder)
    except OSError:
        raise MountFailed(
            'conflict', f'{recovered} is not a folder. Remove it so files found in a lost mount can go there.'
        ) from None
    try:
        os.mkdir(staged, 0o700, dir_fd=recovered_fd)
        staged_fd = os.open(staged, folder, dir_fd=recovered_fd)
        try:
            source_fd = os.open(path, folder)
            try:
                for entry in strays:
                    os.rename(entry, entry, src_dir_fd=source_fd, dst_dir_fd=staged_fd)
            finally:
                os.close(source_fd)
            # The agent's user can then move or remove what was found.
            os.fchmod(staged_fd, 0o755)
            os.fchown(staged_fd, owner.st_uid, owner.st_gid)
        finally:
            os.close(staged_fd)
        os.fchown(recovered_fd, owner.st_uid, owner.st_gid)
    finally:
        os.close(recovered_fd)
    return False, os.path.join(recovered, staged)


def record_expiry(fingerprint: str, expires_at: float) -> None:
    """Remember when the keys a mount last received expire, so another bucket may take its folder after that."""
    with open(f'{STATE}/{fingerprint}.expires', 'w', encoding='utf-8') as file:
        file.write(str(expires_at))


def expired(fingerprint: str) -> bool:
    """Whether the keys the mount of `fingerprint` last received have expired, by this sandbox's clock."""
    try:
        with open(f'{STATE}/{fingerprint}.expires', encoding='utf-8') as file:
            return float(file.read()) <= time.time()
    except (OSError, ValueError):
        return True


def mount(wanted: dict[str, Any], owner: os.stat_result) -> None:
    path, fingerprint = str(wanted['path']), str(wanted['fingerprint'])
    remote = f'pai-{fingerprint}'
    parameters: dict[str, str] = {**wanted['remote'], **wanted['keys']}
    rc('config/delete', {'name': remote})
    rc('config/create', {'name': remote, 'type': parameters.pop('type'), 'parameters': parameters})
    rc(
        'mount/mount',
        {
            'fs': f'{remote}:{wanted["root"]}',
            'mountPoint': path,
            # The folder holds only the marker, which `prepare` checked; the mount hides it.
            'mountOpt': {'AllowOther': True, 'AllowNonEmpty': True, 'DeviceName': remote},
            'vfsOpt': {
                # The write cache is what makes appends, in-place edits and renames work on object storage.
                'CacheMode': 'writes',
                'DirCacheTime': DIR_CACHE_NANOSECONDS,
                'Links': True,
                'ReadOnly': bool(wanted['read_only']),
                'UID': owner.st_uid,
                'GID': owner.st_gid,
                'FilePerms': 0o755,
                'DirPerms': 0o755,
            },
        },
    )
    if not alive(path) or kernel_mounts().get(path) != remote:
        raise MountFailed(
            'mount', f'{path} did not answer right after it was mounted. Check the bucket and its keys, and try again.'
        )
    record_expiry(fingerprint, float(wanted['expires_at']))


def refresh_or_clear(wanted: dict[str, Any], source: str) -> bool:
    """Give the mount already at `wanted`'s path fresh keys and return True, or unmount it and return False."""
    path, fingerprint = str(wanted['path']), str(wanted['fingerprint'])
    if not source.startswith('pai-'):
        raise MountFailed('conflict', f'{path} already has something else mounted on it. Mount at another folder.')
    fs = served().get(path)
    if fs is not None and source == f'pai-{fingerprint}':
        # The same bucket: only the keys change, and commands using the mount carry on. The keys go first, since
        # a mount whose keys expired can't answer the lookup that tells whether it is alive.
        rc('backend/command', {'command': 'set', 'fs': fs, 'opt': wanted['keys']})
        if alive(path):
            record_expiry(fingerprint, float(wanted['expires_at']))
            return True
    elif fs is not None and alive(path) and not expired(source.removeprefix('pai-')):
        raise MountFailed(
            'conflict',
            f'{path} already shows another bucket in this sandbox. Mount at another folder, or destroy the sandbox.',
        )
    unmount(path)
    return False


def place(wanted: dict[str, Any], owner: os.stat_result) -> str | None:
    """Mount `wanted` at its path, or give the mount already there fresh keys; returns where stray files went."""
    path, fingerprint = str(wanted['path']), str(wanted['fingerprint'])
    if os.path.realpath(path) != path:
        # A symlink would put the mount, and the moving aside of strays, somewhere else.
        raise MountFailed('conflict', f'{path} leads through a symlink. Mount at a real folder.')
    source = kernel_mounts().get(path)
    if source is not None and refresh_or_clear(wanted, source):
        return None
    make_folder(path, owner)
    marked, moved = prepare(path, owner)
    try:
        mount(wanted, owner)
    except MountFailed:
        if kernel_mounts().get(path) == f'pai-{fingerprint}':
            # Mounted but not answering. The marker is behind the mount, where removing it would delete a bucket object,
            # so it goes once the mount is off, as `unmount` does.
            unmount(path)
        elif marked:
            # Nothing was mounted, so the marker would later read as a lost mount.
            os.remove(os.path.join(path, MARKER))
        raise
    return moved


def ensure(request: dict[str, Any]) -> dict[str, Any]:
    install_fuse()
    start_daemon(install_rclone())
    forget_lost()
    # Mount folders belong to the agent's user, who owns the working directory.
    owner = os.stat(request['working_dir'])
    return {'recovered': [moved for wanted in request['mounts'] if (moved := place(wanted, owner))]}


def out_of_time(signum: int, frame: object) -> None:
    raise MountFailed('timeout', f'Mounting took longer than {DEADLINE_SECONDS} seconds. Try again.')


def main() -> None:
    os.environ['PATH'] = ROOT_PATH
    signal.signal(signal.SIGALRM, out_of_time)
    signal.alarm(DEADLINE_SECONDS)
    request = json.loads(sys.stdin.readline())
    try:
        result = locked(request)
    except MountFailed as error:
        result = {'error': error.code, 'message': str(error)}
    print(json.dumps(result))


def locked(request: dict[str, Any]) -> dict[str, Any]:
    """`ensure`, one call at a time: two at once would mount twice on one path."""
    private_state()
    with os.fdopen(os.open(f'{STATE}/lock', os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600), 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return ensure(request)


def private_state() -> None:
    """Make `STATE` a folder only this user (root) can open, refusing one someone else made.

    Its parent is open to every user, so another could make it first, with links in it to files root would then write.
    """
    os.makedirs(STATE, mode=0o700, exist_ok=True)
    info = os.lstat(STATE)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
        raise MountFailed('conflict', f'{STATE} was not made by root. Remove it, or restart the sandbox.')
    os.chmod(STATE, 0o700)


if __name__ == '__main__':
    main()
