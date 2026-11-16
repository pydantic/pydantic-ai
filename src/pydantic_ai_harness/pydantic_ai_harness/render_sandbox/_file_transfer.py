"""Small stdlib helper executed inside the sandbox, never imported by the host.

Only file preparation and atomic commit run here. Contents travel through Render's
public file API. A private snapshot bounds reads even if the source keeps growing.
"""

import errno
import json
import os
import shutil
import stat
import sys


def regular_file(path: str) -> int:
    # O_NONBLOCK prevents a path swapped for a FIFO from hanging before fstat.
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    mode = os.fstat(fd).st_mode
    if not stat.S_ISREG(mode):
        os.close(fd)
        code = errno.EISDIR if stat.S_ISDIR(mode) else errno.EINVAL
        raise OSError(code, 'Expected a regular file', path)
    return fd


def snapshot(path: str, staged: str) -> int:
    with os.fdopen(regular_file(path), 'rb') as source:
        remaining = os.fstat(source.fileno()).st_size
        size = remaining
        with open(staged, 'xb') as target:
            while remaining:
                chunk = source.read(min(remaining, 1024 * 1024))
                if not chunk:
                    raise OSError(errno.EIO, 'File shrank during read', path)
                target.write(chunk)
                remaining -= len(chunk)
    return size


def destination(path: str) -> str:
    # strict=False retains missing components and follows a dangling leaf link.
    target = os.path.realpath(path)
    try:
        info = os.stat(target)
    except FileNotFoundError:
        pass
    else:
        if stat.S_ISDIR(info.st_mode):
            raise IsADirectoryError(errno.EISDIR, 'Destination is a directory', path)
        if not stat.S_ISREG(info.st_mode):
            raise OSError(errno.EINVAL, 'Expected a regular file', path)
        if not os.access(target, os.W_OK):
            raise PermissionError(errno.EACCES, 'Destination is not writable', path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    return target


def umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


def sibling(path: str, directory: str) -> str:
    # Named after the staging directory, so `cleanup` can remove it even if a stop killed `commit`.
    return os.path.join(os.path.dirname(os.path.realpath(path)), '.render-pydantic-' + os.path.basename(directory))


def commit(path: str, directory: str, size: int) -> None:
    target = destination(path)
    staged = os.path.join(directory, 'data')
    # Stage beside the destination so replacement is atomic across mount points.
    # A failed upload or interrupted copy cannot truncate the user's old file.
    temporary = sibling(path, directory)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'wb') as output, os.fdopen(regular_file(staged), 'rb') as source:
            if os.fstat(source.fileno()).st_size != size:
                raise OSError(errno.EIO, 'Incomplete uploaded file', path)
            shutil.copyfileobj(source, output, 1024 * 1024)
        # Private while it fills; then the old file's mode, or the umask default for a new file.
        os.chmod(temporary, stat.S_IMODE(os.stat(target).st_mode) if os.path.exists(target) else 0o666 & ~umask())
        os.replace(temporary, target)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)


def main() -> None:
    action, path, directory = sys.argv[1:4]
    staged = os.path.join(directory, 'data')
    if action == 'cleanup':
        shutil.rmtree(directory, ignore_errors=True)
        try:
            os.unlink(sibling(path, directory))
        except OSError:
            pass
        return
    try:
        if action == 'read':
            os.mkdir(directory, 0o700)
            result = snapshot(path, staged)
        elif action == 'prepare':
            destination(path)
            os.mkdir(directory, 0o700)
            result = 0
        elif action == 'commit':
            commit(path, directory, int(sys.argv[4]))
            result = 0
        else:
            raise ValueError('Unknown file operation')  # pragma: no cover - private dispatch selects a fixed operation
        print(json.dumps({'size': result}))
    except OSError as error:
        print(json.dumps({'errno': error.errno, 'message': error.strerror}))
        sys.exit(1)


if __name__ == '__main__':
    main()
