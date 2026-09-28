"""Child-only file limits, installed before executing the user's shell."""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path


def validate_file_limit(limit: object, *, persistent: bool) -> None:
    if limit is None:
        return
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError('max_file_bytes must be a positive integer.')
    if os.name != 'posix':
        raise ValueError('max_file_bytes is unavailable on this platform; POSIX RLIMIT_FSIZE is required.')
    import resource

    if not hasattr(resource, 'RLIMIT_FSIZE'):
        raise ValueError('max_file_bytes is unavailable on this platform; RLIMIT_FSIZE is required.')
    if persistent:
        raise ValueError('max_file_bytes is not supported with the persistent shell tool.')
    if limit > sys.maxsize:
        raise ValueError('max_file_bytes must not exceed sys.maxsize.')


def limited_command(command: str, limit: int | None) -> str | list[str]:
    if limit is None:
        return command
    return [sys.executable, '-I', str(Path(__file__).resolve()), str(limit), command]


def file_limit_status(exit_code: int, limit: int | None) -> str:
    if limit is None or exit_code == 0:
        return ''
    if exit_code in (-signal.SIGXFSZ, 128 + signal.SIGXFSZ):
        return f'\n[File-size limit exceeded: max_file_bytes={limit}. Reduce file output before retrying.]'
    return f'\n[Command failed with max_file_bytes={limit}; file writes may have reached the per-file limit.]'


def main() -> None:  # pragma: no cover
    """Run in a fresh interpreter so the parent and supervisor remain unaffected."""
    import resource

    limit = int(sys.argv[1])
    _, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))
        signal.signal(signal.SIGXFSZ, signal.SIG_DFL)
        os.execv('/bin/sh', ['/bin/sh', '-c', sys.argv[2]])
    except (OSError, ValueError) as exc:
        print(f'Unable to apply max_file_bytes or execute shell: {exc}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':  # pragma: no cover
    main()
