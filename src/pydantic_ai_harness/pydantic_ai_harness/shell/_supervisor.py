"""Detached shell supervisor; owns exit-status publication, not agent scheduling."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path


def main() -> None:  # pragma: no cover
    """Run in a separate interpreter so commands survive run and loop teardown."""
    directory = Path(sys.argv[1])
    command = sys.argv[2]
    status = directory / 'status.json'
    pending = directory / 'status.tmp'

    def publish(*, pid: int, exit_code: int | None) -> None:
        pending.write_text(json.dumps({'pid': pid, 'exit_code': exit_code}), encoding='utf-8')
        pending.replace(status)

    with (directory / 'output.log').open('ab', buffering=0) as output:
        process = subprocess.Popen(command, shell=True, stdin=subprocess.DEVNULL, stdout=output, stderr=output)
        publish(pid=process.pid, exit_code=None)
        exit_code = process.wait()
        publish(pid=process.pid, exit_code=exit_code)


if __name__ == '__main__':  # pragma: no cover
    # The caller starts a new session. Ignore terminal hangups without changing
    # the child command's termination handling.
    if os.name != 'nt':
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    main()
