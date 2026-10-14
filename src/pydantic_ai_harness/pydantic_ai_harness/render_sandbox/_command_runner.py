"""Stdlib-only supervisor sent through the public Render exec API.

Render keeps an exec alive until its inherited output pipes close. User processes
therefore write to private pipes; this supervisor frames their bytes as base64 and
can finish after the direct child exits even when a background job retains a pipe.
The claim directory and flock also let a separate public exec stop an invocation
whose start response was lost. This is process management, not a trust boundary
between commands running as the same sandbox user.
"""

from __future__ import annotations

import base64
import fcntl
import json
import os
import selectors
import signal
import subprocess
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, TypeGuard

_CHUNK_SIZE = 16_384


def _json_object(value: object) -> TypeGuard[dict[str, object]]:
    # Callers obtain objects from json.loads, which produces string dictionary keys.
    return isinstance(value, dict)


def _mapping(value: object) -> dict[str, object]:
    if not _json_object(value):  # pragma: no cover - only host-generated JSON objects and owned state reach here
        raise TypeError('Expected an object')
    # These objects come from JSON decoding, which already guarantees text keys.
    return value


def _text(value: object) -> str:
    if not isinstance(value, str):  # pragma: no cover - the host validates all text fields before dispatch
        raise TypeError('Expected text')
    return value


def _frame(kind: str, nonce: str, **fields: object) -> None:
    # Only the supervisor owns these pipes. User bytes cannot forge control frames.
    print(json.dumps({'kind': kind, 'nonce': nonce, **fields}, separators=(',', ':')), flush=True)


def _identity(pid: int) -> str | None:
    """Linux start ticks are stable even while the unreaped leader is a zombie."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text(encoding='utf-8').rsplit(')', 1)[1].split()
        return fields[19]
    except (FileNotFoundError, IndexError):  # pragma: no cover - local macOS fallback; Render uses Linux /proc
        # A local POSIX test can run on macOS. Render uses the Linux branch above.
        result = subprocess.run(
            ['ps', '-o', 'lstart=', '-p', str(pid)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            env={**os.environ, 'LC_ALL': 'C'},
        )
        return result.stdout.decode().strip() or None


@contextmanager
def _locked(claim: Path) -> Generator[None]:
    with (claim / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _state(claim: Path, **fields: object) -> None:
    (claim / 'state.json').write_text(json.dumps(fields))


def _signal_owned(claim: Path) -> None:
    """Call under the registration/reaping lock, preventing a PID reuse race."""
    try:
        record = _mapping(json.loads((claim / 'state.json').read_text(encoding='utf-8')))
    except FileNotFoundError:
        return  # The stop fence prevents a not-yet-registered launcher from starting.
    if record.get('done') is True:
        return  # Direct command already exited. Its background jobs may continue.
    pid, birth = record.get('pid'), record.get('birth')
    if not isinstance(pid, int) or not isinstance(birth, str):  # pragma: no cover - private registration corruption
        raise TypeError('Invalid process registration')
    current = _identity(pid)
    if current is None:  # pragma: no cover - the registered child is still owned and unreaped
        return
    if current != birth:  # pragma: no cover - ownership lock prevents PID reuse until state is marked done
        raise RuntimeError('Process identity no longer matches; no signal was sent')
    try:
        if os.getpgid(pid) != pid:  # pragma: no cover - private registration always records the new group leader
            raise RuntimeError('Registered process is not its own group leader')
        # The supervisor cannot reap this leader while we hold the lock. Even if
        # it exits between the identity check and killpg, its PID cannot be reused.
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _stop(config: dict[str, object]) -> None:
    claim = Path(_text(config['claim']))
    nonce = _text(config['nonce'])
    try:
        claim.mkdir(mode=0o700)
    except FileExistsError:
        with _locked(claim):
            (claim / 'stopped').touch()
            _signal_owned(claim)
    # If stop won mkdir, leave the directory. A delayed launcher must never run.
    _frame('stopped', nonce)


def _json_array(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _spawn(config: dict[str, object]) -> subprocess.Popen[bytes]:
    values = config['argv']
    if not _json_array(values):  # pragma: no cover - the host always supplies its validated argument list
        raise TypeError('argv must be a list')
    arguments: list[object] = values
    argv = [_text(value) for value in arguments]
    supplied = _mapping(config['env'])
    env = {key: _text(value) for key, value in supplied.items()}
    return subprocess.Popen(
        argv,
        cwd=_text(config['cwd']),
        env={**os.environ, **env},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


def _run(config: dict[str, object]) -> None:
    claim = Path(_text(config['claim']))
    nonce = _text(config['nonce'])
    limit = config['limit']
    if not isinstance(limit, int) or limit < 1:  # pragma: no cover - public config rejects invalid limits
        raise ValueError('Invalid output limit')
    timeout = config.get('timeout')
    if timeout is not None and not isinstance(timeout, (float, int)):  # pragma: no cover - host validated timeout
        raise ValueError('Invalid timeout')
    try:
        claim.mkdir(mode=0o700)
    except FileExistsError:
        _frame('fenced', nonce)
        return
    process: subprocess.Popen[bytes] | None = None
    try:
        # Registration and stop are mutually exclusive. A stop that acquires this
        # lock before registration fences the launch, including a lost start ACK.
        with _locked(claim):
            if (claim / 'stopped').exists():  # pragma: lax no cover - the stop fence can win mkdir or registration
                _frame('fenced', nonce)
                return
            try:
                process = _spawn(config)
            except FileNotFoundError:
                _state(claim, done=True)
                _frame('exit', nonce, code=127)
                return
            except PermissionError:
                _state(claim, done=True)
                _frame('exit', nonce, code=126)
                return
            birth = _identity(process.pid)
            if birth is None:  # pragma: no cover - the new Linux child cannot be reaped before registration
                raise RuntimeError('Cannot record process identity')
            _state(claim, pid=process.pid, birth=birth, done=False)
        _frame('started', nonce)
        assert process.stdout is not None and process.stderr is not None
        _capture(process, claim, nonce, limit, timeout)
    finally:
        if process is not None:
            with _locked(claim):
                if process.poll() is None:  # pragma: no cover - fallback for a private supervisor machinery failure
                    # This Popen is still our unreaped child, including if
                    # registration failed. Never rely on a missing state file.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=2)
                _state(claim, done=True)
        # A consumed claim cannot be used by a late launcher: each UUID is invoked
        # once. Keep the small state/tombstone until sandbox deletion so late stop
        # calls remain idempotent and cannot recreate an ambiguous registration.


def _capture(
    process: subprocess.Popen[bytes], claim: Path, nonce: str, limit: int, timeout: float | int | None
) -> None:
    assert process.stdout is not None and process.stderr is not None
    pipes: dict[int, tuple[str, IO[bytes]]] = {
        process.stdout.fileno(): ('stdout', process.stdout),
        process.stderr.fileno(): ('stderr', process.stderr),
    }
    started = time.monotonic()
    direct_exit: int | None = None
    terminal: str | None = None
    output_bytes = 0
    with selectors.DefaultSelector() as selector:
        for fd in pipes:
            os.set_blocking(fd, False)
            selector.register(fd, selectors.EVENT_READ)
        while True:
            now = time.monotonic()
            with _locked(claim):
                # poll() reaps. The state changes in the same critical section,
                # so the stop process never signals a reused PID after reaping.
                direct_exit = process.poll()
                if direct_exit is not None:
                    _state(claim, done=True)
                elif terminal is None and timeout is not None and now - started >= timeout:
                    terminal = 'timeout'
                    _signal_owned(claim)
            # Once the direct child has exited, all of its bytes are already
            # in these pipes. Drain ready bytes through EAGAIN, not a clock
            # deadline: slow transport must never cause truncated success.
            ready = selector.select(0 if direct_exit is not None else 0.02)
            if direct_exit is not None and (not ready or terminal == 'limit'):
                break
            for key, _ in ready:
                fd = key.fd
                stream_name, pipe = pipes[fd]
                try:
                    data = os.read(fd, _CHUNK_SIZE)
                except BlockingIOError:  # pragma: lax no cover - nonblocking readiness can race the kernel
                    continue
                if not data:
                    selector.unregister(fd)
                    pipe.close()
                    continue
                remaining = max(0, limit - output_bytes)
                captured = data[:remaining]
                output_bytes += len(data)
                if captured:  # pragma: no branch - pipe draining can finish before another post-limit read
                    _frame(
                        'output',
                        nonce,
                        stream=stream_name,
                        data=base64.b64encode(captured).decode('ascii'),
                    )
                if output_bytes > limit and terminal is None:
                    terminal = 'limit'
                    with _locked(claim):
                        _signal_owned(claim)
        # Closing our pipe ends, instead of waiting for EOF, is what permits a
        # normal direct exit while a background child still holds its ends.
        for _, pipe in pipes.values():
            pipe.close()
    _frame(terminal or 'exit', nonce, code=direct_exit)


def main() -> None:
    config = _mapping(json.loads(sys.argv[2]))
    nonce = _text(config['nonce'])
    try:
        if sys.argv[1] == 'stop':
            _stop(config)
        elif sys.argv[1] == 'run':
            _run(config)
        else:
            raise ValueError(
                'Invalid supervisor operation'
            )  # pragma: no cover - private dispatch uses only run and stop
    except Exception as error:
        # Never echo argv, env, or file contents into control-plane diagnostics.
        _frame('error', nonce, category=type(error).__name__)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
