import json
import os
import shlex
import sys
from pathlib import Path

import anyio
import pytest
from anyio.abc import SocketAttribute, SocketStream

from .test_tools import call

pytestmark = [pytest.mark.anyio, pytest.mark.skipif(os.name == 'nt', reason='POSIX process groups')]


class TestCoder:
    @pytest.mark.parametrize('mode', ['foreground', 'background'])
    async def test_process_survives_run(self, tmp_path: Path, mode: str) -> None:
        connected = anyio.Event()
        release = anyio.Event()
        completed = anyio.Event()
        status: Path | None = None
        listener = await anyio.create_tcp_listener(local_host='127.0.0.1')
        port = listener.extra(SocketAttribute.local_address)[1]

        async def serve(stream: SocketStream) -> None:
            async with stream:
                assert await stream.receive() == b'ready'
                connected.set()
                await release.wait()
                await stream.send(b'finish')
                assert await stream.receive() == b'done'
                completed.set()

        script = (
            f'import socket; s=socket.create_connection(("127.0.0.1", {port})); '
            's.sendall(b"ready"); s.recv(100); s.sendall(b"done"); s.close(); print("completed")'
        )
        async with listener, anyio.create_task_group() as group:
            group.start_soon(listener.serve, serve)
            output = await call(
                tmp_path,
                'shell',
                {
                    'command': f'{shlex.quote(sys.executable)} -c {shlex.quote(script)}',
                    'mode': mode,
                    'timeout': 0.01,
                },
            )
            pid = int(output.split('PID: ')[1].split()[0])
            status = Path(output.split('Status: ')[1].splitlines()[0])
            try:
                with anyio.fail_after(10):
                    await connected.wait()
                    # The Agent.run above has returned; the command still waits on our event.
                    os.kill(pid, 0)
                    state = json.loads(status.read_text())
                    assert state['exit_code'] is None
                    release.set()
                    await completed.wait()
            finally:
                release.set()
                group.cancel_scope.cancel()
        assert status is not None
        # Inspect completion using the same shell surface available to the agent.
        script = (
            'import json, pathlib, time; '
            f'p=pathlib.Path({str(status)!r}); '
            '\nwhile json.loads(p.read_text())["exit_code"] is None: time.sleep(0.01)'
            '\nprint(p.read_text())'
        )
        result = await call(tmp_path, 'shell', {'command': f'{shlex.quote(sys.executable)} -c {shlex.quote(script)}'})
        assert '"exit_code": 0' in result
        assert status.with_name('output.log').read_text() == 'completed\n'

    async def test_cancelled_foreground_terminates_process(self, tmp_path: Path) -> None:
        connected = anyio.Event()
        listener = await anyio.create_tcp_listener(local_host='127.0.0.1')
        port = listener.extra(SocketAttribute.local_address)[1]
        pid_file = tmp_path / 'pid'

        async def serve(stream: SocketStream) -> None:
            async with stream:
                assert await stream.receive() == b'ready'
                connected.set()
                await anyio.sleep_forever()

        script = (
            'import os, pathlib, socket; '
            f'pathlib.Path({str(pid_file)!r}).write_text(str(os.getpgrp())); '
            f's=socket.create_connection(("127.0.0.1", {port})); s.sendall(b"ready"); s.recv(1)'
        )

        async def run() -> None:
            await call(tmp_path, 'shell', {'command': f'{shlex.quote(sys.executable)} -c {shlex.quote(script)}'})

        async with listener, anyio.create_task_group() as group:
            group.start_soon(listener.serve, serve)
            group.start_soon(run)
            with anyio.fail_after(10):
                await connected.wait()
            group.cancel_scope.cancel()
        pid = int(pid_file.read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
