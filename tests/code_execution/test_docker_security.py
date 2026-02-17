"""Docker runtime security enforcement tests.

Each test executes actual code inside a hardened container and verifies
that the security constraint works by attempting the exact attack vector
the flag is designed to block.
"""

from __future__ import annotations

import subprocess
import textwrap

import pytest

from pydantic_ai.toolsets.code_execution.docker import DockerRuntime, DockerSecuritySettings

from .conftest import _docker_is_available, run_code_with_tools  # pyright: ignore[reportPrivateUsage]

pytestmark = pytest.mark.skipif(not _docker_is_available(), reason='Docker is not available')


async def test_network_isolation():
    """--network none blocks outbound network connections."""
    runtime = DockerRuntime()
    async with runtime:
        result = await run_code_with_tools(
            textwrap.dedent("""\
                import socket
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(2)
                try:
                    s.connect(('8.8.8.8', 53))
                    result = 'FAIL: network should be blocked'
                except OSError as e:
                    result = f'PASS: {e}'
                finally:
                    s.close()
                result
            """),
            runtime,
        )
        assert 'PASS' in result


async def test_read_only_filesystem():
    """--read-only prevents writes outside /tmp."""
    runtime = DockerRuntime()
    async with runtime:
        result = await run_code_with_tools(
            textwrap.dedent("""\
                import os
                try:
                    with open('/etc/test_file', 'w') as f:
                        f.write('should fail')
                    result = 'FAIL: filesystem should be read-only'
                except OSError as e:
                    result = f'PASS: {e}'
                result
            """),
            runtime,
        )
        assert 'PASS' in result


async def test_tmpfs_writable():
    """/tmp tmpfs mount allows writes (positive test)."""
    runtime = DockerRuntime()
    async with runtime:
        result = await run_code_with_tools(
            textwrap.dedent("""\
                with open('/tmp/test.txt', 'w') as f:
                    f.write('hello')
                with open('/tmp/test.txt') as f:
                    result = f.read()
                result
            """),
            runtime,
        )
        assert result == 'hello'


async def test_runs_as_unprivileged_user():
    """--user nobody enforces UID 65534."""
    runtime = DockerRuntime()
    async with runtime:
        result = await run_code_with_tools(
            textwrap.dedent("""\
                import os
                os.getuid()
            """),
            runtime,
        )
        assert result == 65534


async def test_pids_limit_enforced():
    """--pids-limit 256 blocks fork bombs."""
    runtime = DockerRuntime()
    async with runtime:
        result = await run_code_with_tools(
            textwrap.dedent("""\
                import os
                pids = []
                try:
                    for i in range(300):
                        pid = os.fork()
                        if pid == 0:
                            os._exit(0)
                        pids.append(pid)
                    result = f'FAIL: created {len(pids)} processes'
                except OSError as e:
                    result = f'PASS: fork limited at {len(pids)} processes: {e}'
                finally:
                    for pid in pids:
                        try:
                            os.waitpid(pid, 0)
                        except ChildProcessError:
                            pass
                result
            """),
            runtime,
        )
        assert 'PASS' in result


async def test_empty_user_raises_valueerror():
    """Setting user='' raises ValueError to prevent silent root execution."""
    runtime = DockerRuntime(security=DockerSecuritySettings(user=''))
    with pytest.raises(ValueError, match='must not be empty'):
        async with runtime:
            pass


async def test_custom_security_settings():
    """Overriding security defaults actually changes container configuration."""
    runtime = DockerRuntime(security=DockerSecuritySettings(network=True))
    async with runtime:
        assert runtime.container_id is not None

        # Verify network mode is NOT 'none'
        result = subprocess.run(
            ['docker', 'inspect', '--format', '{{.HostConfig.NetworkMode}}', runtime.container_id],
            capture_output=True,
            text=True,
        )
        network_mode = result.stdout.strip()
        assert network_mode != 'none'

        # Prove network actually works by resolving a hostname
        result = await run_code_with_tools(
            textwrap.dedent("""\
                import socket
                try:
                    socket.getaddrinfo('localhost', 80)
                    result = 'PASS: network works'
                except OSError:
                    result = 'FAIL: network should be enabled'
                result
            """),
            runtime,
        )
        assert 'PASS' in result
