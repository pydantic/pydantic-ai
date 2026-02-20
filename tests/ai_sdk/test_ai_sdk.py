"""Pytest orchestration for AI SDK <-> Pydantic AI integration tests.

Starts a real HTTP server per test, runs TypeScript tests against it, and fails
if the tests fail. Requires node >= 22.6 (built-in TypeScript strip).
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

SDK_DIR = Path(__file__).parent
REPO_ROOT = SDK_DIR.parents[1]
SERVER_MODULE = 'tests.ai_sdk.server'
STARTUP_TIMEOUT = 10.0
STARTUP_POLL = 0.25

pytestmark = pytest.mark.skipif(not shutil.which('node'), reason='node not installed')


@pytest.fixture(scope='module')
def _npm_install() -> None:
    if not (SDK_DIR / 'node_modules').is_dir():
        subprocess.run(['npm', 'install'], cwd=SDK_DIR, check=True, capture_output=True)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def _wait_for_server(port: int, timeout: float = STARTUP_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=0.5):
                return
        except OSError:
            time.sleep(STARTUP_POLL)
    raise TimeoutError(f'Server on port {port} did not start within {timeout}s')


@pytest.fixture
def server_url(request: pytest.FixtureRequest) -> Iterator[str]:
    agent_name: str = request.param
    port = _free_port()
    log = tempfile.TemporaryFile()
    proc = subprocess.Popen(
        [sys.executable, '-m', SERVER_MODULE, agent_name, str(port)],
        cwd=REPO_ROOT,
        stdout=log,
        stderr=log,
    )
    try:
        _wait_for_server(port)
        yield f'http://127.0.0.1:{port}'
    finally:
        proc.terminate()
        proc.wait(timeout=5)
        log.seek(0)
        output = log.read().decode(errors='replace')
        log.close()
        if output:
            print(f'\n--- server log ---\n{output}--- end server log ---')


TEST_FILES = sorted(SDK_DIR.glob('test_*.ts'))


def test_agents_match_test_files() -> None:
    from tests.ai_sdk.server import AGENTS

    agent_names = set(AGENTS.keys())
    test_names = {f.stem.removeprefix('test_') for f in TEST_FILES}
    assert agent_names == test_names


@pytest.mark.parametrize(
    ('test_file', 'server_url'),
    [(f, f.stem.removeprefix('test_')) for f in TEST_FILES],
    ids=[f.name for f in TEST_FILES],
    indirect=['server_url'],
)
def test_ai_sdk(_npm_install: None, server_url: str, test_file: Path) -> None:
    result = subprocess.run(
        ['node', '--test', str(test_file)],
        env={**os.environ, 'SERVER_URL': server_url},
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        pytest.fail(
            f'node --test {test_file.name} exited {result.returncode}\n\nstdout:\n{result.stdout}\n\nstderr:\n{result.stderr}'
        )
