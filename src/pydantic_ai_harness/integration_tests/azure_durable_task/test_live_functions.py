"""Opt-in real Functions host + isolated Azurite non-HTTP orchestration verification.

Set AZURE_FUNCTIONS_REQUIRE_LIVE=1 and run this module with Python 3.13+ and the
preview extra installed. Requires Core Tools and Azurite on PATH. Processes,
ports, task hub and storage belong to this fixture, not shared developer services.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from uuid import uuid4

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get('AZURE_FUNCTIONS_REQUIRE_LIVE') != '1', reason='Opt in with AZURE_FUNCTIONS_REQUIRE_LIVE=1.'
)
_TIMEOUT = 90.0


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def _command(executable: str) -> list[str]:
    if os.name == 'nt':
        if Path(executable).stem.lower() == 'func':
            direct = Path(executable).parent / 'node_modules' / 'azure-functions-core-tools' / 'bin' / 'func.exe'
            if direct.is_file():
                return [str(direct)]
        if executable.lower().endswith(('.cmd', '.bat')):
            return [os.environ['COMSPEC'], '/c', executable]
    return [executable]


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == 'nt':
        # Core Tools owns a .NET host and Python worker; terminate only this fixture's tree.
        script = (
            '$ErrorActionPreference="Stop"; function Stop-Tree([int]$parentId) { '
            'Get-CimInstance Win32_Process -Filter "ParentProcessId=$parentId" | '
            'ForEach-Object { Stop-Tree $_.ProcessId }; '
            'try { Stop-Process -Id $parentId -Force -ErrorAction Stop } catch { '
            'if ($_.FullyQualifiedErrorId -notlike "NoProcessFoundForGivenId,*") { throw } } }; '
            f'Stop-Tree {process.pid}'
        )
        subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', script], check=True, timeout=20)
    else:
        os.killpg(process.pid, signal.SIGTERM)
    process.wait(timeout=20)


@contextmanager
def _process(command: list[str], *, cwd: Path, env: dict[str, str], log: Path) -> Generator[subprocess.Popen[bytes]]:
    with log.open('wb') as output:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=os.name != 'nt',
        )
        try:
            yield process
        finally:
            _stop(process)


@pytest.fixture
def functions_host(tmp_path: Path) -> Generator[Path]:
    """Own storage, the real host, their process trees and per-test configuration."""
    assert sys.version_info >= (3, 13), 'Functions preview requires Python 3.13+.'
    func = shutil.which('func')
    azurite = shutil.which('azurite')
    assert func is not None, 'Install Azure Functions Core Tools.'
    assert azurite is not None, 'Install Azurite.'
    app_dir = tmp_path / 'app'
    sample = Path(__file__).parents[2] / 'examples' / 'azure_functions'
    app_dir.mkdir()
    shutil.copyfile(sample / 'function_app.py', app_dir / 'function_app.py')
    config = json.loads((sample / 'host.json').read_text())
    config['extensions']['durableTask']['hubName'] = f'Pydantic{uuid4().hex}'
    (app_dir / 'host.json').write_text(json.dumps(config))
    blob_port, queue_port, table_port, http_port = _port(), _port(), _port(), _port()
    env = os.environ.copy()
    env['VIRTUAL_ENV'] = sys.prefix
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    env['FUNCTIONS_WORKER_RUNTIME'] = 'python'
    env['AzureWebJobsFeatureFlags'] = 'EnableWorkerIndexing'
    env['AzureWebJobsStorage'] = (
        'DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;'
        'AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;'
        f'BlobEndpoint=http://127.0.0.1:{blob_port}/devstoreaccount1;'
        f'QueueEndpoint=http://127.0.0.1:{queue_port}/devstoreaccount1;'
        f'TableEndpoint=http://127.0.0.1:{table_port}/devstoreaccount1;'
    )
    storage_log, host_log = tmp_path / 'azurite.log', tmp_path / 'functions.log'
    with ExitStack() as stack:
        storage = stack.enter_context(
            _process(
                [
                    *_command(azurite),
                    '--silent',
                    '--skipApiVersionCheck',
                    '--location',
                    str(tmp_path / 'storage'),
                    '--blobPort',
                    str(blob_port),
                    '--queuePort',
                    str(queue_port),
                    '--tablePort',
                    str(table_port),
                ],
                cwd=tmp_path,
                env=env,
                log=storage_log,
            )
        )
        deadline = time.monotonic() + _TIMEOUT
        while True:
            assert storage.poll() is None, storage_log.read_text(errors='replace')
            try:
                with socket.create_connection(('127.0.0.1', table_port), timeout=1):
                    break
            except OSError:
                assert time.monotonic() < deadline, storage_log.read_text(errors='replace')
                time.sleep(0.1)
        host = stack.enter_context(
            _process(
                [*_command(func), 'start', '--port', str(http_port)],
                cwd=app_dir,
                env=env,
                log=host_log,
            )
        )
        deadline = time.monotonic() + _TIMEOUT
        while True:
            assert host.poll() is None, host_log.read_text(errors='replace')
            if 'Agent output: "Hosted durably"' in host_log.read_text(errors='replace'):
                break
            assert time.monotonic() < deadline, host_log.read_text(errors='replace')
            time.sleep(0.1)
        yield host_log


def test_live_functions_native_orchestration(functions_host: Path) -> None:
    """Real host indexing, async activities and the SDK client are the behavior under test."""
    log = functions_host.read_text(encoding='utf-8', errors='replace')
    assert "Executing 'Functions.support'" in log
    assert "Executing 'Functions.pydantic_ai_" in log
    assert 'Agent output: "Hosted durably"' in log
