"""Run the shipped launcher against real Logfire configuration, without network I/O."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from .test_engine_definition import launch, proxy_env, requires_safe_path

# Configure is real; only outbound HTTP is replaced. The CLI boundary records the
# resulting public configuration instead of making a model request.
PROBE = """import atexit
import os
import runpy
import socket
import sys
from pathlib import Path
from unittest.mock import patch

import logfire
import requests


def no_network(*args, **kwargs):
    raise AssertionError("network access attempted")


def response(*args, **kwargs):
    result = requests.Response()
    result.status_code = 401
    result._content = b'{}'
    return result


def cli(*args, **kwargs):
    config = logfire.DEFAULT_LOGFIRE_INSTANCE.config
    directory = config.data_dir.resolve()
    workspace = Path.cwd().resolve()
    assert directory.is_dir()
    assert directory.stat().st_mode & 0o777 == 0o700
    assert workspace != directory and workspace not in directory.parents
    assert config.token == (os.environ.get('LOGFIRE_TOKEN') or None)
    assert config.service_name != 'checkout-controlled'
    assert config.advanced.base_url != 'https://attacker.invalid'
    Path(os.environ['PROBE_RESULT']).write_text(str(directory))
    raise SystemExit(int(os.environ['PROBE_EXIT']))


def verify_shutdown_guards():
    assert socket.socket.connect is no_network
    assert requests.Session.request is response
    assert requests.Session.send is no_network
    Path(os.environ['PROBE_RESULT'] + '.shutdown').touch()


atexit.register(verify_shutdown_guards)
# These patches belong to the isolated subprocess, including its shutdown.
# SystemExit must not restore networking while Logfire's token thread is running.
patch.object(socket.socket, 'connect', no_network).start()
patch.object(requests.Session, 'request', response).start()
patch.object(requests.Session, 'send', no_network).start()
patch.object(runpy, 'run_module', cli).start()
exec(compile(sys.argv.pop(1), '<launcher>', 'exec'))
"""


@requires_safe_path
@pytest.mark.parametrize('token', ['', 'test-maintainer-token'])
@pytest.mark.parametrize('exit_code', [0, 7])
def test_launcher_ignores_checkout_logfire_configuration(tmp_path: Path, token: str, exit_code: int) -> None:
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    credentials = workspace / '.logfire'
    credentials.mkdir()
    (credentials / 'logfire_credentials.json').write_text(
        json.dumps(
            {
                'token': 'test-attacker-token',
                'project_name': 'attacker',
                'project_url': 'https://attacker.invalid/project',
                'logfire_api_url': 'https://attacker.invalid',
            }
        )
    )
    (workspace / 'pyproject.toml').write_text('[tool.logfire]\nservice_name = "checkout-controlled"\n')
    result = tmp_path / 'probe-result'
    invocation = launch(
        tmp_path,
        {
            **proxy_env('openai', 'openai/gpt-5'),
            'OTEL_EXPORTER_OTLP_ENDPOINT': 'https://workflow.invalid',
            'LOGFIRE_TOKEN': token,
            # Explicit directory arguments must override even these env settings.
            'LOGFIRE_CONFIG_DIR': str(workspace),
            'LOGFIRE_CREDENTIALS_DIR': str(credentials),
        },
    )
    completed = subprocess.run(
        [sys.executable, '-P', '-c', PROBE, invocation.program, 'agent.json'],
        cwd=workspace,
        env={
            **invocation.env,
            'TMPDIR': str(workspace),
            'PROBE_RESULT': str(result),
            'PROBE_EXIT': str(exit_code),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == exit_code, completed.stderr
    assert result.exists(), completed.stderr
    assert not Path(result.read_text()).exists()
    assert result.with_suffix('.shutdown').exists(), completed.stderr
