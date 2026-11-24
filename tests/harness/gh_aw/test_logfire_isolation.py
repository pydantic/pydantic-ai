"""Run the shipped launcher against real Logfire configuration, without network I/O."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import NotRequired

import pytest
from pydantic import TypeAdapter
from typing_extensions import TypedDict

from tests.conftest import try_import

from .test_engine_definition import CLAI2_SOURCE, launch, proxy_env

with try_import() as exporter_installed:
    importlib.import_module('opentelemetry.exporter.otlp.json.file')

pytestmark = [
    pytest.mark.subprocess(reason='runs the generated gh-aw launcher program as a real script'),
    # The first `node` start on a runner is slow; `started_node` pays it outside the test's budget.
    pytest.mark.usefixtures('started_node'),
]

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


# Each case is a fresh interpreter importing `pydantic_ai` and Logfire, so the two
# axes are paired rather than crossed: the token decides whether Logfire starts its
# exporter and token-validation thread, the exit code only which `SystemExit` the
# launcher re-raises, and neither changes how the other reaches shutdown.
@pytest.mark.parametrize(
    ('exit_code', 'token'),
    [(0, 'test-maintainer-token'), (7, '')],
    ids=['token-success', 'no-token-failure'],
)
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
        extra_python_path=CLAI2_SOURCE,
    )
    completed = subprocess.run(
        [
            sys.executable,
            '-P',
            '-c',
            PROBE,
            invocation.program,
            invocation.target,
            str(invocation.prompt_file),
            *invocation.cli_args,
        ],
        cwd=workspace,
        env={
            **invocation.env,
            'TMPDIR': str(workspace),
            'PROBE_RESULT': str(result),
            'PROBE_EXIT': str(exit_code),
        },
        capture_output=True,
        text=True,
        input=invocation.stdin,
        timeout=30,
        check=False,
    )
    assert completed.returncode == exit_code, completed.stderr
    assert result.exists(), completed.stderr
    assert not Path(result.read_text()).exists()
    assert result.with_suffix('.shutdown').exists(), completed.stderr


class _Attribute(TypedDict):
    key: str
    value: dict[str, object]


class _Span(TypedDict):
    name: str
    attributes: NotRequired[list[_Attribute]]


class _Scope(TypedDict):
    spans: list[_Span]


class _Resource(TypedDict):
    scopeSpans: list[_Scope]


class _Payload(TypedDict):
    resourceSpans: list[_Resource]


# The CLI boundary opens a root and delegate span, an instrumented request, and a cut-off request.
TRACING_PROBE = """import runpy
import sys
from unittest.mock import patch

import httpx2
import logfire


def cli(*args, **kwargs):
    with logfire.span('workflow-root'):
        with logfire.span('workflow-delegate'):
            with httpx2.Client(transport=httpx2.MockTransport(lambda request: httpx2.Response(200))) as client:
                logfire.instrument_httpx(client)
                client.get('https://provider.invalid/probe')
        try:
            with logfire.span('cut-off-request') as span:
                span.set_attribute('error.type', 'timeout')
                raise TimeoutError('run budget')
        except TimeoutError:
            pass
    logfire.force_flush()


with patch.object(runpy, 'run_module', cli):
    exec(compile(sys.argv.pop(1), '<launcher>', 'exec'))
"""


@pytest.mark.skipif(not exporter_installed(), reason='needs the `gh-aw-tracing` dependency group')
def test_tracing_exports_locally_without_sandbox_credentials(tmp_path: Path) -> None:
    """`PAI_OTEL_EXPORT_PATH` writes spans to a local file when the sandbox has no OTLP endpoint or token."""
    destination = tmp_path / 'private-spans.jsonl'
    invocation = launch(
        tmp_path,
        {**proxy_env('openai', 'openai/gpt-5'), 'PAI_OTEL_EXPORT_PATH': str(destination)},
        extra_python_path=CLAI2_SOURCE,
    )
    environment = {
        key: value
        for key, value in invocation.env.items()
        if not key.startswith('OTEL_EXPORTER_OTLP') and key != 'LOGFIRE_TOKEN'
    }
    completed = subprocess.run(
        [sys.executable, '-P', '-c', TRACING_PROBE, invocation.program, *invocation.argv[3:]],
        cwd=invocation.cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    spans: list[_Span] = []
    for line in destination.read_text().splitlines():
        payload = TypeAdapter(_Payload).validate_json(line)
        spans.extend(
            span for resource in payload['resourceSpans'] for scope in resource['scopeSpans'] for span in scope['spans']
        )
    assert {'workflow-root', 'workflow-delegate', 'cut-off-request', 'GET'}.issubset({span['name'] for span in spans})
    assert any(
        attribute == {'key': 'error.type', 'value': {'stringValue': 'timeout'}}
        for span in spans
        for attribute in span.get('attributes', [])
    )
