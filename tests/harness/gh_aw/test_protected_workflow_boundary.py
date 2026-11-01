"""Exercise the generated launcher across the protected-capture boundary."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import NotRequired

import pytest
from pydantic import TypeAdapter
from typing_extensions import TypedDict

from .test_engine_definition import CLAI2_SOURCE, definition, launch, parse_log, proxy_env

pytestmark = pytest.mark.subprocess(reason='executes the generated protected launcher and its real tracing exporter')


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


def test_host_installer_prepares_exporter_without_agent_only_environment(tmp_path: Path) -> None:
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    python = binaries / 'python3'
    python.write_text('#!/bin/sh\nprintf \'%s\\n\' "$*" >> "$INSTALL_LOG"\n')
    python.chmod(0o755)
    home = tmp_path / 'home'
    cli = home / '.local/bin/clai2'
    cli.parent.mkdir(parents=True)
    cli.write_text('#!/bin/sh\nexit 0\n')
    cli.chmod(0o755)
    log = tmp_path / 'installer.log'
    script = next(step.run for step in definition().pre_agent_steps if 'pip install' in step.run)
    completed = subprocess.run(
        ['bash', '-e', '-c', script],
        env={
            'PATH': str(binaries) + os.pathsep + os.environ['PATH'],
            'HOME': str(home),
            'INSTALL_LOG': str(log),
            'GH_AW_ENGINE_VERSION': '0.64.0',
            'OTEL_EXPORTER_OTLP_ENDPOINT': 'https://collector.invalid',
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert 'opentelemetry-exporter-otlp-json-file>=0.64b0' in log.read_text()


def test_prewritten_noop_has_an_authenticated_terminal_record(tmp_path: Path) -> None:
    outputs = tmp_path / 'safeoutputs.jsonl'
    outputs.write_text('{"type":"noop","message":"The PR moved"}\n')
    invocation = launch(tmp_path, {'GH_AW_SAFE_OUTPUTS': str(outputs)})
    completed = subprocess.run(
        [sys.executable, '-P', *invocation.argv],
        cwd=invocation.cwd,
        env=invocation.env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    parsed = parse_log(tmp_path, completed.stderr)
    assert len(parsed.log_entries) == 1
    assert parsed.log_entries[0].type == 'session.result'
    assert parsed.log_entries[0].data['status'] == 'success'
    assert parsed.diagnostic_log_entries == []


def test_tracing_exports_locally_without_sandbox_credentials(tmp_path: Path) -> None:
    destination = tmp_path / 'private-spans.jsonl'
    invocation = launch(
        tmp_path,
        {**proxy_env('openai', 'openai/gpt-5'), 'PAI_OTEL_EXPORT_PATH': str(destination)},
        extra_python_path=CLAI2_SOURCE,
    )
    probe = """import runpy
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
    environment = {
        key: value
        for key, value in invocation.env.items()
        if not key.startswith('OTEL_EXPORTER_OTLP') and key != 'LOGFIRE_TOKEN'
    }
    completed = subprocess.run(
        [sys.executable, '-P', '-c', probe, invocation.program, *invocation.argv[3:]],
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


def test_terminal_metadata_is_signed_with_the_observed_result(tmp_path: Path) -> None:
    invocation = launch(
        tmp_path,
        {
            **proxy_env('openai', 'openai/gpt-5'),
            'PAI_AGENT': 'workflow_fixture:agent',
            'PAI_RESULT_METADATA': 'workflow_fixture:result_metadata',
        },
        extra_python_path=CLAI2_SOURCE,
    )
    (Path(invocation.cwd) / 'workflow_fixture.py').write_text(
        'from pydantic_ai import Agent\nagent = Agent()\nresult_metadata = {}\n'
    )
    probe = """import runpy
import sys
from unittest.mock import patch


def cli(*args, **kwargs):
    import workflow_fixture
    workflow_fixture.result_metadata['provider_health'] = {'failure': {'kind': 'plan_limit', 'http_status': 429}}
    raise SystemExit(1)


with patch.object(runpy, 'run_module', cli):
    exec(compile(sys.argv.pop(1), '<launcher>', 'exec'))
"""
    completed = subprocess.run(
        [sys.executable, '-P', '-c', probe, invocation.program, *invocation.argv[3:]],
        cwd=invocation.cwd,
        env=invocation.env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 1, completed.stdout + completed.stderr
    terminal = parse_log(tmp_path, completed.stderr).log_entries[-1]
    assert terminal.type == 'session.result'
    assert terminal.data['status'] == 'failure'
    assert terminal.data['metadata'] == {'provider_health': {'failure': {'kind': 'plan_limit', 'http_status': 429}}}
