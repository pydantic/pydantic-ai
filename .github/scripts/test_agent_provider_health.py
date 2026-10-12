"""Focused boundaries for the Z.ai inference gate and incident controller."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.usage import RunUsage

sys.path.insert(0, str(Path(__file__).parent))

import agent_provider_health as health
from agent_provider_health import (
    _create_or_reuse_incident,  # pyright: ignore[reportPrivateUsage]
    _fetch_zai_quota,  # pyright: ignore[reportPrivateUsage]
    _health_from_json,  # pyright: ignore[reportPrivateUsage]
    _mapping,  # pyright: ignore[reportPrivateUsage]
    _marker_from_body,  # pyright: ignore[reportPrivateUsage]
    _monitor_command,  # pyright: ignore[reportPrivateUsage]
    _parse_zai_quota,  # pyright: ignore[reportPrivateUsage]
    _reconcile_recovery,  # pyright: ignore[reportPrivateUsage]
    _run_result_from,  # pyright: ignore[reportPrivateUsage]
    _scope_for,  # pyright: ignore[reportPrivateUsage]
    _write_health,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai_gh_aw_shim import cli as shim


def _quota_response(
    *,
    interval_percentage: object = 1,
    weekly_percentage: object = 16,
    interval_reset: object = 1_791_012_926_973,
    weekly_reset: object = 1_791_471_269_972,
    limits: list[object] | None = None,
    code: object = 200,
    success: object = True,
) -> dict[str, object]:
    if limits is None:
        limits = [
            {
                'type': 'TOKENS_LIMIT',
                'unit': 3,
                'number': 5,
                'percentage': interval_percentage,
                'nextResetTime': interval_reset,
            },
            {
                'type': 'TOKENS_LIMIT',
                'unit': 6,
                'number': 1,
                'percentage': weekly_percentage,
                'nextResetTime': weekly_reset,
            },
            {
                'type': 'TIME_LIMIT',
                'unit': 5,
                'number': 1,
                'usage': 1000,
                'currentValue': 0,
                'remaining': 1000,
                'percentage': 0,
                'nextResetTime': 1_793_372_069_999,
            },
        ]
    return {'code': code, 'success': success, 'data': {'limits': limits}}


def _issue(number: int, marker: health.IncidentMarker, *, body_prefix: str = '') -> health.Issue:
    marker_json = json.dumps(
        {
            'version': 1,
            'scope': marker.scope,
            'key': marker.key,
            'kind': marker.kind,
            'run_id': marker.run_id,
            'reset_at': marker.reset_at,
        },
        separators=(',', ':'),
    )
    body = f'{body_prefix}{health.MARKER_PREFIX}{marker_json} -->\n'
    return health.Issue(number, 'Agent workflow incident', body, 'open', f'https://github.com/org/repo/issues/{number}')


def _marker_from_payload(payload: object) -> health.IncidentMarker | None:
    data = _mapping(payload)
    body = data.get('body') if data is not None else None
    if not isinstance(body, str):
        return None
    return _marker_from_body(body)


class FakeGitHub(health.GitHubClient):
    """Capture issue requests in memory for reconciler boundary tests."""

    def __init__(self, issues: list[health.Issue] | None = None) -> None:
        super().__init__('org/repo', 'token')
        self.issues = list(issues or [])
        self.posts: list[object] = []
        self.closed: list[int] = []

    def request(self, method: str, path: str, payload: object | None = None) -> object:
        if method == 'GET':
            return [
                {
                    'number': item.number,
                    'title': item.title,
                    'body': item.body,
                    'state': item.state,
                    'html_url': item.html_url,
                }
                for item in self.issues
            ]
        if method == 'POST':
            self.posts.append(payload)
            data = _mapping(payload) or {}
            issue = health.Issue(
                100 + len(self.posts),
                str(data.get('title')),
                str(data.get('body')),
                'open',
                'https://github.com/org/repo/issues/100',
            )
            self.issues.append(issue)
            return {
                'number': issue.number,
                'title': issue.title,
                'body': issue.body,
                'state': issue.state,
                'html_url': issue.html_url,
            }
        if method == 'PATCH':
            self.closed.append(int(path.split('/')[-1]))
            return {}
        raise AssertionError(f'unexpected request {method} {path}')

    def open_incidents(self) -> list[health.Issue]:
        return self.issues


class FakeHTTPResponse:
    """A small Z.ai response for the trusted-runner command tests."""

    def __init__(self, payload: dict[str, object] | bytes, status: int = 200) -> None:
        self.payload = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.status = status

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def test_zai_quota_parses_observed_windows_and_ignores_mcp_limit() -> None:
    """The observed Coding Plan windows map into the trusted quota DTO."""
    quota = _parse_zai_quota(_quota_response())

    assert quota.status == 'healthy'
    assert quota.interval_remaining_percent == 99
    assert quota.weekly_remaining_percent == 84
    assert quota.interval_reset_at == '2026-10-03T07:35:26Z'
    assert quota.weekly_reset_at == '2026-10-08T14:54:29Z'
    assert quota.reset_at is None


@pytest.mark.parametrize(
    ('interval_percentage', 'weekly_percentage', 'expected_reset'),
    [
        (100, 84, '2026-10-03T07:35:26Z'),
        (1, 100, '2026-10-08T14:54:29Z'),
        (100, 100, '2026-10-08T14:54:29Z'),
    ],
)
def test_zai_quota_exhaustion_uses_latest_exhausted_window_reset(
    interval_percentage: object, weekly_percentage: object, expected_reset: str
) -> None:
    """Exhausted windows keep the latest reset needed to recover."""
    quota = _parse_zai_quota(
        _quota_response(interval_percentage=interval_percentage, weekly_percentage=weekly_percentage)
    )

    assert quota.status == 'exhausted'
    assert quota.reset_at == expected_reset


@pytest.mark.parametrize(
    'payload',
    [
        {},
        _quota_response(code=201),
        _quota_response(code=True),
        _quota_response(success=False),
        _quota_response(success=1),
        {'code': 200, 'success': True, 'data': {}},
        _quota_response(limits=[]),
        _quota_response(
            limits=[
                {
                    'type': 'TOKENS_LIMIT',
                    'unit': 6,
                    'number': 1,
                    'percentage': 16,
                    'nextResetTime': 1_791_471_269_972,
                }
            ]
        ),
        _quota_response(
            limits=[
                {
                    'type': 'TOKENS_LIMIT',
                    'unit': 3,
                    'number': 5,
                    'percentage': 1,
                    'nextResetTime': 1_791_012_926_973,
                }
            ]
        ),
        _quota_response(
            limits=[
                {'type': 'TOKENS_LIMIT', 'unit': 3, 'number': 5, 'percentage': 1, 'nextResetTime': 1_791_012_926_973},
                {'type': 'TOKENS_LIMIT', 'unit': 3, 'number': 5, 'percentage': 1, 'nextResetTime': 1_791_012_926_973},
                {'type': 'TOKENS_LIMIT', 'unit': 6, 'number': 1, 'percentage': 16, 'nextResetTime': 1_791_471_269_972},
            ]
        ),
        _quota_response(
            limits=[
                {'type': 'TOKENS_LIMIT', 'unit': 3, 'number': 4, 'percentage': 1, 'nextResetTime': 1_791_012_926_973},
                {'type': 'TOKENS_LIMIT', 'unit': 6, 'number': 1, 'percentage': 16, 'nextResetTime': 1_791_471_269_972},
            ]
        ),
        _quota_response(interval_percentage=True),
        _quota_response(interval_percentage=float('nan')),
        _quota_response(interval_percentage=-1),
        _quota_response(interval_percentage=101),
        _quota_response(interval_reset=True),
        _quota_response(interval_reset=10**1000),
        _quota_response(weekly_reset=None),
    ],
)
def test_zai_quota_unavailable_or_ambiguous_is_unknown(payload: object) -> None:
    """Missing, malformed, duplicate, or unsupported windows fail closed."""
    assert _parse_zai_quota(payload).status == 'unknown'


def test_zai_quota_probe_returns_unknown_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    response = FakeHTTPResponse(_quota_response(), status=503)
    monkeypatch.setattr(health.urllib.request.OpenerDirector, 'open', lambda *_args, **_kwargs: response)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]

    assert _fetch_zai_quota('private-zai-fixture-key').status == 'unknown'


@pytest.mark.parametrize('error', [OSError('offline'), health.urllib.error.URLError('offline')])
def test_zai_quota_probe_returns_unknown_on_transport_error(
    error: OSError | health.urllib.error.URLError, monkeypatch: pytest.MonkeyPatch
) -> None:
    def open_url(
        _opener: health.urllib.request.OpenerDirector, *_args: object, **_kwargs: object
    ) -> FakeHTTPResponse:
        raise error

    monkeypatch.setattr(health.urllib.request.OpenerDirector, 'open', open_url)

    assert _fetch_zai_quota('private-zai-fixture-key').status == 'unknown'


def test_zai_quota_probe_returns_unknown_on_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    response = FakeHTTPResponse(b'{')
    monkeypatch.setattr(health.urllib.request.OpenerDirector, 'open', lambda *_args, **_kwargs: response)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]

    assert _fetch_zai_quota('private-zai-fixture-key').status == 'unknown'


def test_zai_quota_probe_does_not_forward_authorization_on_cross_origin_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_authorization: list[str | None] = []
    target_authorization: list[str | None] = []

    class QuotaHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            target_authorization.append(self.headers.get('Authorization'))
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(_quota_response()).encode())

        def log_message(self, format: str, *args: object) -> None:
            pass

    target_server = ThreadingHTTPServer(('127.0.0.1', 0), QuotaHandler)
    target_thread = threading.Thread(target=target_server.serve_forever, daemon=True)
    target_thread.start()
    try:
        class RedirectHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                source_authorization.append(self.headers.get('Authorization'))
                self.send_response(302)
                self.send_header('Location', f'http://127.0.0.1:{target_server.server_address[1]}/quota')
                self.end_headers()

            def log_message(self, format: str, *args: object) -> None:
                pass

        source_server = ThreadingHTTPServer(('127.0.0.1', 0), RedirectHandler)
        source_thread = threading.Thread(target=source_server.serve_forever, daemon=True)
        source_thread.start()
        try:
            monkeypatch.setattr(health, 'ZAI_QUOTA_URL', f'http://127.0.0.1:{source_server.server_address[1]}/quota')
            quota = _fetch_zai_quota('dummy-zai-fixture-key')
        finally:
            source_server.shutdown()
            source_thread.join()
            source_server.server_close()
    finally:
        target_server.shutdown()
        target_thread.join()
        target_server.server_close()

    assert source_authorization == ['dummy-zai-fixture-key']
    assert target_authorization == []
    assert quota.status == 'unknown'


def test_zai_quota_probe_returns_unknown_on_oversized_json_integer(monkeypatch: pytest.MonkeyPatch) -> None:
    response = FakeHTTPResponse(b'{"code":' + b'9' * 5000 + b'}')
    monkeypatch.setattr(health.urllib.request.OpenerDirector, 'open', lambda *_args, **_kwargs: response)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]

    assert _fetch_zai_quota('private-zai-fixture-key').status == 'unknown'


def test_check_writes_blocked_artifact_for_out_of_range_zai_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unrepresentable provider reset time blocks the gate without losing its artifact."""
    payload = _quota_response(interval_reset=10**1000)
    monkeypatch.setattr(health.urllib.request.OpenerDirector, 'open', lambda *_args, **_kwargs: FakeHTTPResponse(payload))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: FakeGitHub())  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    secret = 'zai-key-private-fixture'
    monkeypatch.setenv('ZAI_API_KEY', secret)
    monkeypatch.setenv('GITHUB_WORKFLOW', 'nightly-sweep')
    monkeypatch.setenv('PYDANTIC_AI_TASK_KEY', 'task-1')
    monkeypatch.setenv('PYDANTIC_AI_TRIGGER_EVENT', 'schedule')
    monkeypatch.setenv('PYDANTIC_AI_RUN_ATTEMPT', '1')
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'token')
    output = tmp_path / 'provider-health.json'

    assert health.main(['check', '--output', str(output)]) == 0

    artifact = output.read_text()
    result = _health_from_json(json.loads(artifact))
    assert not result.ready
    assert result.quota.status == 'unknown'
    assert secret not in artifact


def test_health_check_blocks_unknown_empty_and_matching_incident() -> None:
    """Unknown and exhausted quotas or a provider incident block every workflow."""
    context = ('nightly-sweep', 'pr-45-head-a1b2', 'schedule', 1)

    assert not health.check_health(*context, health.Quota('unknown'), []).ready
    assert not health.check_health(*context, health.Quota('exhausted'), []).ready
    assert health.check_health(*context, health.Quota('healthy'), []).ready
    issue = _issue(7, health.IncidentMarker('provider', 'zai', 'authentication', '12', None))
    blocked = health.check_health(*context, health.Quota('healthy'), [issue])
    assert not blocked.ready
    assert blocked.reason == 'Open operational incident #7 blocks inference'
    other_context = ('different-workflow', 'other-task', 'workflow_dispatch', 1)
    assert not health.check_health(*other_context, health.Quota('healthy'), [issue]).ready


def test_health_check_ignores_retired_provider_incident() -> None:
    """A retired provider incident does not block inference through Z.ai."""
    context = ('nightly-sweep', 'task-1', 'schedule', 1)
    issue = _issue(13, health.IncidentMarker('provider', 'minimax', 'authentication', '13', None))

    assert health.check_health(*context, health.Quota('healthy'), [issue]).ready


def test_workflow_and_task_incidents_match_only_their_scope() -> None:
    """Workflow and task incidents block matching identities only."""
    context = ('nightly-sweep', 'pr-45-head-a1b2', 'schedule', 1)
    quota = health.Quota('healthy')

    workflow_issue = _issue(8, health.IncidentMarker('workflow', 'nightly-sweep', 'execution', '8', None))
    other_workflow_issue = _issue(9, health.IncidentMarker('workflow', 'weekly-sweep', 'execution', '9', None))
    task_issue = _issue(10, health.IncidentMarker('task', 'nightly-sweep:pr-45-head-a1b2', 'timeout', '10', None))
    other_task_issue = _issue(11, health.IncidentMarker('task', 'nightly-sweep:pr-46-head-c3d4', 'timeout', '11', None))

    assert not health.check_health(*context, quota, [workflow_issue]).ready
    assert health.check_health(*context, quota, [other_workflow_issue]).ready
    assert not health.check_health(*context, quota, [task_issue]).ready
    assert health.check_health(*context, quota, [other_task_issue]).ready
    provider_issue = _issue(12, health.IncidentMarker('provider', 'zai', 'balance', '12', None))
    assert not health.check_health(*context, quota, [provider_issue]).ready


def test_health_artifact_round_trips_versioned_gate_dto(tmp_path: Path) -> None:
    """The runner gate artifact round-trips its versioned validated DTO."""
    original = health.Health(
        'nightly-sweep',
        'task-1',
        'schedule',
        1,
        False,
        'Z.ai quota health is unknown',
        '2026-10-01T12:00:00Z',
        health.Quota(
            'unknown',
            interval_remaining_percent=12.5,
            weekly_remaining_percent=30,
            interval_reset_at='2026-10-01T13:00:00Z',
            weekly_reset_at='2026-10-07T00:00:00Z',
        ),
    )
    path = tmp_path / 'provider-health.json'

    _write_health(path, original)

    payload: object = json.loads(path.read_text())
    payload_data = _mapping(payload)
    assert payload_data is not None
    quota_data = _mapping(payload_data.get('quota'))
    assert quota_data is not None
    assert 'interval_unlimited' not in quota_data
    assert 'weekly_unlimited' not in quota_data
    quota_data['interval_unlimited'] = 'legacy value'
    quota_data['weekly_unlimited'] = {'legacy': True}
    assert _health_from_json(payload_data) == original


def test_check_requires_the_actual_run_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The check command never guesses a run attempt when runner metadata is missing."""
    client = FakeGitHub()
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: client)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'token')
    monkeypatch.delenv('PYDANTIC_AI_RUN_ATTEMPT', raising=False)
    output = tmp_path / 'provider-health.json'

    assert health.main(['check', '--output', str(output)]) == 1
    assert not output.exists()


@pytest.mark.parametrize('run_attempt', [None, True, 0, -1, '2'])
def test_attempt_contract_rejects_missing_or_invalid_values(run_attempt: object, tmp_path: Path) -> None:
    """Health artifacts require positive integer attempts; invalid agent attempts stay untyped."""
    original = health.Health(
        'nightly-sweep',
        'task-1',
        'schedule',
        1,
        True,
        'Provider health check passed',
        '2026-10-01T12:00:00Z',
        health.Quota('healthy'),
    )
    path = tmp_path / 'provider-health.json'
    _write_health(path, original)
    health_payload: dict[str, object] = json.loads(path.read_text())
    health_payload['run_attempt'] = run_attempt
    with pytest.raises(ValueError, match='missing required validated fields'):
        _health_from_json(health_payload)

    provider_health: dict[str, object] = {
        'workflow': 'nightly-sweep',
        'task_key': 'task-1',
        'trigger_event': 'schedule',
        'run_attempt': run_attempt,
        'failure': {'kind': 'timeout'},
    }
    assert _run_result_from({'type': 'result', 'provider_health': provider_health}) is None


def test_result_parser_reads_shim_terminal_jsonl_at_root(tmp_path: Path) -> None:
    """Terminal result parsing reads the nested shim metadata from JSONL."""
    result_file = tmp_path / 'agent-stdio.log'
    provider_health: dict[str, object] = {
        'workflow': 'nightly-sweep',
        'task_key': 'task-1',
        'trigger_event': 'schedule',
        'run_attempt': 1,
        'failure': {'kind': 'rate_limit', 'http_status': 429},
    }
    result_file.write_text(
        '\n'.join(
            [
                'starting agent',
                json.dumps({'type': 'result', 'provider_health': provider_health}),
            ]
        )
    )

    assert health.parse_run_result(result_file) == health.RunResult(
        'nightly-sweep', 'task-1', 'schedule', 1, health.Failure('rate_limit', 429, None)
    )


def test_result_parser_rejects_retired_balance_failure_kind() -> None:
    """The removed MiniMax failure kind is not accepted as current runner metadata."""
    provider_health: dict[str, object] = {
        'workflow': 'nightly-sweep',
        'task_key': 'task-1',
        'trigger_event': 'schedule',
        'run_attempt': 1,
        'failure': {'kind': 'balance'},
    }
    assert _run_result_from({'type': 'result', 'provider_health': provider_health}) is None


def test_monitor_creates_one_assigned_labeled_incident_then_reuses_it() -> None:
    """An assigned incident is created once and reused without comment growth."""
    client = FakeGitHub()
    result = health.RunResult('nightly-sweep', 'task-1', 'schedule', 1, health.Failure('authentication', 401, None))

    first = _create_or_reuse_incident(client, result, '88', client.repo, 'Z.ai rejected credentials', dry_run=False)
    redelivery = _create_or_reuse_incident(
        client, result, '88', client.repo, 'Z.ai rejected credentials', dry_run=False
    )
    later_failure = _create_or_reuse_incident(
        client, result, '89', client.repo, 'Z.ai rejected credentials', dry_run=False
    )

    assert first is not None and redelivery is not None and later_failure is not None
    assert first.number == redelivery.number == later_failure.number
    assert len(client.posts) == 1
    payload: object = client.posts[0]
    payload_data = _mapping(payload) or {}
    assert payload_data['labels'] == ['agentic-workflows', 'pydanty:meta']
    assert payload_data['assignees'] == ['dsfaccini']
    body = payload_data.get('body')
    assert isinstance(body, str)
    assert 'First failing run: https://github.com/org/repo/actions/runs/88' in body
    assert 'amount' not in body.lower()


def test_real_shim_result_is_idempotently_monitored_and_blocks_other_workflows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A Z.ai quota error is monitored once and blocks new workflows."""
    workflow = 'Pydantic AI CI Review'
    task_key = 'CI Review:workflow_run:pr-123:head-a1b2'
    trigger_event = 'workflow_run'
    monkeypatch.setenv('GITHUB_WORKFLOW', workflow)
    monkeypatch.setenv('PYDANTIC_AI_TASK_KEY', task_key)
    monkeypatch.setenv('PYDANTIC_AI_TRIGGER_EVENT', trigger_event)
    monkeypatch.setenv('PYDANTIC_AI_RUN_ATTEMPT', '1')
    shim.emit_result(
        'agent run failed',
        usage=RunUsage(requests=5, input_tokens=23, output_tokens=9, cache_read_tokens=4, cache_write_tokens=2),
        session_id='workflow-run-1',
        is_error=True,
        error=ModelHTTPError(
            429,
            'glm-5.3-flash',
            {'error': {'code': 1316, 'message': 'private provider detail'}},
        ),
    )
    emitted = capsys.readouterr().out
    agent_artifact = tmp_path / 'agent-stdio.log'
    agent_artifact.write_text(emitted)
    result_event = json.loads(emitted.strip())
    assert result_event['provider_health'] == {
        'workflow': workflow,
        'task_key': task_key,
        'trigger_event': trigger_event,
        'run_attempt': 1,
        'failure': {'kind': 'rate_limit', 'http_status': 429},
    }
    usage = result_event['usage']
    assert isinstance(usage, dict)
    assert usage['input_tokens'] == 23
    assert usage['output_tokens'] == 9
    assert usage['cache_read_input_tokens'] == 4
    assert usage['cache_creation_input_tokens'] == 2
    assert health.parse_run_result(agent_artifact) == health.RunResult(
        workflow, task_key, trigger_event, 1, health.Failure('rate_limit', 429, None)
    )

    health_artifact = tmp_path / 'provider-health.json'
    _write_health(
        health_artifact,
        health.Health(
            workflow,
            task_key,
            trigger_event,
            1,
            True,
            'Provider health check passed',
            '2026-10-01T12:00:00Z',
            health.Quota('healthy'),
        ),
    )
    client = FakeGitHub()
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: client)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'token')
    monitor_args = argparse.Namespace(
        repository=None,
        run_id=70,
        run_attempt=1,
        conclusion='failure',
        health_artifact=health_artifact,
        agent_artifact=agent_artifact,
        dry_run=False,
        recover_issue=None,
    )

    _monitor_command(monitor_args)
    _monitor_command(monitor_args)
    _monitor_command(
        argparse.Namespace(
            repository=None,
            run_id=71,
            run_attempt=1,
            conclusion='failure',
            health_artifact=health_artifact,
            agent_artifact=agent_artifact,
            dry_run=False,
            recover_issue=None,
        )
    )

    assert len(client.posts) == 1
    issue = client.issues[0]
    marker = _marker_from_body(issue.body)
    assert marker == health.IncidentMarker('provider', 'zai', 'rate_limit', '70', None)
    assert 'private provider detail' not in issue.body
    monkeypatch.setattr(health, '_fetch_zai_quota', lambda *_: health.Quota('healthy'))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setenv('GITHUB_WORKFLOW', 'Another workflow')
    monkeypatch.setenv('PYDANTIC_AI_TASK_KEY', 'different-task')
    monkeypatch.setenv('PYDANTIC_AI_TRIGGER_EVENT', 'schedule')
    monkeypatch.setenv('ZAI_API_KEY', 'zai-fresh-healthy-key')
    monkeypatch.delenv('GITHUB_OUTPUT', raising=False)
    monkeypatch.delenv('GITHUB_STEP_SUMMARY', raising=False)
    second_workflow_health = tmp_path / 'other-workflow-provider-health.json'

    assert health.main(['check', '--output', str(second_workflow_health)]) == 0
    assert not _health_from_json(json.loads(second_workflow_health.read_text())).ready
    assert len(client.posts) == 1


@pytest.mark.parametrize('stale_is_error', [False, True])
def test_monitor_uses_fresh_quota_when_agent_result_is_from_prior_attempt(
    stale_is_error: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Attempt-one success or failure cannot override the current attempt-two quota block."""
    workflow = 'nightly-sweep'
    task_key = 'stable-task-key'
    trigger_event = 'schedule'
    monkeypatch.setenv('GITHUB_WORKFLOW', workflow)
    monkeypatch.setenv('PYDANTIC_AI_TASK_KEY', task_key)
    monkeypatch.setenv('PYDANTIC_AI_TRIGGER_EVENT', trigger_event)
    monkeypatch.setenv('PYDANTIC_AI_RUN_ATTEMPT', '1')
    shim.emit_result(
        'stale attempt result',
        usage=RunUsage(requests=1),
        session_id='old-attempt',
        is_error=stale_is_error,
        error=TimeoutError('stale attempt timed out') if stale_is_error else None,
    )
    log = tmp_path / 'agent-stdio.log'
    log.write_text(capsys.readouterr().out)
    stale_result = health.parse_run_result(log)
    assert stale_result is not None
    assert stale_result.run_attempt == 1
    assert (stale_result.failure is not None) is stale_is_error

    health_artifact = tmp_path / 'provider-health.json'
    _write_health(
        health_artifact,
        health.Health(
            workflow,
            task_key,
            trigger_event,
            2,
            False,
            'Z.ai quota is exhausted',
            '2026-10-01T12:00:00Z',
            health.Quota('exhausted', '2026-10-01T13:00:00Z'),
        ),
    )
    client = FakeGitHub()
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: client)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'token')

    assert (
        _monitor_command(
            argparse.Namespace(
                repository=None,
                run_id=80,
                run_attempt=2,
                conclusion='failure',
                health_artifact=health_artifact,
                agent_artifact=log,
                dry_run=False,
                recover_issue=None,
            )
        )
        == 0
    )

    payload: object = client.posts[0]
    assert _marker_from_payload(payload) == health.IncidentMarker(
        'provider', 'zai', 'quota_exhausted', '80', '2026-10-01T13:00:00Z'
    )


def test_monitor_ignores_stale_failure_on_current_healthy_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale attempt-one failure cannot resurrect an incident after attempt-two success."""
    health_artifact = tmp_path / 'provider-health.json'
    _write_health(
        health_artifact,
        health.Health(
            'nightly-sweep',
            'stable-task-key',
            'schedule',
            2,
            True,
            'Provider health check passed',
            '2026-10-01T12:00:00Z',
            health.Quota('healthy'),
        ),
    )
    agent_log = tmp_path / 'agent-stdio.log'
    agent_log.write_text(
        json.dumps(
            {
                'type': 'result',
                'provider_health': {
                    'workflow': 'nightly-sweep',
                    'task_key': 'stable-task-key',
                    'trigger_event': 'schedule',
                    'run_attempt': 1,
                    'failure': {'kind': 'timeout'},
                },
            }
        )
    )
    client = FakeGitHub()
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: client)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'token')

    assert (
        _monitor_command(
            argparse.Namespace(
                repository=None,
                run_id=81,
                run_attempt=2,
                conclusion='success',
                health_artifact=health_artifact,
                agent_artifact=agent_log,
                dry_run=False,
                recover_issue=None,
            )
        )
        == 0
    )
    assert client.posts == []


def test_monitor_reuses_existing_provider_block_when_agent_result_is_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stale terminal metadata does not duplicate the provider issue that blocked this run."""
    existing = _issue(82, health.IncidentMarker('provider', 'zai', 'authentication', '82', None))
    health_artifact = tmp_path / 'provider-health.json'
    _write_health(
        health_artifact,
        health.Health(
            'nightly-sweep',
            'stable-task-key',
            'schedule',
            2,
            False,
            'Open operational incident #82 blocks inference',
            '2026-10-01T12:00:00Z',
            health.Quota('healthy'),
        ),
    )
    agent_log = tmp_path / 'agent-stdio.log'
    agent_log.write_text(
        json.dumps(
            {
                'type': 'result',
                'provider_health': {
                    'workflow': 'nightly-sweep',
                    'task_key': 'stable-task-key',
                    'trigger_event': 'schedule',
                    'run_attempt': 1,
                    'failure': {'kind': 'timeout'},
                },
            }
        )
    )
    client = FakeGitHub([existing])
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: client)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'token')

    assert (
        _monitor_command(
            argparse.Namespace(
                repository=None,
                run_id=83,
                run_attempt=2,
                conclusion='failure',
                health_artifact=health_artifact,
                agent_artifact=agent_log,
                dry_run=False,
                recover_issue=None,
            )
        )
        == 0
    )
    assert client.posts == []
    assert client.issues == [existing]


def test_monitor_rejects_health_artifact_from_different_run_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A trusted health artifact must match the triggering workflow attempt."""
    health_artifact = tmp_path / 'provider-health.json'
    _write_health(
        health_artifact,
        health.Health(
            'nightly-sweep',
            'stable-task-key',
            'schedule',
            1,
            False,
            'Z.ai quota is exhausted',
            '2026-10-01T12:00:00Z',
            health.Quota('exhausted'),
        ),
    )
    client = FakeGitHub()
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: client)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'token')

    with pytest.raises(ValueError, match='does not match the triggering'):
        _monitor_command(
            argparse.Namespace(
                repository=None,
                run_id=84,
                run_attempt=2,
                conclusion='failure',
                health_artifact=health_artifact,
                agent_artifact=None,
                dry_run=False,
                recover_issue=None,
            )
        )
    assert client.posts == []


@pytest.mark.parametrize(
    ('payload', 'expected_ready'),
    [
        (_quota_response(), True),
        (_quota_response(success=False), False),
    ],
)
def test_check_command_reads_zai_quota_and_writes_secret_free_artifact(
    payload: dict[str, object], expected_ready: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trusted check reads the quota endpoint with raw auth and emits secret-free state."""
    requests: list[health.urllib.request.Request] = []

    def open_url(
        _opener: health.urllib.request.OpenerDirector, request: health.urllib.request.Request, *, timeout: int
    ) -> FakeHTTPResponse:
        requests.append(request)
        assert timeout == 20
        return FakeHTTPResponse(payload)

    monkeypatch.setattr(health.urllib.request.OpenerDirector, 'open', open_url)
    client = FakeGitHub()
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: client)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    key = 'private-zai-fixture-key'
    monkeypatch.setenv('ZAI_API_KEY', key)
    monkeypatch.setenv('MINIMAX_API_KEY', 'retired-fixture-key')
    monkeypatch.setenv('GITHUB_WORKFLOW', 'Pydantic AI CI Review')
    monkeypatch.setenv('PYDANTIC_AI_TASK_KEY', 'task-1')
    monkeypatch.setenv('PYDANTIC_AI_TRIGGER_EVENT', 'workflow_run')
    monkeypatch.setenv('PYDANTIC_AI_RUN_ATTEMPT', '1')
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'github-token')
    output = tmp_path / 'provider-health.json'
    github_output = tmp_path / 'output'
    summary = tmp_path / 'summary'
    monkeypatch.setenv('GITHUB_OUTPUT', str(github_output))
    monkeypatch.setenv('GITHUB_STEP_SUMMARY', str(summary))

    assert health.main(['check', '--output', str(output)]) == 0

    dto_text = output.read_text()
    assert _health_from_json(json.loads(dto_text)).ready is expected_ready
    ready_line = 'ready=true' if expected_ready else 'ready=false'
    assert ready_line in github_output.read_text()
    assert requests[0].full_url == health.ZAI_QUOTA_URL
    assert requests[0].get_method() == 'GET'
    assert requests[0].get_header('Authorization') == key
    assert key not in dto_text
    assert key not in github_output.read_text()
    assert key not in summary.read_text()
    if expected_ready:
        assert 'interval: 99% remaining; resets at 2026-10-03T07:35:26Z' in summary.read_text()
        assert 'weekly: 84% remaining; resets at 2026-10-08T14:54:29Z' in summary.read_text()
    else:
        assert 'interval: ' not in summary.read_text()


@pytest.mark.parametrize('minimax_key', ['', 'retired-fixture-key'])
@pytest.mark.parametrize('conclusion', ['success', 'failure'])
def test_retired_provider_configuration_skips_without_an_incident(
    minimax_key: str, conclusion: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Old locks skip inference and incident reporting without querying provider health."""
    client = FakeGitHub()

    def unexpected_quota(_key: str) -> health.Quota:
        raise AssertionError('retired configuration must not query Z.ai')

    def unexpected_incidents() -> list[health.Issue]:
        raise AssertionError('retired configuration must not query operational incidents')

    monkeypatch.setattr(health, '_fetch_zai_quota', unexpected_quota)
    monkeypatch.setattr(client, 'open_incidents', unexpected_incidents)
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: client)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.delenv('ZAI_API_KEY', raising=False)
    monkeypatch.setenv('MINIMAX_API_KEY', minimax_key)
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'github-token')
    monkeypatch.setenv('GITHUB_WORKFLOW', 'Pydantic AI UI Security Review')
    monkeypatch.setenv('PYDANTIC_AI_TASK_KEY', 'old-head')
    monkeypatch.setenv('PYDANTIC_AI_TRIGGER_EVENT', 'pull_request')
    monkeypatch.setenv('PYDANTIC_AI_RUN_ATTEMPT', '1')
    github_output = tmp_path / 'output'
    summary = tmp_path / 'summary'
    monkeypatch.setenv('GITHUB_OUTPUT', str(github_output))
    monkeypatch.setenv('GITHUB_STEP_SUMMARY', str(summary))
    artifact = tmp_path / 'provider-health.json'

    assert health.main(['check', '--output', str(artifact)]) == 0
    decision = _health_from_json(json.loads(artifact.read_text()))
    assert decision.ready is False
    assert 'retired `MINIMAX_API_KEY` instead of `ZAI_API_KEY`' in decision.reason
    assert 'update the workflow from `main`' in decision.reason
    assert 'ready=false' in github_output.read_text()
    assert decision.reason in summary.read_text()
    assert 'retired-fixture-key' not in artifact.read_text()
    assert (
        health.main(
            [
                'monitor',
                '--run-id',
                '43',
                '--run-attempt',
                '1',
                '--conclusion',
                conclusion,
                '--health-artifact',
                str(artifact),
            ]
        )
        == 0
    )
    assert client.posts == []


@pytest.mark.parametrize(('zai_key', 'minimax_key'), [(None, None), ('', None), ('', 'retired-fixture-key')])
def test_current_configuration_with_missing_credentials_still_creates_an_incident(
    zai_key: str | None, minimax_key: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing current credentials remain an outage even when a retired key is available."""
    client = FakeGitHub()
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: client)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    for name, value in [('ZAI_API_KEY', zai_key), ('MINIMAX_API_KEY', minimax_key)]:
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'github-token')
    monkeypatch.setenv('GITHUB_WORKFLOW', 'Pydantic AI UI Security Review')
    monkeypatch.setenv('PYDANTIC_AI_TASK_KEY', 'current-head')
    monkeypatch.setenv('PYDANTIC_AI_TRIGGER_EVENT', 'pull_request')
    monkeypatch.setenv('PYDANTIC_AI_RUN_ATTEMPT', '1')
    monkeypatch.delenv('GITHUB_OUTPUT', raising=False)
    monkeypatch.delenv('GITHUB_STEP_SUMMARY', raising=False)
    artifact = tmp_path / 'provider-health.json'

    assert health.main(['check', '--output', str(artifact)]) == 0
    decision = _health_from_json(json.loads(artifact.read_text()))
    assert decision.ready is False
    assert decision.reason == 'Z.ai quota health is unknown'
    assert health.main(['monitor', '--run-id', '43', '--run-attempt', '1', '--health-artifact', str(artifact)]) == 0
    assert len(client.posts) == 1
    assert _marker_from_payload(client.posts[0]) == health.IncidentMarker(
        'provider', 'zai', 'quota_unknown', '43', None
    )


def test_scope_rules_for_typed_and_untyped_failures() -> None:
    """Terminal failure kinds map to provider, task, or scheduled workflow scope."""
    assert _scope_for(health.RunResult('w', 't', 'schedule', 1, health.Failure('authentication', None, None))) == (
        'provider',
        'zai',
        'authentication',
    )
    assert _scope_for(health.RunResult('w', 't', 'schedule', 1, health.Failure('other', 403, None))) == (
        'provider',
        'zai',
        'other',
    )
    rate_limited = health.RunResult('w', 't', 'workflow_dispatch', 1, health.Failure('rate_limit', 429, None))
    assert _scope_for(rate_limited) == ('provider', 'zai', 'rate_limit')
    scheduled_timeout = health.RunResult('w', 'target-head-a1b2', 'schedule', 1, health.Failure('timeout', None, None))
    repeated_scheduled_timeout = health.RunResult(
        'w', 'target-head-a1b2', 'schedule', 1, health.Failure('timeout', None, None)
    )
    assert _scope_for(scheduled_timeout) == ('workflow', 'w', 'timeout')
    assert _scope_for(repeated_scheduled_timeout) == _scope_for(scheduled_timeout)
    assert _scope_for(
        health.RunResult('w', 'target-head-a1b2', 'workflow_dispatch', 1, health.Failure('timeout', None, None))
    ) == ('task', 'w:target-head-a1b2', 'timeout')
    assert _scope_for(health.RunResult('w', 't', 'schedule', 1, None)) == ('workflow', 'w', 'execution')


def test_scheduled_recovery_closes_only_elapsed_known_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scheduled recovery closes elapsed windows but leaves future and unknown resets."""
    elapsed = _issue(1, health.IncidentMarker('provider', 'zai', 'rate_limit', '1', '2026-09-30T00:00:00Z'))
    future = _issue(2, health.IncidentMarker('provider', 'zai', 'quota_exhausted', '2', '2026-12-01T00:00:00Z'))
    unknown = _issue(3, health.IncidentMarker('provider', 'zai', 'quota_unknown', None, None))
    retired = _issue(4, health.IncidentMarker('provider', 'minimax', 'rate_limit', '4', '2026-09-30T00:00:00Z'))
    client = FakeGitHub([elapsed, future, unknown, retired])
    monkeypatch.setenv('ZAI_API_KEY', 'zai-key')
    monkeypatch.setattr(health, '_fetch_zai_quota', lambda *_: health.Quota('healthy'))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setattr(health, '_now', lambda: dt.datetime(2026, 10, 1, tzinfo=dt.timezone.utc))

    assert _reconcile_recovery(client, argparse.Namespace(dry_run=False, recover_issue=None)) == 0
    assert client.closed == [1]


def test_scheduled_monitor_creates_or_reuses_unknown_quota_incident(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated scheduled checks reuse one incident when quota remains unknown."""
    client = FakeGitHub()
    monkeypatch.setenv('ZAI_API_KEY', 'zai-key')
    monkeypatch.setenv('PYDANTIC_AI_RUN_ATTEMPT', '1')
    monkeypatch.setattr(health, '_fetch_zai_quota', lambda *_: health.Quota('unknown'))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    args = argparse.Namespace(dry_run=False, recover_issue=None)

    _reconcile_recovery(client, args)
    _reconcile_recovery(client, args)

    assert len(client.posts) == 1
    payload: object = client.posts[0]
    assert _marker_from_payload(payload) == health.IncidentMarker('provider', 'zai', 'quota_unknown', None, None)


def test_manual_recovery_requires_healthy_provider_and_targets_one_issue(monkeypatch: pytest.MonkeyPatch) -> None:
    """Manual provider recovery requires health and closes only the selected issue."""
    incident = _issue(5, health.IncidentMarker('provider', 'zai', 'authentication', '5', None))
    other = _issue(6, health.IncidentMarker('provider', 'zai', 'quota_unknown', None, None))
    client = FakeGitHub([incident, other])
    monkeypatch.setenv('ZAI_API_KEY', 'zai-key')
    monkeypatch.setattr(health, '_fetch_zai_quota', lambda *_: health.Quota('unknown'))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]

    _reconcile_recovery(client, argparse.Namespace(dry_run=False, recover_issue=5))
    assert client.closed == []

    monkeypatch.setattr(health, '_fetch_zai_quota', lambda *_: health.Quota('healthy'))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    _reconcile_recovery(client, argparse.Namespace(dry_run=False, recover_issue=5))
    assert client.closed == [5]


def test_manual_recovery_rejects_retired_provider_incident(monkeypatch: pytest.MonkeyPatch) -> None:
    """Z.ai health cannot be used to close a retired MiniMax provider incident."""
    issue = _issue(14, health.IncidentMarker('provider', 'minimax', 'authentication', '14', None))
    client = FakeGitHub([issue])
    monkeypatch.setenv('ZAI_API_KEY', 'zai-key')
    checked: list[str] = []

    def fetch_quota(*_args: object) -> health.Quota:
        checked.append('zai')
        return health.Quota('healthy')

    monkeypatch.setattr(health, '_fetch_zai_quota', fetch_quota)

    with pytest.raises(ValueError, match='not a Z.ai provider-health incident'):
        _reconcile_recovery(client, argparse.Namespace(dry_run=False, recover_issue=14))

    assert checked == []
    assert client.closed == []


def test_named_recovery_without_provider_key_stays_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing provider credentials stay unknown even when the fetch stub is healthy."""
    issue = _issue(4, health.IncidentMarker('workflow', 'nightly-sweep', 'timeout', '4', None))
    client = FakeGitHub([issue])
    monkeypatch.delenv('ZAI_API_KEY', raising=False)
    monkeypatch.setattr(health, '_fetch_zai_quota', lambda *_: health.Quota('healthy'))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]

    _reconcile_recovery(client, argparse.Namespace(dry_run=False, recover_issue=4))

    assert client.closed == []


def test_named_workflow_and_task_recovery_also_requires_healthy_quota(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every selected issue stays latched until a fresh quota check is healthy."""
    workflow_issue = _issue(7, health.IncidentMarker('workflow', 'nightly-sweep', 'timeout', '7', None))
    task_issue = _issue(8, health.IncidentMarker('task', 'nightly-sweep:task-1', 'timeout', '8', None))
    client = FakeGitHub([workflow_issue, task_issue])
    monkeypatch.setenv('ZAI_API_KEY', 'zai-key')
    monkeypatch.setattr(health, '_fetch_zai_quota', lambda *_: health.Quota('unknown'))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]

    _reconcile_recovery(client, argparse.Namespace(dry_run=False, recover_issue=7))
    _reconcile_recovery(client, argparse.Namespace(dry_run=False, recover_issue=8))
    assert client.closed == []

    monkeypatch.setattr(health, '_fetch_zai_quota', lambda *_: health.Quota('healthy'))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    _reconcile_recovery(client, argparse.Namespace(dry_run=False, recover_issue=7))
    assert client.closed == [7]
    _reconcile_recovery(client, argparse.Namespace(dry_run=False, recover_issue=8))
    assert client.closed == [7, 8]


def test_marker_from_body_rejects_deeply_nested_json() -> None:
    """Malformed nested issue markers cannot abort monitoring."""
    body = health.MARKER_PREFIX + '{"nested":' + '[' * 10_000 + '0' + ']' * 10_000 + '} -->'

    assert _marker_from_body(body) is None


def test_monitor_ignores_successful_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A healthy successful workflow completion creates no incident."""
    artifact = tmp_path / 'provider-health.json'
    _write_health(
        artifact,
        health.Health(
            'nightly-sweep',
            'task-1',
            'schedule',
            1,
            True,
            'Provider health check passed',
            '2026-10-01T12:00:00Z',
            health.Quota('healthy'),
        ),
    )
    client = FakeGitHub()
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: client)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'token')

    result = _monitor_command(
        argparse.Namespace(
            repository=None,
            run_id=42,
            run_attempt=1,
            conclusion='success',
            health_artifact=artifact,
            agent_artifact=None,
            dry_run=False,
            recover_issue=None,
        )
    )

    assert result == 0
    assert client.posts == []


@pytest.mark.parametrize(('conclusion', 'creates_incident'), [('failure', True), ('success', False)])
def test_monitor_handles_missing_terminal_log(
    conclusion: str, creates_incident: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing terminal log is untyped and follows the trusted workflow outcome."""
    health_artifact = tmp_path / 'provider-health.json'
    _write_health(
        health_artifact,
        health.Health(
            'nightly-sweep',
            'task-1',
            'schedule',
            1,
            True,
            'Provider health check passed',
            '2026-10-01T12:00:00Z',
            health.Quota('healthy'),
        ),
    )
    downloaded_artifact = tmp_path / 'agent-artifact'
    downloaded_artifact.mkdir()
    (downloaded_artifact / 'prompt.txt').write_text('prompt')
    missing_agent_log = downloaded_artifact / 'agent-stdio.log'
    assert not missing_agent_log.exists()

    client = FakeGitHub()
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: client)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'token')

    result = _monitor_command(
        argparse.Namespace(
            repository=None,
            run_id=45,
            run_attempt=1,
            conclusion=conclusion,
            health_artifact=health_artifact,
            agent_artifact=missing_agent_log,
            dry_run=False,
            recover_issue=None,
        )
    )

    assert result == 0
    assert bool(client.posts) is creates_incident
    if creates_incident:
        payload: object = client.posts[0]
        assert _marker_from_payload(payload) == health.IncidentMarker(
            'workflow', 'nightly-sweep', 'execution', '45', None
        )


def test_blocked_run_without_agent_artifact_creates_provider_incident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blocked gate creates an incident even when the model artifact is absent."""
    artifact = tmp_path / 'provider-health.json'
    _write_health(
        artifact,
        health.Health(
            'nightly-sweep',
            'task-1',
            'schedule',
            1,
            False,
            'Z.ai quota is exhausted',
            '2026-10-01T12:00:00Z',
            health.Quota('exhausted', '2026-10-01T11:00:00Z'),
        ),
    )
    client = FakeGitHub()
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: client)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'token')

    result = _monitor_command(
        argparse.Namespace(
            repository=None,
            run_id=43,
            run_attempt=1,
            conclusion='success',
            health_artifact=artifact,
            agent_artifact=None,
            dry_run=False,
            recover_issue=None,
        )
    )

    assert result == 0
    assert len(client.posts) == 1
    payload: object = client.posts[0]
    issue_marker = _marker_from_payload(payload)
    assert issue_marker == health.IncidentMarker('provider', 'zai', 'quota_exhausted', '43', '2026-10-01T11:00:00Z')


def test_monitor_uses_trusted_context_when_terminal_metadata_mismatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Uncorrelated shim metadata falls back to workflow scope from the trusted DTO."""
    health_artifact = tmp_path / 'provider-health.json'
    _write_health(
        health_artifact,
        health.Health(
            'nightly-sweep',
            'task-1',
            'schedule',
            1,
            True,
            'Provider health check passed',
            '2026-10-01T12:00:00Z',
            health.Quota('healthy'),
        ),
    )
    agent_log = tmp_path / 'agent-stdio.log'
    agent_log.write_text(
        json.dumps(
            {
                'type': 'result',
                'provider_health': {
                    'workflow': 'other-workflow',
                    'task_key': 'other-task',
                    'trigger_event': 'workflow_dispatch',
                    'run_attempt': 1,
                    'failure': {'kind': 'authentication'},
                },
            }
        )
    )
    client = FakeGitHub()
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: client)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'token')

    _monitor_command(
        argparse.Namespace(
            repository=None,
            run_id=44,
            run_attempt=1,
            conclusion='failure',
            health_artifact=health_artifact,
            agent_artifact=agent_log,
            dry_run=False,
            recover_issue=None,
        )
    )

    payload: object = client.posts[0]
    assert _marker_from_payload(payload) == health.IncidentMarker('workflow', 'nightly-sweep', 'execution', '44', None)
