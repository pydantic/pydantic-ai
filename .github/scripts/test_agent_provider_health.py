"""Focused boundaries for the Z.ai inference gate and incident controller."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import ssl
import sys
import threading
from dataclasses import replace
from email.message import Message
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
    _monitor_workflow_outcome,  # pyright: ignore[reportPrivateUsage]
    _parse_zai_quota,  # pyright: ignore[reportPrivateUsage]
    _reconcile_recovery,  # pyright: ignore[reportPrivateUsage]
    _run_result_from,  # pyright: ignore[reportPrivateUsage]
    _scope_for,  # pyright: ignore[reportPrivateUsage]
    _write_health,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai_gh_aw_shim import cli as shim


@pytest.fixture(autouse=True)
def _fake_github_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the synthetic workflow run ID unless a test sets its own."""
    monkeypatch.setenv('GITHUB_RUN_ID', '43')


@pytest.fixture(autouse=True)
def _no_incident_reread_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-read open incidents without waiting unless a test records the delay."""
    monkeypatch.setattr(health, '_INCIDENT_READ_DELAY_SECONDS', 0)


@pytest.fixture(autouse=True)
def _no_ci_step_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep controller output out of the summary of the CI job running these tests."""
    monkeypatch.delenv('GITHUB_STEP_SUMMARY', raising=False)


def _quota_response(
    *,
    limit_type: str = 'TOKENS_LIMIT',
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
                'type': limit_type,
                'unit': 3,
                'number': 5,
                'percentage': interval_percentage,
                'nextResetTime': interval_reset,
            },
            {
                'type': limit_type,
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


def _issue(
    number: int, marker: health.IncidentMarker, *, body_prefix: str = '', created_at: str | None = None
) -> health.Issue:
    marker_json = json.dumps(
        {
            'version': 1,
            'scope': marker.scope,
            'key': marker.key,
            'kind': marker.kind,
            'run_id': marker.run_id,
            'reset_at': marker.reset_at,
            **({'stage': marker.stage} if marker.stage is not None else {}),
            **({'run_attempt': marker.run_attempt} if marker.run_attempt is not None else {}),
        },
        separators=(',', ':'),
    )
    body = f'{body_prefix}{health.MARKER_PREFIX}{marker_json} -->\n'
    return health.Issue(
        number, 'Agent workflow incident', body, 'open', f'https://github.com/org/repo/issues/{number}', created_at
    )


def _marker_from_payload(payload: object) -> health.IncidentMarker | None:
    data = _mapping(payload)
    body = data.get('body') if data is not None else None
    if not isinstance(body, str):
        return None
    return _marker_from_body(body)


@pytest.mark.parametrize('stage', ['setup', 'validation', 'publication', 'queue', 'inference'])
def test_host_stage_incident_body_points_to_stage_logs(stage: health.Stage):
    """Point trusted host-stage incidents to their logs without changing recovery metadata."""
    marker = health.IncidentMarker('task', 'nightly-sweep:pr-45', 'host_failure', '43', None, stage, 2)

    body = health._incident_body(marker, 'nightly-sweep', 'pr-45', '43', 'org/repo', 'Host stage failed')  # pyright: ignore[reportPrivateUsage]

    assert f'Inspect the trusted {stage} stage logs and correct the host-side failure.' in body
    assert 'Close this incident after the cause is fixed and the required recovery check passes.' in body
    assert _marker_from_body(body) == marker


class FakeGitHub(health.GitHubClient):
    """Capture issue requests in memory for reconciler boundary tests."""

    def __init__(self, issues: list[health.Issue] | None = None) -> None:
        super().__init__('org/repo', 'token')
        self.issues = list(issues or [])
        self.posts: list[object] = []
        self.closed: list[int] = []
        self.source_event = 'workflow_dispatch'
        self.source_status = 'completed'
        self.source_conclusion = 'success'
        self.source_workflow_id = 5
        self.source_workflow_path = '.github/workflows/nightly-sweep.md'
        self.original_workflow_id = 5
        self.source_name = 'nightly-sweep'
        self.comments: list[tuple[int, str]] = []
        self.jobs: dict[tuple[str, int], list[dict[str, object]]] = {}
        self.pull_requests: dict[int, dict[str, object]] = {}
        self.commit_pulls: dict[str, list[dict[str, object]]] = {}
        self.reviews: dict[int, list[dict[str, object]]] = {}
        self.default_branch_calls = 0

    def request(self, method: str, path: str, payload: object | None = None) -> object:
        if method == 'GET':
            return [
                {
                    'number': item.number,
                    'title': item.title,
                    'body': item.body,
                    'state': item.state,
                    'html_url': item.html_url,
                    'created_at': item.created_at,
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
                None,
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

    def send(self, method: str, path: str, payload: object | None = None) -> health.GitHubResponse:
        return health.GitHubResponse(200, 'fake-request', self.request(method, path, payload))

    def attempt_jobs(self, run_id: str, attempt: int) -> list[dict[str, object]]:
        return self.jobs.get((run_id, attempt), [])

    def default_branch(self) -> str | None:
        self.default_branch_calls += 1
        return 'main'

    def open_incidents(self) -> list[health.Issue]:
        return self.issues


class FakeHTTPResponse:
    """A small Z.ai response for the trusted-runner command tests."""

    def __init__(
        self, payload: dict[str, object] | list[object] | bytes, status: int = 200, request_id: str | None = None
    ) -> None:
        self.payload = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.status = status
        self.headers = Message()
        if request_id is not None:
            self.headers['X-GitHub-Request-Id'] = request_id

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


@pytest.mark.parametrize('limit_type', ['TOKENS_LIMIT', 'CREDIT_LIMIT'])
def test_zai_quota_parses_observed_windows_and_ignores_mcp_limit(limit_type: str) -> None:
    """The observed Coding Plan windows map into the trusted quota DTO."""
    quota = _parse_zai_quota(_quota_response(limit_type=limit_type))

    assert quota.status == 'healthy'
    assert quota.interval_remaining_percent == 99
    assert quota.weekly_remaining_percent == 84
    assert quota.interval_reset_at == '2026-10-03T07:35:26Z'
    assert quota.weekly_reset_at == '2026-10-08T14:54:29Z'
    assert quota.reset_at is None


def test_zai_quota_exact_null_reset_probe_round_trips_exhausted_weekly_health(tmp_path: Path) -> None:
    """The observed idle interval and exhausted weekly row retain known quota state."""
    payload: dict[str, object] = {
        'code': 200,
        'success': True,
        'data': {
            'limits': [
                {
                    'type': 'TOKENS_LIMIT',
                    'unit': 3,
                    'number': 5,
                    'percentage': 0,
                    'nextResetTime': None,
                },
                {
                    'type': 'TOKENS_LIMIT',
                    'unit': 6,
                    'number': 1,
                    'percentage': 100,
                    'nextResetTime': 1_792_517_882_984,
                },
            ]
        },
    }

    quota = _parse_zai_quota(payload)
    decision = health.check_health('nightly-sweep', 'task-1', 'schedule', 1, quota, [])
    path = tmp_path / 'provider-health.json'
    _write_health(path, decision)
    result = _health_from_json(json.loads(path.read_text()))

    assert not result.ready
    assert result.quota.status == 'exhausted'
    assert result.quota.interval_remaining_percent == 100
    assert result.quota.interval_reset_at is None
    assert result.quota.weekly_remaining_percent == 0
    assert result.quota.weekly_reset_at == '2026-10-20T17:38:02Z'
    assert result.quota.reset_at == '2026-10-20T17:38:02Z'


@pytest.mark.parametrize('percentage', [0, 1, 100])
def test_zai_quota_missing_reset_is_valid_only_on_idle_window(percentage: int) -> None:
    """An omitted reset is valid only before the token window has consumed quota."""
    payload: dict[str, object] = {
        'code': 200,
        'success': True,
        'data': {
            'limits': [
                {'type': 'TOKENS_LIMIT', 'unit': 3, 'number': 5, 'percentage': percentage},
                {
                    'type': 'TOKENS_LIMIT',
                    'unit': 6,
                    'number': 1,
                    'percentage': 16,
                    'nextResetTime': 1_791_471_269_972,
                },
            ]
        },
    }

    quota = _parse_zai_quota(payload)

    decision = health.check_health('nightly-sweep', 'task-1', 'schedule', 1, quota, [])

    if percentage == 0:
        assert quota.status == 'healthy'
        assert quota.interval_remaining_percent == 100
        assert quota.interval_reset_at is None
        assert quota.weekly_remaining_percent == 84
        assert quota.weekly_reset_at == '2026-10-08T14:54:29Z'
        assert quota.diagnostic is None
        assert decision.ready
    else:
        assert quota.status == 'unknown'
        assert quota.diagnostic == 'invalid_reset_time'
        assert not decision.ready


def test_zai_quota_idle_interval_with_remaining_weekly_quota_is_healthy() -> None:
    """An idle interval without a reset does not invalidate a non-exhausted weekly quota."""
    quota = _parse_zai_quota(_quota_response(interval_percentage=0, weekly_percentage=16, interval_reset=None))
    decision = health.check_health('nightly-sweep', 'task-1', 'schedule', 1, quota, [])

    assert quota.status == 'healthy'
    assert quota.interval_remaining_percent == 100
    assert quota.interval_reset_at is None
    assert quota.weekly_remaining_percent == 84
    assert quota.weekly_reset_at == '2026-10-08T14:54:29Z'
    assert decision.ready


def test_zai_quota_null_reset_on_consumed_window_stays_invalid() -> None:
    """A consumed or exhausted token window still requires a valid reset timestamp."""
    for percentage in (1, 100):
        quota = _parse_zai_quota(_quota_response(interval_percentage=percentage, interval_reset=None))
        assert quota.status == 'unknown'
        assert quota.diagnostic == 'invalid_reset_time'


@pytest.mark.parametrize('limit_type', ['TOKENS_LIMIT', 'CREDIT_LIMIT'])
@pytest.mark.parametrize(
    ('interval_percentage', 'weekly_percentage', 'expected_reset'),
    [
        (100, 84, '2026-10-03T07:35:26Z'),
        (1, 100, '2026-10-08T14:54:29Z'),
        (100, 100, '2026-10-08T14:54:29Z'),
    ],
)
def test_zai_quota_exhaustion_uses_latest_exhausted_window_reset(
    limit_type: str, interval_percentage: object, weekly_percentage: object, expected_reset: str
) -> None:
    """Exhausted windows keep the latest reset needed to recover."""
    quota = _parse_zai_quota(
        _quota_response(
            limit_type=limit_type, interval_percentage=interval_percentage, weekly_percentage=weekly_percentage
        )
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


@pytest.mark.parametrize(
    ('payload', 'diagnostic'),
    [
        ([], 'invalid_response'),
        ({'code': 201}, 'invalid_response_code'),
        (_quota_response(success=False), 'unsuccessful_response'),
        ({'code': 200, 'success': True, 'data': {}}, 'invalid_limits'),
        (_quota_response(limits=[None]), 'invalid_limit_row'),
        (_quota_response(limits=[{'type': 'TOKENS_LIMIT', 'unit': 2}]), 'invalid_window_unit'),
        (
            _quota_response(limits=[{'type': 'TOKENS_LIMIT', 'unit': 3, 'number': 4}]),
            'invalid_window_number',
        ),
        (
            _quota_response(
                limits=[
                    {
                        'type': 'TOKENS_LIMIT',
                        'unit': 3,
                        'number': 5,
                        'percentage': 1,
                        'nextResetTime': 1_791_012_926_973,
                    },
                    {
                        'type': 'TOKENS_LIMIT',
                        'unit': 3,
                        'number': 5,
                        'percentage': 1,
                        'nextResetTime': 1_791_012_926_973,
                    },
                ]
            ),
            'duplicate_window',
        ),
        (_quota_response(interval_percentage=None), 'invalid_percentage'),
        (_quota_response(interval_percentage=101), 'invalid_percentage'),
        (_quota_response(interval_percentage=10**1000), 'invalid_percentage'),
        (_quota_response(interval_reset=10**1000), 'invalid_reset_time'),
        (_quota_response(limits=[]), 'missing_windows'),
        (_quota_response(limit_type='CREDIT_LIMIT', interval_percentage=None), 'invalid_percentage'),
        (_quota_response(limit_type='CREDIT_LIMIT', interval_reset=None), 'invalid_reset_time'),
        (
            _quota_response(
                limits=[
                    {
                        'type': 'CREDIT_LIMIT',
                        'unit': 3,
                        'number': 5,
                        'percentage': 1,
                        'nextResetTime': 1_792_478_501_101,
                    },
                ]
            ),
            'missing_windows',
        ),
    ],
)
def test_zai_quota_parser_attaches_fixed_diagnostic(payload: object, diagnostic: str) -> None:
    """Malformed response branches expose only their fixed diagnostic code."""
    quota = _parse_zai_quota(payload)

    assert quota.status == 'unknown'
    assert quota.diagnostic == diagnostic


def test_zai_quota_probe_returns_unknown_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-error HTTP response with the wrong status remains unknown."""
    response = FakeHTTPResponse(_quota_response(), status=503)
    monkeypatch.setattr(health.urllib.request.OpenerDirector, 'open', lambda *_args, **_kwargs: response)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]

    quota = _fetch_zai_quota('private-zai-fixture-key')

    assert quota.status == 'unknown'
    assert quota.diagnostic == 'invalid_response_code'
    assert quota.http_status == 503


@pytest.mark.parametrize('status', [401, 429, 503])
def test_zai_quota_probe_records_only_http_error_category_and_status(
    status: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HTTP error metadata is reduced to a fixed category and numeric status."""
    sentinel = 'provider-private-response-sentinel'
    headers = Message()
    headers['X-Private'] = sentinel
    error = health.urllib.error.HTTPError(
        health.ZAI_QUOTA_URL,
        status,
        f'reason {sentinel}',
        headers,
        None,
    )

    def open_url(*_args: object, **_kwargs: object) -> FakeHTTPResponse:
        raise error

    monkeypatch.setattr(health.urllib.request.OpenerDirector, 'open', open_url)

    quota = _fetch_zai_quota(f'key-{sentinel}')

    assert quota.status == 'unknown'
    assert quota.diagnostic == 'http_error'
    assert quota.http_status == status
    assert sentinel not in repr(quota)


@pytest.mark.parametrize(
    ('error', 'diagnostic'),
    [
        (TimeoutError('private timeout message'), 'timeout'),
        (ssl.SSLError('private TLS message'), 'tls_error'),
        (OSError('private network message'), 'network_error'),
        (health.urllib.error.URLError(TimeoutError('wrapped timeout secret')), 'timeout'),
        (health.urllib.error.URLError(ssl.SSLError('wrapped TLS secret')), 'tls_error'),
        (health.urllib.error.URLError('private network message'), 'network_error'),
    ],
)
def test_zai_quota_probe_returns_unknown_on_transport_error(
    error: OSError | health.urllib.error.URLError, diagnostic: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Transport exceptions map to fixed network diagnostic categories."""

    def open_url(_opener: health.urllib.request.OpenerDirector, *_args: object, **_kwargs: object) -> FakeHTTPResponse:
        raise error

    monkeypatch.setattr(health.urllib.request.OpenerDirector, 'open', open_url)

    quota = _fetch_zai_quota('private-zai-fixture-key')

    assert quota.status == 'unknown'
    assert quota.diagnostic == diagnostic
    assert quota.http_status is None


def test_zai_quota_probe_returns_unknown_on_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed response JSON reports its fixed category and successful HTTP status."""
    response = FakeHTTPResponse(b'{')
    monkeypatch.setattr(health.urllib.request.OpenerDirector, 'open', lambda *_args, **_kwargs: response)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]

    quota = _fetch_zai_quota('private-zai-fixture-key')

    assert quota.status == 'unknown'
    assert quota.diagnostic == 'invalid_json'
    assert quota.http_status == 200


def test_zai_quota_probe_classifies_encoding_error_without_payload_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid UTF-8 reports a fixed category without including response bytes."""
    response = FakeHTTPResponse(b'\xffprivate-response-sentinel')
    monkeypatch.setattr(health.urllib.request.OpenerDirector, 'open', lambda *_args, **_kwargs: response)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]

    quota = _fetch_zai_quota('private-zai-fixture-key')

    assert quota.diagnostic == 'invalid_encoding'
    assert quota.http_status == 200
    assert 'private-response-sentinel' not in repr(quota)


def test_zai_quota_probe_does_not_request_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """An absent key returns a diagnostic without making a network request."""

    def unexpected_open(*_args: object, **_kwargs: object) -> FakeHTTPResponse:
        raise AssertionError('missing key must not make a request')

    monkeypatch.setattr(health.urllib.request.OpenerDirector, 'open', unexpected_open)

    assert _fetch_zai_quota('').diagnostic == 'missing_api_key'


def test_zai_quota_probe_does_not_forward_authorization_on_cross_origin_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The quota probe never forwards credentials across origins."""
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
    """Oversized response integers produce an unknown quota diagnostic."""
    response = FakeHTTPResponse(b'{"code":' + b'9' * 5000 + b'}')

    def open_url(*_args: object, **_kwargs: object) -> FakeHTTPResponse:
        return response

    monkeypatch.setattr(health.urllib.request.OpenerDirector, 'open', open_url)

    assert _fetch_zai_quota('private-zai-fixture-key').status == 'unknown'


def test_check_writes_blocked_artifact_for_out_of_range_zai_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unrepresentable provider reset time blocks the gate without losing its artifact."""
    payload = _quota_response(interval_reset=10**1000)

    def open_url(*_args: object, **_kwargs: object) -> FakeHTTPResponse:
        return FakeHTTPResponse(payload)

    monkeypatch.setattr(health.urllib.request.OpenerDirector, 'open', open_url)
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
    assert result.quota.diagnostic == 'invalid_reset_time'
    assert result.quota.http_status == 200
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


def test_fresh_manual_task_identity_does_not_match_old_pr_task_incident() -> None:
    """A new manual run has a distinct task key while workflow and provider incidents still block."""
    workflow = 'CI Review'
    old_task_key = f'{workflow}:workflow_run:feature-branch:head-a1b2:'
    fresh_task_key = f'{workflow}:workflow_dispatch:main:head-a1b2:12345'
    quota = health.Quota('healthy')
    old_task_incident = _issue(21, health.IncidentMarker('task', f'{workflow}:{old_task_key}', 'timeout', '90', None))
    old_task_result = health.check_health(workflow, old_task_key, 'workflow_run', 1, quota, [old_task_incident])
    fresh_manual_result = health.check_health(
        workflow, fresh_task_key, 'workflow_dispatch', 1, quota, [old_task_incident]
    )

    assert not old_task_result.ready
    assert fresh_manual_result.ready
    workflow_incident = _issue(22, health.IncidentMarker('workflow', workflow, 'execution', '91', None))
    provider_incident = _issue(23, health.IncidentMarker('provider', 'zai', 'rate_limit', '92', None))
    assert not health.check_health(workflow, fresh_task_key, 'workflow_dispatch', 1, quota, [workflow_incident]).ready
    assert not health.check_health(workflow, fresh_task_key, 'workflow_dispatch', 1, quota, [provider_incident]).ready


def test_health_recovery_bypasses_only_the_selected_single_workflow_incident() -> None:
    """A healthy gate passes one explicitly selected workflow incident only."""
    incident = _issue(7, health.IncidentMarker('workflow', 'nightly-sweep', 'execution', '8', None))
    selected = health.Health(
        'nightly-sweep',
        'task-1',
        'workflow_dispatch',
        1,
        True,
        '',
        '2026-10-01T12:00:00Z',
        health.Quota('healthy'),
        recovery_issue=7,
    )

    assert health._health_decision(selected, [incident]).ready  # pyright: ignore[reportPrivateUsage]
    other_incident = _issue(8, health.IncidentMarker('workflow', 'nightly-sweep', 'execution', '9', None))
    assert not health._health_decision(selected, [incident, other_incident]).ready  # pyright: ignore[reportPrivateUsage]
    task = _issue(7, health.IncidentMarker('task', 'nightly-sweep:task-1', 'timeout', '8', None))
    assert not health._health_decision(selected, [task]).ready  # pyright: ignore[reportPrivateUsage]
    provider = _issue(7, health.IncidentMarker('provider', 'zai', 'rate_limit', '8', None))
    assert not health._health_decision(selected, [provider]).ready  # pyright: ignore[reportPrivateUsage]


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
            diagnostic='invalid_json',
            http_status=200,
        ),
        '45',
        13,
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
    payload_data['quota'] = quota_data
    assert _health_from_json(payload_data) == original

    del quota_data['diagnostic']
    del quota_data['http_status']
    legacy = _health_from_json(payload_data)
    assert legacy.quota.diagnostic is None
    assert legacy.quota.http_status is None


@pytest.mark.parametrize(
    ('diagnostic', 'http_status', 'expected_diagnostic', 'expected_http_status'),
    [
        ('private provider message', 200, None, 200),
        ('invalid_json', True, 'invalid_json', None),
        ('invalid_json', 99, 'invalid_json', None),
        ('invalid_json', 600, 'invalid_json', None),
    ],
)
def test_health_artifact_discards_invalid_optional_diagnostics(
    diagnostic: object, http_status: object, expected_diagnostic: str | None, expected_http_status: int | None
) -> None:
    """Invalid optional metadata is discarded without rejecting the artifact."""
    payload: dict[str, object] = {
        'version': 1,
        'workflow': 'nightly-sweep',
        'task_key': 'task-1',
        'trigger_event': 'schedule',
        'run_attempt': 1,
        'ready': False,
        'reason': 'Z.ai quota health is unknown',
        'checked_at': '2026-10-01T12:00:00Z',
        'quota': {'status': 'unknown', 'diagnostic': diagnostic, 'http_status': http_status},
    }

    result = _health_from_json(payload)

    assert result.quota.status == 'unknown'
    assert result.quota.diagnostic == expected_diagnostic
    assert result.quota.http_status == expected_http_status


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


def test_result_parser_rejects_ambiguous_agent_artifacts(tmp_path: Path) -> None:
    """Two distinct terminal records from eligible agent logs do not select an authority."""
    first_log = tmp_path / 'agent' / 'agent-stdio.log'
    second_log = tmp_path / 'c84aadcc-agent' / 'agent-stdio.log'
    first_log.parent.mkdir(parents=True)
    second_log.parent.mkdir(parents=True)
    first_log.write_text(
        json.dumps(
            {
                'type': 'result',
                'provider_health': {
                    'workflow': 'nightly-sweep',
                    'task_key': 'task-1',
                    'trigger_event': 'schedule',
                    'run_attempt': 1,
                },
            }
        )
    )
    assert health.parse_run_result(tmp_path) == health.RunResult('nightly-sweep', 'task-1', 'schedule', 1, None)

    first_log.unlink()
    second_log.write_text(
        json.dumps(
            {
                'type': 'result',
                'provider_health': {
                    'workflow': 'nightly-sweep',
                    'task_key': 'task-1',
                    'trigger_event': 'schedule',
                    'run_attempt': 2,
                },
            }
        )
    )
    assert health.parse_run_result(tmp_path) == health.RunResult('nightly-sweep', 'task-1', 'schedule', 2, None)

    first_log.write_text(
        json.dumps(
            {
                'type': 'result',
                'provider_health': {
                    'workflow': 'nightly-sweep',
                    'task_key': 'task-1',
                    'trigger_event': 'schedule',
                    'run_attempt': 1,
                },
            }
        )
    )
    assert health.parse_run_result(tmp_path) is None

    second_log.write_text(
        json.dumps(
            {
                'type': 'result',
                'provider_health': {
                    'workflow': 'nightly-sweep',
                    'task_key': 'task-1',
                    'trigger_event': 'schedule',
                    'run_attempt': 1,
                    'failure': {'kind': 'timeout'},
                },
            }
        )
    )
    assert health.parse_run_result(tmp_path, expected_attempt=1) is None

    second_log.unlink()
    second_log.write_text(
        '\n'.join(
            [
                json.dumps(
                    {
                        'type': 'result',
                        'provider_health': {
                            'workflow': 'nightly-sweep',
                            'task_key': 'task-1',
                            'trigger_event': 'schedule',
                            'run_attempt': 1,
                        },
                    }
                ),
                json.dumps(
                    {
                        'type': 'result',
                        'provider_health': {
                            'workflow': 'nightly-sweep',
                            'task_key': 'task-1',
                            'trigger_event': 'schedule',
                            'run_attempt': 1,
                            'failure': {'kind': 'timeout'},
                        },
                    }
                ),
            ]
        )
    )
    assert health.parse_run_result(second_log, expected_attempt=1) is None


@pytest.mark.parametrize(
    ('failure', 'expected_scope', 'expected_kind'),
    [
        ({'kind': 'timeout'}, None, None),
        ({'kind': 'rate_limit', 'http_status': 429}, None, None),
        ({'kind': 'plan_limit', 'http_status': 429}, 'provider', 'plan_limit'),
    ],
)
def test_monitor_selects_current_attempt_before_resolving_agent_artifacts(
    failure: dict[str, object] | None,
    expected_scope: health.Scope | None,
    expected_kind: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An older terminal artifact cannot hide the triggering attempt's typed result."""
    workflow = 'CI Review'
    task_key = 'CI Review:workflow_run:pr-123:head-a1b2'
    artifacts = tmp_path / 'downloaded-agent-artifacts'
    for prefix, attempt, artifact_failure in (
        ('a1b2c3d4', 1, {'kind': 'rate_limit', 'http_status': 429}),
        ('e5f6a7b8', 2, failure),
    ):
        artifact_dir = artifacts / f'{prefix}-agent'
        artifact_dir.mkdir(parents=True)
        provider_health: dict[str, object] = {
            'workflow': workflow,
            'task_key': task_key,
            'trigger_event': 'workflow_run',
            'run_attempt': attempt,
        }
        if artifact_failure is not None:
            provider_health['failure'] = artifact_failure
        (artifact_dir / 'agent-stdio.log').write_text(
            json.dumps({'type': 'result', 'provider_health': provider_health})
        )

    health_artifact = tmp_path / 'provider-health.json'
    _write_health(
        health_artifact,
        health.Health(
            workflow,
            task_key,
            'workflow_run',
            2,
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

    assert (
        _monitor_command(
            argparse.Namespace(
                repository=None,
                run_id=85,
                run_attempt=2,
                conclusion='failure',
                health_artifact=health_artifact,
                agent_artifact=artifacts,
                dry_run=False,
                recover_issue=None,
            )
        )
        == 0
    )

    if expected_scope is None:
        assert client.posts == []
        assert client.issues == []
    else:
        assert expected_kind is not None
        [issue] = client.issues
        assert _marker_from_body(issue.body) == health.IncidentMarker(
            expected_scope, 'zai', expected_kind, '85', None, run_attempt=2
        )


def test_monitor_rejects_current_attempt_artifact_identity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A terminal result from another task cannot be trusted for the triggering run."""
    health_artifact = tmp_path / 'provider-health.json'
    _write_health(
        health_artifact,
        health.Health(
            'CI Review',
            'current-task',
            'workflow_run',
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
                    'workflow': 'CI Review',
                    'task_key': 'different-task',
                    'trigger_event': 'workflow_run',
                    'run_attempt': 2,
                    'failure': {'kind': 'rate_limit', 'http_status': 429},
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
            run_id=86,
            run_attempt=2,
            conclusion='failure',
            health_artifact=health_artifact,
            agent_artifact=agent_log,
            dry_run=False,
            recover_issue=None,
        )
    )

    assert client.posts == []
    assert client.issues == []


def test_result_parser_does_not_treat_session_result_as_terminal_authority(tmp_path: Path) -> None:
    """An engine session result cannot stand in for the authenticated wrapper terminal record."""
    session_result = tmp_path / 'session.result'
    session_result.write_text(
        json.dumps(
            {
                'type': 'result',
                'provider_health': {
                    'workflow': 'nightly-sweep',
                    'task_key': 'task-1',
                    'trigger_event': 'schedule',
                    'run_attempt': 1,
                },
            }
        )
    )

    assert health.parse_run_result(session_result) is None


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
        'failure': {'kind': 'plan_limit', 'http_status': 429},
    }
    usage = result_event['usage']
    assert isinstance(usage, dict)
    assert usage['input_tokens'] == 23
    assert usage['output_tokens'] == 9
    assert usage['cache_read_input_tokens'] == 4
    assert usage['cache_creation_input_tokens'] == 2
    assert health.parse_run_result(agent_artifact) == health.RunResult(
        workflow, task_key, trigger_event, 1, health.Failure('plan_limit', 429, None)
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
    assert marker == health.IncidentMarker('provider', 'zai', 'plan_limit', '70', None, run_attempt=1)
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


def test_shim_time_limit_opens_a_task_incident_that_does_not_blame_the_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run that used its whole time budget is reported as its own limit, not a provider timeout."""
    workflow = 'CI Review'
    task_key = 'CI Review:workflow_run:pr-123:head-a1b2'
    monkeypatch.setenv('GITHUB_WORKFLOW', workflow)
    monkeypatch.setenv('PYDANTIC_AI_TASK_KEY', task_key)
    monkeypatch.setenv('PYDANTIC_AI_TRIGGER_EVENT', 'workflow_run')
    monkeypatch.setenv('PYDANTIC_AI_RUN_ATTEMPT', '1')
    shim.emit_result(
        'run timed out after 28min', usage=None, session_id='s', is_error=True, error=shim.RunTimeLimitExceeded()
    )
    agent_artifact = tmp_path / 'agent-stdio.log'
    agent_artifact.write_text(capsys.readouterr().out)
    health_artifact = tmp_path / 'provider-health.json'
    _write_health(
        health_artifact,
        health.Health(
            workflow,
            task_key,
            'workflow_run',
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

    _monitor_command(
        argparse.Namespace(
            repository=None,
            run_id=70,
            run_attempt=1,
            conclusion='failure',
            health_artifact=health_artifact,
            agent_artifact=agent_artifact,
            dry_run=False,
            recover_issue=None,
        )
    )

    [issue] = client.issues
    assert _marker_from_body(issue.body) == health.IncidentMarker(
        'task', f'{workflow}:{task_key}', 'time_limit', '70', None, run_attempt=1
    )
    assert issue.body.startswith('## Operational failure\n\nThe agent reached its per-run time limit\n\n')
    assert 'The run stopped at its own limit, not on a provider error.' in issue.body
    assert 'Confirm provider credentials' not in issue.body


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


def test_monitor_rejects_health_artifact_from_different_run_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A health artifact from another run is rejected before incident writes."""
    health_artifact = tmp_path / 'provider-health.json'
    _write_health(
        health_artifact,
        health.Health(
            'nightly-sweep',
            'stable-task-key',
            'schedule',
            2,
            False,
            'Z.ai quota is exhausted',
            '2026-10-01T12:00:00Z',
            health.Quota('exhausted'),
            run_id='83',
        ),
    )
    client = FakeGitHub()
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: client)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'token')

    with pytest.raises(ValueError, match='does not match the triggering workflow_run'):
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
    assert client.comments == []
    assert client.closed == []


@pytest.mark.parametrize(
    ('payload', 'expected_ready'),
    [
        (_quota_response(), True),
        (_quota_response(limit_type='CREDIT_LIMIT'), True),
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


@pytest.mark.parametrize(('ref', 'selected'), [('refs/heads/main', 7), ('refs/heads/topic', None)])
def test_check_command_accepts_recovery_issue_only_on_default_dispatch_ref(
    ref: str, selected: int | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recovery issue is carried into the artifact only for a default-branch dispatch."""
    issue = _issue(7, health.IncidentMarker('workflow', 'nightly-sweep', 'inference', '90', None, 'inference'))
    client = FakeGitHub([issue])
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: client)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setattr(health, '_fetch_zai_quota', lambda *_: health.Quota('healthy'))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setenv('ZAI_API_KEY', 'zai-key')
    monkeypatch.setenv('GITHUB_WORKFLOW', 'nightly-sweep')
    monkeypatch.setenv('PYDANTIC_AI_TASK_KEY', 'task-1')
    monkeypatch.setenv('PYDANTIC_AI_TRIGGER_EVENT', 'workflow_dispatch')
    monkeypatch.setenv('PYDANTIC_AI_RUN_ATTEMPT', '1')
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'github-token')
    monkeypatch.setenv('GITHUB_EVENT_NAME', 'workflow_dispatch')
    monkeypatch.setenv('GITHUB_REF', ref)
    monkeypatch.setenv('GITHUB_RUN_ID', '101')
    monkeypatch.setenv('PYDANTIC_AI_RECOVERY_ISSUE', '7')
    output = tmp_path / 'provider-health.json'

    assert health.main(['check', '--output', str(output)]) == 0
    decision = _health_from_json(json.loads(output.read_text()))
    assert decision.recovery_issue == selected
    assert client.default_branch_calls == 1
    assert decision.run_id == '101'
    assert decision.ready is (selected is not None)


@pytest.mark.parametrize('blocker', ['provider_incident', 'quota_exhausted'])
def test_agent_start_recheck_blocks_a_stale_ready_gate_and_replaces_its_decision(
    blocker: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A queued agent acts on provider state when it starts, and the monitor reconciles that decision."""
    client = FakeGitHub()
    quota = health.Quota('healthy')
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: client)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setattr(health, '_fetch_zai_quota', lambda *_: quota)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setenv('ZAI_API_KEY', 'zai-key')
    monkeypatch.setenv('GITHUB_WORKFLOW', 'CI Review')
    monkeypatch.setenv('PYDANTIC_AI_TASK_KEY', 'CI Review:workflow_run:feature:head-a1b2:')
    monkeypatch.setenv('PYDANTIC_AI_TRIGGER_EVENT', 'workflow_run')
    monkeypatch.setenv('PYDANTIC_AI_RUN_ATTEMPT', '1')
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'github-token')
    github_output = tmp_path / 'output'
    monkeypatch.setenv('GITHUB_OUTPUT', str(github_output))
    gate_decision = tmp_path / 'gate' / 'provider-health.json'
    start_decision = tmp_path / 'agent-start' / 'provider-health.json'

    assert health.main(['check', '--output', str(gate_decision)]) == 0
    assert github_output.read_text() == 'ready=true\nreason=Provider health check passed\n'

    if blocker == 'provider_incident':
        client.issues.append(_issue(10572, health.IncidentMarker('provider', 'zai', 'rate_limit', '39', None)))
    else:
        quota = health.Quota('exhausted', reset_at='2026-10-24T00:00:00Z')
    github_output.write_text('')
    assert health.main(['check', '--output', str(start_decision)]) == 0
    assert github_output.read_text().startswith('ready=false\n')

    # The stop step fails the agent job after the recheck and before the engine step.
    client.jobs[('43', 1)] = [
        {
            'name': 'agent',
            'conclusion': 'failure',
            'steps': [
                {'name': health.AGENT_START_HEALTH_STEPS[0], 'conclusion': 'success'},
                {'name': health.AGENT_START_HEALTH_STEPS[1], 'conclusion': 'success'},
                {'name': health.AGENT_START_HEALTH_STEPS[2], 'conclusion': 'failure'},
                {'name': 'Execute Claude Code CLI', 'conclusion': 'skipped'},
            ],
        }
    ]
    agent_artifact = tmp_path / 'agent'
    agent_artifact.mkdir()
    (agent_artifact / 'agent-stdio.log').write_text('')

    def monitor(decision: Path) -> list[object]:
        client.posts.clear()
        args = argparse.Namespace(
            repository=None,
            run_id=43,
            run_attempt=1,
            conclusion='failure',
            health_artifact=decision,
            agent_artifact=agent_artifact,
            dry_run=False,
            recover_issue=None,
        )
        assert _monitor_command(args) == 0
        return list(client.posts)

    # Read against the stale gate decision, as when the replacement upload fails, the stop opens nothing.
    assert monitor(gate_decision) == []
    posts = monitor(start_decision)
    if blocker == 'provider_incident':
        assert posts == []
    else:
        assert [_marker_from_payload(post) for post in posts] == [
            health.IncidentMarker('provider', 'zai', 'quota_exhausted', '43', '2026-10-24T00:00:00Z')
        ]


def test_fresh_dispatch_task_key_runs_without_recovery_issue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Manual task keys include the new run ID, so old task incidents do not block them."""
    task_key = 'CI Review:workflow_dispatch:main:head-a1b2:38977267972'
    task_issue = _issue(
        8,
        health.IncidentMarker(
            'task', 'CI Review:CI Review:workflow_dispatch:main:head-a1b2:38977267971', 'timeout', '90', None
        ),
    )
    client = FakeGitHub([task_issue])
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: client)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setattr(health, '_fetch_zai_quota', lambda *_: health.Quota('healthy'))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setenv('ZAI_API_KEY', 'zai-key')
    monkeypatch.setenv('GITHUB_WORKFLOW', 'CI Review')
    monkeypatch.setenv('PYDANTIC_AI_TASK_KEY', task_key)
    monkeypatch.setenv('PYDANTIC_AI_TRIGGER_EVENT', 'workflow_dispatch')
    monkeypatch.setenv('PYDANTIC_AI_RUN_ATTEMPT', '1')
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'github-token')
    monkeypatch.setenv('GITHUB_EVENT_NAME', 'workflow_dispatch')
    monkeypatch.setenv('GITHUB_REF', 'refs/heads/main')
    monkeypatch.setenv('GITHUB_RUN_ID', '38977267972')
    monkeypatch.delenv('PYDANTIC_AI_RECOVERY_ISSUE', raising=False)
    output = tmp_path / 'provider-health.json'

    assert health.main(['check', '--output', str(output)]) == 0
    decision = _health_from_json(json.loads(output.read_text()))
    assert decision.ready
    assert decision.recovery_issue is None


@pytest.mark.parametrize(
    ('event_name', 'raw_issue', 'ref', 'lookup_default_branch'),
    [
        ('workflow_dispatch', None, 'refs/heads/main', False),
        ('workflow_dispatch', '', 'refs/heads/main', False),
        ('workflow_dispatch', '0', 'refs/heads/main', False),
        ('workflow_dispatch', '-7', 'refs/heads/main', False),
        ('workflow_dispatch', 'bad', 'refs/heads/main', False),
        ('workflow_dispatch', '７', 'refs/heads/main', False),
        ('workflow_dispatch', '7' * 5000, 'refs/heads/main', False),
        ('workflow_run', '7', 'refs/heads/main', False),
        ('workflow_dispatch', '7', 'refs/heads/topic', True),
    ],
)
def test_check_command_rejects_invalid_manual_recovery_without_leaking_input(
    event_name: str,
    raw_issue: str | None,
    ref: str,
    lookup_default_branch: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Only a positive manual recovery issue on the default branch is admitted."""
    client = FakeGitHub()
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: client)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setattr(health, '_fetch_zai_quota', lambda *_: health.Quota('healthy'))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setenv('ZAI_API_KEY', 'zai-key')
    monkeypatch.setenv('GITHUB_WORKFLOW', 'nightly-sweep')
    monkeypatch.setenv('PYDANTIC_AI_TASK_KEY', 'task-1')
    monkeypatch.setenv('PYDANTIC_AI_TRIGGER_EVENT', event_name)
    monkeypatch.setenv('PYDANTIC_AI_RUN_ATTEMPT', '1')
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'github-token')
    monkeypatch.setenv('GITHUB_EVENT_NAME', event_name)
    monkeypatch.setenv('GITHUB_REF', ref)
    monkeypatch.setenv('GITHUB_RUN_ID', '101')
    if raw_issue is None:
        monkeypatch.delenv('PYDANTIC_AI_RECOVERY_ISSUE', raising=False)
    else:
        monkeypatch.setenv('PYDANTIC_AI_RECOVERY_ISSUE', raw_issue)
    output = tmp_path / 'provider-health.json'

    assert health.main(['check', '--output', str(output)]) == 0
    decision = _health_from_json(json.loads(output.read_text()))
    assert decision.recovery_issue is None
    assert decision.ready
    assert client.default_branch_calls == int(lookup_default_branch)
    captured = capsys.readouterr().out
    if raw_issue:
        assert 'Recovery input ignored; use a positive issue number on a default-branch workflow_dispatch.' in captured
        assert raw_issue not in captured
    else:
        assert 'Recovery input ignored' not in captured


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('run_id', 7),
        ('run_id', ''),
        ('run_id', '0'),
        ('run_id', '７'),
        ('run_id', '7' * 5000),
        ('recovery_issue', '7'),
        ('recovery_issue', True),
        ('recovery_issue', 0),
    ],
)
def test_health_artifact_rejects_malformed_present_optional_identity_fields(field: str, value: object) -> None:
    """Present malformed identity metadata remains a DTO validation error."""
    payload: dict[str, object] = {
        'version': 1,
        'workflow': 'nightly-sweep',
        'task_key': 'task-1',
        'trigger_event': 'schedule',
        'run_attempt': 1,
        'ready': True,
        'reason': 'Provider health check passed',
        'checked_at': '2026-10-01T12:00:00Z',
        'quota': {'status': 'healthy'},
        field: value,
    }

    with pytest.raises(ValueError, match='missing required validated fields'):
        _health_from_json(payload)


def test_check_and_incident_outputs_keep_hostile_provider_data_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Provider secrets and exception data never enter outputs or incidents."""
    sentinel = 'private-provider-sentinel-7bf3'
    client = FakeGitHub()
    headers = Message()
    headers['X-Private-Header'] = sentinel
    error = health.urllib.error.HTTPError(
        health.ZAI_QUOTA_URL,
        429,
        f'reason {sentinel}',
        headers,
        None,
    )

    def open_url(*_args: object, **_kwargs: object) -> FakeHTTPResponse:
        raise error

    monkeypatch.setattr(health.urllib.request.OpenerDirector, 'open', open_url)
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: client)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setenv('ZAI_API_KEY', f'key-{sentinel}')
    monkeypatch.setenv('GITHUB_WORKFLOW', 'nightly-sweep')
    monkeypatch.setenv('PYDANTIC_AI_TASK_KEY', 'task-1')
    monkeypatch.setenv('PYDANTIC_AI_TRIGGER_EVENT', 'schedule')
    monkeypatch.setenv('PYDANTIC_AI_RUN_ATTEMPT', '1')
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'github-token')
    output = tmp_path / 'provider-health.json'
    github_output = tmp_path / 'output'
    summary = tmp_path / 'summary'
    monkeypatch.setenv('GITHUB_OUTPUT', str(github_output))
    monkeypatch.setenv('GITHUB_STEP_SUMMARY', str(summary))

    assert health.main(['check', '--output', str(output)]) == 0
    check_stdout = capsys.readouterr().out
    assert 'quota diagnostic: http_error; HTTP status 429' in check_stdout
    for text in (output.read_text(), github_output.read_text(), summary.read_text(), check_stdout):
        assert sentinel not in text

    assert health.main(['monitor', '--run-id', '43', '--run-attempt', '1', '--health-artifact', str(output)]) == 0
    post_data = _mapping(client.posts[0])
    assert post_data is not None
    incident_body = str(post_data.get('body'))
    assert 'quota diagnostic: http_error; HTTP status 429' in incident_body
    assert sentinel not in incident_body


def test_check_command_sanitizes_request_validation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Request validation errors remain blocked without exposing the credential."""
    sentinel = 'credential-header-secret-9d25'

    def open_url(*_args: object, **_kwargs: object) -> FakeHTTPResponse:
        raise ValueError(f'Invalid header value: {sentinel}')

    monkeypatch.setattr(health.urllib.request.OpenerDirector, 'open', open_url)
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: FakeGitHub())  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setenv('ZAI_API_KEY', sentinel)
    monkeypatch.setenv('GITHUB_WORKFLOW', 'nightly-sweep')
    monkeypatch.setenv('PYDANTIC_AI_TASK_KEY', 'task-1')
    monkeypatch.setenv('PYDANTIC_AI_TRIGGER_EVENT', 'schedule')
    monkeypatch.setenv('PYDANTIC_AI_RUN_ATTEMPT', '1')
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'github-token')
    artifact = tmp_path / 'provider-health.json'
    output = tmp_path / 'output'
    summary = tmp_path / 'summary'
    monkeypatch.setenv('GITHUB_OUTPUT', str(output))
    monkeypatch.setenv('GITHUB_STEP_SUMMARY', str(summary))

    assert health.main(['check', '--output', str(artifact)]) == 0
    decision = _health_from_json(json.loads(artifact.read_text()))
    assert not decision.ready
    assert decision.quota.diagnostic == 'invalid_request'
    for text in (artifact.read_text(), output.read_text(), summary.read_text(), capsys.readouterr().out):
        assert sentinel not in text


def test_monitor_reports_only_valid_http_status_from_untrusted_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An invalid artifact category is dropped while its valid status remains safe."""
    client = FakeGitHub()
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: client)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'github-token')
    artifact = tmp_path / 'provider-health.json'
    artifact.write_text(
        json.dumps(
            {
                'version': 1,
                'workflow': 'nightly-sweep',
                'task_key': 'task-1',
                'trigger_event': 'schedule',
                'run_attempt': 1,
                'ready': False,
                'reason': 'Z.ai quota health is unknown',
                'checked_at': '2026-10-01T12:00:00Z',
                'quota': {'status': 'unknown', 'diagnostic': 'provider-private-category', 'http_status': 200},
            }
        )
    )

    assert (
        health.main(
            ['monitor', '--run-id', '43', '--run-attempt', '1', '--health-artifact', str(artifact), '--dry-run']
        )
        == 0
    )

    output = capsys.readouterr().out
    assert 'quota diagnostic: HTTP status 200' in output
    assert 'provider-private-category' not in output
    assert 'http_error' not in output
    assert client.posts == []


def test_monitor_existing_incident_reuse_prints_quota_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Existing-incident blocks print the safe diagnostic suffix and return."""
    client = FakeGitHub()
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: client)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'github-token')
    artifact = tmp_path / 'provider-health.json'
    _write_health(
        artifact,
        health.Health(
            'nightly-sweep',
            'task-1',
            'schedule',
            1,
            False,
            'Open operational incident #77 blocks inference',
            '2026-10-01T12:00:00Z',
            health.Quota('unknown', diagnostic='timeout', http_status=429),
        ),
    )

    assert health.main(['monitor', '--run-id', '43', '--run-attempt', '1', '--health-artifact', str(artifact)]) == 0

    assert capsys.readouterr().out.strip() == (
        'Activation was blocked by an existing incident: Open operational incident #77 blocks inference '
        '(quota diagnostic: timeout; HTTP status 429)'
    )
    assert client.posts == []


@pytest.mark.parametrize('existing_run_id', ['43', '99'], ids=['exact-marker', 'same-scope-different-run'])
def test_monitor_reuses_matching_quota_incident_with_safe_diagnostic(
    existing_run_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A matching quota incident is reused and the safe diagnostic is logged."""
    quota = health.Quota('unknown', diagnostic='timeout', http_status=503)
    marker = health.IncidentMarker('provider', 'zai', 'quota_unknown', existing_run_id, quota.reset_at)
    client = FakeGitHub([_issue(77, marker)])
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: client)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'github-token')
    artifact = tmp_path / 'provider-health.json'
    _write_health(
        artifact,
        health.Health(
            'nightly-sweep',
            'task-1',
            'schedule',
            1,
            False,
            'Z.ai quota health is unknown',
            '2026-10-01T12:00:00Z',
            quota,
        ),
    )

    assert health.main(['monitor', '--run-id', '43', '--run-attempt', '1', '--health-artifact', str(artifact)]) == 0

    assert capsys.readouterr().out.strip() == (
        'Reusing operational incident #77: https://github.com/org/repo/issues/77 '
        '(quota diagnostic: timeout; HTTP status 503)'
    )
    assert client.posts == []


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
    """Missing current credentials on an event that receives secrets remain an outage."""
    client = FakeGitHub()
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: client)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    for name, value in [('ZAI_API_KEY', zai_key), ('MINIMAX_API_KEY', minimax_key)]:
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'github-token')
    monkeypatch.setenv('GITHUB_WORKFLOW', 'CI Review')
    monkeypatch.setenv('PYDANTIC_AI_TASK_KEY', 'CI Review:workflow_run:pr-123:head-a1b2:')
    monkeypatch.setenv('PYDANTIC_AI_TRIGGER_EVENT', 'workflow_run')
    monkeypatch.setenv('PYDANTIC_AI_RUN_ATTEMPT', '1')
    monkeypatch.delenv('GITHUB_OUTPUT', raising=False)
    monkeypatch.delenv('GITHUB_STEP_SUMMARY', raising=False)
    artifact = tmp_path / 'provider-health.json'

    assert health.main(['check', '--output', str(artifact)]) == 0
    assert _health_from_json(json.loads(artifact.read_text())).quota.diagnostic == 'missing_api_key'
    assert health.main(['monitor', '--run-id', '43', '--run-attempt', '1', '--health-artifact', str(artifact)]) == 0
    assert len(client.posts) == 1
    assert 'quota diagnostic: missing_api_key' in str((_mapping(client.posts[0]) or {}).get('body'))
    assert _marker_from_payload(client.posts[0]) == health.IncidentMarker(
        'provider', 'zai', 'quota_unknown', '43', None
    )


@pytest.mark.parametrize('trigger_event', ['pull_request', 'pull_request_review', 'pull_request_review_comment'])
def test_missing_credentials_on_secretless_pull_request_events_leave_the_alert_to_the_scheduled_check(
    trigger_event: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GitHub withholds secrets from fork and Dependabot pull request events, so their blocked gate opens nothing."""
    client = FakeGitHub()
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: client)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.delenv('ZAI_API_KEY', raising=False)
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'github-token')
    monkeypatch.setenv('GITHUB_WORKFLOW', 'CI Review')
    monkeypatch.setenv('PYDANTIC_AI_TASK_KEY', f'CI Review:{trigger_event}:10979:head-a1b2:')
    monkeypatch.setenv('PYDANTIC_AI_TRIGGER_EVENT', trigger_event)
    monkeypatch.setenv('PYDANTIC_AI_RUN_ATTEMPT', '1')
    monkeypatch.delenv('GITHUB_OUTPUT', raising=False)
    monkeypatch.delenv('GITHUB_STEP_SUMMARY', raising=False)
    artifact = tmp_path / 'provider-health.json'

    assert health.main(['check', '--output', str(artifact)]) == 0
    decision = _health_from_json(json.loads(artifact.read_text()))
    assert decision.ready is False
    assert decision.reason == 'Z.ai quota health is unknown'
    assert decision.quota.diagnostic == 'missing_api_key'
    assert health.main(['monitor', '--run-id', '43', '--run-attempt', '1', '--health-artifact', str(artifact)]) == 0
    assert client.posts == []

    monkeypatch.setenv('GITHUB_WORKFLOW', 'Agent Provider Health')
    monkeypatch.setenv('PYDANTIC_AI_TRIGGER_EVENT', 'schedule')
    assert health.main(['monitor', '--repository', 'org/repo']) == 0
    assert len(client.posts) == 1
    post_data = _mapping(client.posts[0])
    assert post_data is not None
    assert 'quota diagnostic: missing_api_key' in str(post_data.get('body'))
    assert _marker_from_payload(client.posts[0]) == health.IncidentMarker(
        'provider', 'zai', 'quota_unknown', None, None
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
    plan_limited = health.RunResult('w', 't', 'workflow_dispatch', 1, health.Failure('plan_limit', 429, None))
    assert _scope_for(plan_limited) == ('provider', 'zai', 'plan_limit')
    scheduled_timeout = health.RunResult('w', 'target-head-a1b2', 'schedule', 1, health.Failure('timeout', None, None))
    repeated_scheduled_timeout = health.RunResult(
        'w', 'target-head-a1b2', 'schedule', 1, health.Failure('timeout', None, None)
    )
    assert _scope_for(scheduled_timeout) == ('task', 'w:target-head-a1b2', 'timeout')
    assert _scope_for(repeated_scheduled_timeout) == _scope_for(scheduled_timeout)
    assert _scope_for(
        health.RunResult('w', 'target-head-a1b2', 'workflow_dispatch', 1, health.Failure('timeout', None, None))
    ) == ('task', 'w:target-head-a1b2', 'timeout')
    assert _scope_for(health.RunResult('w', 't', 'schedule', 1, None)) == ('workflow', 'w', 'execution')


def test_monitor_uses_attempt_validation_failure_after_successful_model_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trusted failed validation steps remain distinct from absent model terminal data."""
    artifact = tmp_path / 'provider-health.json'
    _write_health(
        artifact,
        health.Health(
            'Attention',
            'issue-42',
            'issues',
            1,
            True,
            'Provider health check passed',
            '2026-10-01T12:00:00Z',
            health.Quota('healthy'),
        ),
    )
    log = tmp_path / 'agent-stdio.log'
    log.write_text(
        json.dumps(
            {
                'type': 'result',
                'provider_health': {
                    'workflow': 'Attention',
                    'task_key': 'issue-42',
                    'trigger_event': 'issues',
                    'run_attempt': 1,
                },
            }
        )
    )
    client = FakeGitHub()
    client.jobs[('42', 1)] = [
        {
            'name': 'safe_outputs',
            'conclusion': 'failure',
            'steps': [{'name': 'Apply validated maintainer attention', 'conclusion': 'failure'}],
        }
    ]
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: client)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'token')

    _monitor_command(
        argparse.Namespace(
            repository=None,
            run_id=42,
            run_attempt=1,
            conclusion='failure',
            health_artifact=artifact,
            agent_artifact=log,
            dry_run=False,
            recover_issue=None,
        )
    )

    marker = _marker_from_payload(client.posts[0])
    assert marker == health.IncidentMarker('task', 'Attention:issue-42', 'validation', '42', None, 'validation', 1)


def test_monitor_classifies_post_agent_allowlist_restore_as_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The trusted post-agent decision job reports a failed allowlist restore as validation."""
    artifact = tmp_path / 'provider-health.json'
    _write_health(
        artifact,
        health.Health(
            'Attention',
            'issue-42',
            'issues',
            1,
            True,
            'Provider health check passed',
            '2026-10-01T12:00:00Z',
            health.Quota('healthy'),
        ),
    )
    log = tmp_path / 'agent-stdio.log'
    log.write_text(
        json.dumps(
            {
                'type': 'result',
                'provider_health': {
                    'workflow': 'Attention',
                    'task_key': 'issue-42',
                    'trigger_event': 'issues',
                    'run_attempt': 1,
                },
            }
        )
    )
    client = FakeGitHub()
    client.jobs[('42', 1)] = [
        {
            'name': 'record_attention_decision',
            'conclusion': 'failure',
            'steps': [
                {'name': 'Checkout repository', 'conclusion': 'success'},
                {'name': 'Install dependencies', 'conclusion': 'success'},
                {'name': 'Restore exact candidate allowlist', 'conclusion': 'failure'},
            ],
        }
    ]
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: client)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'token')

    _monitor_command(
        argparse.Namespace(
            repository=None,
            run_id=42,
            run_attempt=1,
            conclusion='failure',
            health_artifact=artifact,
            agent_artifact=log,
            dry_run=False,
            recover_issue=None,
        )
    )

    marker = _marker_from_payload(client.posts[0])
    assert marker == health.IncidentMarker('task', 'Attention:issue-42', 'validation', '42', None, 'validation', 1)


@pytest.mark.parametrize(
    ('jobs', 'expected'),
    [  # pyright: ignore[reportUnknownArgumentType]
        (
            [
                {
                    'name': 'agent',
                    'conclusion': 'failure',
                    'steps': [
                        {'name': 'Execute Claude Code CLI', 'conclusion': 'success'},
                        {'name': 'Restore full history and tags for dynamic versioning', 'conclusion': 'failure'},
                    ],
                }
            ],
            'setup',
        ),
        (
            [
                {
                    'name': 'agent',
                    'conclusion': 'failure',
                    'steps': [
                        {'name': 'Download activation artifact', 'conclusion': 'failure'},
                        {'name': 'Execute Pydantic AI CLI', 'conclusion': 'skipped'},
                    ],
                }
            ],
            'setup',
        ),
        *(
            (
                [
                    {
                        'name': 'agent',
                        'conclusion': 'failure',
                        'steps': [
                            {'name': name, 'conclusion': 'failure'},
                            {'name': 'Execute Claude Code CLI', 'conclusion': 'skipped'},
                        ],
                    }
                ],
                None,
            )
            for name in health.AGENT_START_HEALTH_STEPS
        ),
        ([{'name': 'agent', 'conclusion': 'failure', 'runner_name': None, 'steps': []}], None),
        ([{'name': 'provider_health', 'conclusion': 'failure', 'steps': []}], None),
        (
            [
                {
                    'name': 'safe_outputs',
                    'conclusion': 'failure',
                    'steps': [{'name': 'Restore exact candidate allowlist', 'conclusion': 'failure'}],
                }
            ],
            'setup',
        ),
    ],
)
def test_attempt_stage_uses_real_setup_and_no_step_job_shapes(
    jobs: list[dict[str, object]], expected: health.Stage | None
) -> None:
    """Moved-tag setup failures are distinct from queued and pre-execution no-step failures."""
    assert health._stage_from_attempt_jobs(jobs) == expected  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize('job_name', ['pre_activation', 'unrelated_job'])
def test_monitor_records_trusted_pre_activation_setup_failure(job_name: str) -> None:
    """Recognize the compiled pre-activation setup step without classifying arbitrary jobs."""
    client = FakeGitHub()
    client.jobs[('42', 1)] = [
        {'name': job_name, 'conclusion': 'failure', 'steps': [{'name': 'Setup Scripts', 'conclusion': 'failure'}]}
    ]
    gate = health.Health(
        'Pydantic AI Attention Triage',
        'task-1',
        'schedule',
        1,
        True,
        'Provider health check passed',
        '2026-10-17T00:00:00Z',
        health.Quota('healthy'),
        run_id='42',
    )

    assert (
        _monitor_workflow_outcome(
            client, argparse.Namespace(run_id=42, conclusion='failure', dry_run=False), gate, 1, None, 'org/repo'
        )
        == 0
    )

    if job_name == 'pre_activation':
        assert len(client.posts) == 1
        assert _marker_from_payload(client.posts[0]) == health.IncidentMarker(
            'workflow', gate.workflow, 'setup', '42', None, 'setup', 1
        )
    else:
        assert client.posts == []


@pytest.mark.parametrize(
    'name', ['Execute Claude Code CLI', 'Execute Pydantic AI CLI', 'Run agent', 'Run legacy engine']
)
def test_failed_execution_step_is_inference(name: str) -> None:
    """Recognized execution anchors classify a failed execution step as inference."""
    jobs: list[dict[str, object]] = [
        {'name': 'agent', 'conclusion': 'failure', 'steps': [{'name': name, 'conclusion': 'failure'}]}
    ]
    assert health._stage_from_attempt_jobs(jobs) == 'inference'  # pyright: ignore[reportPrivateUsage]


def test_failed_post_execution_step_is_publication() -> None:
    """A failure after Pydantic AI CLI execution is classified as publication."""
    jobs: list[dict[str, object]] = [
        {
            'name': 'agent',
            'conclusion': 'failure',
            'steps': [
                {'name': 'Execute Pydantic AI CLI', 'conclusion': 'success'},
                {'name': 'Collect execution diagnostics', 'conclusion': 'failure'},
            ],
        }
    ]
    assert health._stage_from_attempt_jobs(jobs) == 'publication'  # pyright: ignore[reportPrivateUsage]


def test_attempt_jobs_paginates_exact_attempt_and_uses_raw_page_length(monkeypatch: pytest.MonkeyPatch) -> None:
    """Attempt jobs include later pages and stop on a short raw jobs page."""
    requests: list[health.urllib.request.Request] = []
    pages: list[dict[str, object]] = [
        {'jobs': [{'id': index} for index in range(100)]},
        {'jobs': [{'id': 100, 'conclusion': 'failure'}]},
    ]

    def open_url(
        _opener: health.urllib.request.OpenerDirector,
        request: health.urllib.request.Request,
        _data: object | None = None,
        timeout: float = 30,
    ) -> FakeHTTPResponse:
        requests.append(request)
        return FakeHTTPResponse(pages[len(requests) - 1])

    monkeypatch.setattr(health.urllib.request.OpenerDirector, 'open', open_url)
    client = health.GitHubClient('org/repo', 'token')

    jobs = client.attempt_jobs('run-42', 3)

    assert len(jobs) == 101
    assert jobs[-1] == {'id': 100, 'conclusion': 'failure'}
    assert [request.full_url for request in requests] == [
        f'{health.API_ROOT}/repos/org/repo/actions/runs/run-42/attempts/3/jobs?per_page=100&page=1',
        f'{health.API_ROOT}/repos/org/repo/actions/runs/run-42/attempts/3/jobs?per_page=100&page=2',
    ]


def test_attempt_jobs_rejects_malformed_later_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """A malformed later page fails closed instead of returning partial jobs."""
    pages: list[dict[str, object]] = [{'jobs': [{'id': index} for index in range(100)]}, {'jobs': None}]

    def open_url(
        _opener: health.urllib.request.OpenerDirector,
        _request: health.urllib.request.Request,
        _data: object | None = None,
        timeout: float = 30,
    ) -> FakeHTTPResponse:
        return FakeHTTPResponse(pages.pop(0))

    monkeypatch.setattr(health.urllib.request.OpenerDirector, 'open', open_url)
    client = health.GitHubClient('org/repo', 'token')

    with pytest.raises(ValueError, match='invalid attempt-jobs response'):
        client.attempt_jobs('run-42', 3)


def test_scheduled_recovery_closes_only_elapsed_known_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scheduled recovery closes elapsed windows but leaves future ones, however old the incident."""
    elapsed = _issue(1, health.IncidentMarker('provider', 'zai', 'rate_limit', '1', '2026-09-30T00:00:00Z'))
    future = _issue(
        2,
        health.IncidentMarker('provider', 'zai', 'quota_exhausted', '2', '2026-12-01T00:00:00Z'),
        created_at='2026-09-29T00:00:00Z',
    )
    retired = _issue(4, health.IncidentMarker('provider', 'minimax', 'rate_limit', '4', '2026-09-30T00:00:00Z'))
    client = FakeGitHub([elapsed, future, retired])
    monkeypatch.setenv('ZAI_API_KEY', 'zai-key')
    monkeypatch.setattr(health, '_fetch_zai_quota', lambda *_: health.Quota('healthy'))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setattr(health, '_now', lambda: dt.datetime(2026, 10, 1, tzinfo=dt.UTC))

    assert _reconcile_recovery(client, argparse.Namespace(dry_run=False, recover_issue=None)) == 0
    assert client.closed == [1]


@pytest.mark.parametrize('kind', ['rate_limit', 'plan_limit', 'quota_exhausted', 'quota_unknown'])
def test_scheduled_recovery_closes_null_reset_incident_after_cool_down(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A null-reset quota or rate-limit incident closes on a healthy check once it is an hour old."""
    aged = _issue(1, health.IncidentMarker('provider', 'zai', kind, '1', None), created_at='2026-10-15T11:00:00Z')
    young = _issue(2, health.IncidentMarker('provider', 'zai', kind, '2', None), created_at='2026-10-15T11:01:00Z')
    undated = _issue(3, health.IncidentMarker('provider', 'zai', kind, '3', None))
    client = FakeGitHub([aged, young, undated])
    monkeypatch.setenv('ZAI_API_KEY', 'zai-key')

    def fetch_healthy(*_args: object) -> health.Quota:
        return health.Quota('healthy')

    monkeypatch.setattr(health, '_fetch_zai_quota', fetch_healthy)
    monkeypatch.setattr(health, '_now', lambda: dt.datetime(2026, 10, 15, 12, tzinfo=dt.UTC))

    assert _reconcile_recovery(client, argparse.Namespace(dry_run=False, recover_issue=None)) == 0
    assert client.closed == [1]


@pytest.mark.parametrize(
    'quota', [health.Quota('unknown', diagnostic='timeout'), health.Quota('exhausted', '2026-10-15T17:00:00Z')]
)
def test_scheduled_recovery_keeps_aged_null_reset_incident_while_unhealthy(
    quota: health.Quota, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cool-down alone never closes an incident; the fresh check must also be healthy."""
    incident = _issue(
        1, health.IncidentMarker('provider', 'zai', 'rate_limit', '1', None), created_at='2026-10-15T08:00:00Z'
    )
    client = FakeGitHub([incident])
    monkeypatch.setenv('ZAI_API_KEY', 'zai-key')
    monkeypatch.setenv('PYDANTIC_AI_RUN_ATTEMPT', '1')

    def fetch_unhealthy(*_args: object) -> health.Quota:
        return quota

    monkeypatch.setattr(health, '_fetch_zai_quota', fetch_unhealthy)
    monkeypatch.setattr(health, '_now', lambda: dt.datetime(2026, 10, 15, 12, tzinfo=dt.UTC))

    assert _reconcile_recovery(client, argparse.Namespace(dry_run=False, recover_issue=None)) == 0
    assert client.closed == []
    assert client.posts == []


def test_scheduled_recovery_leaves_credential_and_scoped_incidents_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """Credential, other provider failures, and workflow or task incidents stay manual after the cool-down."""
    created_at = '2026-10-15T08:00:00Z'
    client = FakeGitHub(
        [
            _issue(1, health.IncidentMarker('provider', 'zai', 'authentication', '1', None), created_at=created_at),
            _issue(2, health.IncidentMarker('provider', 'zai', 'other', '2', None), created_at=created_at),
            _issue(3, health.IncidentMarker('workflow', 'nightly-sweep', 'timeout', '3', None), created_at=created_at),
            _issue(4, health.IncidentMarker('task', 'nightly-sweep:t', 'rate_limit', '4', None), created_at=created_at),
        ]
    )
    monkeypatch.setenv('ZAI_API_KEY', 'zai-key')

    def fetch_healthy(*_args: object) -> health.Quota:
        return health.Quota('healthy')

    monkeypatch.setattr(health, '_fetch_zai_quota', fetch_healthy)
    monkeypatch.setattr(health, '_now', lambda: dt.datetime(2026, 10, 15, 12, tzinfo=dt.UTC))

    assert _reconcile_recovery(client, argparse.Namespace(dry_run=False, recover_issue=None)) == 0
    assert client.closed == []


def test_open_incidents_read_normalized_issue_creation_time() -> None:
    """Incident age comes from the GitHub issue's `created_at`; a malformed value leaves it unknown."""
    marker = health.IncidentMarker('provider', 'zai', 'rate_limit', '1', None)
    client = FakeGitHub(
        [_issue(1, marker, created_at='2026-10-15T11:00:00+00:00'), _issue(2, marker, created_at='yesterday')]
    )

    issues = health.GitHubClient.open_incidents(client)

    assert [issue.created_at for issue in issues] == ['2026-10-15T11:00:00Z', None]


def test_open_incidents_records_each_page_and_the_incidents_it_returned(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The job log and summary show what the Actions token's read returned, page by page."""
    marker = health.IncidentMarker('provider', 'zai', 'quota_unknown', '1', None)

    def entry(issue: health.Issue) -> dict[str, object]:
        return {'number': issue.number, 'title': 't', 'body': issue.body, 'state': 'open', 'html_url': 'u'}

    plain = [entry(health.Issue(number, 't', 'no marker', 'open', 'u', None)) for number in range(1, 99)]
    unreadable: dict[str, object] = {'number': 99, 'title': 't', 'body': None, 'state': 'open', 'html_url': 'u'}
    pages: list[FakeHTTPResponse] = [
        FakeHTTPResponse([*plain, unreadable, entry(_issue(200, marker))], request_id='AAAA:1'),
        FakeHTTPResponse([entry(_issue(201, marker))]),
    ]

    def open_url(
        _opener: health.urllib.request.OpenerDirector,
        _request: health.urllib.request.Request,
        _data: object | None = None,
        timeout: float = 30,
    ) -> FakeHTTPResponse:
        return pages.pop(0)

    monkeypatch.setattr(health.urllib.request.OpenerDirector, 'open', open_url)
    summary = tmp_path / 'summary.md'
    monkeypatch.setenv('GITHUB_STEP_SUMMARY', str(summary))

    issues = health.GitHubClient('org/repo', 'token').open_incidents()

    evidence = (
        'Open-incident read: page 1 HTTP 200 with 100 issues, 1 unreadable (request AAAA:1); '
        'page 2 HTTP 200 with 1 issues, 0 unreadable (request unknown); incidents: #200, #201'
    )
    assert len(issues) == 100
    assert capsys.readouterr().out == f'{evidence}\n'
    assert summary.read_text(encoding='utf-8') == f'{evidence}\n\n'


def test_empty_open_incident_read_says_none(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An empty read is logged as such instead of looking like a skipped read."""
    monkeypatch.setattr(
        health.urllib.request.OpenerDirector,
        'open',
        lambda *_: FakeHTTPResponse([], request_id='BBBB:2'),  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    )

    assert health.GitHubClient('org/repo', 'token').open_incidents() == []
    assert capsys.readouterr().out == (
        'Open-incident read: page 1 HTTP 200 with 0 issues, 0 unreadable (request BBBB:2); incidents: none\n'
    )


class DegradedGitHub(FakeGitHub):
    """Serve open-incident reads that miss every incident, as GitHub did during its 2026-10-22 Issues degradation."""

    def __init__(self, issues: list[health.Issue], misses: int) -> None:
        super().__init__(issues)
        self.misses = misses
        self.reads = 0

    def open_incidents(self) -> list[health.Issue]:
        self.reads += 1
        return [] if self.reads <= self.misses else self.issues


def _run_gate(client: FakeGitHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: client)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setattr(health, '_fetch_zai_quota', lambda *_: health.Quota('healthy'))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setenv('ZAI_API_KEY', 'zai-key')
    monkeypatch.setenv('GITHUB_WORKFLOW', 'CI Review')
    monkeypatch.setenv('PYDANTIC_AI_TASK_KEY', 'task-1')
    monkeypatch.setenv('PYDANTIC_AI_TRIGGER_EVENT', 'pull_request')
    monkeypatch.setenv('PYDANTIC_AI_RUN_ATTEMPT', '1')
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'github-token')
    github_output = tmp_path / 'output'
    monkeypatch.setenv('GITHUB_OUTPUT', str(github_output))

    assert health.main(['check', '--output', str(tmp_path / 'provider-health.json')]) == 0
    return github_output.read_text()


def test_gate_rereads_an_incident_less_read_before_passing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A read that misses the open provider incident is re-read instead of letting inference through."""
    client = DegradedGitHub([_issue(7, health.IncidentMarker('provider', 'zai', 'quota_unknown', '1', None))], misses=2)
    delays: list[float] = []
    monkeypatch.setattr(health, '_INCIDENT_READ_DELAY_SECONDS', 2)
    monkeypatch.setattr(health.time, 'sleep', delays.append)

    output = _run_gate(client, tmp_path, monkeypatch)

    assert output == 'ready=false\nreason=Open operational incident #7 blocks inference\n'
    assert client.reads == 3
    assert delays == [2, 2]


def test_gate_passes_only_after_every_read_finds_no_incident(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With nothing open, the gate spends its whole re-read budget before it passes."""
    client = DegradedGitHub([], misses=0)

    assert _run_gate(client, tmp_path, monkeypatch) == 'ready=true\nreason=Provider health check passed\n'
    assert client.reads == health._INCIDENT_READS  # pyright: ignore[reportPrivateUsage]


def test_gate_trusts_a_read_that_returns_another_scopes_incident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A degraded read returns no incident at all, so a read with an unrelated incident passes without re-reading."""
    other_task = _issue(8, health.IncidentMarker('task', 'CI Review:other-task', 'execution', '1', None))
    client = DegradedGitHub([other_task], misses=0)

    assert _run_gate(client, tmp_path, monkeypatch) == 'ready=true\nreason=Provider health check passed\n'
    assert client.reads == 1


@pytest.mark.parametrize('override', [False, True])
@pytest.mark.parametrize(('misses', 'created'), [(4, False), (5, True)])
def test_incident_reuse_rereads_until_it_finds_the_open_incident(override: bool, misses: int, created: bool) -> None:
    """Reuse survives incident-less reads up to the re-read budget; only a budget of misses opens a duplicate."""
    open_incident = _issue(7, health.IncidentMarker('provider', 'zai', 'quota_unknown', '1', None))
    client = DegradedGitHub([open_incident], misses=misses)
    result = health.RunResult('CI Review', 'task-1', 'pull_request', 1, health.Failure('rate_limit', 429, None))
    marker = health.IncidentMarker('provider', 'zai', 'quota_unknown', '88', None) if override else None

    issue = _create_or_reuse_incident(
        client, result, '88', client.repo, 'Z.ai quota health is unknown', dry_run=False, marker_override=marker
    )

    assert issue is not None
    assert (issue.number != open_incident.number) is created
    assert len(client.posts) == int(created)
    assert client.reads == misses + int(not created)


def test_scheduled_monitor_creates_or_reuses_unknown_quota_incident(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Repeated scheduled checks reuse one incident when quota remains unknown."""
    client = FakeGitHub()
    monkeypatch.setenv('ZAI_API_KEY', 'zai-key')
    monkeypatch.setenv('PYDANTIC_AI_RUN_ATTEMPT', '1')

    def fetch_unknown(*_args: object) -> health.Quota:
        return health.Quota('unknown', diagnostic='timeout')

    monkeypatch.setattr(health, '_fetch_zai_quota', fetch_unknown)
    args = argparse.Namespace(dry_run=False, recover_issue=None)

    _reconcile_recovery(client, args)
    _reconcile_recovery(client, args)

    assert len(client.posts) == 1
    payload: object = client.posts[0]
    assert _marker_from_payload(payload) == health.IncidentMarker('provider', 'zai', 'quota_unknown', None, None)
    post_data = _mapping(payload)
    assert post_data is not None
    incident_body = str(post_data.get('body'))
    assert 'quota diagnostic: timeout' in incident_body
    assert 'quota diagnostic: timeout' in capsys.readouterr().out


def test_scheduled_recovery_dry_run_reports_unknown_quota_diagnostic(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Dry-run recovery output includes safe quota diagnostics without creating issues."""
    client = FakeGitHub()
    monkeypatch.setenv('ZAI_API_KEY', 'zai-key')
    monkeypatch.setenv('PYDANTIC_AI_RUN_ATTEMPT', '1')

    def fetch_unknown(*_args: object) -> health.Quota:
        return health.Quota('unknown', diagnostic='invalid_json', http_status=200)

    monkeypatch.setattr(health, '_fetch_zai_quota', fetch_unknown)

    assert _reconcile_recovery(client, argparse.Namespace(dry_run=True, recover_issue=None)) == 0

    assert capsys.readouterr().out.strip() == (
        'DRY RUN: provider quota is unknown; would open/reuse an incident '
        '(quota diagnostic: invalid_json; HTTP status 200)'
    )
    assert client.posts == []


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

    with pytest.raises(ValueError, match=r'not a Z\.ai provider-health incident'):
        _reconcile_recovery(client, argparse.Namespace(dry_run=False, recover_issue=14))

    assert checked == []
    assert client.closed == []


def test_named_recovery_without_provider_key_stays_unknown(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Missing provider credentials stay unknown even when the fetch stub is healthy."""
    issue = _issue(4, health.IncidentMarker('provider', 'zai', 'authentication', '4', None))
    client = FakeGitHub([issue])
    monkeypatch.delenv('ZAI_API_KEY', raising=False)
    monkeypatch.setattr(health, '_fetch_zai_quota', lambda *_: health.Quota('healthy'))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]

    _reconcile_recovery(client, argparse.Namespace(dry_run=False, recover_issue=4))

    assert client.closed == []
    assert 'quota diagnostic: missing_api_key' in capsys.readouterr().out


def test_named_workflow_and_task_recovery_refuses_quota_only_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Workflow and task incidents require operator verification and manual closure."""
    workflow_issue = _issue(7, health.IncidentMarker('workflow', 'nightly-sweep', 'timeout', '7', None))
    task_issue = _issue(8, health.IncidentMarker('task', 'nightly-sweep:task-1', 'timeout', '8', None))
    client = FakeGitHub([workflow_issue, task_issue])
    monkeypatch.setenv('ZAI_API_KEY', 'zai-key')
    monkeypatch.setenv('PYDANTIC_AI_RECOVERY_ISSUE', '7')
    monkeypatch.setenv('GITHUB_EVENT_NAME', 'workflow_dispatch')
    monkeypatch.setattr(health, '_fetch_zai_quota', lambda *_: health.Quota('healthy'))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]

    _reconcile_recovery(client, argparse.Namespace(dry_run=False, recover_issue=7))
    _reconcile_recovery(client, argparse.Namespace(dry_run=False, recover_issue=8))
    assert client.closed == []


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
            run_id='42',
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


_HEALTHY_GATE = health.Health(
    'CI Review',
    'task-1',
    'workflow_run',
    1,
    True,
    'Provider health check passed',
    '2026-10-22T16:00:00Z',
    health.Quota('healthy'),
    run_id='42',
)
_FAILED_INFERENCE_JOBS: list[dict[str, object]] = [
    {'name': 'agent', 'conclusion': 'failure', 'steps': [{'name': 'Execute Pydantic AI CLI', 'conclusion': 'failure'}]}
]


@pytest.mark.parametrize(
    ('gate', 'conclusion', 'jobs', 'needs_incident'),
    [
        pytest.param(_HEALTHY_GATE, 'success', [], False, id='healthy-success'),
        pytest.param(
            replace(
                _HEALTHY_GATE,
                ready=False,
                reason='Open operational incident #7 blocks inference',
                quota=health.Quota('exhausted', '2026-10-22T18:00:00Z'),
            ),
            'success',
            [],
            False,
            id='blocked-by-open-incident',
        ),
        pytest.param(_HEALTHY_GATE, 'failure', [], False, id='failure-without-host-stage'),
        pytest.param(
            replace(
                _HEALTHY_GATE,
                ready=False,
                reason='Z.ai quota is exhausted',
                quota=health.Quota('exhausted', '2026-10-22T18:00:00Z'),
            ),
            'success',
            [],
            True,
            id='exhausted-quota-without-incident',
        ),
        pytest.param(_HEALTHY_GATE, 'failure', _FAILED_INFERENCE_JOBS, True, id='failed-inference'),
    ],
)
def test_triage_hands_over_exactly_the_completions_monitor_opens_incidents_for(
    gate: health.Health,
    conclusion: str,
    jobs: list[dict[str, object]],
    needs_incident: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Triage never reads incidents and defers only the completions whose reconciliation reaches them."""
    artifact = tmp_path / 'provider-health.json'
    _write_health(artifact, gate)
    github_output = tmp_path / 'github-output'
    github_output.touch()
    monkeypatch.setenv('GITHUB_OUTPUT', str(github_output))
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'token')
    run_args = ['--run-id', '42', '--run-attempt', '1', '--conclusion', conclusion, '--health-artifact', str(artifact)]
    triage_client = FakeGitHub()
    triage_client.jobs[('42', 1)] = jobs

    def unexpected_incidents() -> list[health.Issue]:
        raise AssertionError('triage must not read operational incidents')

    monkeypatch.setattr(triage_client, 'open_incidents', unexpected_incidents)
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: triage_client)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]

    assert health.main(['triage', *run_args]) == 0
    assert github_output.read_text(encoding='utf-8') == ('reconcile=true\n' if needs_incident else '')
    assert triage_client.posts == []

    monitor_client = FakeGitHub()
    monitor_client.jobs[('42', 1)] = jobs
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: monitor_client)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]

    assert health.main(['monitor', *run_args]) == 0
    assert len(monitor_client.posts) == int(needs_incident)


def test_triage_without_github_output_still_reports_deferral(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A local triage run prints its decision when there is no step output to write."""
    artifact = tmp_path / 'provider-health.json'
    _write_health(artifact, _HEALTHY_GATE)
    client = FakeGitHub()
    client.jobs[('42', 1)] = _FAILED_INFERENCE_JOBS
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: client)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.delenv('GITHUB_OUTPUT', raising=False)
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'token')

    assert (
        health.main(
            [
                'triage',
                '--run-id',
                '42',
                '--run-attempt',
                '1',
                '--conclusion',
                'failure',
                '--health-artifact',
                str(artifact),
            ]
        )
        == 0
    )
    assert 'deferring it to the serialized monitor job' in capsys.readouterr().out
    assert client.posts == []


def test_triage_cannot_reach_recovery() -> None:
    """Triage requires a completed source run, so it never runs the recovery path that closes incidents."""
    with pytest.raises(SystemExit) as exc_info:
        health.main(['triage', '--run-attempt', '1', '--health-artifact', 'provider-health.json'])
    assert exc_info.value.code == 2


def test_healthy_cancellation_skips_attempt_jobs() -> None:
    """A healthy manual cancellation is expected and does not need job evidence."""
    client = FakeGitHub()

    def unexpected_attempt_jobs(run_id: str, attempt: int) -> list[dict[str, object]]:
        raise AssertionError(f'unexpected attempt job lookup for {run_id} attempt {attempt}')

    client.attempt_jobs = unexpected_attempt_jobs
    result = _monitor_workflow_outcome(
        client,
        argparse.Namespace(run_id=42, conclusion='cancelled', dry_run=False),
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
        1,
        None,
        'org/repo',
    )

    assert result == 0
    assert client.posts == []


def test_typed_cancellation_failure_still_fetches_attempt_jobs() -> None:
    """Typed terminal failure metadata still requests trusted attempt evidence."""
    client = FakeGitHub()
    attempts: list[tuple[str, int]] = []

    def record_attempt_jobs(run_id: str, attempt: int) -> list[dict[str, object]]:
        attempts.append((run_id, attempt))
        return []

    client.attempt_jobs = record_attempt_jobs
    result = _monitor_workflow_outcome(
        client,
        argparse.Namespace(run_id=43, conclusion='cancelled', dry_run=True),
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
        1,
        health.RunResult('nightly-sweep', 'task-1', 'schedule', 1, health.Failure('rate_limit', 429, None)),
        'org/repo',
    )

    assert result == 0
    assert attempts == [('43', 1)]


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
    if conclusion == 'failure':
        client.jobs[('45', 1)] = [
            {
                'name': 'agent',
                'conclusion': 'failure',
                'steps': [{'name': 'Run agent', 'conclusion': 'failure'}],
            }
        ]
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
            'workflow', 'nightly-sweep', 'inference', '45', None, 'inference', 1
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


def test_monitor_does_not_classify_untrusted_terminal_metadata_without_host_stage(
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

    assert client.posts == []
