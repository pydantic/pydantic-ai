"""Focused boundaries for the MiniMax inference gate and incident controller."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pytest

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.usage import RunUsage

sys.path.insert(0, str(Path(__file__).parent))

import agent_provider_health as health
from agent_provider_health import (
    _create_or_reuse_incident,  # pyright: ignore[reportPrivateUsage]
    _fetch_minimax_quota,  # pyright: ignore[reportPrivateUsage]
    _health_from_json,  # pyright: ignore[reportPrivateUsage]
    _mapping,  # pyright: ignore[reportPrivateUsage]
    _marker_from_body,  # pyright: ignore[reportPrivateUsage]
    _monitor_command,  # pyright: ignore[reportPrivateUsage]
    _parse_balance,  # pyright: ignore[reportPrivateUsage]
    _parse_plan_quota,  # pyright: ignore[reportPrivateUsage]
    _reconcile_recovery,  # pyright: ignore[reportPrivateUsage]
    _run_result_from,  # pyright: ignore[reportPrivateUsage]
    _scope_for,  # pyright: ignore[reportPrivateUsage]
    _write_health,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai_gh_aw_shim import cli as shim


def _quota_entry(
    *,
    name: str = 'general',
    interval_status: int = 1,
    weekly_status: int = 1,
    interval_percent: float = 80,
    weekly_percent: float = 70,
    interval_total: int = 100,
    weekly_total: int = 100,
) -> dict[str, object]:
    return {
        'model_name': name,
        'end_time': 1_800_000_000_000,
        'weekly_end_time': 1_800_000_000_000,
        'current_interval_status': interval_status,
        'current_weekly_status': weekly_status,
        'current_interval_remaining_percent': interval_percent,
        'current_weekly_remaining_percent': weekly_percent,
        'current_interval_total_count': interval_total,
        'current_weekly_total_count': weekly_total,
    }


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
    """A small native MiniMax response for the trusted-runner command tests."""

    status = 200

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = json.dumps(payload).encode()

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def test_plan_quota_requires_explicit_resource_and_complete_status() -> None:
    """Only a configured quota resource with complete values may open inference."""
    payload: dict[str, object] = {'model_remains': [_quota_entry()]}

    assert _parse_plan_quota(payload, None).status == 'unknown'
    assert _parse_plan_quota(payload, 'video').status == 'unknown'
    quota = _parse_plan_quota(payload, 'general')
    assert quota.status == 'healthy'
    assert quota.interval_remaining_percent == 80
    assert quota.weekly_remaining_percent == 70
    assert quota.interval_reset_at == '2027-01-15T08:00:00Z'
    assert quota.weekly_reset_at == '2027-01-15T08:00:00Z'


def test_plan_quota_exhaustion_uses_status_and_preserves_reset() -> None:
    """An exhausted quota carries the provider's window reset into incident state."""
    quota = _parse_plan_quota(
        {'model_remains': [_quota_entry(interval_status=2, interval_percent=0)]},
        'general',
    )

    assert quota.status == 'exhausted'
    assert quota.reset_at == '2027-01-15T08:00:00Z'


def test_unlimited_status_ignores_zero_remaining_percentage() -> None:
    """MiniMax status 3 means unlimited unless the not-in-plan sentinel applies."""
    quota = _parse_plan_quota(
        {
            'model_remains': [
                _quota_entry(
                    interval_status=3,
                    weekly_status=3,
                    interval_percent=0,
                    weekly_percent=0,
                )
            ]
        },
        'general',
    )

    assert quota.status == 'healthy'
    assert quota.interval_remaining_percent is None
    assert quota.weekly_remaining_percent is None
    assert quota.interval_unlimited is True
    assert quota.weekly_unlimited is True


@pytest.mark.parametrize(
    'payload',
    [
        {},
        {'model_remains': []},
        {'model_remains': [_quota_entry(name='video')]},
        {'model_remains': [_quota_entry(interval_status=3, weekly_status=3, interval_total=0, weekly_total=0)]},
        {'model_remains': [_quota_entry(), _quota_entry()]},
        {'model_remains': [{**_quota_entry(), 'current_interval_remaining_percent': None}]},
        {'model_remains': [_quota_entry(interval_total=-1)]},
    ],
)
def test_plan_quota_unavailable_or_not_in_plan_is_unknown(payload: object) -> None:
    """Missing resource rows and non-entitled rows stay unknown."""
    assert _parse_plan_quota(payload, 'general').status == 'unknown'


def test_check_writes_blocked_artifact_for_out_of_range_plan_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unrepresentable provider reset time blocks the gate without losing its artifact."""
    entry: dict[str, object] = _quota_entry()
    entry['end_time'] = 1e300
    entry['weekly_end_time'] = 1e300
    payload: dict[str, object] = {'base_resp': {'status_code': 0}, 'model_remains': [entry]}
    monkeypatch.setattr(health.urllib.request, 'urlopen', lambda *_args, **_kwargs: FakeHTTPResponse(payload))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: FakeGitHub())  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    secret = 'plan-key-private-fixture'
    monkeypatch.setenv('MINIMAX_API_KEY', secret)
    monkeypatch.setenv('MINIMAX_QUOTA_RESOURCE', 'general')
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


def test_payg_balance_validates_native_response_without_exposing_amount() -> None:
    """The native PAYG response validates status and balance without logging it."""
    assert _parse_balance({'base_resp': {'status_code': 0}, 'available_amount': '0.5'}).status == 'healthy'
    assert _parse_balance({'base_resp': {'status_code': 0}, 'available_amount': '0'}).status == 'exhausted'
    assert _parse_balance({'base_resp': {'status_code': 3}, 'available_amount': '50'}).status == 'unknown'


def test_check_summary_reports_plan_usage_left_and_window_resets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The trusted check reports validated remaining percentages and their reset times."""
    payload: dict[str, object] = {'base_resp': {'status_code': 0}, 'model_remains': [_quota_entry()]}
    monkeypatch.setattr(health.urllib.request, 'urlopen', lambda *_args, **_kwargs: FakeHTTPResponse(payload))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: FakeGitHub())  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setenv('MINIMAX_API_KEY', 'plan-key')
    monkeypatch.setenv('MINIMAX_QUOTA_RESOURCE', 'general')
    monkeypatch.setenv('GITHUB_WORKFLOW', 'nightly-sweep')
    monkeypatch.setenv('PYDANTIC_AI_TASK_KEY', 'task-1')
    monkeypatch.setenv('PYDANTIC_AI_TRIGGER_EVENT', 'schedule')
    monkeypatch.setenv('PYDANTIC_AI_RUN_ATTEMPT', '1')
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'github-token')
    summary = tmp_path / 'summary'
    monkeypatch.setenv('GITHUB_STEP_SUMMARY', str(summary))

    assert health.main(['check', '--output', str(tmp_path / 'provider-health.json')]) == 0

    summary_text = summary.read_text()
    assert 'interval: 80% remaining; resets at 2027-01-15T08:00:00Z' in summary_text
    assert 'weekly: 70% remaining; resets at 2027-01-15T08:00:00Z' in summary_text


def test_plan_key_without_active_plan_falls_back_to_native_balance(monkeypatch: pytest.MonkeyPatch) -> None:
    """MiniMax's no-plan response triggers a fresh native balance query."""
    responses = iter(
        [
            FakeHTTPResponse({'base_resp': {'status_code': 2062, 'status_msg': 'no active token plan subscription'}}),
            FakeHTTPResponse({'base_resp': {'status_code': 0}, 'available_amount': '0'}),
        ]
    )
    seen_urls: list[str] = []

    def open_url(request: health.urllib.request.Request, *, timeout: int) -> FakeHTTPResponse:
        seen_urls.append(request.full_url)
        assert timeout == 20
        return next(responses)

    monkeypatch.setattr(health.urllib.request, 'urlopen', open_url)

    assert _fetch_minimax_quota('legacy-key', 'general').status == 'exhausted'
    assert seen_urls == [health.MINIMAX_PLAN_URL, health.MINIMAX_BALANCE_URL]


def test_plan_key_fallback_with_positive_native_balance_is_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful fallback balance query can confirm a PAYG credential."""
    responses = iter(
        [
            FakeHTTPResponse({'base_resp': {'status_code': 2062}}),
            FakeHTTPResponse({'base_resp': {'status_code': 0}, 'available_amount': '2.50'}),
        ]
    )
    monkeypatch.setattr(health.urllib.request, 'urlopen', lambda *_args, **_kwargs: next(responses))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]

    assert _fetch_minimax_quota('legacy-key', 'general').status == 'healthy'


def test_health_check_blocks_unknown_empty_and_matching_incident() -> None:
    """Unknown and exhausted quotas or a provider incident block every workflow."""
    context = ('nightly-sweep', 'pr-45-head-a1b2', 'schedule', 1)

    assert not health.check_health(*context, health.Quota('unknown'), []).ready
    assert not health.check_health(*context, health.Quota('exhausted'), []).ready
    assert health.check_health(*context, health.Quota('healthy'), []).ready
    issue = _issue(7, health.IncidentMarker('provider', 'minimax', 'authentication', '12', None))
    blocked = health.check_health(*context, health.Quota('healthy'), [issue])
    assert not blocked.ready
    assert blocked.reason == 'Open operational incident #7 blocks inference'
    other_context = ('different-workflow', 'other-task', 'workflow_dispatch', 1)
    assert not health.check_health(*other_context, health.Quota('healthy'), [issue]).ready


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
    provider_issue = _issue(12, health.IncidentMarker('provider', 'minimax', 'balance', '12', None))
    assert not health.check_health(*context, quota, [provider_issue]).ready


def test_health_artifact_round_trips_versioned_gate_dto(tmp_path: Path) -> None:
    """The runner gate artifact round-trips its versioned validated DTO."""
    original = health.Health(
        'nightly-sweep',
        'task-1',
        'schedule',
        1,
        False,
        'MiniMax quota health is unknown',
        '2026-10-01T12:00:00Z',
        health.Quota(
            'unknown',
            interval_remaining_percent=12.5,
            weekly_remaining_percent=30,
            interval_reset_at='2026-10-01T13:00:00Z',
            weekly_reset_at='2026-10-07T00:00:00Z',
            interval_unlimited=False,
            weekly_unlimited=False,
        ),
    )
    path = tmp_path / 'provider-health.json'

    _write_health(path, original)

    assert _health_from_json(json.loads(path.read_text())) == original


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


def test_monitor_creates_one_assigned_labeled_incident_then_reuses_it() -> None:
    """An assigned incident is created once and reused without comment growth."""
    client = FakeGitHub()
    result = health.RunResult('nightly-sweep', 'task-1', 'schedule', 1, health.Failure('authentication', 401, None))

    first = _create_or_reuse_incident(client, result, '88', client.repo, 'MiniMax rejected credentials', dry_run=False)
    redelivery = _create_or_reuse_incident(
        client, result, '88', client.repo, 'MiniMax rejected credentials', dry_run=False
    )
    later_failure = _create_or_reuse_incident(
        client, result, '89', client.repo, 'MiniMax rejected credentials', dry_run=False
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
    """Real shim JSONL preserves usage, reconciles once, and blocks new workflows."""
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
            402,
            'MiniMax-M3',
            {'error': {'type': 'insufficient_balance_error', 'message': 'private provider detail'}},
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
        'failure': {'kind': 'balance', 'http_status': 402},
    }
    usage = result_event['usage']
    assert isinstance(usage, dict)
    assert usage['input_tokens'] == 23
    assert usage['output_tokens'] == 9
    assert usage['cache_read_input_tokens'] == 4
    assert usage['cache_creation_input_tokens'] == 2
    assert health.parse_run_result(agent_artifact) == health.RunResult(
        workflow, task_key, trigger_event, 1, health.Failure('balance', 402, None)
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
    assert marker == health.IncidentMarker('provider', 'minimax', 'balance', '70', None)
    assert 'private provider detail' not in issue.body
    monkeypatch.setattr(health, '_fetch_minimax_quota', lambda *_: health.Quota('healthy'))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setenv('GITHUB_WORKFLOW', 'Another workflow')
    monkeypatch.setenv('PYDANTIC_AI_TASK_KEY', 'different-task')
    monkeypatch.setenv('PYDANTIC_AI_TRIGGER_EVENT', 'schedule')
    monkeypatch.setenv('MINIMAX_API_KEY', 'sk-api-fresh-healthy-key')
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
            'MiniMax quota is exhausted',
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
        'provider', 'minimax', 'quota_exhausted', '80', '2026-10-01T13:00:00Z'
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
    existing = _issue(82, health.IncidentMarker('provider', 'minimax', 'authentication', '82', None))
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
            'MiniMax quota is exhausted',
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
        ({'base_resp': {'status_code': 0}, 'available_amount': '1.25'}, True),
        ({'base_resp': {'status_code': 7}, 'available_amount': '1.25'}, False),
    ],
)
def test_check_command_reads_bearer_balance_and_writes_secret_free_artifact(
    payload: dict[str, object], expected_ready: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trusted check reads native PAYG balance and emits only the gate decision."""
    requests: list[health.urllib.request.Request] = []

    def open_url(request: health.urllib.request.Request, *, timeout: int) -> FakeHTTPResponse:
        requests.append(request)
        assert timeout == 20
        return FakeHTTPResponse(payload)

    monkeypatch.setattr(health.urllib.request, 'urlopen', open_url)
    client = FakeGitHub()
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: client)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    key = 'sk-api-private-fixture-key'
    monkeypatch.setenv('MINIMAX_API_KEY', key)
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
    assert requests[0].full_url == health.MINIMAX_BALANCE_URL
    assert requests[0].get_header('Authorization') == f'Bearer {key}'
    assert key not in dto_text
    assert key not in github_output.read_text()
    assert key not in summary.read_text()
    balance_label = 'PAYG balance status: positive' if expected_ready else 'PAYG balance status: unknown'
    assert balance_label in summary.read_text()


def test_scope_rules_for_typed_and_untyped_failures() -> None:
    """Terminal failure kinds map to provider, task, or scheduled workflow scope."""
    assert _scope_for(health.RunResult('w', 't', 'schedule', 1, health.Failure('balance', None, None))) == (
        'provider',
        'minimax',
        'balance',
    )
    assert _scope_for(health.RunResult('w', 't', 'schedule', 1, health.Failure('other', 403, None))) == (
        'provider',
        'minimax',
        'other',
    )
    rate_limited = health.RunResult('w', 't', 'workflow_dispatch', 1, health.Failure('rate_limit', 429, None))
    assert _scope_for(rate_limited) == ('provider', 'minimax', 'rate_limit')
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
    elapsed = _issue(1, health.IncidentMarker('provider', 'minimax', 'rate_limit', '1', '2026-09-30T00:00:00Z'))
    future = _issue(2, health.IncidentMarker('provider', 'minimax', 'quota_exhausted', '2', '2026-12-01T00:00:00Z'))
    unknown = _issue(3, health.IncidentMarker('provider', 'minimax', 'quota_unknown', None, None))
    client = FakeGitHub([elapsed, future, unknown])
    monkeypatch.setenv('MINIMAX_API_KEY', 'plan-key')
    monkeypatch.setattr(health, '_fetch_minimax_quota', lambda *_: health.Quota('healthy'))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    monkeypatch.setattr(health, '_now', lambda: dt.datetime(2026, 10, 1, tzinfo=dt.timezone.utc))

    assert _reconcile_recovery(client, argparse.Namespace(dry_run=False, recover_issue=None)) == 0
    assert client.closed == [1]


def test_scheduled_monitor_creates_or_reuses_unknown_quota_incident(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated scheduled checks reuse one incident when quota remains unknown."""
    client = FakeGitHub()
    monkeypatch.setenv('MINIMAX_API_KEY', 'plan-key')
    monkeypatch.setenv('PYDANTIC_AI_RUN_ATTEMPT', '1')
    monkeypatch.setattr(health, '_fetch_minimax_quota', lambda *_: health.Quota('unknown'))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    args = argparse.Namespace(dry_run=False, recover_issue=None)

    _reconcile_recovery(client, args)
    _reconcile_recovery(client, args)

    assert len(client.posts) == 1
    payload: object = client.posts[0]
    assert _marker_from_payload(payload) == health.IncidentMarker('provider', 'minimax', 'quota_unknown', None, None)


def test_manual_recovery_requires_healthy_provider_and_targets_one_issue(monkeypatch: pytest.MonkeyPatch) -> None:
    """Manual provider recovery requires health and closes only the selected issue."""
    incident = _issue(5, health.IncidentMarker('provider', 'minimax', 'authentication', '5', None))
    other = _issue(6, health.IncidentMarker('provider', 'minimax', 'quota_unknown', None, None))
    client = FakeGitHub([incident, other])
    monkeypatch.setenv('MINIMAX_API_KEY', 'plan-key')
    monkeypatch.setattr(health, '_fetch_minimax_quota', lambda *_: health.Quota('unknown'))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]

    _reconcile_recovery(client, argparse.Namespace(dry_run=False, recover_issue=5))
    assert client.closed == []

    monkeypatch.setattr(health, '_fetch_minimax_quota', lambda *_: health.Quota('healthy'))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    _reconcile_recovery(client, argparse.Namespace(dry_run=False, recover_issue=5))
    assert client.closed == [5]


def test_named_recovery_without_provider_key_stays_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing provider credentials stay unknown even when the fetch stub is healthy."""
    issue = _issue(4, health.IncidentMarker('workflow', 'nightly-sweep', 'timeout', '4', None))
    client = FakeGitHub([issue])
    monkeypatch.delenv('MINIMAX_API_KEY', raising=False)
    monkeypatch.setattr(health, '_fetch_minimax_quota', lambda *_: health.Quota('healthy'))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]

    _reconcile_recovery(client, argparse.Namespace(dry_run=False, recover_issue=4))

    assert client.closed == []


def test_named_workflow_and_task_recovery_also_requires_healthy_quota(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every selected issue stays latched until a fresh quota check is healthy."""
    workflow_issue = _issue(7, health.IncidentMarker('workflow', 'nightly-sweep', 'timeout', '7', None))
    task_issue = _issue(8, health.IncidentMarker('task', 'nightly-sweep:task-1', 'timeout', '8', None))
    client = FakeGitHub([workflow_issue, task_issue])
    monkeypatch.setenv('MINIMAX_API_KEY', 'plan-key')
    monkeypatch.setattr(health, '_fetch_minimax_quota', lambda *_: health.Quota('unknown'))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]

    _reconcile_recovery(client, argparse.Namespace(dry_run=False, recover_issue=7))
    _reconcile_recovery(client, argparse.Namespace(dry_run=False, recover_issue=8))
    assert client.closed == []

    monkeypatch.setattr(health, '_fetch_minimax_quota', lambda *_: health.Quota('healthy'))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
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
            'MiniMax quota is exhausted',
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
    assert issue_marker == health.IncidentMarker('provider', 'minimax', 'quota_exhausted', '43', '2026-10-01T11:00:00Z')


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
                    'failure': {'kind': 'balance'},
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
