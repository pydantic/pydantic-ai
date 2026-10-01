"""Focused boundaries for the MiniMax inference gate and incident controller."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.usage import RunUsage

sys.path.insert(0, str(Path(__file__).parent))

import agent_provider_health as health
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
            data = payload if isinstance(payload, dict) else {}
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
    payload = {'model_remains': [_quota_entry()]}

    assert health._parse_plan_quota(payload, None).status == 'unknown'
    assert health._parse_plan_quota(payload, 'video').status == 'unknown'
    quota = health._parse_plan_quota(payload, 'general')
    assert quota.status == 'healthy'
    assert quota.interval_remaining_percent == 80
    assert quota.weekly_remaining_percent == 70
    assert quota.interval_reset_at == '2027-01-15T08:00:00Z'
    assert quota.weekly_reset_at == '2027-01-15T08:00:00Z'


def test_plan_quota_exhaustion_uses_status_and_preserves_reset() -> None:
    """An exhausted quota carries the provider's window reset into incident state."""
    quota = health._parse_plan_quota(
        {'model_remains': [_quota_entry(interval_status=2, interval_percent=0)]},
        'general',
    )

    assert quota.status == 'exhausted'
    assert quota.reset_at == '2027-01-15T08:00:00Z'


def test_unlimited_status_ignores_zero_remaining_percentage() -> None:
    """MiniMax status 3 means unlimited unless the not-in-plan sentinel applies."""
    quota = health._parse_plan_quota(
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
    assert health._parse_plan_quota(payload, 'general').status == 'unknown'


def test_payg_balance_validates_native_response_without_exposing_amount() -> None:
    """The native PAYG response validates status and balance without logging it."""
    assert health._parse_balance({'base_resp': {'status_code': 0}, 'available_amount': '0.5'}).status == 'healthy'
    assert health._parse_balance({'base_resp': {'status_code': 0}, 'available_amount': '0'}).status == 'exhausted'
    assert health._parse_balance({'base_resp': {'status_code': 3}, 'available_amount': '50'}).status == 'unknown'


def test_check_summary_reports_plan_usage_left_and_window_resets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The trusted check reports validated remaining percentages and their reset times."""
    payload = {'base_resp': {'status_code': 0}, 'model_remains': [_quota_entry()]}
    monkeypatch.setattr(health.urllib.request, 'urlopen', lambda *_args, **_kwargs: FakeHTTPResponse(payload))
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: FakeGitHub())
    monkeypatch.setenv('MINIMAX_API_KEY', 'plan-key')
    monkeypatch.setenv('MINIMAX_QUOTA_RESOURCE', 'general')
    monkeypatch.setenv('GITHUB_WORKFLOW', 'nightly-sweep')
    monkeypatch.setenv('PYDANTIC_AI_TASK_KEY', 'task-1')
    monkeypatch.setenv('PYDANTIC_AI_TRIGGER_EVENT', 'schedule')
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

    assert health._fetch_minimax_quota('legacy-key', 'general').status == 'exhausted'
    assert seen_urls == [health.MINIMAX_PLAN_URL, health.MINIMAX_BALANCE_URL]


def test_plan_key_fallback_with_positive_native_balance_is_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful fallback balance query can confirm a PAYG credential."""
    responses = iter(
        [
            FakeHTTPResponse({'base_resp': {'status_code': 2062}}),
            FakeHTTPResponse({'base_resp': {'status_code': 0}, 'available_amount': '2.50'}),
        ]
    )
    monkeypatch.setattr(health.urllib.request, 'urlopen', lambda *_args, **_kwargs: next(responses))

    assert health._fetch_minimax_quota('legacy-key', 'general').status == 'healthy'


def test_health_check_blocks_unknown_empty_and_matching_incident() -> None:
    """Unknown and exhausted quotas or a provider incident block every workflow."""
    context = ('nightly-sweep', 'pr-45-head-a1b2', 'schedule')

    assert not health.check_health(*context, health.Quota('unknown'), []).ready
    assert not health.check_health(*context, health.Quota('exhausted'), []).ready
    assert health.check_health(*context, health.Quota('healthy'), []).ready
    issue = _issue(7, health.IncidentMarker('provider', 'minimax', 'authentication', '12', None))
    blocked = health.check_health(*context, health.Quota('healthy'), [issue])
    assert not blocked.ready
    assert blocked.reason == 'Open operational incident #7 blocks inference'
    other_context = ('different-workflow', 'other-task', 'workflow_dispatch')
    assert not health.check_health(*other_context, health.Quota('healthy'), [issue]).ready


def test_workflow_and_task_incidents_match_only_their_scope() -> None:
    """Workflow and task incidents block matching identities only."""
    context = ('nightly-sweep', 'pr-45-head-a1b2', 'schedule')
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

    health._write_health(path, original)

    assert health._health_from_json(json.loads(path.read_text())) == original


def test_result_parser_reads_shim_terminal_jsonl_at_root(tmp_path: Path) -> None:
    """Terminal result parsing reads the nested shim metadata from JSONL."""
    result_file = tmp_path / 'agent-stdio.log'
    provider_health = {
        'workflow': 'nightly-sweep',
        'task_key': 'task-1',
        'trigger_event': 'schedule',
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
        'nightly-sweep', 'task-1', 'schedule', health.Failure('rate_limit', 429, None)
    )


def test_monitor_creates_one_assigned_labeled_incident_then_reuses_it() -> None:
    """An assigned incident is created once and reused without comment growth."""
    client = FakeGitHub()
    result = health.RunResult('nightly-sweep', 'task-1', 'schedule', health.Failure('authentication', 401, None))

    first = health._create_or_reuse_incident(
        client, result, '88', client.repo, 'MiniMax rejected credentials', dry_run=False
    )
    redelivery = health._create_or_reuse_incident(
        client, result, '88', client.repo, 'MiniMax rejected credentials', dry_run=False
    )
    later_failure = health._create_or_reuse_incident(
        client, result, '89', client.repo, 'MiniMax rejected credentials', dry_run=False
    )

    assert first is not None and redelivery is not None and later_failure is not None
    assert first.number == redelivery.number == later_failure.number
    assert len(client.posts) == 1
    payload = client.posts[0]
    assert isinstance(payload, dict)
    assert payload['labels'] == ['agentic-workflows', 'pydanty:meta']
    assert payload['assignees'] == ['dsfaccini']
    assert 'First failing run: https://github.com/org/repo/actions/runs/88' in str(payload['body'])
    assert 'amount' not in str(payload['body']).lower()


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
        'failure': {'kind': 'balance', 'http_status': 402},
    }
    usage = result_event['usage']
    assert isinstance(usage, dict)
    assert usage['input_tokens'] == 23
    assert usage['output_tokens'] == 9
    assert usage['cache_read_input_tokens'] == 4
    assert usage['cache_creation_input_tokens'] == 2
    assert health.parse_run_result(agent_artifact) == health.RunResult(
        workflow, task_key, trigger_event, health.Failure('balance', 402, None)
    )

    health_artifact = tmp_path / 'provider-health.json'
    health._write_health(
        health_artifact,
        health.Health(
            workflow,
            task_key,
            trigger_event,
            True,
            'Provider health check passed',
            '2026-10-01T12:00:00Z',
            health.Quota('healthy'),
        ),
    )
    client = FakeGitHub()
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: client)
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'token')
    monitor_args = SimpleNamespace(
        repository=None,
        run_id=70,
        conclusion='failure',
        health_artifact=health_artifact,
        agent_artifact=agent_artifact,
        dry_run=False,
        recover_issue=None,
    )

    health._monitor_command(monitor_args)
    health._monitor_command(monitor_args)
    health._monitor_command(
        SimpleNamespace(
            repository=None,
            run_id=71,
            conclusion='failure',
            health_artifact=health_artifact,
            agent_artifact=agent_artifact,
            dry_run=False,
            recover_issue=None,
        )
    )

    assert len(client.posts) == 1
    issue = client.issues[0]
    marker = health._marker_from_body(issue.body)
    assert marker == health.IncidentMarker('provider', 'minimax', 'balance', '70', None)
    assert 'private provider detail' not in issue.body
    monkeypatch.setattr(health, '_fetch_minimax_quota', lambda *_: health.Quota('healthy'))
    monkeypatch.setenv('GITHUB_WORKFLOW', 'Another workflow')
    monkeypatch.setenv('PYDANTIC_AI_TASK_KEY', 'different-task')
    monkeypatch.setenv('PYDANTIC_AI_TRIGGER_EVENT', 'schedule')
    monkeypatch.setenv('MINIMAX_API_KEY', 'sk-api-fresh-healthy-key')
    monkeypatch.delenv('GITHUB_OUTPUT', raising=False)
    monkeypatch.delenv('GITHUB_STEP_SUMMARY', raising=False)
    second_workflow_health = tmp_path / 'other-workflow-provider-health.json'

    assert health.main(['check', '--output', str(second_workflow_health)]) == 0
    assert not health._health_from_json(json.loads(second_workflow_health.read_text())).ready
    assert len(client.posts) == 1


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
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: client)
    key = 'sk-api-private-fixture-key'
    monkeypatch.setenv('MINIMAX_API_KEY', key)
    monkeypatch.setenv('GITHUB_WORKFLOW', 'Pydantic AI CI Review')
    monkeypatch.setenv('PYDANTIC_AI_TASK_KEY', 'task-1')
    monkeypatch.setenv('PYDANTIC_AI_TRIGGER_EVENT', 'workflow_run')
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'github-token')
    output = tmp_path / 'provider-health.json'
    github_output = tmp_path / 'output'
    summary = tmp_path / 'summary'
    monkeypatch.setenv('GITHUB_OUTPUT', str(github_output))
    monkeypatch.setenv('GITHUB_STEP_SUMMARY', str(summary))

    assert health.main(['check', '--output', str(output)]) == 0

    dto_text = output.read_text()
    assert health._health_from_json(json.loads(dto_text)).ready is expected_ready
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
    assert health._scope_for(health.RunResult('w', 't', 'schedule', health.Failure('balance', None, None))) == (
        'provider',
        'minimax',
        'balance',
    )
    assert health._scope_for(health.RunResult('w', 't', 'schedule', health.Failure('other', 403, None))) == (
        'provider',
        'minimax',
        'other',
    )
    rate_limited = health.RunResult('w', 't', 'workflow_dispatch', health.Failure('rate_limit', 429, None))
    assert health._scope_for(rate_limited) == ('provider', 'minimax', 'rate_limit')
    scheduled_timeout = health.RunResult('w', 'target-head-a1b2', 'schedule', health.Failure('timeout', None, None))
    repeated_scheduled_timeout = health.RunResult(
        'w', 'target-head-a1b2', 'schedule', health.Failure('timeout', None, None)
    )
    assert health._scope_for(scheduled_timeout) == ('workflow', 'w', 'timeout')
    assert health._scope_for(repeated_scheduled_timeout) == health._scope_for(scheduled_timeout)
    assert health._scope_for(
        health.RunResult('w', 'target-head-a1b2', 'workflow_dispatch', health.Failure('timeout', None, None))
    ) == ('task', 'w:target-head-a1b2', 'timeout')
    assert health._scope_for(health.RunResult('w', 't', 'schedule', None)) == ('workflow', 'w', 'execution')


def test_scheduled_recovery_closes_only_elapsed_known_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scheduled recovery closes elapsed windows but leaves future and unknown resets."""
    elapsed = _issue(1, health.IncidentMarker('provider', 'minimax', 'rate_limit', '1', '2026-09-30T00:00:00Z'))
    future = _issue(2, health.IncidentMarker('provider', 'minimax', 'quota_exhausted', '2', '2026-12-01T00:00:00Z'))
    unknown = _issue(3, health.IncidentMarker('provider', 'minimax', 'quota_unknown', None, None))
    client = FakeGitHub([elapsed, future, unknown])
    monkeypatch.setenv('MINIMAX_API_KEY', 'plan-key')
    monkeypatch.setattr(health, '_fetch_minimax_quota', lambda *_: health.Quota('healthy'))

    assert health._reconcile_recovery(client, SimpleNamespace(dry_run=False, recover_issue=None)) == 0
    assert client.closed == [1]


def test_scheduled_monitor_creates_or_reuses_unknown_quota_incident(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated scheduled checks reuse one incident when quota remains unknown."""
    client = FakeGitHub()
    monkeypatch.setenv('MINIMAX_API_KEY', 'plan-key')
    monkeypatch.setattr(health, '_fetch_minimax_quota', lambda *_: health.Quota('unknown'))
    args = SimpleNamespace(dry_run=False, recover_issue=None)

    health._reconcile_recovery(client, args)
    health._reconcile_recovery(client, args)

    assert len(client.posts) == 1
    payload = client.posts[0]
    assert isinstance(payload, dict)
    assert health._marker_from_body(str(payload['body'])) == health.IncidentMarker(
        'provider', 'minimax', 'quota_unknown', None, None
    )


def test_manual_recovery_requires_healthy_provider_and_targets_one_issue(monkeypatch: pytest.MonkeyPatch) -> None:
    """Manual provider recovery requires health and closes only the selected issue."""
    incident = _issue(5, health.IncidentMarker('provider', 'minimax', 'authentication', '5', None))
    other = _issue(6, health.IncidentMarker('provider', 'minimax', 'quota_unknown', None, None))
    client = FakeGitHub([incident, other])
    monkeypatch.setenv('MINIMAX_API_KEY', 'plan-key')
    monkeypatch.setattr(health, '_fetch_minimax_quota', lambda *_: health.Quota('unknown'))

    health._reconcile_recovery(client, SimpleNamespace(dry_run=False, recover_issue=5))
    assert client.closed == []

    monkeypatch.setattr(health, '_fetch_minimax_quota', lambda *_: health.Quota('healthy'))
    health._reconcile_recovery(client, SimpleNamespace(dry_run=False, recover_issue=5))
    assert client.closed == [5]


def test_named_recovery_without_provider_key_stays_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing provider credentials stay unknown even when the fetch stub is healthy."""
    issue = _issue(4, health.IncidentMarker('workflow', 'nightly-sweep', 'timeout', '4', None))
    client = FakeGitHub([issue])
    monkeypatch.delenv('MINIMAX_API_KEY', raising=False)
    monkeypatch.setattr(health, '_fetch_minimax_quota', lambda *_: health.Quota('healthy'))

    health._reconcile_recovery(client, SimpleNamespace(dry_run=False, recover_issue=4))

    assert client.closed == []


def test_named_workflow_and_task_recovery_also_requires_healthy_quota(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every selected issue stays latched until a fresh quota check is healthy."""
    workflow_issue = _issue(7, health.IncidentMarker('workflow', 'nightly-sweep', 'timeout', '7', None))
    task_issue = _issue(8, health.IncidentMarker('task', 'nightly-sweep:task-1', 'timeout', '8', None))
    client = FakeGitHub([workflow_issue, task_issue])
    monkeypatch.setenv('MINIMAX_API_KEY', 'plan-key')
    monkeypatch.setattr(health, '_fetch_minimax_quota', lambda *_: health.Quota('unknown'))

    health._reconcile_recovery(client, SimpleNamespace(dry_run=False, recover_issue=7))
    health._reconcile_recovery(client, SimpleNamespace(dry_run=False, recover_issue=8))
    assert client.closed == []

    monkeypatch.setattr(health, '_fetch_minimax_quota', lambda *_: health.Quota('healthy'))
    health._reconcile_recovery(client, SimpleNamespace(dry_run=False, recover_issue=7))
    assert client.closed == [7]
    health._reconcile_recovery(client, SimpleNamespace(dry_run=False, recover_issue=8))
    assert client.closed == [7, 8]


def test_monitor_ignores_successful_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A healthy successful workflow completion creates no incident."""
    artifact = tmp_path / 'provider-health.json'
    health._write_health(
        artifact,
        health.Health(
            'nightly-sweep',
            'task-1',
            'schedule',
            True,
            'Provider health check passed',
            '2026-10-01T12:00:00Z',
            health.Quota('healthy'),
        ),
    )
    client = FakeGitHub()
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: client)
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'token')

    result = health._monitor_command(
        SimpleNamespace(
            repository=None,
            run_id=42,
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
    health._write_health(
        health_artifact,
        health.Health(
            'nightly-sweep',
            'task-1',
            'schedule',
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
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: client)
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'token')

    result = health._monitor_command(
        SimpleNamespace(
            repository=None,
            run_id=45,
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
        payload = client.posts[0]
        assert isinstance(payload, dict)
        assert health._marker_from_body(str(payload['body'])) == health.IncidentMarker(
            'workflow', 'nightly-sweep', 'execution', '45', None
        )


def test_blocked_run_without_agent_artifact_creates_provider_incident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blocked gate creates an incident even when the model artifact is absent."""
    artifact = tmp_path / 'provider-health.json'
    health._write_health(
        artifact,
        health.Health(
            'nightly-sweep',
            'task-1',
            'schedule',
            False,
            'MiniMax quota is exhausted',
            '2026-10-01T12:00:00Z',
            health.Quota('exhausted', '2026-10-01T11:00:00Z'),
        ),
    )
    client = FakeGitHub()
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: client)
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'token')

    result = health._monitor_command(
        SimpleNamespace(
            repository=None,
            run_id=43,
            conclusion='success',
            health_artifact=artifact,
            agent_artifact=None,
            dry_run=False,
            recover_issue=None,
        )
    )

    assert result == 0
    assert len(client.posts) == 1
    payload = client.posts[0]
    assert isinstance(payload, dict)
    issue_marker = health._marker_from_body(str(payload['body']))
    assert issue_marker == health.IncidentMarker('provider', 'minimax', 'quota_exhausted', '43', '2026-10-01T11:00:00Z')


def test_monitor_uses_trusted_context_when_terminal_metadata_mismatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Uncorrelated shim metadata falls back to workflow scope from the trusted DTO."""
    health_artifact = tmp_path / 'provider-health.json'
    health._write_health(
        health_artifact,
        health.Health(
            'nightly-sweep',
            'task-1',
            'schedule',
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
                    'failure': {'kind': 'balance'},
                },
            }
        )
    )
    client = FakeGitHub()
    monkeypatch.setattr(health, 'GitHubClient', lambda *_: client)
    monkeypatch.setenv('GITHUB_REPOSITORY', 'org/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'token')

    health._monitor_command(
        SimpleNamespace(
            repository=None,
            run_id=44,
            conclusion='failure',
            health_artifact=health_artifact,
            agent_artifact=agent_log,
            dry_run=False,
            recover_issue=None,
        )
    )

    payload = client.posts[0]
    assert isinstance(payload, dict)
    assert health._marker_from_body(str(payload['body'])) == health.IncidentMarker(
        'workflow', 'nightly-sweep', 'execution', '44', None
    )
