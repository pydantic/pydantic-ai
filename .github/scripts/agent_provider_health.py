#!/usr/bin/env python3
"""Gate agent inference on provider health and reconcile operational incidents."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal

API_ROOT = 'https://api.github.com'
MINIMAX_PLAN_URL = 'https://api.minimax.io/v1/token_plan/remains'
MINIMAX_BALANCE_URL = 'https://api.minimax.io/account/query_balance'
MARKER_PREFIX = '<!-- pydantic-ai-provider-health:v1 '
MARKER_RE = re.compile(r'<!-- pydantic-ai-provider-health:v1 (\{[^\n]*\}) -->')
REQUIRED_LABELS = ('agentic-workflows', 'pydanty:meta')
ASSIGNEE = 'dsfaccini'

Scope = Literal['provider', 'workflow', 'task']
QuotaStatus = Literal['healthy', 'exhausted', 'unknown']
FailureKind = Literal['balance', 'authentication', 'rate_limit', 'request_limit', 'timeout', 'other']


@dataclass(frozen=True)
class Quota:
    """Validated provider quota state and an optional reset time."""

    status: QuotaStatus
    reset_at: str | None = None
    interval_remaining_percent: float | None = None
    weekly_remaining_percent: float | None = None
    interval_reset_at: str | None = None
    weekly_reset_at: str | None = None
    interval_unlimited: bool | None = None
    weekly_unlimited: bool | None = None


@dataclass(frozen=True)
class Health:
    """The trusted gate decision written to the workflow artifact."""

    workflow: str
    task_key: str
    trigger_event: str
    run_attempt: int
    ready: bool
    reason: str
    checked_at: str
    quota: Quota

    def __post_init__(self) -> None:
        if _positive_attempt(self.run_attempt) is None:
            raise ValueError('run_attempt must be a positive integer')


@dataclass(frozen=True)
class IncidentMarker:
    """Versioned incident identity stored in an issue-body marker."""

    scope: Scope
    key: str
    kind: str
    run_id: str | None
    reset_at: str | None


@dataclass(frozen=True)
class Issue:
    """The validated GitHub issue fields used by reconciliation."""

    number: int
    title: str
    body: str
    state: str
    html_url: str


@dataclass(frozen=True)
class Failure:
    """Typed terminal provider failure metadata emitted by the runner shim."""

    kind: FailureKind
    http_status: int | None
    retry_at: str | None


@dataclass(frozen=True)
class RunResult:
    """Final result metadata correlated with the trusted health artifact."""

    workflow: str
    task_key: str
    trigger_event: str
    run_attempt: int
    failure: Failure | None

    def __post_init__(self) -> None:
        if _positive_attempt(self.run_attempt) is None:
            raise ValueError('run_attempt must be a positive integer')


def _mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        return None
    return value


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _object_string(data: dict[str, object], key: str) -> str | None:
    return _string(data.get(key))


def _positive_attempt(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _run_attempt_from_env() -> int:
    value = os.environ.get('PYDANTIC_AI_RUN_ATTEMPT')
    if value is None or not value.isascii() or not value.isdecimal():
        raise ValueError('PYDANTIC_AI_RUN_ATTEMPT must be a positive integer')
    attempt = int(value)
    if attempt < 1:
        raise ValueError('PYDANTIC_AI_RUN_ATTEMPT must be a positive integer')
    return attempt


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _timestamp(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


def _iso_time(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return _timestamp(parsed)


def _write_health(path: Path, health: Health) -> None:
    payload: dict[str, object] = {
        'version': 1,
        'workflow': health.workflow,
        'task_key': health.task_key,
        'trigger_event': health.trigger_event,
        'run_attempt': health.run_attempt,
        'ready': health.ready,
        'reason': health.reason,
        'checked_at': health.checked_at,
        'quota': {
            'status': health.quota.status,
            'reset_at': health.quota.reset_at,
            'interval_remaining_percent': health.quota.interval_remaining_percent,
            'weekly_remaining_percent': health.quota.weekly_remaining_percent,
            'interval_reset_at': health.quota.interval_reset_at,
            'weekly_reset_at': health.quota.weekly_reset_at,
            'interval_unlimited': health.quota.interval_unlimited,
            'weekly_unlimited': health.quota.weekly_unlimited,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')


def _health_from_json(value: object) -> Health:
    data = _mapping(value)
    quota_data = _mapping(data.get('quota')) if data else None
    version = data.get('version') if data is not None else None
    if data is None or quota_data is None or isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ValueError('provider-health artifact has an unsupported or invalid shape')
    workflow = _object_string(data, 'workflow')
    task_key = _object_string(data, 'task_key')
    trigger_event = _object_string(data, 'trigger_event')
    run_attempt = _positive_attempt(data.get('run_attempt'))
    reason = _object_string(data, 'reason')
    checked_at = _iso_time(data.get('checked_at'))
    status = quota_data.get('status')
    ready = data.get('ready')
    if status == 'healthy':
        quota_status: QuotaStatus = 'healthy'
    elif status == 'exhausted':
        quota_status = 'exhausted'
    elif status == 'unknown':
        quota_status = 'unknown'
    else:
        raise ValueError('provider-health artifact has an invalid quota status')
    reset_at = _iso_time(quota_data.get('reset_at')) if quota_data.get('reset_at') is not None else None
    interval_reset_at = (
        _iso_time(quota_data.get('interval_reset_at')) if quota_data.get('interval_reset_at') is not None else None
    )
    weekly_reset_at = (
        _iso_time(quota_data.get('weekly_reset_at')) if quota_data.get('weekly_reset_at') is not None else None
    )
    interval_percent = quota_data.get('interval_remaining_percent')
    weekly_percent = quota_data.get('weekly_remaining_percent')
    interval_unlimited = quota_data.get('interval_unlimited')
    weekly_unlimited = quota_data.get('weekly_unlimited')
    for percent in (interval_percent, weekly_percent):
        if percent is not None and (
            isinstance(percent, bool)
            or not isinstance(percent, (int, float))
            or not math.isfinite(percent)
            or not 0 <= percent <= 100
        ):
            raise ValueError('provider-health artifact has an invalid remaining percentage')
    if (
        workflow is None
        or task_key is None
        or trigger_event is None
        or run_attempt is None
        or reason is None
        or checked_at is None
        or not isinstance(ready, bool)
        or (quota_data.get('reset_at') is not None and reset_at is None)
        or (quota_data.get('interval_reset_at') is not None and interval_reset_at is None)
        or (quota_data.get('weekly_reset_at') is not None and weekly_reset_at is None)
        or (interval_unlimited is not None and not isinstance(interval_unlimited, bool))
        or (weekly_unlimited is not None and not isinstance(weekly_unlimited, bool))
    ):
        raise ValueError('provider-health artifact is missing required validated fields')
    return Health(
        workflow,
        task_key,
        trigger_event,
        run_attempt,
        ready,
        reason,
        checked_at,
        Quota(
            quota_status,
            reset_at,
            interval_percent,
            weekly_percent,
            interval_reset_at,
            weekly_reset_at,
            interval_unlimited,
            weekly_unlimited,
        ),
    )


def _quota_window_status(status: object, percent: object, total: object) -> QuotaStatus:
    """Classify one validated MiniMax quota window."""
    if isinstance(status, bool) or not isinstance(status, int) or status not in (1, 2, 3):
        return 'unknown'
    if isinstance(total, bool) or not isinstance(total, (int, float)) or not math.isfinite(total) or total < 0:
        return 'unknown'
    if percent is not None and (
        isinstance(percent, bool)
        or not isinstance(percent, (int, float))
        or not math.isfinite(percent)
        or not 0 <= percent <= 100
    ):
        return 'unknown'
    if status == 1 and percent is None:
        return 'unknown'
    if status == 2 or (status == 1 and isinstance(percent, (int, float)) and percent <= 0):
        return 'exhausted'
    if status == 3 or status == 1 or (isinstance(percent, (int, float)) and percent > 0):
        return 'healthy'
    return 'unknown'


def _parse_plan_quota(value: object, resource: str | None) -> Quota:
    if not resource:
        return Quota('unknown')
    data = _mapping(value)
    response = _mapping(data.get('base_resp')) if data else None
    if response is not None:
        status_code = response.get('status_code')
        if isinstance(status_code, bool) or not isinstance(status_code, int) or status_code != 0:
            return Quota('unknown')
    remains = data.get('model_remains') if data else None
    if not isinstance(remains, list) or not remains:
        return Quota('unknown')
    matching_rows = 0
    exhausted = False
    healthy = False
    resets: list[int | float] = []
    window_resets: list[str] = []
    remaining_percent: list[float | None] = []
    unlimited: list[bool] = []
    for item in remains:
        entry = _mapping(item)
        if entry is None:
            return Quota('unknown')
        if entry.get('model_name') != resource:
            continue
        matching_rows += 1
        statuses = (entry.get('current_interval_status'), entry.get('current_weekly_status'))
        percentages = (entry.get('current_interval_remaining_percent'), entry.get('current_weekly_remaining_percent'))
        totals = (entry.get('current_interval_total_count'), entry.get('current_weekly_total_count'))
        ends = (entry.get('end_time'), entry.get('weekly_end_time'))
        if statuses == (3, 3) and totals == (0, 0):
            return Quota('unknown')
        for status, percent, total, end in zip(statuses, percentages, totals, ends, strict=True):
            if not isinstance(end, (int, float)) or isinstance(end, bool) or not math.isfinite(end):
                return Quota('unknown')
            window_status = _quota_window_status(status, percent, total)
            if window_status == 'unknown':
                return Quota('unknown')
            if window_status == 'exhausted':
                exhausted = True
                resets.append(end)
            else:
                healthy = True
            window_resets.append(_timestamp(dt.datetime.fromtimestamp(end / 1000, tz=dt.timezone.utc)))
            remaining_percent.append(float(percent) if status != 3 and isinstance(percent, (int, float)) else None)
            unlimited.append(status == 3)
    if matching_rows != 1:
        return Quota('unknown')
    reset_at: str | None = None
    if exhausted:
        reset_ms = min(resets) if resets else None
        reset_at = (
            _timestamp(dt.datetime.fromtimestamp(reset_ms / 1000, tz=dt.timezone.utc)) if reset_ms is not None else None
        )
        status: QuotaStatus = 'exhausted'
    else:
        status = 'healthy' if healthy else 'unknown'
    return Quota(
        status,
        reset_at,
        remaining_percent[0],
        remaining_percent[1],
        window_resets[0],
        window_resets[1],
        unlimited[0],
        unlimited[1],
    )


def _parse_balance(value: object) -> Quota:
    data = _mapping(value)
    if data is None:
        return Quota('unknown')
    response = _mapping(data.get('base_resp'))
    status_code = response.get('status_code') if response is not None else None
    if isinstance(status_code, bool) or not isinstance(status_code, int) or status_code != 0:
        return Quota('unknown')
    balance = data.get('available_amount')
    if isinstance(balance, str):
        try:
            amount = Decimal(balance)
        except InvalidOperation:
            return Quota('unknown')
        if amount.is_finite():
            return Quota('healthy' if amount > 0 else 'exhausted')
    elif isinstance(balance, (int, float)) and not isinstance(balance, bool):
        if math.isfinite(balance):
            return Quota('healthy' if balance > 0 else 'exhausted')
    return Quota('unknown')


def _fetch_minimax_quota(api_key: str, resource: str | None = None) -> Quota:
    url = MINIMAX_BALANCE_URL if api_key.startswith('sk-api-') else MINIMAX_PLAN_URL
    headers: dict[str, str] = {'Authorization': f'Bearer {api_key}', 'Accept': 'application/json'}
    request = urllib.request.Request(url, headers=headers, method='GET')
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload: object = json.loads(response.read())
    except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError):
        return Quota('unknown')
    if url == MINIMAX_PLAN_URL:
        data = _mapping(payload)
        plan_status = _mapping(data.get('base_resp')) if data else None
        plan_code = plan_status.get('status_code') if plan_status is not None else None
        if isinstance(plan_code, int) and not isinstance(plan_code, bool) and plan_code == 2062:
            # MiniMax reported no active plan for this credential. Check its separate
            # native account-balance endpoint before treating the provider as unknown.
            balance_request = urllib.request.Request(MINIMAX_BALANCE_URL, headers=headers, method='GET')
            try:
                with urllib.request.urlopen(balance_request, timeout=20) as response:
                    payload = json.loads(response.read())
            except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError):
                return Quota('unknown')
            return _parse_balance(payload)
    return _parse_balance(payload) if url == MINIMAX_BALANCE_URL else _parse_plan_quota(payload, resource)


class GitHubClient:
    """Small GitHub REST client for issue-backed controller state."""

    def __init__(self, repo: str, token: str) -> None:
        self.repo = repo
        self.token = token

    def request(self, method: str, path: str, payload: object | None = None) -> object:
        body = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(f'{API_ROOT}/repos/{self.repo}/{path}', data=body, method=method)
        request.add_header('Authorization', f'Bearer {self.token}')
        request.add_header('Accept', 'application/vnd.github+json')
        request.add_header('X-GitHub-Api-Version', '2022-11-28')
        if body is not None:
            request.add_header('Content-Type', 'application/json')
        with urllib.request.urlopen(request, timeout=30) as response:
            payload: object = json.loads(response.read()) if response.status != 204 else {}
            return payload

    def open_incidents(self) -> list[Issue]:
        found: list[Issue] = []
        page = 1
        labels = urllib.parse.quote(','.join(REQUIRED_LABELS), safe=',')
        while True:
            response = self.request('GET', f'issues?state=open&labels={labels}&per_page=100&page={page}')
            if not isinstance(response, list):
                raise ValueError('GitHub returned an invalid open-issues response')
            batch = [issue for entry in response if (issue := _issue(entry)) is not None]
            found.extend(batch)
            if len(response) < 100:
                return found
            page += 1

    def create_incident(self, title: str, body: str) -> Issue:
        response = self.request(
            'POST',
            'issues',
            {'title': title, 'body': body, 'labels': list(REQUIRED_LABELS), 'assignees': [ASSIGNEE]},
        )
        issue = _issue(response)
        if issue is None:
            raise ValueError('GitHub returned an invalid issue after incident creation')
        return issue

    def close_issue(self, number: int) -> None:
        self.request('PATCH', f'issues/{number}', {'state': 'closed'})


def _issue(value: object) -> Issue | None:
    data = _mapping(value)
    if data is None:
        return None
    number = data.get('number')
    title = _object_string(data, 'title')
    body = data.get('body')
    state = _object_string(data, 'state')
    url = _object_string(data, 'html_url')
    if isinstance(number, bool) or not isinstance(number, int) or title is None or not isinstance(body, str):
        return None
    if state is None or url is None:
        return None
    return Issue(number, title, body, state, url)


def _marker_from_body(body: str) -> IncidentMarker | None:
    match = MARKER_RE.search(body)
    if match is None:
        return None
    try:
        decoded: object = json.loads(match.group(1))
        data = _mapping(decoded)
    except json.JSONDecodeError:
        return None
    version = data.get('version') if data is not None else None
    if data is None or isinstance(version, bool) or not isinstance(version, int) or version != 1:
        return None
    scope = data.get('scope')
    key = _object_string(data, 'key')
    kind = _object_string(data, 'kind')
    run_id = _string(data.get('run_id'))
    reset_at = _iso_time(data.get('reset_at')) if data.get('reset_at') is not None else None
    if scope not in ('provider', 'workflow', 'task') or key is None or kind is None:
        return None
    if data.get('reset_at') is not None and reset_at is None:
        return None
    return IncidentMarker(scope, key, kind, run_id, reset_at)


def _matching_incident(health: Health, issues: list[Issue]) -> Issue | None:
    task_scope_key = f'{health.workflow}:{health.task_key}'
    for issue in issues:
        marker = _marker_from_body(issue.body)
        if marker is None:
            continue
        if marker.scope == 'provider' and marker.key == 'minimax':
            return issue
        if marker.scope == 'workflow' and marker.key == health.workflow:
            return issue
        if marker.scope == 'task' and marker.key == task_scope_key:
            return issue
    return None


def _health_decision(health: Health, issues: list[Issue]) -> Health:
    incident = _matching_incident(health, issues)
    if incident is not None:
        return Health(
            health.workflow,
            health.task_key,
            health.trigger_event,
            health.run_attempt,
            False,
            f'Open operational incident #{incident.number} blocks inference',
            health.checked_at,
            health.quota,
        )
    if health.quota.status != 'healthy':
        message = (
            'MiniMax quota is exhausted' if health.quota.status == 'exhausted' else 'MiniMax quota health is unknown'
        )
        return Health(
            health.workflow,
            health.task_key,
            health.trigger_event,
            health.run_attempt,
            False,
            message,
            health.checked_at,
            health.quota,
        )
    return Health(
        health.workflow,
        health.task_key,
        health.trigger_event,
        health.run_attempt,
        True,
        'Provider health check passed',
        health.checked_at,
        health.quota,
    )


def check_health(
    workflow: str,
    task_key: str,
    trigger_event: str,
    run_attempt: int,
    quota: Quota,
    issues: list[Issue],
    checked_at: dt.datetime | None = None,
) -> Health:
    """Return the inference gate decision from provider and incident state."""
    if _positive_attempt(run_attempt) is None:
        raise ValueError('run_attempt must be a positive integer')
    base = Health(workflow, task_key, trigger_event, run_attempt, True, '', _timestamp(checked_at or _now()), quota)
    return _health_decision(base, issues)


def _failure_from(value: object) -> Failure | None:
    data = _mapping(value)
    if data is None:
        return None
    kind = data.get('kind')
    if kind not in ('balance', 'authentication', 'rate_limit', 'request_limit', 'timeout', 'other'):
        return None
    status = data.get('http_status')
    if status is not None and (isinstance(status, bool) or not isinstance(status, int) or not 100 <= status <= 599):
        return None
    retry_at = _iso_time(data.get('retry_at')) if data.get('retry_at') is not None else None
    if data.get('retry_at') is not None and retry_at is None:
        return None
    return Failure(kind, status, retry_at)


def _run_result_from(value: object) -> RunResult | None:
    data = _mapping(value)
    if data is None or data.get('type') != 'result':
        return None
    data = _mapping(data.get('provider_health'))
    if data is None:
        return None
    workflow = _object_string(data, 'workflow')
    task_key = _object_string(data, 'task_key')
    trigger_event = _object_string(data, 'trigger_event')
    run_attempt = _positive_attempt(data.get('run_attempt'))
    if workflow is None or task_key is None or trigger_event is None or run_attempt is None:
        return None
    failure_value = data.get('failure')
    failure = None if failure_value is None else _failure_from(failure_value)
    if failure_value is not None and failure is None:
        return None
    return RunResult(workflow, task_key, trigger_event, run_attempt, failure)


def _json_values(path: Path) -> list[object]:
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(file for file in path.rglob('*') if file.is_file() and file.suffix in ('.json', '.jsonl'))
    else:
        raise ValueError(f'artifact path does not exist: {path}')
    values: list[object] = []
    for file in files:
        text = file.read_text(encoding='utf-8')
        try:
            decoded: object = json.loads(text)
            values.append(decoded)
        except json.JSONDecodeError:
            for line in text.splitlines():
                try:
                    decoded = json.loads(line)
                    values.append(decoded)
                except json.JSONDecodeError:
                    continue
    return values


def parse_run_result(path: Path) -> RunResult | None:
    """Find validated terminal provider metadata in a downloaded agent artifact."""
    if not path.exists():
        return None
    for value in _json_values(path):
        result = _run_result_from(value)
        if result is not None:
            return result
    return None


def _scope_for(result: RunResult) -> tuple[Scope, str, str]:
    failure = result.failure
    if failure is None:
        return 'workflow', result.workflow, 'execution'
    if failure.kind in ('balance', 'authentication') or failure.http_status in (401, 403):
        return 'provider', 'minimax', failure.kind
    if failure.kind == 'rate_limit':
        return 'provider', 'minimax', failure.kind
    if result.trigger_event == 'schedule':
        return 'workflow', result.workflow, failure.kind
    return 'task', f'{result.workflow}:{result.task_key}', failure.kind


def _failure_reason(failure: Failure | None) -> str:
    if failure is None:
        return 'Agent execution failed before a typed provider result was available'
    reasons: dict[FailureKind, str] = {
        'balance': 'MiniMax reported an insufficient balance',
        'authentication': 'MiniMax rejected the configured credentials',
        'rate_limit': 'MiniMax rate limits stopped the run',
        'request_limit': 'The agent reached its per-run request limit',
        'timeout': 'A provider request timed out',
        'other': 'The provider request failed with a typed error',
    }
    reason = reasons[failure.kind]
    return f'{reason} (HTTP {failure.http_status})' if failure.http_status is not None else reason


def _incident_body(marker: IncidentMarker, workflow: str, task_key: str, run_id: str, repo: str, reason: str) -> str:
    marker_data: dict[str, object] = {
        'version': 1,
        'scope': marker.scope,
        'key': marker.key,
        'kind': marker.kind,
        'run_id': marker.run_id,
        'reset_at': marker.reset_at,
    }
    run_url = f'https://github.com/{repo}/actions/runs/{urllib.parse.quote(run_id)}'
    return (
        f'## Operational failure\n\n{reason}\n\n'
        f'- Workflow: `{workflow}`\n- Task: `{task_key}`\n- First failing run: {run_url}\n\n'
        'Confirm provider credentials, configuration, and quota health. Close this incident after the cause is '
        'fixed and the required recovery check passes.\n\n'
        f'{MARKER_PREFIX}{json.dumps(marker_data, separators=(",", ":"))} -->\n'
    )


def _create_or_reuse_incident(
    client: GitHubClient,
    result: RunResult,
    run_id: str,
    repo: str,
    reason: str,
    *,
    dry_run: bool,
    marker_override: IncidentMarker | None = None,
) -> Issue | None:
    scope, key, kind = _scope_for(result)
    marker = marker_override or IncidentMarker(
        scope, key, kind, run_id, result.failure.retry_at if result.failure else None
    )
    matching_key = Health(
        result.workflow,
        result.task_key,
        result.trigger_event,
        result.run_attempt,
        False,
        '',
        _timestamp(_now()),
        Quota('unknown'),
    )
    if marker_override is not None:
        existing = next(
            (
                issue
                for issue in client.open_incidents()
                if (candidate := _marker_from_body(issue.body)) is not None
                and candidate.scope == marker.scope
                and candidate.key == marker.key
            ),
            None,
        )
    else:
        existing = _matching_incident(matching_key, client.open_incidents())
    if existing is not None:
        print(f'Reusing operational incident #{existing.number}: {existing.html_url}')
        return existing
    title = f'Agent workflow {marker.scope} failure: {result.workflow}'
    body = _incident_body(marker, result.workflow, result.task_key, run_id, repo, reason)
    if dry_run:
        print(f'DRY RUN: would create assigned incident: {title}')
        print(body)
        return None
    issue = client.create_incident(title, body)
    print(f'Created operational incident #{issue.number}: {issue.html_url}')
    return issue


def _health_from_artifact(path: Path) -> Health:
    values = _json_values(path)
    for value in values:
        try:
            return _health_from_json(value)
        except ValueError:
            continue
    raise ValueError('provider-health artifact contains no valid health DTO')


def _check_command(args: argparse.Namespace) -> int:
    workflow = os.environ.get('GITHUB_WORKFLOW', '')
    task_key = os.environ.get('PYDANTIC_AI_TASK_KEY', '')
    trigger_event = os.environ.get('PYDANTIC_AI_TRIGGER_EVENT', '')
    run_attempt = _run_attempt_from_env()
    api_key = os.environ.get('MINIMAX_API_KEY', '')
    repo = os.environ.get('GITHUB_REPOSITORY', '')
    token = os.environ.get('GITHUB_TOKEN', '')
    if not repo or not token:
        raise ValueError('GITHUB_REPOSITORY and GITHUB_TOKEN are required')
    quota = _fetch_minimax_quota(api_key, os.environ.get('MINIMAX_QUOTA_RESOURCE')) if api_key else Quota('unknown')
    client = GitHubClient(repo, token)
    health = check_health(
        workflow or 'unknown',
        task_key or 'unknown',
        trigger_event or 'unknown',
        run_attempt,
        quota,
        client.open_incidents(),
    )
    if not workflow or not task_key or not trigger_event:
        health = Health(
            health.workflow,
            health.task_key,
            health.trigger_event,
            health.run_attempt,
            False,
            'Workflow task identity is unavailable',
            health.checked_at,
            health.quota,
        )
    _write_health(args.output, health)
    if output := os.environ.get('GITHUB_OUTPUT'):
        with open(output, 'a', encoding='utf-8') as handle:
            handle.write(f'ready={str(health.ready).lower()}\nreason={health.reason}\n')
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        quota_detail = ''
        if api_key.startswith('sk-api-'):
            balance_state = (
                'positive'
                if health.quota.status == 'healthy'
                else 'nonpositive'
                if health.quota.status == 'exhausted'
                else 'unknown'
            )
            quota_detail = f'\n\nPAYG balance status: {balance_state}.'
        elif health.quota.interval_reset_at is not None and health.quota.weekly_reset_at is not None:
            windows = (
                (
                    'interval',
                    health.quota.interval_remaining_percent,
                    health.quota.interval_reset_at,
                    health.quota.interval_unlimited,
                ),
                (
                    'weekly',
                    health.quota.weekly_remaining_percent,
                    health.quota.weekly_reset_at,
                    health.quota.weekly_unlimited,
                ),
            )
            details: list[str] = []
            for name, percent, reset_at, is_unlimited in windows:
                remaining = (
                    'unlimited'
                    if is_unlimited
                    else f'{percent:g}% remaining'
                    if percent is not None
                    else 'remaining percentage unknown'
                )
                details.append(f'{name}: {remaining}; resets at {reset_at}')
            quota_detail = '\n\nPlan quota windows: ' + '; '.join(details) + '.'
        with open(summary, 'a', encoding='utf-8') as handle:
            handle.write(
                f'### Provider health: {"ready" if health.ready else "blocked"}\n\n{health.reason}.{quota_detail}\n'
            )
    print(f'{"ready" if health.ready else "blocked"}: {health.reason}')
    return 0


def _monitor_command(args: argparse.Namespace) -> int:
    repo = args.repository or os.environ.get('GITHUB_REPOSITORY', '')
    token = os.environ.get('GITHUB_TOKEN', '')
    if not repo or not token:
        raise ValueError('GITHUB_REPOSITORY and GITHUB_TOKEN are required')
    client = GitHubClient(repo, token)
    if args.run_id is None:
        return _reconcile_recovery(client, args)
    if args.health_artifact is None:
        raise ValueError('--health-artifact is required with --run-id')
    run_attempt = _positive_attempt(args.run_attempt)
    if run_attempt is None:
        raise ValueError('--run-attempt must be a positive integer with --run-id')
    health = _health_from_artifact(args.health_artifact)
    if health.run_attempt != run_attempt:
        raise ValueError('provider-health artifact run attempt does not match the triggering workflow_run attempt')
    result = parse_run_result(args.agent_artifact) if args.agent_artifact is not None else None
    if result is not None and (
        result.workflow != health.workflow
        or result.task_key != health.task_key
        or result.trigger_event != health.trigger_event
        or result.run_attempt != run_attempt
    ):
        print('Agent result identity did not match the trusted provider-health artifact; treating it as untyped')
        result = None
    if (
        args.conclusion is not None
        and args.conclusion not in ('failure', 'timed_out')
        and health.ready
        and (result is None or result.failure is None)
    ):
        print(f'Workflow concluded {args.conclusion}; no operational incident was created')
        return 0
    if result is None:
        if not health.ready:
            if health.reason.startswith('Open operational incident #'):
                print(f'Activation was blocked by an existing incident: {health.reason}')
                return 0
            if health.quota.status == 'healthy':
                print(f'Activation was blocked by existing state: {health.reason}')
                return 0
            kind = 'quota_unknown' if health.quota.status == 'unknown' else 'quota_exhausted'
            marker = IncidentMarker('provider', 'minimax', kind, str(args.run_id), health.quota.reset_at)
            result = RunResult(health.workflow, health.task_key, health.trigger_event, run_attempt, None)
            existing = next(
                (issue for issue in client.open_incidents() if _marker_from_body(issue.body) == marker),
                None,
            )
            if existing is None:
                _create_or_reuse_incident(
                    client,
                    result,
                    str(args.run_id),
                    repo,
                    health.reason,
                    dry_run=args.dry_run,
                    marker_override=marker,
                )
            else:
                print(f'Reusing operational incident #{existing.number}: {existing.html_url}')
            return 0
        result = RunResult(health.workflow, health.task_key, health.trigger_event, run_attempt, None)
    reason = health.reason if not health.ready else _failure_reason(result.failure)
    if health.ready and result.failure is not None:
        reason = _failure_reason(result.failure)
    _create_or_reuse_incident(client, result, str(args.run_id), repo, reason, dry_run=args.dry_run)
    return 0


def _recover_issue(
    client: GitHubClient,
    issue: Issue,
    marker: IncidentMarker,
    quota: Quota,
    now: dt.datetime,
    *,
    dry_run: bool,
    automatic: bool,
) -> bool:
    """Close one incident only when its configured recovery condition holds."""
    if quota.status != 'healthy':
        if not automatic:
            print(f'Incident #{issue.number} remains open: provider health is {quota.status}')
        return False
    if marker.kind in ('rate_limit', 'quota_exhausted'):
        if marker.reset_at is None:
            if automatic:
                return False
        else:
            reset_at = dt.datetime.fromisoformat(marker.reset_at.replace('Z', '+00:00'))
            if now < reset_at:
                if not automatic:
                    print(f'Incident #{issue.number} remains open until the recorded reset time')
                return False
    if dry_run:
        print(f'DRY RUN: recovery checks passed; would close incident #{issue.number}')
    else:
        client.close_issue(issue.number)
        print(f'Closed recovered incident #{issue.number}: {issue.html_url}')
    return True


def _reconcile_recovery(client: GitHubClient, args: argparse.Namespace) -> int:
    api_key = os.environ.get('MINIMAX_API_KEY', '')
    quota = _fetch_minimax_quota(api_key, os.environ.get('MINIMAX_QUOTA_RESOURCE')) if api_key else Quota('unknown')
    now = _now()
    issues = client.open_incidents()
    if args.recover_issue is not None:
        issue = next((issue for issue in issues if issue.number == args.recover_issue), None)
        if issue is None:
            raise ValueError(f'open operational incident #{args.recover_issue} was not found')
        marker = _marker_from_body(issue.body)
        if marker is None:
            raise ValueError(f'issue #{issue.number} is not a provider-health incident')
        _recover_issue(client, issue, marker, quota, now, dry_run=args.dry_run, automatic=False)
        return 0
    if quota.status != 'healthy':
        marker = IncidentMarker(
            'provider',
            'minimax',
            'quota_unknown' if quota.status == 'unknown' else 'quota_exhausted',
            None,
            quota.reset_at if quota.status == 'exhausted' else None,
        )
        result = RunResult(
            os.environ.get('GITHUB_WORKFLOW', 'provider-health-recovery'),
            os.environ.get('PYDANTIC_AI_TASK_KEY', 'provider-health-recovery'),
            os.environ.get('PYDANTIC_AI_TRIGGER_EVENT', 'schedule'),
            _run_attempt_from_env(),
            Failure('other', None, None),
        )
        existing = next(
            (
                issue
                for issue in issues
                if (candidate := _marker_from_body(issue.body)) is not None
                and candidate.scope == marker.scope
                and candidate.key == marker.key
            ),
            None,
        )
        if existing is not None:
            print(f'Provider quota remains {quota.status}; incident #{existing.number} is open')
        elif not args.dry_run:
            _create_or_reuse_incident(
                client,
                result,
                os.environ.get('GITHUB_RUN_ID', 'unknown'),
                client.repo,
                'MiniMax quota could not be confirmed healthy',
                dry_run=False,
                marker_override=marker,
            )
        else:
            print(f'DRY RUN: provider quota is {quota.status}; would open/reuse an incident')
        return 0
    for issue in issues:
        marker = _marker_from_body(issue.body)
        if marker is None or marker.scope != 'provider':
            continue
        if marker.kind not in ('rate_limit', 'quota_exhausted') or marker.reset_at is None:
            continue
        _recover_issue(client, issue, marker, quota, now, dry_run=args.dry_run, automatic=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the selected provider-health command."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('check', help='check incident state and provider quota before inference')
    check.add_argument('--output', type=Path, default=Path('provider-health.json'))
    check.set_defaults(handler=_check_command)
    monitor = commands.add_parser('monitor', help='reconcile a completed agent workflow or provider recovery')
    monitor.add_argument('--repository')
    monitor.add_argument('--run-id', type=int)
    monitor.add_argument('--run-attempt', type=int)
    monitor.add_argument('--conclusion')
    monitor.add_argument('--health-artifact', type=Path)
    monitor.add_argument('--agent-artifact', type=Path)
    monitor.add_argument('--recover-issue', type=int)
    monitor.add_argument('--dry-run', action='store_true')
    monitor.set_defaults(handler=_monitor_command)
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f'provider-health failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
