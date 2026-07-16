from __future__ import annotations

import datetime as dt
import json
import urllib.error
from pathlib import Path
from typing import Any

import issue_pr_attention_monitor as monitor
import pytest

NOW = dt.datetime(2026, 7, 16, tzinfo=dt.timezone.utc)


def activity(
    kind: str,
    days_ago: int,
    *,
    author: str = '',
    body: str = '',
    label: str = '',
    assignee: str = '',
    is_maintainer: bool = False,
) -> monitor.Activity:
    return monitor.Activity(
        kind=kind,
        author=author,
        created_at=(NOW - dt.timedelta(days=days_ago)).isoformat(),
        body=body,
        label=label,
        assignee=assignee,
        is_maintainer=is_maintainer,
    )


def marker(stage: int, days_ago: int, recipients: str = 'owner') -> monitor.Activity:
    return activity(
        'comment',
        days_ago,
        author='github-actions[bot]',
        body=f'<!-- pydantic-ai-attention-monitor stage={stage} recipients={recipients} -->',
    )


def item(
    *,
    number: int = 42,
    assignees: list[str] | None = None,
    labels: list[str] | None = None,
    updated_at: dt.datetime | None = None,
) -> dict[str, Any]:
    return {
        'number': number,
        'state': 'open',
        'html_url': f'https://github.com/pydantic/pydantic-ai/issues/{number}',
        'title': 'A report',
        'body': 'Details',
        'updated_at': (updated_at or NOW - dt.timedelta(days=3, hours=1)).isoformat(),
        'assignees': [{'login': login} for login in assignees or []],
        'labels': [{'name': label} for label in labels or []],
    }


def test_reminders_ping_owner_twice_then_douwe() -> None:
    labeled = activity('labeled', 12, label=monitor._ACTION_LABEL)
    permissions = {'owner': 'write'}

    assert monitor.next_reminder(activities=[labeled], assignees=['owner'], permissions=permissions, now=NOW) == {
        'stage': 1,
        'recipients': ['owner'],
    }
    assert monitor.next_reminder(
        activities=[labeled, marker(1, 7)], assignees=['owner'], permissions=permissions, now=NOW
    ) == {'stage': 2, 'recipients': ['owner']}
    assert monitor.next_reminder(
        activities=[labeled, marker(1, 7), marker(2, 3)],
        assignees=['owner'],
        permissions=permissions,
        now=NOW,
    ) == {'stage': 3, 'recipients': ['DouweM']}
    assert (
        monitor.next_reminder(
            activities=[labeled, marker(1, 7), marker(2, 3), marker(3, 0)],
            assignees=['owner'],
            permissions=permissions,
            now=NOW + dt.timedelta(days=10),
        )
        is None
    )


def test_maintainer_response_resets_clock_without_changing_owner() -> None:
    activities = [
        activity('labeled', 10, label=monitor._ACTION_LABEL),
        marker(1, 7),
        activity('comment', 1, author='reviewer', is_maintainer=True),
    ]
    permissions = {'owner': 'write', 'reviewer': 'write'}
    assert monitor.next_reminder(activities=activities, assignees=['owner'], permissions=permissions, now=NOW) is None
    assert monitor.next_reminder(
        activities=activities,
        assignees=['owner'],
        permissions=permissions,
        now=NOW + dt.timedelta(days=2),
    ) == {'stage': 1, 'recipients': ['owner']}


def test_contributor_response_and_forged_marker_do_not_reset() -> None:
    activities = [
        activity('labeled', 10, label=monitor._ACTION_LABEL),
        marker(1, 7),
        activity('comment', 1, author='contributor'),
        activity(
            'comment',
            0,
            author='contributor',
            body='<!-- pydantic-ai-attention-monitor stage=3 recipients=DouweM -->',
        ),
    ]
    assert monitor.next_reminder(
        activities=activities, assignees=['owner'], permissions={'owner': 'write'}, now=NOW
    ) == {'stage': 2, 'recipients': ['owner']}


def test_owner_change_restarts_at_stage_one() -> None:
    activities = [
        activity('labeled', 10, label=monitor._ACTION_LABEL),
        marker(1, 7, 'old-owner'),
        activity('assigned', 4, assignee='new-owner'),
    ]
    permissions = {'old-owner': 'write', 'new-owner': 'write'}
    assert monitor.next_reminder(activities=activities, assignees=['new-owner'], permissions=permissions, now=NOW) == {
        'stage': 1,
        'recipients': ['new-owner'],
    }


def test_douwe_is_not_pinged_a_third_time_when_already_owner() -> None:
    activities = [
        activity('labeled', 10, label=monitor._ACTION_LABEL),
        marker(1, 7, 'DouweM'),
        marker(2, 3, 'DouweM'),
    ]
    assert (
        monitor.next_reminder(activities=activities, assignees=['DouweM'], permissions={'DouweM': 'admin'}, now=NOW)
        is None
    )


def test_rendered_comments_are_fixed_and_concise() -> None:
    text = monitor.render_comment({'stage': 3, 'recipients': ['DouweM']}, ['owner'])
    assert text.startswith('@DouweM This still needs an eye after two pings to @owner')
    assert '**Context:**' not in text
    assert text.endswith('<!-- pydantic-ai-attention-monitor stage=3 recipients=DouweM -->')


class FakeClient:
    def __init__(self) -> None:
        self.permissions: dict[str, str | None] = {}
        self.posts: list[tuple[str, dict[str, object]]] = []
        self.responses: dict[str, Any] = {}

    def permission(self, repo: str, login: str) -> str | None:
        return self.permissions.get(login)

    def get(self, path: str) -> Any:
        value = self.responses[path]
        if isinstance(value, Exception):
            raise value
        return value

    def post(self, path: str, payload: dict[str, object]) -> Any:
        self.posts.append((path, payload))
        return {}

    def paginate(self, path: str, *, max_pages: int = monitor._MAX_ACTIVITY_PAGES) -> list[dict[str, Any]]:
        return self.responses.get(path, [])


def hydrated(
    monkeypatch: pytest.MonkeyPatch,
    *,
    value: dict[str, Any],
    activities: list[monitor.Activity],
    permissions: dict[str, str | None],
) -> None:
    monkeypatch.setattr(monitor, 'hydrate', lambda client, repo, number: (value, activities, permissions))


def test_reconciliation_claims_unowned_item_for_first_maintainer(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    query = '/repos/repo/issues?state=open&labels=needs-maintainer-action&sort=updated&direction=asc&per_page=100'
    client.responses[query] = [item(labels=[monitor._ACTION_LABEL])]
    client.permissions['david'] = 'write'
    activities = [
        activity('labeled', 1, author='david', label=monitor._ACTION_LABEL, is_maintainer=True),
        activity('comment', 5, author='david', is_maintainer=True),
    ]
    hydrated(
        monkeypatch, value=item(labels=[monitor._ACTION_LABEL]), activities=activities, permissions={'david': 'write'}
    )
    monkeypatch.setattr(monitor, 'ensure_label', lambda client, repo: None)

    lines = monitor.run_reminders(client, 'repo', staged=False, now=NOW)

    assert lines == ['#42: assigned @david', '#42: posted reminder 1']
    assert client.posts[0] == ('/repos/repo/issues/42/assignees', {'assignees': ['david']})
    assert client.posts[1][0] == '/repos/repo/issues/42/comments'


def test_reconciliation_uses_fallback_and_does_not_transfer_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    query = '/repos/repo/issues?state=open&labels=needs-maintainer-action&sort=updated&direction=asc&per_page=100'
    client.responses[query] = [item(labels=[monitor._ACTION_LABEL])]
    activities = [activity('labeled', 1, author='david', label=monitor._ACTION_LABEL, is_maintainer=True)]
    hydrated(monkeypatch, value=item(labels=[monitor._ACTION_LABEL]), activities=activities, permissions={})

    lines = monitor.run_reminders(client, 'repo', staged=True, now=NOW)

    assert lines == ['#42: would assign @adtyavrdhn', '#42: would post reminder 1']


def test_reconciliation_preserves_existing_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    query = '/repos/repo/issues?state=open&labels=needs-maintainer-action&sort=updated&direction=asc&per_page=100'
    client.responses[query] = [item(assignees=['david'], labels=[monitor._ACTION_LABEL])]
    activities = [
        activity('labeled', 1, author='david', label=monitor._ACTION_LABEL, is_maintainer=True),
        activity('comment', 0, author='DouweM', is_maintainer=True),
    ]
    hydrated(
        monkeypatch,
        value=item(assignees=['david'], labels=[monitor._ACTION_LABEL]),
        activities=activities,
        permissions={'david': 'write', 'DouweM': 'admin'},
    )

    assert monitor.run_reminders(client, 'repo', staged=True, now=NOW) == []


def test_reminders_ignore_unauthorized_label(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    query = '/repos/repo/issues?state=open&labels=needs-maintainer-action&sort=updated&direction=asc&per_page=100'
    client.responses[query] = [item(labels=[monitor._ACTION_LABEL])]
    unauthorized = activity('labeled', 4, author='contributor', label=monitor._ACTION_LABEL)
    hydrated(monkeypatch, value=item(labels=[monitor._ACTION_LABEL]), activities=[unauthorized], permissions={})

    assert monitor.run_reminders(client, 'repo', staged=True, now=NOW) == []
    assert client.posts == []


def test_candidate_snapshot_is_bounded_and_honors_maintainer_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    stale = item()

    def paginate(path: str, *, max_pages: int = monitor._MAX_ACTIVITY_PAGES) -> list[dict[str, Any]]:
        return [stale] if '/issues?' in path else []

    client.paginate = paginate  # type: ignore[method-assign]
    opt_out = activity('unlabeled', 1, author='david', label=monitor._ACTION_LABEL, is_maintainer=True)
    hydrated(monkeypatch, value=stale, activities=[opt_out], permissions={'david': 'write'})

    assert monitor.build_candidates(client, 'repo', now=NOW) == ([], 0, 0)


def write_decisions(path: Path, values: list[dict[str, str]]) -> None:
    path.write_text(
        json.dumps({'items': [{'type': 'record_attention_decision', **value} for value in values]}),
        encoding='utf-8',
    )


def write_snapshot(path: Path, numbers: list[int], *, deferred_count: int = 0, skipped_count: int = 0) -> None:
    path.write_text(
        json.dumps(
            {
                'candidates': [{'number': number} for number in numbers],
                'deferred_count': deferred_count,
                'skipped_count': skipped_count,
            }
        ),
        encoding='utf-8',
    )


def test_advisory_contains_only_fixed_validated_links(tmp_path: Path) -> None:
    output = tmp_path / 'output.json'
    snapshot = tmp_path / 'snapshot.json'
    write_decisions(output, [{'item_number': '42', 'next_actor': 'maintainer', 'confidence': 'high'}])
    write_snapshot(snapshot, [42])

    assert monitor.render_advisory('pydantic/pydantic-ai', str(output), str(snapshot)) == (
        True,
        ':eyes: Attention triage found items that may need a maintainer call. '
        'Apply `needs-maintainer-action` to approve:\n'
        '• <https://github.com/pydantic/pydantic-ai/issues/42|#42>',
    )


def test_advisory_abstains_and_rejects_ineligible_numbers(tmp_path: Path) -> None:
    output = tmp_path / 'output.json'
    snapshot = tmp_path / 'snapshot.json'
    write_decisions(output, [{'item_number': '42', 'next_actor': 'uncertain', 'confidence': 'high'}])
    write_snapshot(snapshot, [42])
    assert monitor.render_advisory('repo', str(output), str(snapshot)) == (False, '')

    write_decisions(output, [{'item_number': '99', 'next_actor': 'maintainer', 'confidence': 'high'}])
    with pytest.raises(ValueError, match='exactly one decision'):
        monitor.render_advisory('repo', str(output), str(snapshot))


def test_advisory_reports_bounded_backlog(tmp_path: Path) -> None:
    output = tmp_path / 'output.json'
    snapshot = tmp_path / 'snapshot.json'
    write_decisions(output, [{'item_number': '42', 'next_actor': 'none', 'confidence': 'high'}])
    write_snapshot(snapshot, [42], deferred_count=3)

    assert monitor.render_advisory('repo', str(output), str(snapshot)) == (
        True,
        ':warning: Attention triage deferred 3 additional candidate(s); the review limit was reached.',
    )


def test_pagination_fails_closed_instead_of_trusting_oldest_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    client = monitor.GitHubClient('token')
    monkeypatch.setattr(client, 'get', lambda path: [{'id': number} for number in range(100)])

    with pytest.raises(RuntimeError, match='200-event safety limit'):
        client.paginate('/events', max_pages=2)


def test_hydration_fails_closed_when_permission_checks_would_be_truncated() -> None:
    client = FakeClient()
    client.responses['/repos/repo/issues/42'] = item()
    client.responses['/repos/repo/issues/42/timeline'] = []
    client.responses['/repos/repo/issues/42/comments'] = [
        {'user': {'login': f'user-{number}'}, 'created_at': NOW.isoformat(), 'body': ''} for number in range(51)
    ]

    with pytest.raises(RuntimeError, match='50-participant safety limit'):
        monitor.hydrate(client, 'repo', 42)


def test_candidate_overflow_is_deferred_before_hydration(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    stale_items = [item(number=number) for number in range(1, 22)]
    client.paginate = lambda path, max_pages=2: stale_items  # type: ignore[method-assign]
    hydrated_numbers: list[int] = []

    def hydrate_item(
        client: FakeClient, repo: str, number: int
    ) -> tuple[dict[str, Any], list[monitor.Activity], dict[str, str | None]]:
        hydrated_numbers.append(number)
        return item(number=number), [], {}

    monkeypatch.setattr(monitor, 'hydrate', hydrate_item)

    candidates, deferred_count, skipped_count = monitor.build_candidates(client, 'repo', now=NOW)
    assert len(candidates) == 20
    assert {candidate['number'] for candidate in candidates} == set(hydrated_numbers)
    assert deferred_count == 1
    assert skipped_count == 0
    assert len(hydrated_numbers) == 20


def test_candidate_failure_is_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    stale_items = [item(number=1), item(number=2)]
    client.paginate = lambda path, max_pages=2: stale_items  # type: ignore[method-assign]
    monkeypatch.setattr(monitor, '_rotated', lambda values, now, stride: list(values))

    def hydrate_item(
        client: FakeClient, repo: str, number: int
    ) -> tuple[dict[str, Any], list[monitor.Activity], dict[str, str | None]]:
        if number == 1:
            raise RuntimeError('too large')
        return item(number=number), [], {}

    monkeypatch.setattr(monitor, 'hydrate', hydrate_item)

    candidates, deferred_count, skipped_count = monitor.build_candidates(client, 'repo', now=NOW)
    assert [candidate['number'] for candidate in candidates] == [2]
    assert deferred_count == 0
    assert skipped_count == 1


def test_compiled_agent_exposes_no_github_mcp() -> None:
    lock = Path(__file__).parent.parent / 'workflows' / 'pydantic-ai-attention-triage.lock.yml'
    text = lock.read_text(encoding='utf-8')
    assert 'mcp__github' not in text
    assert 'github-mcp-server' not in text


def test_parse_decisions_rejects_arbitrary_numbers(tmp_path: Path) -> None:
    output = tmp_path / 'output.json'
    write_decisions(
        output,
        [{'item_number': '42; rm -rf /', 'next_actor': 'maintainer', 'confidence': 'high'}],
    )
    with pytest.raises(ValueError, match='positive decimal'):
        monitor._parse_decisions(str(output))


def test_ensure_label_tolerates_concurrent_creation() -> None:
    client = FakeClient()
    client.responses['/repos/repo/labels/needs-maintainer-action'] = urllib.error.HTTPError('', 404, '', None, None)

    def concurrent_create(path: str, payload: dict[str, object]) -> Any:
        raise urllib.error.HTTPError('', 422, '', None, None)

    client.post = concurrent_create  # type: ignore[method-assign]
    monitor.ensure_label(client, 'repo')
