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


def test_any_maintainer_response_resets_clock_without_changing_owner() -> None:
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


def test_douwe_is_not_pinged_a_third_time_when_already_the_owner() -> None:
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
        self.deletes: list[str] = []
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

    def delete(self, path: str) -> Any:
        self.deletes.append(path)
        return None

    def paginate(self, path: str) -> list[dict[str, Any]]:
        return self.responses.get(path, [])


def item(*, number: int = 42, assignees: list[str] | None = None, labels: list[str] | None = None) -> dict[str, Any]:
    return {
        'number': number,
        'state': 'open',
        'assignees': [{'login': login} for login in assignees or []],
        'labels': [{'name': label} for label in labels or []],
    }


def event(value: dict[str, Any], *, sender: str, action: str, label: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {'issue': value, 'sender': {'login': sender}, 'action': action}
    if label:
        result['label'] = {'name': label}
    return result


def test_first_maintainer_response_claims_unowned_item() -> None:
    client = FakeClient()
    client.permissions['david'] = 'write'
    client.responses['/repos/pydantic/pydantic-ai/issues/42'] = item()
    lines = monitor.handle_event(
        client, 'pydantic/pydantic-ai', event(item(), sender='david', action='created'), 'issue_comment'
    )
    assert lines == ['assigned @david to #42']
    assert client.posts == [('/repos/pydantic/pydantic-ai/issues/42/assignees', {'assignees': ['david']})]


def test_drive_by_maintainer_response_does_not_transfer_owner() -> None:
    client = FakeClient()
    client.permissions.update({'david': 'write', 'DouweM': 'admin'})
    client.responses['/repos/pydantic/pydantic-ai/issues/42'] = item(assignees=['david'])
    lines = monitor.handle_event(
        client,
        'pydantic/pydantic-ai',
        event(item(assignees=['david']), sender='DouweM', action='created'),
        'issue_comment',
    )
    assert lines == ['no deterministic change for #42']
    assert client.posts == []


def test_non_maintainer_cannot_add_or_remove_policy_label() -> None:
    client = FakeClient()
    added = monitor.handle_event(
        client,
        'pydantic/pydantic-ai',
        event(
            item(labels=[monitor._ACTION_LABEL]), sender='contributor', action='labeled', label=monitor._ACTION_LABEL
        ),
        'issues',
    )
    removed = monitor.handle_event(
        client,
        'pydantic/pydantic-ai',
        event(item(), sender='contributor', action='unlabeled', label=monitor._ACTION_LABEL),
        'issues',
    )
    assert added == ['ignored non-maintainer label addition on #42']
    assert removed == ['restored non-maintainer label removal on #42']
    assert client.deletes == ['/repos/pydantic/pydantic-ai/issues/42/labels/needs-maintainer-action']
    assert client.posts == [('/repos/pydantic/pydantic-ai/issues/42/labels', {'labels': ['needs-maintainer-action']})]


def test_maintainer_adding_label_assigns_fallback_only_when_unowned() -> None:
    client = FakeClient()
    client.permissions['david'] = 'write'
    client.responses['/repos/pydantic/pydantic-ai/issues/42'] = item(labels=[monitor._ACTION_LABEL])
    lines = monitor.handle_event(
        client,
        'pydantic/pydantic-ai',
        event(item(labels=[monitor._ACTION_LABEL]), sender='david', action='labeled', label=monitor._ACTION_LABEL),
        'issues',
    )
    assert lines == ['assigned @adtyavrdhn to #42']


def hydrated(
    monkeypatch: pytest.MonkeyPatch,
    *,
    value: dict[str, Any],
    activities: list[monitor.Activity],
    permissions: dict[str, str | None],
) -> None:
    monkeypatch.setattr(monitor, 'hydrate', lambda client, repo, number: (value, activities, permissions))


def decision(actor: str, confidence: str = 'high') -> monitor.Decision:
    return monitor.Decision(
        item_number=42,
        next_actor=actor,  # type: ignore[typeddict-item]
        confidence=confidence,  # type: ignore[typeddict-item]
    )


def test_agent_cannot_override_latest_maintainer_label_action(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    maintainer_add = activity('labeled', 1, author='david', label=monitor._ACTION_LABEL, is_maintainer=True)
    hydrated(
        monkeypatch,
        value=item(labels=[monitor._ACTION_LABEL]),
        activities=[maintainer_add],
        permissions={},
    )
    assert monitor.apply_decision(client, 'repo', decision('none'), staged=False) == (
        '#42: preserved maintainer label decision'
    )
    assert client.deletes == []

    maintainer_remove = activity('unlabeled', 1, author='david', label=monitor._ACTION_LABEL, is_maintainer=True)
    hydrated(monkeypatch, value=item(), activities=[maintainer_remove], permissions={})
    assert monitor.apply_decision(client, 'repo', decision('maintainer'), staged=False) == (
        '#42: preserved maintainer label decision'
    )
    assert client.posts == []


def test_bot_correction_does_not_erase_maintainer_override(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    maintainer_add = activity('labeled', 3, author='david', label=monitor._ACTION_LABEL, is_maintainer=True)
    contributor_remove = activity('unlabeled', 2, author='contributor', label=monitor._ACTION_LABEL)
    bot_restore = activity('labeled', 1, author='github-actions[bot]', label=monitor._ACTION_LABEL)
    hydrated(
        monkeypatch,
        value=item(labels=[monitor._ACTION_LABEL]),
        activities=[maintainer_add, contributor_remove, bot_restore],
        permissions={},
    )
    assert monitor.apply_decision(client, 'repo', decision('none'), staged=False) == (
        '#42: preserved maintainer label decision'
    )
    assert client.deletes == []


def test_high_confidence_agent_adds_label_and_fallback_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    hydrated(monkeypatch, value=item(), activities=[], permissions={})
    monkeypatch.setattr(monitor, 'ensure_label', lambda client, repo: None)
    assert monitor.apply_decision(client, 'repo', decision('maintainer'), staged=False) == (
        '#42: added needs-maintainer-action'
    )
    assert client.posts == [
        ('/repos/repo/issues/42/labels', {'labels': ['needs-maintainer-action']}),
        ('/repos/repo/issues/42/assignees', {'assignees': ['adtyavrdhn']}),
    ]


def test_assignment_failure_rolls_back_new_label(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    hydrated(monkeypatch, value=item(), activities=[], permissions={})
    monkeypatch.setattr(monitor, 'ensure_label', lambda client, repo: None)

    def fail_assign(client: FakeClient, repo: str, number: int, login: str) -> None:
        raise RuntimeError('assignment failed')

    monkeypatch.setattr(monitor, '_assign', fail_assign)
    with pytest.raises(RuntimeError, match='assignment failed'):
        monitor.apply_decision(client, 'repo', decision('maintainer'), staged=False)
    assert client.deletes == ['/repos/repo/issues/42/labels/needs-maintainer-action']


def test_agent_abstains_below_high_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    hydrated(monkeypatch, value=item(), activities=[], permissions={})
    assert monitor.apply_decision(client, 'repo', decision('maintainer', 'medium'), staged=False) == (
        '#42: abstained at medium confidence'
    )
    assert client.posts == []


def test_reminders_ignore_an_unauthorized_label_race(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    encoded = 'needs-maintainer-action'
    client.responses[f'/repos/repo/issues?state=open&labels={encoded}&sort=updated&direction=asc'] = [
        item(labels=[monitor._ACTION_LABEL])
    ]
    unauthorized = activity('labeled', 4, author='triage-user', label=monitor._ACTION_LABEL)
    hydrated(
        monkeypatch,
        value=item(labels=[monitor._ACTION_LABEL]),
        activities=[unauthorized],
        permissions={},
    )
    assert monitor.run_reminders(client, 'repo', staged=True, now=NOW) == []
    assert client.posts == []


def test_eligible_numbers_are_bounded_to_trigger() -> None:
    assert monitor.eligible_numbers({'issue': {'number': 42}}, 'issues') == {42}
    assert monitor.eligible_numbers(
        {'check_suite': {'pull_requests': [{'number': number} for number in range(1, 9)]}}, 'check_suite'
    ) == {1, 2, 3, 4, 5}
    assert monitor.eligible_numbers({}, 'workflow_dispatch') == set()


def test_parse_decisions_rejects_arbitrary_numbers(tmp_path: Path) -> None:
    output = tmp_path / 'output.json'
    output.write_text(
        json.dumps(
            {
                'items': [
                    {
                        'type': 'record_attention_decision',
                        'item_number': '42; rm -rf /',
                        'next_actor': 'maintainer',
                        'confidence': 'high',
                    }
                ]
            }
        )
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
