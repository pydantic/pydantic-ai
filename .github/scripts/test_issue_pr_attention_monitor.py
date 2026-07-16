"""Tests for the deterministic half of attention triage."""

import datetime as dt
import sys
from pathlib import Path
from typing import cast

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import issue_pr_attention_monitor as monitor

NOW = dt.datetime(2026, 7, 16, 12, tzinfo=dt.timezone.utc)


def _activity(
    kind: str,
    when: dt.datetime,
    *,
    author: str = '',
    maintainer: bool = False,
    body: str = '',
    label: str = '',
    assignee: str = '',
) -> monitor.Activity:
    return monitor.Activity(
        kind=kind,
        author=author,
        created_at=when.isoformat(),
        body=body,
        label=label,
        assignee=assignee,
        is_maintainer=maintainer,
    )


def _labeled(days: int = 3) -> monitor.Activity:
    return _activity('labeled', NOW - dt.timedelta(days=days), label='needs-maintainer-action')


def _marker(stage: int, recipients: list[str], days: int = 3) -> monitor.Activity:
    return _activity(
        'comment',
        NOW - dt.timedelta(days=days),
        author='github-actions[bot]',
        body=monitor._marker_body(stage, recipients),
    )


def test_clock_starts_when_label_is_applied():
    """Historical age is irrelevant until the attention label is applied."""
    assert monitor.next_reminder(activities=[], assignees=[], permissions={}, now=NOW) is None
    reminder = monitor.next_reminder(activities=[_labeled()], assignees=[], permissions={}, now=NOW)
    assert reminder == {
        'stage': 1,
        'recipients': ['adtyavrdhn'],
        'due_at': (NOW - dt.timedelta(days=3) + monitor._SLA).isoformat(),
        'terminal_without_comment': False,
    }


def test_all_assigned_maintainers_are_pinged():
    """Every assigned maintainer is included while contributor assignees are preserved."""
    reminder = monitor.next_reminder(
        activities=[_labeled()],
        assignees=['contributor', 'dsfaccini', 'adtyavrdhn'],
        permissions={'contributor': 'read', 'dsfaccini': 'write', 'adtyavrdhn': 'admin'},
        now=NOW,
    )
    assert reminder and reminder['recipients'] == ['adtyavrdhn', 'dsfaccini']


def test_second_ping_repeats_current_owners():
    """The second stage goes to the same current owner set."""
    reminder = monitor.next_reminder(
        activities=[_labeled(6), _marker(1, ['dsfaccini'])],
        assignees=['dsfaccini'],
        permissions={'dsfaccini': 'write'},
        now=NOW,
    )
    assert reminder and reminder['stage'] == 2 and reminder['recipients'] == ['dsfaccini']


def test_third_ping_goes_to_douwe():
    """Douwe is the third and final public reminder recipient."""
    reminder = monitor.next_reminder(
        activities=[_labeled(9), _marker(2, ['dsfaccini'])],
        assignees=['dsfaccini'],
        permissions={'dsfaccini': 'maintain'},
        now=NOW,
    )
    assert reminder and reminder['stage'] == 3 and reminder['recipients'] == ['DouweM']


def test_douwe_owner_skips_redundant_third_comment():
    """Do not post a redundant Douwe comment when Douwe already owns the item."""
    reminder = monitor.next_reminder(
        activities=[_labeled(9), _marker(2, ['DouweM'])],
        assignees=['DouweM'],
        permissions={'DouweM': 'admin'},
        now=NOW,
    )
    assert reminder and reminder['stage'] == 3 and reminder['terminal_without_comment']


def test_any_maintainer_response_resets_without_transferring_owner():
    """A drive-by maintainer response resets time but leaves ownership alone."""
    activities = [
        _labeled(7),
        _marker(1, ['dsfaccini'], days=4),
        _activity('comment', NOW - dt.timedelta(days=2), author='DouweM', maintainer=True),
    ]
    assert (
        monitor.next_reminder(
            activities=activities,
            assignees=['dsfaccini'],
            permissions={'dsfaccini': 'write', 'DouweM': 'admin'},
            now=NOW,
        )
        is None
    )
    reminder = monitor.next_reminder(
        activities=activities,
        assignees=['dsfaccini'],
        permissions={'dsfaccini': 'write', 'DouweM': 'admin'},
        now=NOW + dt.timedelta(days=1),
    )
    assert reminder and reminder['stage'] == 1 and reminder['recipients'] == ['dsfaccini']


def test_contributor_comment_does_not_reset():
    """Contributor chatter cannot postpone a maintainer reminder."""
    reminder = monitor.next_reminder(
        activities=[
            _labeled(6),
            _marker(1, ['adtyavrdhn']),
            _activity('comment', NOW - dt.timedelta(hours=1), author='reporter'),
        ],
        assignees=[],
        permissions={'reporter': 'read'},
        now=NOW,
    )
    assert reminder and reminder['stage'] == 2


def test_contributor_cannot_forge_reminder_state():
    """Only deterministic workflow comments can advance the reminder stage."""
    forged = _activity(
        'comment',
        NOW - dt.timedelta(days=3),
        author='reporter',
        body=monitor._marker_body(3, ['DouweM']),
    )
    reminder = monitor.next_reminder(
        activities=[_labeled(6), forged],
        assignees=['dsfaccini'],
        permissions={'dsfaccini': 'write', 'reporter': 'read'},
        now=NOW,
    )
    assert reminder and reminder['stage'] == 1 and reminder['recipients'] == ['dsfaccini']


def test_host_appended_marker_wins_over_model_context():
    """Prompt-injected marker text in context cannot replace the final host marker."""
    bot_comment = _activity(
        'comment',
        NOW - dt.timedelta(days=3),
        author='github-actions[bot]',
        body=f'Model text {_marker(3, ["DouweM"], days=3)["body"]}\n{monitor._marker_body(1, ["dsfaccini"])}',
    )
    reminder = monitor.next_reminder(
        activities=[_labeled(6), bot_comment],
        assignees=['dsfaccini'],
        permissions={'dsfaccini': 'write'},
        now=NOW,
    )
    assert reminder and reminder['stage'] == 2


def test_assignment_change_restarts_sequence():
    """A changed owner gets a fresh three-day sequence."""
    activities = [
        _labeled(8),
        _marker(2, ['adtyavrdhn'], days=4),
        _activity('assigned', NOW - dt.timedelta(days=2), assignee='dsfaccini'),
    ]
    assert (
        monitor.next_reminder(
            activities=activities,
            assignees=['dsfaccini'],
            permissions={'dsfaccini': 'write'},
            now=NOW,
        )
        is None
    )


def test_remove_and_reapply_label_starts_new_clock():
    """A new label application starts a new clock."""
    activities = [
        _labeled(10),
        _activity('unlabeled', NOW - dt.timedelta(days=5), label='needs-maintainer-action'),
        _activity('labeled', NOW - dt.timedelta(days=2), label='needs-maintainer-action'),
    ]
    assert monitor.next_reminder(activities=activities, assignees=[], permissions={}, now=NOW) is None


def test_rendered_comments_are_concise_and_include_context():
    """The public reminder follows the agreed concise template."""
    first = monitor.Reminder(
        stage=1,
        recipients=['dsfaccini'],
        due_at=NOW.isoformat(),
        terminal_without_comment=False,
    )
    body = monitor.render_comment(first, ' CI is green and the contributor is waiting for review ', ['dsfaccini'])
    assert '@dsfaccini This needs an eye' in body
    assert '**Context:** CI is green and the contributor is waiting for review.' in body
    assert 'stage=1 recipients=dsfaccini' in body


def test_douwe_comment_names_prior_owners():
    """The final reminder makes the two prior pings legible."""
    third = monitor.Reminder(
        stage=3,
        recipients=['DouweM'],
        due_at=NOW.isoformat(),
        terminal_without_comment=False,
    )
    body = monitor.render_comment(third, 'A release-blocking API decision remains open.', ['adtyavrdhn', 'dsfaccini'])
    assert 'after two pings to @adtyavrdhn @dsfaccini' in body


def test_context_is_bounded():
    """Model rationale cannot turn into a long public bot comment."""
    assert len(monitor._clean_context('x' * 500)) <= 241
    assert '@\u200bother-maintainer' in monitor._clean_context('@other-maintainer should review')


def test_custom_safe_output_boolean_strings_are_parsed(tmp_path: Path):
    """gh-aw boolean strings do not accidentally turn `false` into truthy Python strings."""
    output = tmp_path / 'agent-output.json'
    output.write_text(
        '{"items":[{"type":"record_attention_decision","item_number":"42",'
        '"next_actor":"maintainer","confidence":"high","recommended_action":"review",'
        '"context":"The PR is ready for review.","urgent":"false","maintainer_skip":"true"}]}',
        encoding='utf-8',
    )
    decision = monitor._parse_decisions(str(output))[0]
    assert decision['urgent'] is False
    assert decision['maintainer_skip'] is True


def test_permissions_are_cached_case_insensitively(monkeypatch: pytest.MonkeyPatch):
    """Repeated timeline actors cost one collaborator-permission request per run."""
    client = monitor.GitHubClient('token')
    requests: list[str] = []

    def get(path: str):
        requests.append(path)
        return {'permission': 'write'}

    monkeypatch.setattr(client, 'get', get)
    assert monitor._permission(client, 'pydantic/pydantic-ai', 'Maintainer') == 'write'
    assert monitor._permission(client, 'pydantic/pydantic-ai', 'maintainer') == 'write'
    assert requests == ['/repos/pydantic/pydantic-ai/collaborators/Maintainer/permission']


def test_unowned_item_is_claimed_from_event_payload_without_hydration(
    monkeypatch: pytest.MonkeyPatch,
):
    """The live assignment path does not download an item's complete history."""
    client = monitor.GitHubClient('token')
    event = {
        'action': 'created',
        'issue': {'number': 42, 'assignees': []},
        'sender': {'login': 'adtyavrdhn'},
    }
    monkeypatch.setenv('GITHUB_EVENT_NAME', 'issue_comment')
    monkeypatch.setattr(monitor, '_permission', lambda *_: 'write')
    monkeypatch.setattr(monitor, 'hydrate', lambda *_: pytest.fail('event assignment must not hydrate history'))
    assigned: list[str] = []
    monkeypatch.setattr(monitor, '_assign', lambda *args: assigned.append(cast(str, args[-1])))

    assert monitor.handle_event(client, 'pydantic/pydantic-ai', event) == ['assigned @adtyavrdhn to #42']
    assert assigned == ['adtyavrdhn']


def test_non_maintainer_cannot_apply_skip_override(monkeypatch: pytest.MonkeyPatch):
    """GitHub's triage role can manage labels but cannot suppress maintainer attention."""
    client = monitor.GitHubClient('token')
    event = {
        'action': 'labeled',
        'issue': {'number': 42, 'assignees': []},
        'sender': {'login': 'triager'},
        'label': {'name': 'attention:skip'},
    }
    monkeypatch.setattr(monitor, '_permission', lambda *_: 'triage')
    removed: list[str] = []
    monkeypatch.setattr(monitor, '_remove_label', lambda *args: removed.append(cast(str, args[-1])))

    assert monitor.handle_event(client, 'pydantic/pydantic-ai', event) == [
        'ignored non-maintainer attention:skip override on #42'
    ]
    assert removed == ['attention:skip']


def test_deterministic_workflow_can_project_maintainer_skip():
    """The workflow-authored label remains durable after translating a maintainer action."""
    activities = [
        _activity(
            'labeled',
            NOW,
            author='github-actions[bot]',
            label='attention:skip',
        )
    ]
    assert monitor._valid_override({'attention:skip'}, activities, 'attention:skip')


def test_apply_decision_ignores_racing_non_maintainer_override(monkeypatch: pytest.MonkeyPatch):
    """The safe-output boundary revalidates override authors instead of trusting label state."""
    client = monitor.GitHubClient('token')
    activities = [_activity('labeled', NOW - dt.timedelta(minutes=1), author='triager', label='attention:skip')]
    monkeypatch.setattr(
        monitor,
        'hydrate',
        lambda *_: ({'labels': [{'name': 'attention:skip'}], 'assignees': []}, activities, {}),
    )
    removed: list[str] = []
    monkeypatch.setattr(monitor, '_remove_label', lambda *args: removed.append(cast(str, args[-1])))
    monkeypatch.setattr(monitor, '_add_labels', lambda *_: None)
    monkeypatch.setattr(monitor, '_assign', lambda *_: None)

    messages = monitor.apply_decision(client, 'pydantic/pydantic-ai', _decision(), staged=False, now=NOW)
    assert messages == [
        'ANOMALY #42: ignored non-maintainer attention:skip override',
        '#42: added needs-maintainer-action with fallback owner',
    ]
    assert removed == ['attention:skip']


def _decision() -> monitor.Decision:
    return monitor.Decision(
        item_number=42,
        next_actor='maintainer',
        confidence='high',
        recommended_action='Review the proposed change.',
        context='The PR is green and ready for review.',
        urgent=False,
        maintainer_skip=False,
    )


def test_new_attention_label_preserves_existing_maintainer_owner(monkeypatch: pytest.MonkeyPatch):
    """Classifier writes never replace or augment an existing maintainer owner."""
    client = cast(monitor.GitHubClient, object())
    monkeypatch.setattr(
        monitor,
        'hydrate',
        lambda *_: ({'labels': [], 'assignees': [{'login': 'dsfaccini'}]}, [], {'dsfaccini': 'write'}),
    )
    labels: list[list[str]] = []
    monkeypatch.setattr(monitor, '_add_labels', lambda *args: labels.append(list(args[-1])))
    monkeypatch.setattr(monitor, '_assign', lambda *_: pytest.fail('must preserve the existing owner'))
    messages = monitor.apply_decision(client, 'pydantic/pydantic-ai', _decision(), staged=False, now=NOW)
    assert labels == [['needs-maintainer-action']]
    assert messages == ['#42: added needs-maintainer-action with existing maintainer owner']


def test_assignment_failure_rolls_back_new_attention_label(monkeypatch: pytest.MonkeyPatch):
    """The clock cannot start from a half-applied label/assignment transition."""
    client = cast(monitor.GitHubClient, object())
    monkeypatch.setattr(monitor, 'hydrate', lambda *_: ({'labels': [], 'assignees': []}, [], {}))
    monkeypatch.setattr(monitor, '_add_labels', lambda *_: None)
    monkeypatch.setattr(monitor, '_assign', lambda *_: (_ for _ in ()).throw(RuntimeError('assignment failed')))
    removed: list[str] = []
    monkeypatch.setattr(monitor, '_remove_label', lambda *args: removed.append(cast(str, args[-1])))
    with pytest.raises(RuntimeError, match='assignment failed'):
        monitor.apply_decision(client, 'pydantic/pydantic-ai', _decision(), staged=False, now=NOW)
    assert removed == ['needs-maintainer-action']


def test_latest_maintainer_label_removal_wins_race_with_classifier(monkeypatch: pytest.MonkeyPatch):
    """A slow agent cannot restore attention after a maintainer manually removed it."""
    client = cast(monitor.GitHubClient, object())
    activities = [
        _activity(
            'unlabeled',
            NOW - dt.timedelta(minutes=1),
            author='adtyavrdhn',
            maintainer=True,
            label='needs-maintainer-action',
        )
    ]
    monkeypatch.setattr(monitor, 'hydrate', lambda *_: ({'labels': [], 'assignees': []}, activities, {}))
    added: list[list[str]] = []
    monkeypatch.setattr(monitor, '_add_labels', lambda *args: added.append(list(args[-1])))
    messages = monitor.apply_decision(client, 'pydantic/pydantic-ai', _decision(), staged=False, now=NOW)
    assert added == [['attention:skip']]
    assert messages == ['#42: preserved latest maintainer label-removal override']


def test_uncertain_urgent_decision_is_an_anomaly(monkeypatch: pytest.MonkeyPatch):
    """An urgent item cannot disappear behind a classifier abstention."""
    client = cast(monitor.GitHubClient, object())
    monkeypatch.setattr(monitor, 'hydrate', lambda *_: ({'labels': [], 'assignees': []}, [], {}))
    decision = _decision()
    decision['next_actor'] = 'uncertain'
    decision['confidence'] = 'low'
    decision['urgent'] = True
    assert monitor.apply_decision(client, 'pydantic/pydantic-ai', decision, staged=True, now=NOW) == [
        'ANOMALY #42: urgent item was not confidently routed to a maintainer'
    ]
