"""Tests for deterministic recipient and escalation selection."""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import issue_pr_attention_monitor as monitor  # noqa: I001


NOW = dt.datetime(2026, 7, 16, 12, tzinfo=dt.timezone.utc)


def _assignee(login: str, *, maintainer: bool) -> monitor.Assignee:
    return monitor.Assignee(login=login, permission='write' if maintainer else 'read', is_maintainer=maintainer)


def _activity(
    *,
    author: str,
    created_at: dt.datetime,
    association: str = 'NONE',
    body: str = '',
) -> monitor.Activity:
    return monitor.Activity(
        kind='comment',
        author=author,
        author_association=association,
        created_at=created_at.isoformat(),
        body=body,
    )


def test_primary_recipient_prefers_assigned_maintainer():
    """Assigned maintainers take precedence over the fallback recipient."""
    assignees = [_assignee('contributor', maintainer=False), _assignee('dsfaccini', maintainer=True)]
    assert monitor.choose_primary_recipient(assignees) == 'dsfaccini'


def test_primary_recipient_falls_back_to_aditya():
    """Aditya receives Stage 1 when no maintainer is assigned."""
    assert monitor.choose_primary_recipient([]) == 'adtyavrdhn'
    assert monitor.choose_primary_recipient([_assignee('contributor', maintainer=False)]) == 'adtyavrdhn'


def test_stage_one_is_due_after_three_days():
    """The first notification becomes due at the three-day boundary."""
    escalation = monitor.next_escalation(
        activities=[],
        assignees=[_assignee('dsfaccini', maintainer=True)],
        last_activity_at=NOW - dt.timedelta(days=3),
        now=NOW,
    )
    assert escalation == (1, 'dsfaccini')


def test_stage_two_is_douwe_after_another_three_days():
    """An unanswered Stage 1 escalates to Douwe after three more days."""
    marker_time = NOW - dt.timedelta(days=3)
    activities = [
        _activity(
            author='github-actions[bot]',
            created_at=marker_time,
            body='<!-- pydantic-ai-attention-monitor stage=1 recipient=dsfaccini -->',
        )
    ]
    escalation = monitor.next_escalation(
        activities=activities,
        assignees=[_assignee('dsfaccini', maintainer=True)],
        last_activity_at=NOW - dt.timedelta(days=6),
        now=NOW,
    )
    assert escalation == (2, 'DouweM')


def test_stage_two_waits_for_full_second_window():
    """The escalation does not fire before the second window ends."""
    marker_time = NOW - dt.timedelta(days=2, hours=23)
    activities = [
        _activity(
            author='github-actions[bot]',
            created_at=marker_time,
            body='<!-- pydantic-ai-attention-monitor stage=1 recipient=adtyavrdhn -->',
        )
    ]
    assert (
        monitor.next_escalation(
            activities=activities,
            assignees=[],
            last_activity_at=NOW - dt.timedelta(days=6),
            now=NOW,
        )
        is None
    )


def test_contributor_activity_does_not_delay_unanswered_stage_one():
    """Only a maintainer response stops the escalation clock."""
    marker_time = NOW - dt.timedelta(days=3)
    contributor_time = NOW - dt.timedelta(hours=1)
    activities = [
        _activity(
            author='github-actions[bot]',
            created_at=marker_time,
            body='<!-- pydantic-ai-attention-monitor stage=1 recipient=adtyavrdhn -->',
        ),
        _activity(author='reporter', created_at=contributor_time, body='One more detail.'),
    ]
    escalation = monitor.next_escalation(
        activities=activities,
        assignees=[],
        last_activity_at=contributor_time,
        now=NOW,
    )
    assert escalation == (2, 'DouweM')


def test_recipient_response_resets_even_without_association():
    """A direct response from the pinged assignee resets Stage 1."""
    marker_time = NOW - dt.timedelta(days=6)
    response_time = NOW - dt.timedelta(days=3)
    activities = [
        _activity(
            author='github-actions[bot]',
            created_at=marker_time,
            body='<!-- pydantic-ai-attention-monitor stage=1 recipient=dsfaccini -->',
        ),
        _activity(author='dsfaccini', created_at=response_time, body='I will take a look.'),
    ]
    escalation = monitor.next_escalation(
        activities=activities,
        assignees=[_assignee('dsfaccini', maintainer=True)],
        last_activity_at=response_time,
        now=NOW,
    )
    assert escalation == (1, 'dsfaccini')


def test_maintainer_response_resets_the_escalation():
    """Maintainer activity starts a fresh Stage 1 window."""
    marker_time = NOW - dt.timedelta(days=6)
    response_time = NOW - dt.timedelta(days=3)
    activities = [
        _activity(
            author='github-actions[bot]',
            created_at=marker_time,
            body='<!-- pydantic-ai-attention-monitor stage=1 recipient=dsfaccini -->',
        ),
        _activity(author='dsfaccini', created_at=response_time, association='MEMBER', body='I will take a look.'),
    ]
    escalation = monitor.next_escalation(
        activities=activities,
        assignees=[_assignee('dsfaccini', maintainer=True)],
        last_activity_at=response_time,
        now=NOW,
    )
    assert escalation == (1, 'dsfaccini')


def test_reassignment_restarts_at_stage_one_for_new_maintainer():
    """A changed assigned maintainer invalidates the previous Stage 1."""
    activities = [
        _activity(
            author='github-actions[bot]',
            created_at=NOW - dt.timedelta(days=4),
            body='<!-- pydantic-ai-attention-monitor stage=1 recipient=adtyavrdhn -->',
        )
    ]
    escalation = monitor.next_escalation(
        activities=activities,
        assignees=[_assignee('dsfaccini', maintainer=True)],
        last_activity_at=NOW - dt.timedelta(days=4),
        now=NOW,
    )
    assert escalation == (1, 'dsfaccini')


def test_stage_two_is_terminal_until_new_maintainer_activity():
    """No reminders follow Stage 2 until maintainer activity resets state."""
    activities = [
        _activity(
            author='github-actions[bot]',
            created_at=NOW - dt.timedelta(days=4),
            body='<!-- pydantic-ai-attention-monitor stage=2 recipient=DouweM -->',
        )
    ]
    assert (
        monitor.next_escalation(
            activities=activities,
            assignees=[],
            last_activity_at=NOW - dt.timedelta(days=7),
            now=NOW,
        )
        is None
    )


def test_comment_contains_state_marker():
    """Posted comments carry the state needed by later scheduled runs."""
    item = {
        'kind': 'issue',
        'last_activity_at': (NOW - dt.timedelta(days=3)).isoformat(),
    }
    body = monitor.render_comment(item, 1, 'adtyavrdhn')
    assert '@adtyavrdhn' in body
    assert '<!-- pydantic-ai-attention-monitor stage=1 recipient=adtyavrdhn -->' in body
