"""Tests for deterministic recipient and reminder selection."""

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


def _marker(stage: int, recipient: str, created_at: dt.datetime) -> monitor.Activity:
    return _activity(
        author='github-actions[bot]',
        created_at=created_at,
        body=f'<!-- pydantic-ai-attention-monitor stage={stage} recipient={recipient} -->',
    )


def test_primary_recipient_prefers_assigned_maintainer():
    """Assigned maintainers take precedence over the fallback recipient."""
    assignees = [_assignee('contributor', maintainer=False), _assignee('dsfaccini', maintainer=True)]
    assert monitor.choose_primary_recipient(assignees) == 'dsfaccini'


def test_primary_recipient_falls_back_to_aditya():
    """Aditya receives the first reminder when no maintainer is assigned."""
    assert monitor.choose_primary_recipient([]) == 'adtyavrdhn'
    assert monitor.choose_primary_recipient([_assignee('contributor', maintainer=False)]) == 'adtyavrdhn'


def test_first_reminder_is_due_after_three_days():
    """The first notification becomes due at the three-day boundary."""
    reminder = monitor.next_reminder(
        activities=[],
        assignees=[_assignee('dsfaccini', maintainer=True)],
        last_activity_at=NOW - dt.timedelta(days=3),
        now=NOW,
    )
    assert reminder == (1, 'dsfaccini')


def test_second_reminder_repeats_primary_recipient_after_three_more_days():
    """The assigned maintainer gets a second ping before Douwe does."""
    activities = [_marker(1, 'dsfaccini', NOW - dt.timedelta(days=3))]
    reminder = monitor.next_reminder(
        activities=activities,
        assignees=[_assignee('dsfaccini', maintainer=True)],
        last_activity_at=NOW - dt.timedelta(days=6),
        now=NOW,
    )
    assert reminder == (2, 'dsfaccini')


def test_second_reminder_waits_for_full_second_window():
    """The second ping does not fire before its three-day window ends."""
    activities = [_marker(1, 'adtyavrdhn', NOW - dt.timedelta(days=2, hours=23))]
    assert (
        monitor.next_reminder(
            activities=activities,
            assignees=[],
            last_activity_at=NOW - dt.timedelta(days=6),
            now=NOW,
        )
        is None
    )


def test_contributor_activity_does_not_delay_second_reminder():
    """Only a maintainer response stops the reminder clock."""
    contributor_time = NOW - dt.timedelta(hours=1)
    activities = [
        _marker(1, 'adtyavrdhn', NOW - dt.timedelta(days=3)),
        _activity(author='reporter', created_at=contributor_time, body='One more detail.'),
    ]
    reminder = monitor.next_reminder(
        activities=activities,
        assignees=[],
        last_activity_at=contributor_time,
        now=NOW,
    )
    assert reminder == (2, 'adtyavrdhn')


def test_third_reminder_pings_douwe_after_three_more_days():
    """Douwe is pinged only after the second primary-recipient reminder."""
    activities = [_marker(2, 'dsfaccini', NOW - dt.timedelta(days=3))]
    reminder = monitor.next_reminder(
        activities=activities,
        assignees=[_assignee('dsfaccini', maintainer=True)],
        last_activity_at=NOW - dt.timedelta(days=9),
        now=NOW,
    )
    assert reminder == (3, 'DouweM')


def test_third_reminder_waits_for_full_third_window():
    """Douwe is not pinged before the final three-day window ends."""
    activities = [_marker(2, 'adtyavrdhn', NOW - dt.timedelta(days=2, hours=23))]
    assert (
        monitor.next_reminder(
            activities=activities,
            assignees=[],
            last_activity_at=NOW - dt.timedelta(days=9),
            now=NOW,
        )
        is None
    )


def test_recipient_response_resets_even_without_association():
    """A direct response from the pinged assignee resets reminders."""
    response_time = NOW - dt.timedelta(days=3)
    activities = [
        _marker(1, 'dsfaccini', NOW - dt.timedelta(days=6)),
        _activity(author='dsfaccini', created_at=response_time, body='I will take a look.'),
    ]
    reminder = monitor.next_reminder(
        activities=activities,
        assignees=[_assignee('dsfaccini', maintainer=True)],
        last_activity_at=response_time,
        now=NOW,
    )
    assert reminder == (1, 'dsfaccini')


def test_maintainer_response_resets_reminders():
    """Maintainer activity starts a fresh first-reminder window."""
    response_time = NOW - dt.timedelta(days=3)
    activities = [
        _marker(1, 'dsfaccini', NOW - dt.timedelta(days=6)),
        _activity(author='another-maintainer', created_at=response_time, association='MEMBER', body='Looking.'),
    ]
    reminder = monitor.next_reminder(
        activities=activities,
        assignees=[_assignee('dsfaccini', maintainer=True)],
        last_activity_at=response_time,
        now=NOW,
    )
    assert reminder == (1, 'dsfaccini')


def test_reassignment_restarts_with_new_maintainer():
    """A changed assigned maintainer invalidates either primary reminder."""
    activities = [_marker(2, 'adtyavrdhn', NOW - dt.timedelta(days=4))]
    reminder = monitor.next_reminder(
        activities=activities,
        assignees=[_assignee('dsfaccini', maintainer=True)],
        last_activity_at=NOW - dt.timedelta(days=4),
        now=NOW,
    )
    assert reminder == (1, 'dsfaccini')


def test_douwe_reminder_is_terminal_until_new_maintainer_activity():
    """No reminders follow the Douwe ping until maintainer activity resets state."""
    activities = [_marker(3, 'DouweM', NOW - dt.timedelta(days=4))]
    assert (
        monitor.next_reminder(
            activities=activities,
            assignees=[],
            last_activity_at=NOW - dt.timedelta(days=10),
            now=NOW,
        )
        is None
    )


def test_comments_contain_state_markers_and_ping_wording():
    """Comments carry state and describe the second and Douwe pings."""
    item = {
        'kind': 'issue',
        'last_activity_at': (NOW - dt.timedelta(days=3)).isoformat(),
    }
    second = monitor.render_comment(item, 2, 'adtyavrdhn')
    douwe = monitor.render_comment(item, 3, 'DouweM')
    assert '@adtyavrdhn second ping' in second
    assert '<!-- pydantic-ai-attention-monitor stage=2 recipient=adtyavrdhn -->' in second
    assert '@DouweM pinging you' in douwe
    assert '<!-- pydantic-ai-attention-monitor stage=3 recipient=DouweM -->' in douwe
