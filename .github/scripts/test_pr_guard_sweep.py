"""Tests for how PR Guard's sweep chooses the pull requests and events `pr_guard.sh` judges."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from pr_guard_sweep import PullRequest, deciding_event, guard_env, sweep_start

NOW = datetime(2026, 11, 16, 18, 0, tzinfo=UTC)
SINCE = NOW - timedelta(minutes=15)
WORKFLOW = Path(__file__).parent.parent / 'workflows' / 'pr-guard.yml'
CONTRIBUTOR = {'__typename': 'User', 'login': 'contributor'}
MAINTAINER = {'__typename': 'User', 'login': 'maintainer'}
DESCRIPTION_BOT = {'__typename': 'Bot', 'login': 'cubic-dev-ai'}


def _at(minutes_ago: int) -> str:
    return (NOW - timedelta(minutes=minutes_ago)).isoformat()


def _pull(
    *,
    created: int = 10,
    edited: tuple[int, dict[str, str]] | None = None,
    timeline: list[tuple[int, dict[str, str] | None]] | None = None,
    author: dict[str, str] | None = CONTRIBUTOR,
) -> PullRequest:
    return PullRequest.model_validate(
        {
            'number': 7,
            'state': 'OPEN',
            'isDraft': False,
            'createdAt': _at(created),
            'authorAssociation': 'CONTRIBUTOR',
            'baseRefName': 'main',
            'lastEditedAt': _at(edited[0]) if edited else None,
            'author': author,
            'editor': edited[1] if edited else None,
            'timelineItems': {'nodes': [{'createdAt': _at(ago), 'actor': actor} for ago, actor in timeline or []]},
        }
    )


def test_a_bot_editing_the_description_does_not_hide_the_opening() -> None:
    """Checked event by event, the opening was judged before the bot's edit could claim the trusted-actor bypass."""
    event = deciding_event(_pull(created=10, edited=(5, DESCRIPTION_BOT)), SINCE)
    assert event is not None
    assert guard_env(_pull(), event)['GITHUB_EVENT_SENDER_LOGIN'] == 'contributor'


def test_a_maintainer_acting_alone_is_judged_as_the_one_who_acted() -> None:
    """With no event of the author's, the latest actor decides, so a maintainer's reopen keeps its bypass."""
    pull = _pull(created=600, edited=(400, CONTRIBUTOR), timeline=[(12, DESCRIPTION_BOT), (3, MAINTAINER)])
    event = deciding_event(pull, SINCE)
    assert event is not None
    assert guard_env(pull, event)['GITHUB_EVENT_SENDER_LOGIN'] == 'maintainer'


def test_a_pull_request_with_nothing_new_is_skipped() -> None:
    """An update the guard never reacted to, such as a push or a comment, starts no check."""
    assert deciding_event(_pull(created=600, edited=(300, CONTRIBUTOR)), SINCE) is None


def test_a_deleted_author_is_judged_as_acting_on_their_own_pull_request() -> None:
    pull = _pull(author=None, timeline=[(2, None)])
    event = deciding_event(pull, SINCE)
    assert event is not None
    assert guard_env(pull, event)['GITHUB_EVENT_SENDER_LOGIN'] == 'ghost'


def test_guard_env_matches_the_pull_request_event_payload() -> None:
    """GraphQL omits a bot login's `[bot]` suffix and capitalizes the state; the guard expects REST's forms."""
    pull = _pull(author={'__typename': 'Bot', 'login': 'pydanty'})
    event = deciding_event(pull, SINCE)
    assert event is not None
    assert guard_env(pull, event) == {
        'GITHUB_EVENT_SENDER_LOGIN': 'pydanty[bot]',
        'GITHUB_EVENT_PULL_REQUEST_NUMBER': '7',
        'GITHUB_EVENT_PULL_REQUEST_USER_LOGIN': 'pydanty[bot]',
        'GITHUB_EVENT_PULL_REQUEST_AUTHOR_ASSOCIATION': 'CONTRIBUTOR',
        'GITHUB_EVENT_PULL_REQUEST_CREATED_AT': '2026-11-16T17:50:00Z',
        'GITHUB_EVENT_PULL_REQUEST_DRAFT': 'false',
        'GITHUB_EVENT_PULL_REQUEST_STATE': 'open',
        'GITHUB_EVENT_PULL_REQUEST_BASE_REF': 'main',
    }


@pytest.mark.parametrize(
    ('last_success', 'expected'),
    [
        (None, NOW - timedelta(hours=1)),
        (NOW - timedelta(minutes=10), NOW - timedelta(minutes=15)),
        (NOW - timedelta(days=3), NOW - timedelta(hours=24)),
    ],
    ids=['first-sweep', 'overlaps-the-last-sweep', 'capped-at-a-day'],
)
def test_a_sweep_reads_back_to_just_before_the_last_successful_one(
    last_success: datetime | None, expected: datetime
) -> None:
    assert sweep_start(last_success, NOW) == expected


def test_the_sweep_runs_default_branch_code_one_at_a_time() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding='utf-8'))
    # YAML 1.1 resolves the bare `on:` key to the boolean True.
    assert set(workflow[True]) == {'schedule', 'workflow_dispatch'}
    assert workflow['concurrency'] == {'group': 'pr-guard'}
    guard = workflow['jobs']['guard']
    assert guard['name'] == 'Duplicate & Issue-Link Guard'
    assert guard['permissions'] == {
        'actions': 'read',
        'contents': 'read',
        'issues': 'write',
        'pull-requests': 'write',
    }
    checkout = guard['steps'][0]['with']
    assert 'ref' not in checkout
    assert checkout['sparse-checkout'].split() == ['.github/scripts/pr_guard.sh', '.github/scripts/pr_guard_sweep.py']
