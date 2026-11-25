"""Tests for how PR Guard's sweep chooses the pull requests and events `pr_guard.sh` judges."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pr_guard_sweep
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
    edited: list[tuple[int, dict[str, str]]] | None = None,
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
            'author': author,
            'userContentEdits': {'nodes': [{'editedAt': _at(ago), 'editor': editor} for ago, editor in edited or []]},
            'timelineItems': {'nodes': [{'createdAt': _at(ago), 'actor': actor} for ago, actor in timeline or []]},
        }
    )


def test_a_bot_editing_the_description_does_not_hide_the_opening() -> None:
    """The author's opening is judged even when a bot's later edit would claim the trusted-actor bypass."""
    event = deciding_event(_pull(created=10, edited=[(5, DESCRIPTION_BOT)]), SINCE)
    assert event is not None
    assert guard_env(_pull(), event)['GITHUB_EVENT_SENDER_LOGIN'] == 'contributor'


def test_a_bot_rewriting_the_description_does_not_hide_the_authors_edit() -> None:
    """Every description edit counts, not only the latest, so the author's own edit is still judged."""
    pull = _pull(created=600, edited=[(8, CONTRIBUTOR), (4, DESCRIPTION_BOT)])
    event = deciding_event(pull, SINCE)
    assert event is not None
    assert guard_env(pull, event)['GITHUB_EVENT_SENDER_LOGIN'] == 'contributor'


def test_a_maintainer_acting_alone_is_judged_as_the_one_who_acted() -> None:
    """With no event of the author's, the latest actor decides, so a maintainer's reopen keeps its bypass."""
    pull = _pull(created=600, edited=[(400, CONTRIBUTOR)], timeline=[(12, DESCRIPTION_BOT), (3, MAINTAINER)])
    event = deciding_event(pull, SINCE)
    assert event is not None
    assert guard_env(pull, event)['GITHUB_EVENT_SENDER_LOGIN'] == 'maintainer'


def test_a_pull_request_with_nothing_new_is_skipped() -> None:
    """An update the guard never reacted to, such as a push or a comment, starts no check."""
    assert deciding_event(_pull(created=600, edited=[(300, CONTRIBUTOR)]), SINCE) is None


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
        (NOW - timedelta(days=3), NOW - timedelta(hours=3)),
    ],
    ids=['first-sweep', 'overlaps-the-last-sweep', 'capped'],
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
    assert checkout['ref'] == '${{ github.event.repository.default_branch }}'
    assert checkout['sparse-checkout'].split() == ['.github/scripts/pr_guard.sh', '.github/scripts/pr_guard_sweep.py']


def test_a_failed_check_leaves_the_rest_checked_and_fails_the_sweep(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One pull request's failed check does not stop the sweep, and the failed sweep leaves its window to the next."""
    now = datetime.now(UTC)
    last_sweep = now - timedelta(minutes=10)

    def node(number: int, created: datetime) -> dict[str, object]:
        return {
            'number': number,
            'state': 'OPEN',
            'isDraft': False,
            'createdAt': created.isoformat(),
            'authorAssociation': 'CONTRIBUTOR',
            'baseRefName': 'main',
            'author': CONTRIBUTOR,
            'userContentEdits': {'nodes': []},
            'timelineItems': {'nodes': []},
        }

    def page(*nodes: dict[str, object]) -> dict[str, object]:
        return {'data': {'search': {'pageInfo': {'hasNextPage': False, 'endCursor': None}, 'nodes': list(nodes)}}}

    responses: dict[str, object] = {
        'repos/pydantic/pydantic-ai': {'default_branch': 'main'},
        'repos/pydantic/pydantic-ai/actions/workflows/pr-guard.yml/runs?status=success&branch=main&per_page=1': {
            'workflow_runs': [{'run_started_at': last_sweep.isoformat()}]
        },
        # `--paginate --slurp` returns one object per search page.
        'graphql': [
            page(node(7, now - timedelta(minutes=2)), node(8, now - timedelta(days=2))),
            page(node(9, now - timedelta(minutes=1))),
        ],
    }
    searches: list[tuple[str, ...]] = []

    def gh(*args: str) -> str:
        if args[0] == 'graphql':
            searches.append(args)
        return json.dumps(responses[args[0]])

    checked: list[str] = []

    def run_guard(args: list[str], *, env: dict[str, str], check: bool) -> subprocess.CompletedProcess[str]:
        number = env['GITHUB_EVENT_PULL_REQUEST_NUMBER']
        checked.append(number)
        assert env['DEFAULT_BRANCH'] == 'main'
        return subprocess.CompletedProcess(args, returncode=1 if number == '7' else 0)

    monkeypatch.setenv('REPO', 'pydantic/pydantic-ai')
    monkeypatch.setattr(pr_guard_sweep, '_gh', gh)
    monkeypatch.setattr(pr_guard_sweep.subprocess, 'run', run_guard)

    assert pr_guard_sweep.main() == 1
    # #8 had no event in the window, so the sweep did not check it.
    assert checked == ['7', '9']
    stamp = (last_sweep - timedelta(minutes=5)).strftime('%Y-%m-%dT%H:%M:%SZ')
    assert f'q=repo:pydantic/pydantic-ai is:pr is:open updated:>={stamp}' in searches[0]
    assert '::error::PR Guard could not check #7.' in capsys.readouterr().out
