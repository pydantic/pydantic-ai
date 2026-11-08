"""Gate each pull request event and publish the `PR Review Checks` status for the head it reviewed."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

from pr_gates import Commit, GateError, Label
from publish_pr_review_checks import (
    ACCEPTED_LABEL,
    MARKER,
    RUN_LABEL,
    Actor,
    CheckResult,
    CommitStatus,
    LivePullRequest,
    ReviewFinding,
    gate,
    publish,
    review_payload,
    verdict,
)

HEAD = 'a' * 40
GROUP = 'b' * 40
MAINTAINER = Actor(login='maintainer', type='User')
CONTRIBUTOR = Actor(login='contributor', type='User')
ENV = {
    'GITHUB_SERVER_URL': 'https://github.com',
    'GITHUB_REPOSITORY': 'o/r',
    'GITHUB_RUN_ID': '100',
    'HEAD_SHA': HEAD,
    'EVENT_ACTION': 'opened',
    'EVENT_SENDER': 'contributor',
    'EVENT_SENDER_TYPE': 'User',
}


class PostedReview(BaseModel):
    """The review payload the publisher sent."""

    commit_id: str
    event: str
    body: str
    comments: list[dict[str, str | int]]


@dataclass
class FakeGitHub:
    """Live pull request state, permissions and the writes a gate or publisher made."""

    author: Actor = field(default_factory=lambda: MAINTAINER)
    labels: list[str] = field(default_factory=list[str])
    head: str = HEAD
    labeled_by: Actor | None = None
    permissions: dict[str, str] = field(default_factory=lambda: {'maintainer': 'write', 'contributor': 'read'})
    reject_inline: bool = False
    statuses: list[tuple[str, CommitStatus, str, str]] = field(default_factory=list[tuple[str, CommitStatus, str, str]])
    reviews: list[PostedReview] = field(default_factory=list[PostedReview])
    removed: list[str] = field(default_factory=list[str])

    def pull_request(self) -> LivePullRequest:
        return LivePullRequest(
            state='open',
            head=Commit(sha=self.head),
            user=self.author,
            labels=[Label(name=name) for name in self.labels],
        )

    def permission(self, login: str) -> str:
        return self.permissions.get(login, 'none')

    def labeler(self, label: str) -> Actor | None:
        assert label == RUN_LABEL
        return self.labeled_by

    def remove_label(self, label: str) -> None:
        self.removed.append(label)

    def status(self, head: str, state: CommitStatus, description: str, url: str) -> None:
        self.statuses.append((head, state, description, url))

    def post_review(self, payload: str) -> str | None:
        review = PostedReview.model_validate_json(payload)
        if self.reject_inline and review.comments:
            return None
        self.reviews.append(review)
        return 'https://github.com/o/r/pull/1#pullrequestreview-1'


def _states(github: FakeGitHub) -> list[tuple[str, CommitStatus]]:
    return [(head, state) for head, state, _, _ in github.statuses]


def test_merge_group_passes_without_a_review() -> None:
    """Each head was reviewed before admission, so the queue must not wait out its timeout."""
    github = FakeGitHub()
    assert not gate(github, {**ENV, 'MERGE_GROUP_SHA': GROUP})
    assert _states(github) == [(GROUP, 'success')]


def test_author_with_write_access_is_reviewed() -> None:
    """A maintainer's pull request is reviewed at once and shows pending until published."""
    github = FakeGitHub()
    assert gate(github, ENV)
    assert _states(github) == [(HEAD, 'pending')]


@pytest.mark.parametrize(
    'labeled_by',
    [None, CONTRIBUTOR, Actor(login='maintainer', type='Bot')],
    ids=['unlabeled', 'read-access-labeler', 'bot-labeler'],
)
def test_outside_author_waits_for_a_maintainer(labeled_by: Actor | None) -> None:
    """Without a write user's run label, the required status fails instead of passing unreviewed."""
    github = FakeGitHub(author=CONTRIBUTOR, labels=[RUN_LABEL] if labeled_by else [], labeled_by=labeled_by)
    assert not gate(github, ENV)
    assert [(state, description) for _, state, description, _ in github.statuses] == [
        ('failure', f'Awaiting maintainer: a user with write access adds `{RUN_LABEL}`.')
    ]


@pytest.mark.parametrize('author', [CONTRIBUTOR, Actor(login='maintainer', type='Bot')], ids=['user', 'app'])
def test_run_label_from_a_write_user_starts_the_review(author: Actor) -> None:
    """An app's branch needs the label too: its content can come from outside text."""
    github = FakeGitHub(author=author, labels=[RUN_LABEL], labeled_by=MAINTAINER)
    assert gate(github, {**ENV, 'EVENT_ACTION': 'labeled', 'EVENT_LABEL': RUN_LABEL})
    assert _states(github) == [(HEAD, 'pending')]


def test_run_label_changes_nothing_for_a_trusted_author() -> None:
    """The label does not buy a second review of a head that was already reviewed."""
    github = FakeGitHub(labels=[RUN_LABEL], labeled_by=MAINTAINER)
    assert not gate(github, {**ENV, 'EVENT_ACTION': 'labeled', 'EVENT_LABEL': RUN_LABEL})
    assert github.statuses == []


def test_acceptance_from_a_write_user_passes_this_head() -> None:
    """A maintainer can accept the findings on the current head, and nothing else."""
    github = FakeGitHub(labels=[ACCEPTED_LABEL])
    env = {**ENV, 'EVENT_ACTION': 'labeled', 'EVENT_LABEL': ACCEPTED_LABEL, 'EVENT_SENDER': 'maintainer'}
    assert not gate(github, env)
    assert [(head, state, description) for head, state, description, _ in github.statuses] == [
        (HEAD, 'success', 'Findings accepted by @maintainer for this head.')
    ]


@pytest.mark.parametrize(
    ('labels', 'sender', 'sender_type'),
    [([ACCEPTED_LABEL], 'contributor', 'User'), ([ACCEPTED_LABEL], 'maintainer', 'Bot'), ([], 'maintainer', 'User')],
    ids=['read-access', 'bot', 'label-removed'],
)
def test_acceptance_is_refused(labels: list[str], sender: str, sender_type: str) -> None:
    """Only a present label from a user with write access accepts findings."""
    github = FakeGitHub(labels=labels)
    env = {
        **ENV,
        'EVENT_ACTION': 'labeled',
        'EVENT_LABEL': ACCEPTED_LABEL,
        'EVENT_SENDER': sender,
        'EVENT_SENDER_TYPE': sender_type,
    }
    with pytest.raises(GateError):
        gate(github, env)
    assert github.statuses == []


def test_push_removes_the_acceptance_and_reviews_again() -> None:
    """An acceptance covers only the head it was given for."""
    github = FakeGitHub(labels=[ACCEPTED_LABEL])
    assert gate(github, {**ENV, 'EVENT_ACTION': 'synchronize'})
    assert github.removed == [ACCEPTED_LABEL]


def test_moved_head_writes_nothing() -> None:
    """The event for the new head decides it."""
    github = FakeGitHub(head='c' * 40)
    assert not gate(github, ENV)
    assert github.statuses == []


def _result(check: str, *severities: str, outcome: str = 'COMPLETE') -> CheckResult:
    findings = [
        ReviewFinding.model_validate({'path': 'a.py', 'line': 3, 'side': 'RIGHT', 'severity': s, 'body': 'Fix.'})
        for s in severities
    ]
    return CheckResult.model_validate(
        {
            'check': check,
            'files': ['a.py'],
            'outcome': outcome,
            'findings': findings,
            'coverage': 'Reviewed a.py.',
            'steps': 4,
            'tokens': 1000,
            'seconds': 60,
        }
    )


@pytest.mark.parametrize(
    ('results', 'expected'),
    [
        ([_result('general'), _result('ui')], ('success', 'Reviewed by general, ui.')),
        (
            [_result('general', 'medium'), _result('ui')],
            ('success', 'Reviewed by general, ui; 1 non-blocking findings.'),
        ),
        ([_result('general', 'high'), _result('ui', 'critical')], ('failure', 'Blocking findings: general, ui')),
        ([_result('general', outcome='INCOMPLETE'), _result('ui')], ('failure', 'Incomplete: general')),
        ([_result('general')], ('failure', 'No result: ui')),
    ],
    ids=['clean', 'medium', 'blocking', 'incomplete', 'missing'],
)
def test_verdict(results: list[CheckResult], expected: tuple[str, str]) -> None:
    """Critical, high, incomplete and missing checks fail; medium findings are reported but pass."""
    assert verdict(['general', 'ui'], {result.check: result for result in results}) == expected


def test_no_selected_check_passes() -> None:
    """A change no check covers, such as documentation, passes at once."""
    assert verdict([], {}) == ('success', 'No file this review covers changed.')


def _publish_env(tmp_path: Path, *results: CheckResult) -> dict[str, str]:
    for result in results:
        (tmp_path / f'{result.check}.json').write_text(result.model_dump_json())
    return {
        **ENV,
        'REVIEW_CHECKS': json.dumps([result.check for result in results]),
        'DETECT_RESULT': 'success',
        'RESULTS_DIR': str(tmp_path),
    }


def test_findings_are_one_review_bound_to_the_run_and_head(tmp_path: Path) -> None:
    """The receipt lets the douwebot gate tell these comments from douwebot's own."""
    github = FakeGitHub()
    publish(github, _publish_env(tmp_path, _result('general', 'high'), _result('ui')))
    [review] = github.reviews
    assert (review.commit_id, review.event) == (HEAD, 'COMMENT')
    marker = MARKER.search(review.body)
    assert marker is not None and (marker[1], marker[2]) == ('100', HEAD)
    assert [comment['path'] for comment in review.comments] == ['a.py']
    assert '| `general` | COMPLETE | 1 | 4 | 1,000 | 60 |' in review.body
    assert github.statuses == [
        (HEAD, 'failure', 'Blocking findings: general', 'https://github.com/o/r/pull/1#pullrequestreview-1')
    ]


def test_rejected_inline_review_moves_findings_into_the_body(tmp_path: Path) -> None:
    """GitHub rejects a line outside the diff; the findings still reach the pull request."""
    github = FakeGitHub(reject_inline=True)
    publish(github, _publish_env(tmp_path, _result('general', 'medium')))
    [review] = github.reviews
    assert review.comments == []
    assert '### `a.py:3` (`RIGHT`)' in review.body


def test_clean_review_posts_only_the_status(tmp_path: Path) -> None:
    """A clean head gets no review comment on every pull request."""
    github = FakeGitHub()
    publish(github, _publish_env(tmp_path, _result('general')))
    assert github.reviews == []
    assert _states(github) == [(HEAD, 'success')]


def test_finding_outside_the_checks_files_fails(tmp_path: Path) -> None:
    """The publisher checks every finding again before it posts."""
    result = _result('general', 'high')
    result.findings[0].path = 'b.py'
    with pytest.raises(GateError, match='outside its files'):
        publish(FakeGitHub(), _publish_env(tmp_path, result))


def test_failed_selection_fails_the_status(tmp_path: Path) -> None:
    """A head whose checks could not be selected is not reviewed, so it cannot pass."""
    github = FakeGitHub()
    publish(github, {**_publish_env(tmp_path), 'DETECT_RESULT': 'failure', 'REVIEW_CHECKS': ''})
    assert _states(github) == [(HEAD, 'failure')]


def test_moved_head_publishes_nothing(tmp_path: Path) -> None:
    """A review of an old head must not replace the status of the new one."""
    github = FakeGitHub(head='c' * 40)
    publish(github, _publish_env(tmp_path, _result('general', 'high')))
    assert (github.reviews, github.statuses) == ([], [])


def test_review_payload_prefixes_each_finding_with_its_check() -> None:
    """A reader can tell which check raised a finding and how severe it is."""
    payload = json.loads(review_payload([_result('url-fetch', 'critical')], HEAD, 5, inline=True))
    assert payload['comments'][0]['body'] == '**critical** · `url-fetch` security check\n\nFix.'
