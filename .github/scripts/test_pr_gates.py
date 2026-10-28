"""Tests for the PR gates in `.github/workflows/pr-gates.yml`."""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pr_gates
import publish_douwebot_review
import pytest
import yaml
from pr_gates import ChangedFile, GateError, GhCli, compatibility_waiver, run
from publish_douwebot_review import Finding, ReviewResult, publication
from pydantic import BaseModel, Field

WORKFLOW = Path(__file__).parent.parent / 'workflows' / 'pr-gates.yml'
ALLOWLIST = '.github/api-compatibility-allowlist.json'
WAIVER = {
    'against': 'v2.59.0',
    'fingerprint': 'a' * 64,
    'reason': 'Deprecated in v2.40, removed on schedule.',
    'pull_request': 'https://github.com/pydantic/pydantic-ai/pull/1',
}


def _allowlist(*waivers: dict[str, str]) -> str:
    return json.dumps({'allowed_breakages': list(waivers)})


@dataclass
class FakeGitHub:
    files: list[str]
    current_labels: set[str] = field(default_factory=set[str])
    texts: dict[str, str | None] = field(default_factory=dict[str, str | None])

    def changed_files(self) -> list[ChangedFile]:
        return [ChangedFile(filename=path) for path in self.files]

    def labels(self) -> set[str]:
        return set(self.current_labels)

    def merge_base(self, base: str, head: str) -> str:
        return 'merge-base'

    def file_text(self, path: str, ref: str) -> str | None:
        assert path == ALLOWLIST
        return self.texts.get(ref)


@pytest.mark.parametrize(
    ('head', 'base', 'labels', 'fails'),
    [
        pytest.param(_allowlist(WAIVER), _allowlist(), set[str](), True, id='new-waiver-unlabeled'),
        pytest.param(_allowlist(WAIVER), _allowlist(), {'compatibility impact'}, False, id='new-waiver-labeled'),
        # A file added by the PR has no version at the merge base.
        pytest.param(_allowlist(WAIVER), None, set[str](), True, id='new-file'),
        pytest.param(_allowlist(), _allowlist(WAIVER), set[str](), False, id='waiver-removed'),
        pytest.param(_allowlist(WAIVER), _allowlist(WAIVER), set[str](), False, id='reformatted'),
        pytest.param('{"allowed_breakages": [{}]}', _allowlist(), {'compatibility impact'}, True, id='invalid'),
    ],
)
def test_compatibility_waiver(head: str, base: str | None, labels: set[str], fails: bool):
    github = FakeGitHub(files=[ALLOWLIST], current_labels=labels, texts={'head': head, 'merge-base': base})

    assert bool(compatibility_waiver(github, base_sha='base', head_sha='head').failures) is fails


def test_an_untouched_allowlist_passes_without_reading_it():
    github = FakeGitHub(files=['pydantic_ai_slim/pydantic_ai/agent.py'])

    assert compatibility_waiver(github, base_sha='base', head_sha='head').failures == []


def test_run_reports_failures_as_annotations_and_a_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    summary = tmp_path / 'summary.md'
    github = FakeGitHub(files=[ALLOWLIST], texts={'head': _allowlist(WAIVER), 'merge-base': _allowlist()})

    status = run(
        'compatibility-waiver', github, {'BASE_SHA': 'base', 'HEAD_SHA': 'head', 'GITHUB_STEP_SUMMARY': str(summary)}
    )

    assert status == 1
    assert capsys.readouterr().out.startswith('::error title=Compatibility waiver labeled::Adds 1 API compatibility')
    assert summary.read_text().startswith('## Compatibility waiver labeled: failed')


def test_run_fails_closed_when_github_cannot_answer(capsys: pytest.CaptureFixture[str]):
    class Unreachable(FakeGitHub):
        def changed_files(self) -> list[ChangedFile]:
            raise GateError('Retrieved only 3000 of 3001 changed files')

    assert run('compatibility-waiver', Unreachable(files=[]), {'BASE_SHA': 'base', 'HEAD_SHA': 'head'}) == 1
    assert 'Retrieved only 3000 of 3001' in capsys.readouterr().out


def test_gh_cli_parses_github_responses(monkeypatch: pytest.MonkeyPatch):
    responses: dict[str, str | None] = {
        'repos/o/r/pulls/7': '{"changed_files": 2}',
        'repos/o/r/pulls/7/files?per_page=100': '[[{"filename": "a"}], [{"filename": "b", "previous_filename": "c"}]]',
        'repos/o/r/compare/x...y': '{"merge_base_commit": {"sha": "m"}}',
        'repos/o/r/issues/7/labels?per_page=100': '[[{"name": "compatibility impact"}]]',
        f'repos/o/r/contents/{ALLOWLIST}?ref=y': '{}',
        f'repos/o/r/contents/{ALLOWLIST}?ref=m': None,
    }

    def fake_run(args: list[str], **_: bool) -> subprocess.CompletedProcess[str]:
        output = responses[args[-1]]
        if output is None:
            return subprocess.CompletedProcess(args, 1, '', 'gh: Not Found (HTTP 404)')
        return subprocess.CompletedProcess(args, 0, output, '')

    monkeypatch.setattr(pr_gates.subprocess, 'run', fake_run)
    github = GhCli('o/r', 7)

    assert github.changed_files() == [ChangedFile(filename='a'), ChangedFile(filename='b', previous_filename='c')]
    assert github.merge_base('x', 'y') == 'm'
    assert github.labels() == {'compatibility impact'}
    assert github.file_text(ALLOWLIST, 'y') == '{}'
    assert github.file_text(ALLOWLIST, 'm') is None

    responses['repos/o/r/pulls/7'] = '{"changed_files": 3}'
    with pytest.raises(GateError, match='Retrieved only 2 of 3'):
        github.changed_files()


class _Trigger(BaseModel):
    types: list[str] = Field(default_factory=list[str])
    workflows: list[str] = Field(default_factory=list[str])
    paths: list[str] | None = None


class _Step(BaseModel):
    run: str = ''
    uses: str = ''
    settings: dict[str, str] = Field(default_factory=dict[str, str], alias='with')


class _Job(BaseModel):
    condition: str = Field(default='', alias='if')
    permissions: dict[str, str] = Field(default_factory=dict[str, str])
    steps: list[_Step]


class _Workflow(BaseModel):
    on: dict[str, _Trigger]
    jobs: dict[str, _Job]


def test_the_workflow_runs_every_gate_on_every_pr_event_and_skips_only_the_merge_queue():
    # `BaseLoader` keeps the `on` key a string; the default loader reads it as `True`.
    workflow = _Workflow.model_validate(yaml.load(WORKFLOW.read_text(encoding='utf-8'), Loader=yaml.BaseLoader))

    assert set(workflow.on) == {
        'pull_request',
        'pull_request_target',
        'workflow_run',
        'merge_group',
    }
    assert workflow.on['pull_request'].paths is None
    assert workflow.on['workflow_run'].workflows == ['PR Bots', 'PR Gates Events']
    assert workflow.on['pull_request_target'].types == workflow.on['pull_request'].types
    assert {'opened', 'synchronize', 'reopened', 'labeled', 'unlabeled', 'edited'} <= set(
        workflow.on['pull_request'].types
    )
    assert workflow.jobs['compatibility-waiver'].condition == "github.event_name == 'pull_request'"
    publisher = workflow.jobs['review-statuses']
    assert publisher.permissions['statuses'] == 'write'
    assert all(context in publisher.steps[0].settings['script'] for context in pr_gates.REVIEW_CONTEXTS.values())
    checkout = next(step for step in publisher.steps if step.uses.startswith('actions/checkout@'))
    assert checkout.settings['ref'] == '${{ github.event.repository.default_branch }}'
    for job in workflow.jobs.values():
        for step in job.steps:
            assert '${{' not in step.run


def test_review_observer_is_read_only_and_schema_matches_the_trusted_publisher():
    observer = _Workflow.model_validate(
        yaml.load(WORKFLOW.with_name('pr-gates-events.yml').read_text(), Loader=yaml.BaseLoader)
    )
    assert set(observer.on) == {'pull_request_review', 'pull_request_review_comment'}
    assert observer.on['pull_request_review'].types == ['submitted', 'edited', 'dismissed']
    assert observer.on['pull_request_review_comment'].types == ['created', 'edited', 'deleted']
    for job in observer.jobs.values():
        assert not job.permissions
        assert all(not step.uses for step in job.steps)
    bots = _Workflow.model_validate(yaml.load(WORKFLOW.with_name('bots.yml').read_text(), Loader=yaml.BaseLoader))
    reviewer = bots.jobs['review']
    assert set(reviewer.permissions.values()) == {'read'}
    args = shlex.split(
        next(
            step.settings['claude_args']
            for step in reviewer.steps
            if step.uses.startswith('anthropics/claude-code-action@')
        )
    )
    assert json.loads(args[args.index('--json-schema') + 1]) == ReviewResult.model_json_schema()


HEAD = 'a' * 40
OLD_HEAD = 'b' * 40
BOT = pr_gates.ReviewUser(id=1, login='github-actions[bot]', type='Bot')
AUTHOR = pr_gates.ReviewUser(id=2, login='author', type='User')
MAINTAINER = pr_gates.ReviewUser(id=3, login='maintainer', type='User')


def review(
    number: int = 10,
    user: pr_gates.ReviewUser = BOT,
    state: str = 'APPROVED',
    head: str = HEAD,
    body: str | None = None,
) -> pr_gates.Review:
    return pr_gates.Review.model_validate(
        {
            'id': number,
            'user': user,
            'state': state,
            'commit_id': head,
            'submitted_at': datetime(2026, 10, 27, 0, 0, number, tzinfo=UTC),
            'body': body
            if body is not None
            else f'No blocking findings.\n\n<!-- douwebot:review run=20 head={head} -->',
        }
    )


@dataclass
class ReviewGitHub:
    size_label: pr_gates.Size = 'size: M'
    labels: list[str] | None = None
    current_reviews: list[pr_gates.Review] = field(default_factory=lambda: [review()])
    comments: list[pr_gates.ReviewComment] = field(default_factory=list[pr_gates.ReviewComment])
    permissions: dict[str, str] = field(default_factory=lambda: {'maintainer': 'write'})
    verified: bool = True
    head: str = HEAD
    statuses: list[tuple[str, pr_gates.ReviewGate, str]] = field(
        default_factory=list[tuple[str, pr_gates.ReviewGate, str]]
    )

    def pull_request(self) -> pr_gates.ReviewPullRequest:
        return pr_gates.ReviewPullRequest.model_validate(
            {
                'state': 'open',
                'head': {'sha': self.head},
                'base': {'sha': OLD_HEAD},
                'user': AUTHOR,
                'changed_files': 1,
                'labels': [{'name': label} for label in self.labels if label]
                if self.labels is not None
                else [{'name': self.size_label}],
            }
        )

    def size(self, pull: pr_gates.ReviewPullRequest) -> pr_gates.Size:
        return self.size_label

    def reviews(self) -> list[pr_gates.Review]:
        return self.current_reviews.copy()

    def review_comments(self) -> list[pr_gates.ReviewComment]:
        return self.comments.copy()

    def permission(self, login: str) -> str:
        return self.permissions.get(login, 'read')

    def verified_review_run(self, run_id: int, head: str) -> bool:
        assert (run_id, head) == (20, HEAD)
        return self.verified

    def status(self, head: str, gate: pr_gates.ReviewGate, state: str, description: str, url: str) -> None:
        self.statuses.append((head, gate, state))


def verdicts(github: ReviewGitHub) -> dict[pr_gates.ReviewGate, pr_gates.Verdict]:
    return pr_gates.review_verdicts(github, github.pull_request(), github.reviews(), github.review_comments())


@pytest.mark.parametrize('size', ['size: S', 'size: M', 'size: L', 'size: XL'])
def test_review_bar_by_size(size: pr_gates.Size):
    github = ReviewGitHub(size_label=size, current_reviews=[])
    result = verdicts(github)
    assert bool(result['douwebot'].failures) is (size != 'size: S')
    assert bool(result['human'].failures) is (size in ('size: L', 'size: XL'))
    github.current_reviews = [review(), review(11, MAINTAINER, head=OLD_HEAD)]
    assert all(not item.failures for item in verdicts(github).values())


@pytest.mark.parametrize('labels', [[], ['size: S'], ['size: M', 'size: L']])
def test_size_must_match_exactly_one_recomputed_label(labels: list[str]):
    assert all(item.failures for item in verdicts(ReviewGitHub(labels=labels)).values())


@pytest.mark.parametrize(
    'candidate',
    [
        review(head=OLD_HEAD),
        review(body='No blocking findings.'),
        review(state='CHANGES_REQUESTED'),
        review(state='COMMENTED'),
        review(state='DISMISSED'),
        review(state='PENDING'),
        review(body=f'No blocking findings.\n\nOptional discussion\n\n<!-- douwebot:review run=20 head={HEAD} -->'),
        review(body=f'Reviewed at {HEAD}\n\nNo findings.'),
    ],
)
def test_unqualified_bot_reviews_do_not_pass(candidate: pr_gates.Review):
    assert verdicts(ReviewGitHub(current_reviews=[candidate]))['douwebot'].failures


def test_run_marker_must_have_successful_trusted_workflow_provenance():
    assert verdicts(ReviewGitHub(verified=False))['douwebot'].failures


def test_failed_douwebot_publication_cannot_hide_a_finding_behind_another_receipt():
    class FailedRun(ReviewGitHub):
        def verified_review_run(self, run_id: int, head: str) -> bool:
            return run_id == 20

    parent = review(9, body=f'No blocking findings.\n\n<!-- douwebot:review run=21 head={HEAD} -->')
    comment = pr_gates.ReviewComment(
        pull_request_review_id=9,
        user=BOT,
        original_commit_id=HEAD,
        body='Quote: <!-- gh-aw-workflow-call-id: pydantic/pydantic-ai/pydantic-ai-pr-review -->',
    )
    assert verdicts(FailedRun(current_reviews=[parent, review()], comments=[comment]))['douwebot'].failures


@pytest.mark.parametrize('current', [False, True])
def test_relocated_comment_uses_its_parent_review_head(current: bool):
    # GitHub forwards commit_id for non-outdated comments. The immutable parent
    # still says this comment was posted on the older head.
    parent = review(9, head=HEAD if current else OLD_HEAD)
    comment = pr_gates.ReviewComment.model_validate(
        {
            'pull_request_review_id': parent.id,
            'user': BOT,
            'body': 'Optional: finding',
            'commit_id': HEAD,
            'original_commit_id': OLD_HEAD,
        }
    )
    github = ReviewGitHub(current_reviews=[parent, review()], comments=[comment])
    assert bool(verdicts(github)['douwebot'].failures) is current


@pytest.mark.parametrize('workflow', ['pydantic-ai-pr-review', 'pydantic-ai-ui-security-review'])
def test_other_review_receipt_does_not_count_as_douwebot_finding(workflow: str):
    parent = review(8, body=f'Reviewed at head\n<!-- gh-aw-workflow-call-id: pydantic/pydantic-ai/{workflow} -->')
    comment = pr_gates.ReviewComment(pull_request_review_id=8, user=BOT, original_commit_id=HEAD, body='CI finding')
    github = ReviewGitHub(current_reviews=[parent, review()], comments=[comment])
    assert not verdicts(github)['douwebot'].failures
    github.current_reviews = [review(8, body='Unidentified review'), review()]
    assert verdicts(github)['douwebot'].failures
    github.current_reviews = [review()]
    assert verdicts(github)['douwebot'].failures
    github.comments[0] = comment.model_copy(update={'original_commit_id': OLD_HEAD})
    assert not verdicts(github)['douwebot'].failures


@pytest.mark.parametrize(
    ('user', 'permission', 'states', 'passes'),
    [
        (MAINTAINER, 'write', ['APPROVED'], True),
        (MAINTAINER, 'admin', ['APPROVED'], True),
        (MAINTAINER, 'read', ['APPROVED'], False),
        (AUTHOR, 'admin', ['APPROVED'], False),
        (BOT, 'admin', ['APPROVED'], False),
        (MAINTAINER, 'write', ['DISMISSED'], False),
        (MAINTAINER, 'write', ['APPROVED', 'CHANGES_REQUESTED'], False),
        (MAINTAINER, 'write', ['CHANGES_REQUESTED', 'APPROVED'], True),
        (MAINTAINER, 'write', ['APPROVED', 'COMMENTED'], True),
    ],
)
def test_human_approval_semantics(user: pr_gates.ReviewUser, permission: str, states: list[str], passes: bool):
    github = ReviewGitHub(size_label='size: L', permissions={user.login: permission})
    github.current_reviews += [review(11 + index, user, state, OLD_HEAD, '') for index, state in enumerate(states)]
    assert bool(verdicts(github)['human'].failures) is not passes


ENV = {'HEAD_SHA': HEAD, 'GITHUB_SERVER_URL': 'https://github.com', 'GITHUB_REPOSITORY': 'o/r', 'GITHUB_RUN_ID': '100'}


def test_reevaluation_replaces_a_failure_with_success_on_the_live_head():
    github = ReviewGitHub(current_reviews=[])
    assert pr_gates.reconcile_reviews(github, ENV) == 1
    github.current_reviews = [review()]
    assert pr_gates.reconcile_reviews(github, ENV) == 0
    assert github.statuses[-2:] == [(HEAD, 'douwebot', 'success'), (HEAD, 'human', 'success')]


def test_errors_and_changing_snapshots_fail_closed():
    class Unreachable(ReviewGitHub):
        def review_comments(self) -> list[pr_gates.ReviewComment]:
            raise GateError('API unavailable')

    github = Unreachable()
    assert pr_gates.reconcile_reviews(github, ENV) == 1
    assert all(state == 'failure' for _, _, state in github.statuses[-2:])

    class Moving(ReviewGitHub):
        def pull_request(self) -> pr_gates.ReviewPullRequest:
            self.head = OLD_HEAD if self.head == HEAD else HEAD
            return super().pull_request()

    moving = Moving(current_reviews=[])
    assert pr_gates.reconcile_reviews(moving, ENV) == 1
    assert all(state == 'failure' for _, _, state in moving.statuses[-2:])


def test_merge_group_skips_metadata_and_reports_both_contexts():
    class NoMetadata(ReviewGitHub):
        def pull_request(self) -> pr_gates.ReviewPullRequest:
            raise AssertionError('Merge groups do not inspect queue members')

    github = NoMetadata()
    assert pr_gates.reconcile_reviews(github, {**ENV, 'MERGE_GROUP_SHA': OLD_HEAD}) == 0
    assert github.statuses == [(OLD_HEAD, 'douwebot', 'success'), (OLD_HEAD, 'human', 'success')]


def test_publisher_binds_inline_comments_to_one_marked_review():
    result = ReviewResult(
        verdict='APPROVE', comments=[Finding(path='a.py', line=3, side='RIGHT', body='Optional: fix')], discussion=''
    )
    payload = json.loads(publication(result, HEAD, 20))
    parent = review(body=payload['body'])
    comment = pr_gates.ReviewComment(
        pull_request_review_id=parent.id, user=BOT, original_commit_id=HEAD, body=payload['comments'][0]['body']
    )
    assert payload['commit_id'] == HEAD
    assert payload['event'] == 'APPROVE'
    assert verdicts(ReviewGitHub(current_reviews=[parent], comments=[comment]))['douwebot'].failures
    assert json.loads(publication(result.model_copy(update={'verdict': 'INCOMPLETE'}), HEAD, 20))['event'] == 'COMMENT'


@pytest.mark.parametrize('moved', [False, True])
def test_publisher_checks_live_head_before_publication(monkeypatch: pytest.MonkeyPatch, moved: bool):
    for key, value in {
        **ENV,
        'GH_TOKEN': 'token',
        'PR_NUMBER': '7',
        'REVIEW_RESULT': '{"verdict":"APPROVE","comments":[],"discussion":""}',
    }.items():
        monkeypatch.setenv(key, value)
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[-1].startswith('"'):
            return subprocess.CompletedProcess(args, 0, f'open at {OLD_HEAD if moved else HEAD}', '')
        assert args[-1] == 'repos/o/r/pulls/7/reviews'
        assert '--input' in args
        assert kwargs['input'] == publication(ReviewResult(verdict='APPROVE', comments=[], discussion=''), HEAD, 100)
        return subprocess.CompletedProcess(args, 0, '', '')

    monkeypatch.setattr(publish_douwebot_review.subprocess, 'run', fake_run)
    if moved:
        with pytest.raises(RuntimeError, match='closed or moved'):
            publish_douwebot_review.main()
        assert len(calls) == 1
    else:
        publish_douwebot_review.main()
        assert len(calls) == 2


@pytest.mark.parametrize(
    ('path', 'event', 'title', 'conclusion', 'job_conclusion', 'valid'),
    [
        ('.github/workflows/bots.yml', 'pull_request_target', f'PR Bots #7 @ {HEAD}', 'success', 'success', True),
        ('.github/workflows/other.yml', 'pull_request_target', f'PR Bots #7 @ {HEAD}', 'success', 'success', False),
        ('.github/workflows/bots.yml', 'pull_request', f'PR Bots #7 @ {HEAD}', 'success', 'success', False),
        ('.github/workflows/bots.yml', 'pull_request_target', f'PR Bots #8 @ {HEAD}', 'success', 'success', False),
        ('.github/workflows/bots.yml', 'pull_request_target', f'PR Bots #7 @ {OLD_HEAD}', 'success', 'success', False),
        ('.github/workflows/bots.yml', 'pull_request_target', f'PR Bots #7 @ {HEAD}', 'failure', 'success', False),
        ('.github/workflows/bots.yml', 'pull_request_target', f'PR Bots #7 @ {HEAD}', 'success', 'failure', False),
    ],
)
def test_gh_cli_verifies_both_jobs_and_originating_run(
    path: str, event: str, title: str, conclusion: str, job_conclusion: str, valid: bool
):
    class Metadata(GhCli):
        def _api(self, *args: str, input: str | None = None) -> str:
            if args[-1].endswith('/jobs?per_page=100'):
                assert args[:2] == ('--paginate', '--slurp')
                return json.dumps(
                    [
                        {'jobs': [{'name': 'douwebot', 'conclusion': 'success'}]},
                        {'jobs': [{'name': 'Publish douwebot review', 'conclusion': job_conclusion}]},
                    ]
                )
            assert args[-1] == 'repos/o/r/actions/runs/20'
            return json.dumps(
                {'name': 'PR Bots', 'path': path, 'event': event, 'display_title': title, 'conclusion': conclusion}
            )

    assert Metadata('o/r', 7).verified_review_run(20, HEAD) is valid


def test_gh_cli_reads_all_review_pages_and_posts_exact_context():
    class Metadata(GhCli):
        def _api(self, *args: str, input: str | None = None) -> str:
            if args[-1].endswith('/reviews?per_page=100'):
                assert args[:2] == ('--paginate', '--slurp')
                return json.dumps(
                    [[review().model_dump(mode='json')], [review(11, MAINTAINER).model_dump(mode='json')]]
                )
            if args[-1].endswith('/comments?per_page=100'):
                assert args[:2] == ('--paginate', '--slurp')
                return '[[{"pull_request_review_id":10,"user":{"id":1,"login":"github-actions[bot]","type":"Bot"},"original_commit_id":"old","body":"finding"}]]'
            assert args == ('--method', 'POST', '--input', '-', f'repos/o/r/statuses/{HEAD}')
            assert input is not None
            assert json.loads(input) == {
                'context': 'Gate: clean douwebot approval',
                'state': 'success',
                'description': 'Clean',
                'target_url': 'https://run',
            }
            return '{}'

    github = Metadata('o/r', 7)
    assert len(github.reviews()) == 2
    assert github.review_comments()[0].pull_request_review_id == 10
    github.status(HEAD, 'douwebot', 'success', 'Clean', 'https://run')


@pytest.mark.parametrize(
    ('total', 'git_available', 'calculation', 'fails'),
    [
        (1, True, '{"size":"size: M","score":101}', False),
        (2, True, '{"size":"size: M","score":101}', True),
        (1, False, '{"size":"size: M","score":101}', True),
        (1, True, '{"size":"unknown","score":101}', True),
    ],
)
def test_gate_uses_shared_calculator_with_base_attributes(
    monkeypatch: pytest.MonkeyPatch, total: int, git_available: bool, calculation: str, fails: bool
):
    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[0] == 'gh':
            assert args[2:4] == ['--paginate', '--slurp']
            assert args[-1].endswith('/files?per_page=100')
            return subprocess.CompletedProcess(args, 0, '[[{"filename":"x.py","additions":101,"deletions":0}]]', '')
        if args[:2] == ['git', 'cat-file']:
            assert args[-1] == f'{OLD_HEAD}^{{commit}}'
            return subprocess.CompletedProcess(args, 0 if git_available else 1, '', '')
        if args[:2] == ['git', 'fetch']:
            assert args[-1] == OLD_HEAD
            return subprocess.CompletedProcess(args, 1, '', '')
        assert args == ['bash', str(Path(pr_gates.__file__).with_name('pr_size.sh')), OLD_HEAD]
        assert isinstance(kwargs['input'], str)
        assert json.loads(kwargs['input'])[0]['filename'] == 'x.py'
        return subprocess.CompletedProcess(args, 0, calculation, '')

    monkeypatch.setattr(pr_gates.subprocess, 'run', fake_run)
    pull = ReviewGitHub().pull_request().model_copy(update={'changed_files': total})
    if fails:
        with pytest.raises((GateError, pr_gates.ValidationError)):
            GhCli('o/r', 7).size(pull)
    else:
        assert GhCli('o/r', 7).size(pull) == 'size: M'
