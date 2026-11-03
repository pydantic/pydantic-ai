"""Tests for the compatibility and trusted review gate workflows."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pr_gates
import publish_douwebot_review
import pytest
import review_gates
import yaml
from pr_gates import ChangedFile, GateError, GhCli, compatibility_waiver, run
from publish_douwebot_review import Finding, ReviewResult, publication
from pydantic import BaseModel, Field

WORKFLOW = Path(__file__).parent.parent / 'workflows' / 'pr-gates.yml'
REVIEW_WORKFLOW = WORKFLOW.with_name('pr-review-gates.yml')
PATH = f'.github/api-compatibility-waivers/{"a" * 64}.json'
OTHER_PATH = f'.github/api-compatibility-waivers/{"b" * 64}.json'
ALLOWLIST = '.github/api-compatibility-allowlist.json'
WAIVER = {
    'against': 'v2.59.0',
    'reason': 'Deprecated in v2.40, removed on schedule.',
    'pull_request': 'https://github.com/pydantic/pydantic-ai/pull/1',
}


def _waiver(**changes: str) -> str:
    return json.dumps(WAIVER | changes)


def _legacy_allowlist(against: str) -> str:
    """An allowlist that still lists the waiver for `PATH` under `allowed_breakages`."""
    return json.dumps({'allowed_breakages': [WAIVER | {'against': against, 'fingerprint': 'a' * 64}]})


@dataclass
class FakeGitHub:
    files: list[ChangedFile]
    current_labels: set[str] = field(default_factory=set[str])
    # The text of each file at each ref; a file missing from a ref doesn't exist there.
    texts: dict[tuple[str, str], str] = field(default_factory=dict[tuple[str, str], str])

    def changed_files(self) -> list[ChangedFile]:
        return self.files

    def labels(self) -> set[str]:
        return set(self.current_labels)

    def merge_base(self, base: str, head: str) -> str:
        return 'merge-base'

    def file_text(self, path: str, ref: str) -> str | None:
        return self.texts.get((path, ref))


@pytest.mark.parametrize(
    ('file', 'texts', 'labels', 'fails'),
    [
        pytest.param(ChangedFile(filename=PATH), {(PATH, 'head'): _waiver()}, set[str](), True, id='added-unlabeled'),
        pytest.param(
            ChangedFile(filename=PATH),
            {(PATH, 'head'): _waiver()},
            {'compatibility impact'},
            False,
            id='added-labeled',
        ),
        pytest.param(ChangedFile(filename=PATH), {(PATH, 'merge-base'): _waiver()}, set[str](), False, id='removed'),
        pytest.param(
            ChangedFile(filename=PATH),
            {(PATH, 'head'): _waiver(reason='Reworded.'), (PATH, 'merge-base'): _waiver()},
            set[str](),
            False,
            id='reworded',
        ),
        # Waiving the same finding against a newer release is a new waiver.
        pytest.param(
            ChangedFile(filename=PATH),
            {(PATH, 'head'): _waiver(against='v2.60.0'), (PATH, 'merge-base'): _waiver()},
            set[str](),
            True,
            id='rebased',
        ),
        pytest.param(
            ChangedFile(filename=OTHER_PATH, previous_filename=PATH),
            {(OTHER_PATH, 'head'): _waiver(), (PATH, 'merge-base'): _waiver()},
            set[str](),
            True,
            id='renamed',
        ),
        pytest.param(
            ChangedFile(filename=PATH),
            {(PATH, 'head'): _waiver(), (ALLOWLIST, 'merge-base'): _legacy_allowlist('v2.59.0')},
            set[str](),
            False,
            id='moved-from-legacy-list',
        ),
        pytest.param(
            ChangedFile(filename=PATH),
            {(PATH, 'head'): _waiver(), (ALLOWLIST, 'merge-base'): _legacy_allowlist('v2.58.0')},
            set[str](),
            True,
            id='moved-from-legacy-list-against-a-newer-release',
        ),
        pytest.param(
            ChangedFile(filename=PATH),
            {(PATH, 'head'): '{}'},
            {'compatibility impact'},
            True,
            id='invalid',
        ),
        pytest.param(
            ChangedFile(filename='.github/api-compatibility-waivers/bad.json'),
            {('.github/api-compatibility-waivers/bad.json', 'head'): _waiver()},
            {'compatibility impact'},
            True,
            id='misnamed',
        ),
    ],
)
def test_compatibility_waiver(file: ChangedFile, texts: dict[tuple[str, str], str], labels: set[str], fails: bool):
    github = FakeGitHub(files=[file], current_labels=labels, texts=texts)

    assert bool(compatibility_waiver(github, base_sha='base', head_sha='head').failures) is fails


def test_untouched_waivers_pass_without_reading_them():
    github = FakeGitHub(files=[ChangedFile(filename='.github/api-compatibility-allowlist.json')])

    assert compatibility_waiver(github, base_sha='base', head_sha='head').failures == []


def test_run_reports_failures_as_annotations_and_a_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    summary = tmp_path / 'summary.md'
    github = FakeGitHub(files=[ChangedFile(filename=PATH)], texts={(PATH, 'head'): _waiver()})

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
        f'repos/o/r/contents/{PATH}?ref=y': '{}',
        f'repos/o/r/contents/{PATH}?ref=m': None,
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
    assert github.file_text(PATH, 'y') == '{}'
    assert github.file_text(PATH, 'm') is None

    responses['repos/o/r/pulls/7'] = '{"changed_files": 3}'
    with pytest.raises(GateError, match='Retrieved only 2 of 3'):
        github.changed_files()
    responses['repos/o/r/pulls/7'] = '{"changed_files": 1}'
    assert len(github.changed_files()) == 2


class _Trigger(BaseModel):
    types: list[str] = Field(default_factory=list[str])
    workflows: list[str] = Field(default_factory=list[str])
    paths: list[str] | None = None


class _Step(BaseModel):
    id: str = ''
    name: str = ''
    condition: str = Field(default='', alias='if')
    run: str = ''
    uses: str = ''
    settings: dict[str, str] = Field(default_factory=dict[str, str], alias='with')
    env: dict[str, str] = Field(default_factory=dict[str, str])


class _Job(BaseModel):
    name: str = ''
    condition: str = Field(default='', alias='if')
    permissions: dict[str, str] = Field(default_factory=dict[str, str])
    steps: list[_Step]


class _Workflow(BaseModel):
    run_name: str | None = Field(default=None, alias='run-name')
    on: dict[str, _Trigger]
    jobs: dict[str, _Job]


def test_compatibility_workflow_has_no_privileged_triggers_or_duplicate_check_producers():
    # `BaseLoader` keeps the `on` key a string; the default loader reads it as `True`.
    workflow = _Workflow.model_validate(yaml.load(WORKFLOW.read_text(encoding='utf-8'), Loader=yaml.BaseLoader))

    assert set(workflow.on) == {'pull_request', 'merge_group'}
    assert workflow.on['pull_request'].paths is None
    assert {'opened', 'synchronize', 'reopened', 'labeled', 'unlabeled', 'edited'} <= set(
        workflow.on['pull_request'].types
    )
    assert workflow.jobs['compatibility-waiver'].condition == "github.event_name == 'pull_request'"
    assert set(workflow.jobs) == {'compatibility-waiver'}
    trusted = _Workflow.model_validate(yaml.load(REVIEW_WORKFLOW.read_text(), Loader=yaml.BaseLoader))
    names = [job.name for item in (workflow, trusted) for job in item.jobs.values()]
    assert len(names) == len(set(names))
    assert not set(names) & set(review_gates.REVIEW_CONTEXTS.values())


def test_review_workflow_uses_only_trusted_code_and_pinned_status_targets():
    workflow = _Workflow.model_validate(yaml.load(REVIEW_WORKFLOW.read_text(), Loader=yaml.BaseLoader))
    assert set(workflow.on) == {'pull_request_target', 'workflow_run', 'merge_group'}
    assert workflow.on['workflow_run'].workflows == ['PR Bots', 'PR Review Gates Events']
    assert workflow.on['pull_request_target'].paths is None
    assert set(workflow.on['pull_request_target'].types) == {
        'opened',
        'synchronize',
        'reopened',
        'labeled',
        'unlabeled',
        'edited',
    }
    publisher = workflow.jobs['review-statuses']
    assert publisher.permissions['statuses'] == 'write'
    assert all(step.condition == "steps.pending.outputs.active == 'true'" for step in publisher.steps[1:])
    for step in (publisher.steps[0], publisher.steps[-1]):
        assert step.env['EXPECTED_HEAD_SHA'] == '${{ needs.review-target.outputs.head }}'
        assert step.env['EXPECTED_HEAD_REPOSITORY_ID'] == '${{ needs.review-target.outputs.repository }}'
    assert all(context in publisher.steps[0].settings['script'] for context in review_gates.REVIEW_CONTEXTS.values())
    checkout = next(step for step in publisher.steps if step.uses.startswith('actions/checkout@'))
    assert checkout.settings['ref'] == '${{ github.event.repository.default_branch }}'
    for job in workflow.jobs.values():
        for step in job.steps:
            assert '${{' not in step.run


def test_review_observer_is_read_only_and_schema_matches_the_trusted_publisher():
    observer = _Workflow.model_validate(
        yaml.load(WORKFLOW.with_name('pr-review-gates-events.yml').read_text(), Loader=yaml.BaseLoader)
    )
    assert set(observer.on) == {'pull_request_review', 'pull_request_review_comment'}
    assert observer.on['pull_request_review'].types == ['submitted', 'edited', 'dismissed']
    assert observer.on['pull_request_review_comment'].types == ['created', 'edited', 'deleted']
    for job in observer.jobs.values():
        assert not job.permissions
        assert all(not step.uses for step in job.steps)
    bots = _Workflow.model_validate(yaml.load(WORKFLOW.with_name('bots.yml').read_text(), Loader=yaml.BaseLoader))
    assert bots.run_name is None
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


def test_review_failure_reports_distinguish_cancellation_from_failure():
    workflow = _Workflow.model_validate(yaml.load(WORKFLOW.with_name('bots.yml').read_text(), Loader=yaml.BaseLoader))
    steps = workflow.jobs['publish-review'].steps
    report = next(step.condition for step in steps if step.name == 'Report review failure')
    config_report = next(step.condition for step in steps if step.name == 'Report approval request failure')
    cleanup = next(step.condition for step in steps if step.name == 'Remove douwebot label')
    publisher = next(
        step for step in steps if step.name == 'Publish structured review from trusted default-branch code'
    )
    assert report == (
        "always() && (needs.review.result == 'failure' || steps.publish.outcome == 'failure') && "
        "github.event.action == 'labeled' && github.event.label.name == 'douwebot'"
    )
    assert config_report == (
        "always() && (needs.review.result == 'failure' || steps.publish.outcome == 'failure') && "
        "github.event.action == 'labeled' && github.event.label.name == 'douwebot:review-agent-config'"
    )
    assert cleanup == "always() && github.event.action == 'labeled' && github.event.label.name == 'douwebot'"
    assert publisher.id == 'publish'


@pytest.mark.skipif(shutil.which('node') is None, reason='Node.js is required to execute the workflow JavaScript')
def test_observer_cannot_choose_the_status_target():
    scenarios: list[tuple[str, str]] = [
        ('forged-title', 'active'),
        ('review-comment', 'active'),
        ('empty-event-prs', 'active'),
        ('producer', 'active'),
        ('native', 'active'),
        ('group', 'active'),
        ('foreign-pr', 'error'),
        ('wrong-sha', 'error'),
        ('wrong-repository', 'error'),
        ('ambiguous', 'error'),
        ('closed-association', 'error'),
        ('wrong-source', 'error'),
        ('wrong-path', 'error'),
        ('foreign-workflow-repository', 'error'),
        ('observer-bot-event', 'error'),
        ('producer-review-event', 'error'),
        ('api-error', 'error'),
        ('moved-while-queued', 'stale'),
        ('repository-changed', 'stale'),
        ('closed-while-queued', 'stale'),
    ]
    workflow = _Workflow.model_validate(yaml.load(REVIEW_WORKFLOW.read_text(), Loader=yaml.BaseLoader))
    resolve = workflow.jobs['review-target'].steps[0].settings['script']
    pending = workflow.jobs['review-statuses'].steps[0].settings['script']
    # Execute the actual JavaScript host steps; Python cannot run these in-process.
    harness = (
        """
async function runScenario(data) {
const outputs = {};
const writes = [];
const calls = [];
const core = {setOutput: (name, value) => {outputs[name] = value;}};
core.notice = () => {};
const source = {number: 8, state: 'open', head: {sha: data.head, repo: {id: 888}}};
const serverRun = {
  name: 'PR Review Gates Events #7 @ ' + data.otherHead, event: 'pull_request_review',
  path: '.github/workflows/pr-review-gates-events.yml',
  repository: {full_name: 'pydantic/pydantic-ai'},
  head_sha: data.head, head_branch: 'attacker-branch',
  head_repository: {id: 888, owner: {login: 'attacker'}},
  pull_requests: data.scenario === 'empty-event-prs' ? [] : [{number: 8}]
};
const associations = [structuredClone(source)];
const live = structuredClone(source);
if (data.scenario === 'foreign-pr') {
  associations[0] = {number: 7, state: 'open', head: {sha: data.otherHead, repo: {id: 777}}};
}
if (data.scenario === 'wrong-sha') associations[0].head.sha = data.otherHead;
if (data.scenario === 'wrong-repository') associations[0].head.repo.id = 777;
if (data.scenario === 'closed-association') associations[0].state = 'closed';
if (data.scenario === 'ambiguous') associations.push({...structuredClone(source), number: 9});
if (data.scenario === 'moved-while-queued') live.head.sha = data.otherHead;
if (data.scenario === 'repository-changed') live.head.repo.id = 777;
if (data.scenario === 'closed-while-queued') live.state = 'closed';
if (data.scenario === 'wrong-source') serverRun.event = 'workflow_dispatch';
if (data.scenario === 'review-comment') serverRun.event = 'pull_request_review_comment';
if (data.scenario === 'wrong-path') serverRun.path = '.github/workflows/untrusted.yml';
if (data.scenario === 'foreign-workflow-repository') serverRun.repository.full_name = 'attacker/pydantic-ai';
if (data.scenario === 'observer-bot-event') serverRun.event = 'pull_request_target';
if (['producer', 'producer-review-event'].includes(data.scenario)) {
  serverRun.name = 'PR Bots'; serverRun.path = '.github/workflows/bots.yml';
  serverRun.event = 'pull_request_target';
}
if (data.scenario === 'producer-review-event') serverRun.event = 'pull_request_review';
const context = {
  eventName: 'workflow_run', serverUrl: 'https://github.com', runId: 1,
  repo: {owner: 'pydantic', repo: 'pydantic-ai'},
  payload: {workflow_run: {
    ...serverRun, id: 900, display_title: 'PR Review Gates Events #7 @ ' + data.otherHead,
    pull_requests: [{number: 7}], head_sha: data.otherHead,
    head_repository: {id: 777, owner: {login: 'victim'}}
  }, pull_request: source, merge_group: {head_sha: data.head}}
};
if (data.scenario === 'native') context.eventName = 'pull_request_target';
if (data.scenario === 'group') context.eventName = 'merge_group';
const github = {
  rest: {
    actions: {getWorkflowRun: async args => {calls.push(['run', args]); return {data: serverRun};}},
    pulls: {list: 'pull-list', get: async args => {calls.push(['live', args]); return {data: live};}},
    repos: {createCommitStatus: async value => {writes.push(value);}}
  },
  paginate: async (method, args) => {
    if (method !== 'pull-list') throw new Error('Unexpected API');
    calls.push(['associations', args]);
    if (data.scenario === 'api-error') throw new Error('GitHub unavailable');
    return associations;
  }
};
async function resolve() {
"""
        + resolve
        + """
}
async function pending() {
  process.env.PR_NUMBER = String(outputs.number || '');
  process.env.EXPECTED_HEAD_SHA = outputs.head || '';
  process.env.EXPECTED_HEAD_REPOSITORY_ID = String(outputs.repository || '');
  process.env.MERGE_GROUP_SHA = outputs.group || '';
"""
        + pending
        + """
}
try {
  await resolve();
  await pending();
  return {outputs, writes, calls, returncode: 0};
} catch(error) {
  return {outputs, writes, calls, returncode: 1, error: error.message};
}
}
const results = [];
for (const data of JSON.parse(process.argv[1])) results.push(await runScenario(data));
console.log(JSON.stringify(results));
"""
    )
    data: list[dict[str, str]] = [
        {'scenario': scenario, 'head': OLD_HEAD, 'otherHead': HEAD} for scenario, _ in scenarios
    ]
    result = subprocess.run(
        ['node', '--input-type=module', '-e', harness, json.dumps(data)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    outputs = json.loads(result.stdout)
    for (scenario, outcome), output in zip(scenarios, outputs, strict=True):
        if outcome == 'error':
            assert output['returncode'] == 1
            assert output['writes'] == []
            assert output['outputs'] == {}
            continue
        assert output['returncode'] == 0
        if scenario == 'group':
            assert output['outputs'] == {'group': OLD_HEAD, 'active': 'true'}
            assert output['calls'] == []
        else:
            expected: dict[str, str | int] = {'number': 8, 'head': OLD_HEAD, 'repository': 888}
            if outcome == 'active':
                expected['active'] = 'true'
            assert output['outputs'] == expected
            if scenario != 'native':
                assert output['calls'][:2] == [
                    ['run', {'owner': 'pydantic', 'repo': 'pydantic-ai', 'run_id': 900}],
                    [
                        'associations',
                        {
                            'owner': 'pydantic',
                            'repo': 'pydantic-ai',
                            'state': 'open',
                            'head': 'attacker:attacker-branch',
                            'per_page': 100,
                        },
                    ],
                ]
            assert output['calls'][-1] == ['live', {'owner': 'pydantic', 'repo': 'pydantic-ai', 'pull_number': 8}]
        if outcome == 'active':
            assert [(write['sha'], write['context'], write['state']) for write in output['writes']] == [
                (OLD_HEAD, context, 'pending') for context in review_gates.REVIEW_CONTEXTS.values()
            ]
        else:
            assert output['writes'] == []


HEAD = 'a' * 40
OLD_HEAD = 'b' * 40
BOT = review_gates.ReviewUser(id=1, login='github-actions[bot]', type='Bot')
MAINTAINER = review_gates.ReviewUser(id=3, login='maintainer', type='User')


def review(
    number: int = 10,
    user: review_gates.ReviewUser = BOT,
    state: str = 'APPROVED',
    head: str = HEAD,
    body: str | None = None,
) -> review_gates.Review:
    return review_gates.Review.model_validate(
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
    size_label: review_gates.Size = 'size: M'
    labels: list[str] | None = None
    current_reviews: list[review_gates.Review] = field(default_factory=lambda: [review()])
    comments: list[review_gates.ReviewComment] = field(default_factory=list[review_gates.ReviewComment])
    verified: bool = True
    head: str = HEAD
    repository: int = 888
    closed: bool = False
    statuses: list[tuple[str, review_gates.ReviewGate, review_gates.CommitStatus]] = field(
        default_factory=list[tuple[str, review_gates.ReviewGate, review_gates.CommitStatus]]
    )

    def pull_request(self) -> review_gates.ReviewPullRequest:
        return review_gates.ReviewPullRequest.model_validate(
            {
                'state': 'closed' if self.closed else 'open',
                'head': {'sha': self.head, 'repo': {'id': self.repository}},
                'base': {'sha': OLD_HEAD},
                'changed_files': 1,
                'labels': [{'name': label} for label in self.labels if label]
                if self.labels is not None
                else [{'name': self.size_label}],
            }
        )

    def size(self, pull: review_gates.ReviewPullRequest) -> review_gates.Size:
        return self.size_label

    def reviews(self) -> list[review_gates.Review]:
        return self.current_reviews.copy()

    def review_comments(self) -> list[review_gates.ReviewComment]:
        return self.comments.copy()

    def verified_review_run(self, run_id: int, head: str) -> bool:
        assert (run_id, head) == (20, HEAD)
        return self.verified

    def status(
        self, head: str, gate: review_gates.ReviewGate, state: review_gates.CommitStatus, description: str, url: str
    ) -> None:
        self.statuses.append((head, gate, state))


def verdicts(github: ReviewGitHub) -> dict[review_gates.ReviewGate, pr_gates.Verdict]:
    return review_gates.review_verdicts(github, github.pull_request(), github.reviews(), github.review_comments())


@pytest.mark.parametrize('size', ['size: S', 'size: M', 'size: L', 'size: XL'])
def test_review_bar_by_size(size: review_gates.Size):
    github = ReviewGitHub(size_label=size, current_reviews=[])
    result = verdicts(github)
    assert bool(result['douwebot'].failures) is (size != 'size: S')
    assert set(result) == {'douwebot'}
    github.current_reviews = [review()]
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
@pytest.mark.parametrize('size', ['size: M', 'size: L', 'size: XL'])
def test_unqualified_bot_reviews_do_not_pass(candidate: review_gates.Review, size: review_gates.Size):
    assert verdicts(ReviewGitHub(size_label=size, current_reviews=[candidate]))['douwebot'].failures


@pytest.mark.parametrize('size', ['size: M', 'size: L', 'size: XL'])
def test_run_marker_must_have_successful_trusted_workflow_provenance(size: review_gates.Size):
    assert verdicts(ReviewGitHub(size_label=size, verified=False))['douwebot'].failures


@pytest.mark.parametrize('latest_incomplete', [False, True])
def test_latest_douwebot_verdict_allows_a_retry_after_an_empty_incomplete_review(latest_incomplete: bool):
    result = ReviewResult(verdict='INCOMPLETE', comments=[], discussion='')
    body = json.loads(publication(result, HEAD, 20))['body']
    incomplete = review(11 if latest_incomplete else 9, state='COMMENTED', body=body)
    assert (
        bool(verdicts(ReviewGitHub(current_reviews=[incomplete, review()]))['douwebot'].failures) is latest_incomplete
    )


@pytest.mark.parametrize('state', ['COMMENTED', 'CHANGES_REQUESTED'])
@pytest.mark.parametrize('location', ['inline', 'body'])
def test_findings_before_a_clean_retry_still_make_the_head_unclean(state: str, location: str):
    finding = Finding(path='a.py', line=3, side='RIGHT', body='Finding to address')
    result = ReviewResult(
        verdict='INCOMPLETE' if state == 'COMMENTED' else 'REQUEST_CHANGES',
        comments=[finding] if location == 'inline' else [],
        discussion='Finding to address' if location == 'body' else '',
    )
    parent = review(9, state=state, body=json.loads(publication(result, HEAD, 20))['body'])
    comments = (
        [
            review_gates.ReviewComment(
                pull_request_review_id=parent.id, user=BOT, original_commit_id=HEAD, body=finding.body
            )
        ]
        if location == 'inline'
        else []
    )
    assert verdicts(ReviewGitHub(current_reviews=[parent, review()], comments=comments))['douwebot'].failures


def test_failed_douwebot_publication_cannot_hide_a_finding_behind_another_receipt():
    class FailedRun(ReviewGitHub):
        def verified_review_run(self, run_id: int, head: str) -> bool:
            return run_id == 20

    parent = review(9, body=f'No blocking findings.\n\n<!-- douwebot:review run=21 head={HEAD} -->')
    comment = review_gates.ReviewComment(
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
    comment = review_gates.ReviewComment.model_validate(
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
    comment = review_gates.ReviewComment(pull_request_review_id=8, user=BOT, original_commit_id=HEAD, body='CI finding')
    github = ReviewGitHub(current_reviews=[parent, review()], comments=[comment])
    assert not verdicts(github)['douwebot'].failures
    github.current_reviews = [review(8, body='Unidentified review'), review()]
    assert verdicts(github)['douwebot'].failures
    github.current_reviews = [review()]
    assert verdicts(github)['douwebot'].failures
    github.comments[0] = comment.model_copy(update={'original_commit_id': OLD_HEAD})
    assert not verdicts(github)['douwebot'].failures


ENV = {
    'HEAD_SHA': HEAD,
    'EXPECTED_HEAD_SHA': HEAD,
    'EXPECTED_HEAD_REPOSITORY_ID': '888',
    'GITHUB_SERVER_URL': 'https://github.com',
    'GITHUB_REPOSITORY': 'o/r',
    'GITHUB_RUN_ID': '100',
}


def test_reevaluation_replaces_a_failure_with_success_on_the_live_head():
    github = ReviewGitHub(current_reviews=[])
    assert review_gates.reconcile_reviews(github, ENV) == 0
    assert github.statuses == [(HEAD, 'douwebot', 'failure')]
    github.current_reviews = [review()]
    assert review_gates.reconcile_reviews(github, ENV) == 0
    assert github.statuses == [
        (HEAD, 'douwebot', 'failure'),
        (HEAD, 'douwebot', 'success'),
    ]


@pytest.mark.parametrize(
    'error',
    [GateError('API unavailable'), review_gates.ValidationError('ReviewComment', []), OSError('Cannot read metadata')],
)
def test_errors_and_changing_snapshots_fail_closed(error: Exception):
    class Unreachable(ReviewGitHub):
        def review_comments(self) -> list[review_gates.ReviewComment]:
            raise error

    github = Unreachable()
    assert review_gates.reconcile_reviews(github, ENV) == 1
    assert github.statuses == [(HEAD, 'douwebot', 'failure')]

    class Moving(ReviewGitHub):
        def pull_request(self) -> review_gates.ReviewPullRequest:
            self.labels = ['size: M', 'chore'] if self.labels is None else None
            return super().pull_request()

    moving = Moving(current_reviews=[])
    assert review_gates.reconcile_reviews(moving, ENV) == 1
    assert moving.statuses == [(HEAD, 'douwebot', 'failure')]


def test_review_status_publication_errors_propagate():
    class Unwritable(ReviewGitHub):
        def status(
            self, head: str, gate: review_gates.ReviewGate, state: review_gates.CommitStatus, description: str, url: str
        ) -> None:
            raise GateError('Cannot publish status')

    with pytest.raises(GateError, match='Cannot publish status'):
        review_gates.reconcile_reviews(Unwritable(current_reviews=[]), ENV)


@pytest.mark.parametrize('change', ['head', 'repository', 'closed'])
def test_reevaluation_does_not_follow_an_observer_to_a_different_head(change: str):
    class Superseded(ReviewGitHub):
        def review_comments(self) -> list[review_gates.ReviewComment]:
            if change == 'head':
                self.head = OLD_HEAD
            elif change == 'repository':
                self.repository = 777
            else:
                self.closed = True
            return super().review_comments()

    github = Superseded()
    assert review_gates.reconcile_reviews(github, ENV) == 0
    assert github.statuses == []


def test_no_snapshot_cannot_write_to_an_unverified_environment_head():
    class NoSnapshot(ReviewGitHub):
        def pull_request(self) -> review_gates.ReviewPullRequest:
            raise GateError('PR does not exist or cannot be read')

    github = NoSnapshot()
    assert review_gates.reconcile_reviews(github, {**ENV, 'HEAD_SHA': OLD_HEAD}) == 1
    assert github.statuses == []


def test_merge_group_skips_metadata_and_reports_douwebot_context():
    class NoMetadata(ReviewGitHub):
        def pull_request(self) -> review_gates.ReviewPullRequest:
            raise AssertionError('Merge groups do not inspect queue members')

    github = NoMetadata()
    assert review_gates.reconcile_reviews(github, {**ENV, 'MERGE_GROUP_SHA': OLD_HEAD}) == 0
    assert github.statuses == [(OLD_HEAD, 'douwebot', 'success')]


def test_publisher_binds_inline_comments_to_one_marked_review():
    result = ReviewResult(
        verdict='APPROVE', comments=[Finding(path='a.py', line=3, side='RIGHT', body='Optional: fix')], discussion=''
    )
    payload = json.loads(publication(result, HEAD, 20))
    parent = review(body=payload['body'])
    comment = review_gates.ReviewComment(
        pull_request_review_id=parent.id, user=BOT, original_commit_id=HEAD, body=payload['comments'][0]['body']
    )
    assert payload['commit_id'] == HEAD
    assert payload['event'] == 'APPROVE'
    assert verdicts(ReviewGitHub(current_reviews=[parent], comments=[comment]))['douwebot'].failures
    assert json.loads(publication(result.model_copy(update={'verdict': 'INCOMPLETE'}), HEAD, 20))['event'] == 'COMMENT'


@pytest.mark.parametrize('state', ['open', 'moved', 'closed'])
def test_publisher_checks_live_head_before_publication(monkeypatch: pytest.MonkeyPatch, state: str):
    for key, value in {
        **ENV,
        'GH_TOKEN': 'token',
        'PR_NUMBER': '7',
        'REVIEW_RESULT': '{"verdict":"APPROVE","comments":[],"discussion":""}' if state == 'open' else '',
    }.items():
        monkeypatch.setenv(key, value)
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[-1].startswith('"'):
            live = f'closed at {HEAD}' if state == 'closed' else f'open at {OLD_HEAD if state == "moved" else HEAD}'
            return subprocess.CompletedProcess(args, 0, live, '')
        assert args[-1] == 'repos/o/r/pulls/7/reviews'
        assert '--input' in args
        assert kwargs['input'] == publication(ReviewResult(verdict='APPROVE', comments=[], discussion=''), HEAD, 100)
        return subprocess.CompletedProcess(args, 0, '', '')

    monkeypatch.setattr(publish_douwebot_review.subprocess, 'run', fake_run)
    if state != 'open':
        publish_douwebot_review.main()
        assert len(calls) == 1
    else:
        publish_douwebot_review.main()
        assert len(calls) == 2


@pytest.mark.parametrize(
    ('verdict', 'moved'), [('APPROVE', False), ('REQUEST_CHANGES', False), ('REQUEST_CHANGES', True)]
)
def test_rejected_inline_positions_preserve_the_review(monkeypatch: pytest.MonkeyPatch, verdict: str, moved: bool):
    result = ReviewResult.model_validate(
        {
            'verdict': verdict,
            'comments': [{'path': 'a.py', 'line': 999, 'side': 'RIGHT', 'body': 'Finding details'}],
            'discussion': 'Existing discussion',
        }
    )
    for key, value in {
        **ENV,
        'GITHUB_RUN_ID': '20',
        'PR_NUMBER': '7',
        'REVIEW_RESULT': result.model_dump_json(),
    }.items():
        monkeypatch.setenv(key, value)
    posted: list[str] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[-1].startswith('"'):
            return subprocess.CompletedProcess(args, 0, f'open at {OLD_HEAD if moved and posted else HEAD}', '')
        assert args[-1] == 'repos/o/r/pulls/7/reviews'
        payload = kwargs['input']
        assert isinstance(payload, str)
        posted.append(payload)
        response = (
            subprocess.CompletedProcess(args, 1, '', 'gh: Validation Failed (HTTP 422)')
            if len(posted) == 1
            else subprocess.CompletedProcess(args, 0, '', '')
        )
        if kwargs.get('check'):
            response.check_returncode()
        return response

    monkeypatch.setattr(publish_douwebot_review.subprocess, 'run', fake_run)
    publish_douwebot_review.main()
    if moved:
        assert len(posted) == 1
    else:
        assert len(posted) == 2
        fallback = json.loads(posted[-1])
        assert fallback['comments'] == []
        assert fallback['commit_id'] == HEAD
        assert fallback['event'] == verdict
        assert all(text in fallback['body'] for text in ['a.py:999', 'Finding details', 'Existing discussion'])
        parent = review(body=fallback['body'], state='APPROVED' if verdict == 'APPROVE' else 'CHANGES_REQUESTED')
        assert verdicts(ReviewGitHub(current_reviews=[parent]))['douwebot'].failures


@pytest.mark.parametrize(
    ('status', 'comments', 'attempts'), [(403, True, 1), (500, True, 1), (422, False, 1), (422, True, 2)]
)
def test_review_publication_errors_fail_closed(
    monkeypatch: pytest.MonkeyPatch, status: int, comments: bool, attempts: int
):
    result = ReviewResult(
        verdict='APPROVE',
        discussion='',
        comments=[Finding(path='a.py', line=999, side='RIGHT', body='Finding')] if comments else [],
    )
    for key, value in {**ENV, 'PR_NUMBER': '7', 'REVIEW_RESULT': result.model_dump_json()}.items():
        monkeypatch.setenv(key, value)
    posts: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[-1].startswith('"'):
            return subprocess.CompletedProcess(args, 0, f'open at {HEAD}', '')
        posts.append(args)
        response = subprocess.CompletedProcess(args, 1, '', f'gh: Error (HTTP {status})')
        if kwargs.get('check'):
            response.check_returncode()
        return response

    monkeypatch.setattr(publish_douwebot_review.subprocess, 'run', fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        publish_douwebot_review.main()
    assert len(posts) == attempts


@pytest.mark.parametrize(
    ('path', 'event', 'run_head', 'conclusion', 'job_conclusion', 'valid'),
    [
        ('.github/workflows/bots.yml', 'pull_request_target', HEAD, 'success', 'success', True),
        ('.github/workflows/other.yml', 'pull_request_target', HEAD, 'success', 'success', False),
        ('.github/workflows/bots.yml', 'pull_request', HEAD, 'success', 'success', False),
        ('.github/workflows/bots.yml', 'pull_request_target', OLD_HEAD, 'success', 'success', False),
        ('.github/workflows/bots.yml', 'pull_request_target', HEAD, 'failure', 'success', False),
        ('.github/workflows/bots.yml', 'pull_request_target', HEAD, 'success', 'failure', False),
    ],
)
def test_gh_cli_verifies_both_jobs_and_originating_run(
    path: str, event: str, run_head: str, conclusion: str, job_conclusion: str, valid: bool
):
    class Metadata(review_gates.ReviewGhCli):
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
                {
                    'name': 'PR Bots',
                    'path': path,
                    'event': event,
                    'head_sha': run_head,
                    'display_title': f'Untrusted title at {OLD_HEAD}',
                    'conclusion': conclusion,
                }
            )

    assert Metadata('o/r', 7).verified_review_run(20, HEAD) is valid


def test_gh_cli_reads_all_review_pages_and_posts_exact_context():
    class Metadata(review_gates.ReviewGhCli):
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
        assert args == ['bash', str(Path(review_gates.__file__).with_name('pr_size.sh')), OLD_HEAD]
        assert isinstance(kwargs['input'], str)
        assert json.loads(kwargs['input'])[0]['filename'] == 'x.py'
        return subprocess.CompletedProcess(args, 0, calculation, '')

    monkeypatch.setattr(review_gates.subprocess, 'run', fake_run)
    pull = ReviewGitHub().pull_request().model_copy(update={'changed_files': total})
    if fails:
        with pytest.raises((GateError, review_gates.ValidationError)):
            review_gates.ReviewGhCli('o/r', 7).size(pull)
    else:
        assert review_gates.ReviewGhCli('o/r', 7).size(pull) == 'size: M'
