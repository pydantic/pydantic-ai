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
from publish_pr_review_checks import CheckResult, review_payload as pr_review_checks_payload
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
    assert set(workflow.on) == {'workflow_run', 'merge_group'}
    assert workflow.on['workflow_run'].workflows == ['PR Bots', 'PR Review Gates Events', 'PR Review Checks']
    assert workflow.jobs['review-target'].condition == (
        "github.event_name != 'workflow_run' || (github.event.workflow_run.event != 'merge_group' && "
        "(github.event.workflow_run.path != '.github/workflows/pr-review-checks.yml' || "
        "github.event.workflow_run.conclusion != 'skipped'))"
    )
    publisher = workflow.jobs['review-statuses']
    assert publisher.permissions == {
        'contents': 'read',
        'actions': 'read',
        'issues': 'read',
        'pull-requests': 'read',
        'statuses': 'write',
    }
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
    assert set(observer.on) == {
        'pull_request',
        'pull_request_review',
        'pull_request_review_comment',
    }
    assert observer.on['pull_request'].paths is None
    assert set(observer.on['pull_request'].types) == {
        'opened',
        'synchronize',
        'reopened',
        'labeled',
        'unlabeled',
        'edited',
    }
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
        "always() && (needs.review.result == 'failure' || needs.douwebot-opencode.result == 'failure' || "
        "steps.publish.outcome == 'failure') && "
        "github.event.action == 'labeled' && github.event.label.name == 'douwebot'"
    )
    assert config_report == (
        "always() && (needs.review.result == 'failure' || needs.douwebot-opencode.result == 'failure' || "
        "steps.publish.outcome == 'failure') && "
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
        ('pull-request', 'active'),
        ('producer', 'active'),
        ('checks-producer', 'active'),
        ('group', 'active'),
        ('foreign-pr', 'error'),
        ('wrong-sha', 'error'),
        ('wrong-repository', 'error'),
        ('ambiguous', 'error'),
        ('crowded-branch', 'error'),
        ('closed-association', 'error'),
        ('wrong-source', 'error'),
        ('wrong-path', 'error'),
        ('foreign-workflow-repository', 'error'),
        ('native', 'error'),
        ('observer-bot-event', 'error'),
        ('producer-review-event', 'error'),
        ('checks-producer-review-event', 'error'),
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
if (data.scenario === 'wrong-repository') {
  associations[0].head.repo.id = 777;
  serverRun.event = 'pull_request';
}
if (data.scenario === 'closed-association') associations[0].state = 'closed';
if (data.scenario === 'ambiguous') associations.push({...structuredClone(source), number: 9});
if (data.scenario === 'moved-while-queued') live.head.sha = data.otherHead;
if (data.scenario === 'repository-changed') live.head.repo.id = 777;
if (data.scenario === 'closed-while-queued') live.state = 'closed';
if (data.scenario === 'wrong-source') serverRun.event = 'workflow_dispatch';
if (data.scenario === 'review-comment') serverRun.event = 'pull_request_review_comment';
if (data.scenario === 'pull-request') serverRun.event = 'pull_request';
if (data.scenario === 'wrong-path') serverRun.path = '.github/workflows/untrusted.yml';
if (data.scenario === 'foreign-workflow-repository') serverRun.repository.full_name = 'attacker/pydantic-ai';
if (data.scenario === 'observer-bot-event') serverRun.event = 'pull_request_target';
if (['producer', 'producer-review-event'].includes(data.scenario)) {
  serverRun.name = 'PR Bots'; serverRun.path = '.github/workflows/bots.yml';
  serverRun.event = 'pull_request';
}
if (data.scenario === 'producer-review-event') serverRun.event = 'pull_request_review';
if (['checks-producer', 'checks-producer-review-event'].includes(data.scenario)) {
  serverRun.name = 'PR Review Checks'; serverRun.path = '.github/workflows/pr-review-checks.yml';
  serverRun.event = data.scenario === 'checks-producer' ? 'pull_request' : 'pull_request_review';
}
// GitHub reports the base repository as a review run's head repository, even for a fork PR.
if (['pull_request_review', 'pull_request_review_comment'].includes(serverRun.event)) {
  serverRun.head_repository = {id: 1, owner: {login: 'pydantic'}};
}
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
  },
  graphql: async (query, variables) => {
    calls.push(['branch', variables]);
    if (data.scenario === 'api-error') throw new Error('GitHub unavailable');
    const nodes = associations.map(pull => ({
      number: pull.number, state: pull.state.toUpperCase(), headRefOid: pull.head.sha,
      headRepository: {databaseId: pull.head.repo.id}
    }));
    return {repository: {pullRequests: {pageInfo: {hasNextPage: data.scenario === 'crowded-branch'}, nodes}}};
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
            # A review run cannot name a fork's owner, so it looks the branch up across repositories.
            lookup = (
                [
                    'associations',
                    {
                        'owner': 'pydantic',
                        'repo': 'pydantic-ai',
                        'state': 'open',
                        'head': 'attacker:attacker-branch',
                        'per_page': 100,
                    },
                ]
                if scenario in {'pull-request', 'producer', 'checks-producer'}
                else ['branch', {'owner': 'pydantic', 'name': 'pydantic-ai', 'branch': 'attacker-branch'}]
            )
            assert output['calls'][:2] == [['run', {'owner': 'pydantic', 'repo': 'pydantic-ai', 'run_id': 900}], lookup]
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
CONTRIBUTOR = review_gates.ReviewUser(id=5, login='contributor', type='User')
STRANGER = review_gates.ReviewUser(id=7, login='stranger', type='User')
POSTED = datetime(2026, 12, 1, tzinfo=UTC)


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
    resolved: set[int] = field(default_factory=set[int])
    verified: bool = True
    commits: set[str] = field(default_factory=set[str])
    issues: dict[int, datetime] = field(default_factory=dict[int, datetime])
    writers: set[str] = field(default_factory=lambda: {MAINTAINER.login})
    head: str = HEAD
    repository: int = 888
    closed: bool = False
    author: tuple[str, str] = (CONTRIBUTOR.login, 'User')
    author_permission: str = 'read'
    files: list[ChangedFile] = field(
        default_factory=lambda: [ChangedFile(filename='pydantic_ai_slim/pydantic_ai/agent.py')]
    )
    statuses: list[tuple[str, review_gates.ReviewGate, review_gates.CommitStatus]] = field(
        default_factory=list[tuple[str, review_gates.ReviewGate, review_gates.CommitStatus]]
    )

    def pull_request(self) -> review_gates.ReviewPullRequest:
        return review_gates.ReviewPullRequest.model_validate(
            {
                'state': 'closed' if self.closed else 'open',
                'number': 7,
                'html_url': 'https://github.com/o/r/pull/7',
                'user': {'id': 5, 'login': self.author[0], 'type': self.author[1]},
                'head': {'sha': self.head, 'repo': {'id': self.repository}},
                'base': {'sha': OLD_HEAD, 'repo': {'id': 888}},
                'changed_files': 1,
                'labels': [{'name': label} for label in self.labels if label]
                if self.labels is not None
                else [{'name': self.size_label}],
            }
        )

    def changed_files(self) -> list[ChangedFile]:
        return self.files.copy()

    def size(self, pull: review_gates.ReviewPullRequest) -> review_gates.Size:
        return self.size_label

    def reviews(self) -> list[review_gates.Review]:
        return self.current_reviews.copy()

    def review_comments(self) -> list[review_gates.ReviewComment]:
        return self.comments.copy()

    def resolved_comment_ids(self) -> set[int]:
        return self.resolved.copy()

    def verified_review_run(self, run_id: int, head: str) -> bool:
        assert (run_id, head) == (20, HEAD)
        return self.verified

    def own_commits(self, base: str, head: str, shas: set[str]) -> set[str]:
        return {sha for sha in shas if any(commit.startswith(sha) for commit in self.commits)}

    def issue_created_at(self, number: int) -> datetime | None:
        return self.issues.get(number)

    def permission(self, login: str) -> str:
        if login == self.author[0]:
            return self.author_permission
        return 'write' if login in self.writers else 'read'

    def status(
        self, head: str, gate: review_gates.ReviewGate, state: review_gates.CommitStatus, description: str, url: str
    ) -> None:
        self.statuses.append((head, gate, state))


def verdicts(github: ReviewGitHub) -> dict[review_gates.ReviewGate, pr_gates.Verdict]:
    snapshot = (
        github,
        github.pull_request(),
        github.reviews(),
        github.review_comments(),
        github.resolved_comment_ids(),
    )
    return {
        **review_gates.review_verdicts(*snapshot),
        'dispositions': review_gates.finding_dispositions(*snapshot),
    }


@pytest.mark.parametrize('size', ['size: S', 'size: M', 'size: L', 'size: XL'])
def test_review_bar_by_size(size: review_gates.Size):
    github = ReviewGitHub(size_label=size, current_reviews=[])
    result = verdicts(github)
    assert bool(result['douwebot'].failures) is (size != 'size: S')
    assert set(result) == {'douwebot', 'github-dir', 'dispositions'}
    github.current_reviews = [review()]
    assert all(not item.failures for item in verdicts(github).values())


def test_a_missing_size_label_names_the_activation_label():
    """PR Bots labels only a trusted PR, so an app PR waiting for `trusted:ci-run` has no size label."""
    failures = verdicts(ReviewGitHub(labels=[]))['douwebot'].failures
    assert failures == [
        'No size label (computed `size: M`): PR Bots labels a PR whose author has write access, or with `trusted:ci-run`.'
    ]


@pytest.mark.parametrize('size', ['size: M', 'size: XL'])
def test_a_fork_pr_without_a_size_label_is_recreated(size: review_gates.Size):
    """PR Bots never labels a fork PR, and `trusted:ci-run` cannot change that."""
    failures = verdicts(ReviewGitHub(size_label=size, labels=[], repository=999))['douwebot'].failures
    assert failures == [
        f'No size label (computed `{size}`): PR Bots skips fork PRs. A maintainer recreates the change as a same-repository PR.'
    ]
    assert len(failures[0]) <= 140


@pytest.mark.parametrize('labels', [[], ['size: S'], ['size: M', 'size: L']])
def test_size_must_match_exactly_one_recomputed_label(labels: list[str]):
    result = verdicts(ReviewGitHub(labels=labels))
    assert result['douwebot'].failures
    # The `.github/` gate applies at every size, so a size label cannot fail it.
    assert not result['github-dir'].failures
    assert not result['dispositions'].failures


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
    body = json.loads(publication(result, HEAD, 20, engine=None))['body']
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
    parent = review(9, state=state, body=json.loads(publication(result, HEAD, 20, engine=None))['body'])
    comments = (
        [
            review_gates.ReviewComment(
                created_at=POSTED,
                id=30,
                pull_request_review_id=parent.id,
                user=BOT,
                original_commit_id=HEAD,
                body=finding.body,
            )
        ]
        if location == 'inline'
        else []
    )
    assert verdicts(ReviewGitHub(current_reviews=[parent, review()], comments=comments))['douwebot'].failures


@pytest.mark.parametrize(
    ('state', 'body', 'discussion', 'resolved', 'fails'),
    [
        pytest.param('APPROVED', 'Optional: consider this', '', True, False, id='resolved-optional'),
        pytest.param('UNVERIFIED', 'Optional: consider this', '', True, True, id='unverified-approval'),
        pytest.param('APPROVED', 'Optional: consider this', '', False, True, id='unresolved-optional'),
        pytest.param('APPROVED', 'Fix this', '', True, True, id='resolved-blocking'),
        pytest.param('APPROVED', 'Note: Optional: consider this', '', True, True, id='prefix-not-leading'),
        pytest.param('APPROVED', 'Optional: consider this', 'Discussion', True, True, id='body-discussion'),
        pytest.param('CHANGES_REQUESTED', 'Optional: consider this', '', True, True, id='changes-requested-parent'),
    ],
)
def test_only_resolved_optional_findings_on_an_approval_stay_clean(
    state: str, body: str, discussion: str, resolved: bool, fails: bool
):
    class OnlyRun20(ReviewGitHub):
        def verified_review_run(self, run_id: int, head: str) -> bool:
            return run_id == 20

    verdict = 'REQUEST_CHANGES' if state == 'CHANGES_REQUESTED' else 'APPROVE'
    finding = Finding(path='a.py', line=3, side='RIGHT', body=body)
    result = ReviewResult(verdict=verdict, comments=[finding], discussion=discussion)
    # An unverified review carries a marker for a run that is not a successful PR Bots publication.
    run = 21 if state == 'UNVERIFIED' else 20
    payload = json.loads(publication(result, HEAD, run, engine='claude'))
    parent = review(9, state='APPROVED' if state == 'UNVERIFIED' else state, body=payload['body'])
    comment = review_gates.ReviewComment(
        created_at=POSTED,
        id=30,
        pull_request_review_id=parent.id,
        user=BOT,
        original_commit_id=HEAD,
        body=payload['comments'][0]['body'],
    )
    # Other parents are followed by a clean verified approval, so only their finding can fail the head.
    reviews = [parent] if state == 'APPROVED' else [parent, review()]
    github = OnlyRun20(current_reviews=reviews, comments=[comment], resolved={30} if resolved else set[int]())
    assert bool(verdicts(github)['douwebot'].failures) is fails


def test_failed_douwebot_publication_cannot_hide_a_finding_behind_another_receipt():
    class FailedRun(ReviewGitHub):
        def verified_review_run(self, run_id: int, head: str) -> bool:
            return run_id == 20

    parent = review(9, body=f'No blocking findings.\n\n<!-- douwebot:review run=21 head={HEAD} -->')
    comment = review_gates.ReviewComment(
        created_at=POSTED,
        id=30,
        pull_request_review_id=9,
        user=BOT,
        original_commit_id=HEAD,
        body='Quote: <!-- gh-aw-workflow-call-id: pydantic/pydantic-ai/pydantic-ai-pr-review -->',
    )
    assert verdicts(FailedRun(current_reviews=[parent, review()], comments=[comment]))['douwebot'].failures


@pytest.mark.parametrize('current', [False, True])
def test_relocated_comment_uses_its_parent_review_head(current: bool):
    # GitHub forwards commit_id for the review and its non-outdated comments, as on #12375.
    # Only the parent's marker still says this comment was posted on the older head.
    parent = review(9, head=HEAD if current else OLD_HEAD).model_copy(update={'commit_id': HEAD})
    comment = review_gates.ReviewComment.model_validate(
        {
            'id': 30,
            'pull_request_review_id': parent.id,
            'user': BOT,
            'body': 'Optional: finding',
            'commit_id': HEAD,
            'original_commit_id': OLD_HEAD,
            'created_at': POSTED,
        }
    )
    github = ReviewGitHub(current_reviews=[parent, review()], comments=[comment])
    assert bool(verdicts(github)['douwebot'].failures) is current


@pytest.mark.parametrize('current', [False, True])
def test_relocated_comment_under_an_unmarked_review_uses_its_original_head(current: bool):
    parent = review(9, body='Unidentified review')
    comment = review_gates.ReviewComment(
        created_at=POSTED,
        id=30,
        pull_request_review_id=parent.id,
        user=BOT,
        original_commit_id=HEAD if current else OLD_HEAD,
        body='finding',
    )
    github = ReviewGitHub(current_reviews=[parent, review()], comments=[comment])
    assert bool(verdicts(github)['douwebot'].failures) is current


@pytest.mark.parametrize('workflow', ['pydantic-ai-pr-review', 'pydantic-ai-ui-security-review'])
def test_other_review_receipt_does_not_count_as_douwebot_finding(workflow: str):
    parent = review(8, body=f'Reviewed at head\n<!-- gh-aw-workflow-call-id: pydantic/pydantic-ai/{workflow} -->')
    comment = review_gates.ReviewComment(
        created_at=POSTED, id=30, pull_request_review_id=8, user=BOT, original_commit_id=HEAD, body='CI finding'
    )
    github = ReviewGitHub(current_reviews=[parent, review()], comments=[comment])
    assert not verdicts(github)['douwebot'].failures
    github.current_reviews = [review(8, body='Unidentified review'), review()]
    assert verdicts(github)['douwebot'].failures
    github.current_reviews = [review()]
    assert verdicts(github)['douwebot'].failures
    github.comments[0] = comment.model_copy(update={'original_commit_id': OLD_HEAD})
    assert not verdicts(github)['douwebot'].failures


def test_pr_review_checks_receipt_does_not_count_as_douwebot_finding():
    result = CheckResult(
        check='general',
        files=['a.py'],
        outcome='COMPLETE',
        findings=[],
        dropped={},
        coverage='Reviewed a.py.',
        steps=1,
        tokens=1,
        seconds=1,
    )
    parent = review(8, body=json.loads(pr_review_checks_payload([result], HEAD, 30, inline=True, lines={}))['body'])
    comment = review_gates.ReviewComment(
        created_at=POSTED, id=30, pull_request_review_id=8, user=BOT, original_commit_id=HEAD, body='Security finding'
    )
    github = ReviewGitHub(current_reviews=[parent, review()], comments=[comment])
    assert not verdicts(github)['douwebot'].failures
    github.current_reviews = [review(8, body='Unidentified review'), review()]
    assert verdicts(github)['douwebot'].failures


ENV = {
    'HEAD_SHA': HEAD,
    'EXPECTED_HEAD_SHA': HEAD,
    'EXPECTED_HEAD_REPOSITORY_ID': '888',
    'GITHUB_SERVER_URL': 'https://github.com',
    'GITHUB_REPOSITORY': 'o/r',
    'GITHUB_RUN_ID': '100',
    'REVIEW_ENGINE': 'claude',
}


def test_reevaluation_replaces_a_failure_with_success_on_the_live_head():
    github = ReviewGitHub(current_reviews=[])
    assert review_gates.reconcile_reviews(github, ENV) == 0
    assert github.statuses == [
        (HEAD, 'douwebot', 'failure'),
        (HEAD, 'github-dir', 'success'),
        (HEAD, 'dispositions', 'success'),
    ]
    github.current_reviews = [review()]
    assert review_gates.reconcile_reviews(github, ENV) == 0
    assert github.statuses == [
        (HEAD, 'douwebot', 'failure'),
        (HEAD, 'github-dir', 'success'),
        (HEAD, 'dispositions', 'success'),
        (HEAD, 'douwebot', 'success'),
        (HEAD, 'github-dir', 'success'),
        (HEAD, 'dispositions', 'success'),
    ]


def test_a_failed_disposition_lookup_leaves_the_douwebot_status_its_own():
    class IssuesDown(ReviewGitHub):
        def issue_created_at(self, number: int) -> datetime | None:
            raise GateError('Server Error (HTTP 502)')

    github = IssuesDown(comments=[finding(), reply('Tracked in #12')])
    assert review_gates.reconcile_reviews(github, ENV) == 1
    assert github.statuses == [
        (HEAD, 'douwebot', 'success'),
        (HEAD, 'github-dir', 'success'),
        (HEAD, 'dispositions', 'failure'),
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
    assert github.statuses == [
        (HEAD, 'douwebot', 'failure'),
        (HEAD, 'github-dir', 'failure'),
        (HEAD, 'dispositions', 'failure'),
    ]

    class Moving(ReviewGitHub):
        def pull_request(self) -> review_gates.ReviewPullRequest:
            self.labels = ['size: M', 'chore'] if self.labels is None else None
            return super().pull_request()

    moving = Moving(current_reviews=[])
    assert review_gates.reconcile_reviews(moving, ENV) == 1
    assert moving.statuses == [
        (HEAD, 'douwebot', 'failure'),
        (HEAD, 'github-dir', 'failure'),
        (HEAD, 'dispositions', 'failure'),
    ]


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


def test_merge_group_skips_metadata_and_reports_every_gate():
    class NoMetadata(ReviewGitHub):
        def pull_request(self) -> review_gates.ReviewPullRequest:
            raise AssertionError('Merge groups do not inspect queue members')

    github = NoMetadata()
    assert review_gates.reconcile_reviews(github, {**ENV, 'MERGE_GROUP_SHA': OLD_HEAD}) == 0
    assert github.statuses == [
        (OLD_HEAD, 'douwebot', 'success'),
        (OLD_HEAD, 'github-dir', 'success'),
        (OLD_HEAD, 'dispositions', 'success'),
    ]


VERIA = review_gates.ReviewUser(id=4, login='veria-ai[bot]', type='Bot')
RECORDINGS = Path(__file__).parent / 'fixtures' / 'review_dispositions'


class _Recording(BaseModel):
    """A merged pull request's review metadata, read from the REST and GraphQL endpoints on 2026-11-19."""

    pull: review_gates.ReviewPullRequest
    reviews: list[review_gates.Review]
    comments: list[review_gates.ReviewComment]
    commits: list[str]
    resolved: set[int]


def finding(
    user: review_gates.ReviewUser = VERIA,
    body: str = 'Low: this branch is unreachable',
    parent: int = 50,
    posted: datetime = POSTED,
) -> review_gates.ReviewComment:
    return review_gates.ReviewComment(
        id=40, pull_request_review_id=parent, user=user, original_commit_id=OLD_HEAD, created_at=posted, body=body
    )


def reply(body: str, user: review_gates.ReviewUser = CONTRIBUTOR, id: int = 41) -> review_gates.ReviewComment:
    return review_gates.ReviewComment(
        id=id,
        pull_request_review_id=id + 10,
        in_reply_to_id=40,
        user=user,
        original_commit_id=HEAD,
        created_at=POSTED,
        body=body,
    )


@pytest.mark.parametrize(
    ('number', 'reviewers', 'stacked', 'opened'),
    [
        (13615, ['Veria', 'PR Review Checks'], 13630, datetime(2026, 11, 16, 20, 29, 41, tzinfo=UTC)),
        (13617, ['PR Review Checks'], 13659, datetime(2026, 11, 16, 21, 3, 5, tzinfo=UTC)),
    ],
)
def test_merged_prs_with_unanswered_findings_fail(number: int, reviewers: list[str], stacked: int, opened: datetime):
    """Both PRs merged at size S with bot findings answered only by a pointer to a later stacked PR."""
    recording = _Recording.model_validate_json((RECORDINGS / f'{number}.json').read_text(encoding='utf-8'))
    github = ReviewGitHub(
        size_label='size: S',
        head=recording.pull.head.sha,
        commits=set(recording.commits),
        issues={stacked: opened},
        writers={'mpfaffenberger'},
    )

    def evaluate(comments: list[review_gates.ReviewComment]) -> dict[review_gates.ReviewGate, pr_gates.Verdict]:
        snapshot = (github, recording.pull, recording.reviews, comments, recording.resolved)
        return {
            **review_gates.review_verdicts(*snapshot),
            'dispositions': review_gates.finding_dispositions(*snapshot),
        }

    result = evaluate(recording.comments)
    assert result['douwebot'] == pr_gates.Verdict(
        summary=['Size S: automated review enforcement is skipped; the manual merge bar still applies.']
    )
    assert (
        result['dispositions']
        .failures[0]
        .startswith(f'{len(reviewers)} of {len(reviewers)} review bot findings need a reply')
    )
    assert [failure.split(': the ')[1] for failure in result['dispositions'].failures[1:]] == [
        f'{reviewer} finding has no disposition reply.' for reviewer in reviewers
    ]
    answered = [
        comment.model_copy(update={'body': f'Tracked in #{stacked}\n\n{comment.body}'})
        if comment.in_reply_to_id
        else comment
        for comment in recording.comments
    ]
    assert not evaluate(answered)['dispositions'].failures


@pytest.mark.parametrize(
    ('body', 'user', 'passes'),
    [
        pytest.param('Fixed in 1234abc: the branch now returns early.', CONTRIBUTOR, True, id='fixed'),
        pytest.param('> Posted by an agent.\n\nFixed in 1234abc', CONTRIBUTOR, True, id='after-attribution'),
        pytest.param('Fixed in 7777777', CONTRIBUTOR, False, id='fixed-outside-this-pr'),
        pytest.param('Agreed, fixed in 1234abc', CONTRIBUTOR, False, id='not-line-start'),
        pytest.param('Fixed in 123abc', CONTRIBUTOR, False, id='sha-too-short'),
        pytest.param('Fixed in 1234abc', VERIA, False, id='bot-reply'),
        pytest.param('Tracked in #12', CONTRIBUTOR, True, id='tracked'),
        pytest.param('Tracked in #13', CONTRIBUTOR, False, id='tracked-missing'),
        pytest.param('Tracked in #7', CONTRIBUTOR, False, id='tracked-this-pr'),
        pytest.param('Tracked in #13\nTracked in #12', CONTRIBUTOR, False, id='first-line-decides'),
        pytest.param('Tracked in #' + '9' * 5000, CONTRIBUTOR, False, id='tracked-number-too-long'),
        pytest.param('Tracked in #11', CONTRIBUTOR, False, id='tracked-older-issue'),
        pytest.param('Tracked in #11', MAINTAINER, True, id='tracked-older-issue-writer'),
        pytest.param('Fixed in 1234abc', STRANGER, False, id='fixed-third-party'),
        pytest.param('Tracked in #12', STRANGER, False, id='tracked-third-party'),
        pytest.param('Declined: only reachable from a test double.', MAINTAINER, True, id='declined-writer'),
        pytest.param('Declined: only reachable from a test double.', CONTRIBUTOR, False, id='declined-reader'),
        pytest.param('Declined:', MAINTAINER, False, id='declined-without-reason'),
        pytest.param('Addressed in stacked #12.', MAINTAINER, False, id='prose'),
    ],
)
def test_a_finding_needs_a_disposition_reply(body: str, user: review_gates.ReviewUser, passes: bool):
    github = ReviewGitHub(
        comments=[finding(), reply(body, user)],
        commits={'1234abc' + 'f' * 33},
        issues={12: POSTED, 7: POSTED, 11: POSTED.replace(day=1, month=11)},
    )
    assert not verdicts(github)['douwebot'].failures
    assert bool(verdicts(github)['dispositions'].failures) is not passes


@pytest.mark.parametrize(
    ('body', 'user', 'reason'),
    [
        pytest.param('Fixed in 7777777', CONTRIBUTOR, '`7777777` is not a commit of this PR', id='rewritten-history'),
        pytest.param('Tracked in #13', CONTRIBUTOR, '#13 is not another issue or PR', id='tracked-missing'),
        pytest.param('Tracked in #11', CONTRIBUTOR, '#11 was opened before the finding', id='tracked-older-issue'),
        pytest.param('Declined: noise.', CONTRIBUTOR, '`Declined:` needs write access', id='declined-reader'),
        pytest.param(
            'Fixed in 1234abc',
            STRANGER,
            '`stranger` is neither the PR author nor a user with write access',
            id='third-party',
        ),
    ],
)
def test_a_rejected_reply_names_its_reason(body: str, user: review_gates.ReviewUser, reason: str):
    """A rebase drops the commits earlier `Fixed in` replies name, so the status says why a reply does not count."""
    github = ReviewGitHub(
        comments=[finding(), reply(body, user)],
        commits={'1234abc' + 'f' * 33},
        issues={11: POSTED.replace(day=1, month=11)},
    )
    assert verdicts(github)['dispositions'].failures[1:] == [
        f'https://github.com/o/r/pull/7#discussion_r40: the Veria finding has no accepted disposition reply: {reason}.'
    ]


@pytest.mark.parametrize('size', ['size: S', 'size: M', 'size: L', 'size: XL'])
def test_findings_on_every_head_and_size_need_an_answer(size: review_gates.Size):
    github = ReviewGitHub(size_label=size, comments=[finding()])
    failures = verdicts(github)['dispositions'].failures
    assert failures == [
        '1 of 1 review bot findings need a reply: `Fixed in <sha>`, `Tracked in #N` or `Declined: <reason>`.',
        'https://github.com/o/r/pull/7#discussion_r40: the Veria finding has no disposition reply.',
    ]
    github.comments.append(reply('Declined: covered by the type checker.', MAINTAINER))
    assert not verdicts(github)['dispositions'].failures


@pytest.mark.parametrize(
    ('posted', 'resolved', 'passes'),
    [
        (review_gates.RESOLVED_THREADS_STARTED_BEFORE.replace(hour=1), True, False),
        (review_gates.RESOLVED_THREADS_STARTED_BEFORE.replace(day=18), True, True),
        (review_gates.RESOLVED_THREADS_STARTED_BEFORE.replace(day=18), False, False),
    ],
)
def test_only_threads_resolved_before_the_cutoff_need_no_answer(posted: datetime, resolved: bool, passes: bool):
    github = ReviewGitHub(comments=[finding(posted=posted)], resolved={40} if resolved else set[int]())
    assert bool(verdicts(github)['dispositions'].failures) is not passes


def test_only_write_access_answers_an_unregistered_bot_thread():
    """A maintainer can always clear a thread the gate cannot attribute, and the run still names it."""
    stranger = review_gates.ReviewUser(id=6, login='new-reviewer[bot]', type='Bot')
    github = ReviewGitHub(
        comments=[finding(stranger), reply('Fixed in 1234abc', CONTRIBUTOR)], commits={'1234abc' + 'f' * 33}
    )
    unregistered = [
        '1 review bot threads have no registered reviewer, so only a reply from write access answers them. '
        'Register the bot in `REVIEWER_LOGINS`, or see #14014 for a `github-actions[bot]` thread without a receipt.',
        'https://github.com/o/r/pull/7#discussion_r40: no registered reviewer started this `new-reviewer[bot]` thread.',
    ]
    assert verdicts(github)['dispositions'] == pr_gates.Verdict(
        summary=['1 review bot findings.', *unregistered],
        failures=[
            '1 of 1 review bot findings need a reply: `Fixed in <sha>`, `Tracked in #N` or `Declined: <reason>`.',
            'https://github.com/o/r/pull/7#discussion_r40: the `new-reviewer[bot]` thread has no accepted disposition '
            'reply: only write access answers a thread no registered reviewer started, and `contributor` has none.',
        ],
    )
    github.comments.append(reply('Declined: noise.', MAINTAINER, id=42))
    assert verdicts(github)['dispositions'] == pr_gates.Verdict(summary=['1 review bot findings.', *unregistered])
    github.comments = [finding(stranger, posted=review_gates.RESOLVED_THREADS_STARTED_BEFORE.replace(day=18))]
    github.resolved = {40}
    assert not verdicts(github)['dispositions'].failures


def test_at_claude_threads_need_a_disposition():
    """The `@claude` workflow can start inline threads as `claude[bot]`."""
    claude = review_gates.ReviewUser(id=8, login='claude[bot]', type='Bot')
    github = ReviewGitHub(comments=[finding(claude)])
    assert verdicts(github)['dispositions'].failures[1:] == [
        'https://github.com/o/r/pull/7#discussion_r40: the Claude finding has no disposition reply.'
    ]
    github.comments.append(reply('Declined: outside this change.', MAINTAINER))
    assert not verdicts(github)['dispositions'].failures


def test_threads_a_person_starts_are_not_findings():
    assert not verdicts(ReviewGitHub(comments=[finding(MAINTAINER)]))['dispositions'].failures


def _douwebot_finding(body: str) -> tuple[review_gates.Review, review_gates.ReviewComment]:
    result = ReviewResult(
        verdict='APPROVE', comments=[Finding(path='a.py', line=3, side='RIGHT', body=body)], discussion=''
    )
    payload = json.loads(publication(result, HEAD, 20, engine=None))
    return review(9, body=payload['body']), finding(BOT, payload['comments'][0]['body'], parent=9)


@pytest.mark.parametrize(
    ('body', 'user', 'passes'),
    [
        ('Optional: rename this', CONTRIBUTOR, True),
        ('Fix this', CONTRIBUTOR, False),
        ('Optional: rename', STRANGER, False),
    ],
)
def test_the_author_may_decline_an_optional_douwebot_finding(body: str, user: review_gates.ReviewUser, passes: bool):
    parent, comment = _douwebot_finding(body)
    github = ReviewGitHub(
        current_reviews=[parent], comments=[comment, reply('Declined: the name matches the API.', user)]
    )
    assert bool(verdicts(github)['dispositions'].failures) is not passes


@pytest.mark.parametrize(
    ('parent_body', 'reviewer'),
    [
        (f'<!-- douwebot:review run=20 head={HEAD} -->', 'douwebot'),
        (f'<!-- pr-review-checks run=30 head={HEAD} -->', 'PR Review Checks'),
        ('<!-- gh-aw-workflow-call-id: pydantic/pydantic-ai/pydantic-ai-pr-review -->', 'CI Review'),
        ('Unidentified review', None),
    ],
)
def test_github_actions_findings_are_told_apart_by_receipt(parent_body: str, reviewer: str | None):
    parent = review(9, body=f'Findings\n\n{parent_body}')
    assert review_gates.finding_reviewer(finding(BOT, parent=9), parent) == reviewer


@pytest.mark.parametrize('listed', [1, 250])
def test_gh_cli_answers_disposition_lookups(listed: int):
    """The PR commits endpoint stops at 250; only then does an unmatched SHA cost comparisons."""
    own = '1234abc' + 'f' * 33
    later = 'abcdef0' + '0' * 33
    calls: list[str] = []

    class Metadata(review_gates.ReviewGhCli):
        def _api(self, *args: str, input: str | None = None) -> str:
            calls.append(args[-1])
            match args[-1]:
                case 'repos/o/r/pulls/7/commits?per_page=100':
                    commits = [own, *(f'{index:040x}' for index in range(listed - 1))]
                    return json.dumps(
                        [[{'sha': sha} for sha in commits[start : start + 100]] for start in (0, 100, 200)]
                    )
                case 'repos/o/r/commits/abcdef0':
                    return json.dumps({'sha': later})
                case 'repos/o/r/commits/baseba5':
                    return json.dumps({'sha': 'baseba5' + '0' * 33})
                case 'repos/o/r/commits/7777777':
                    raise GateError('No commit found for SHA: 7777777 (HTTP 422)')
                case path if path == f'repos/o/r/compare/{OLD_HEAD}...{later}?per_page=1':
                    return '{"status": "ahead"}'
                case path if path == f'repos/o/r/compare/{later}...{HEAD}?per_page=1':
                    return '{"status": "ahead"}'
                case path if path == f'repos/o/r/compare/{OLD_HEAD}...baseba5{"0" * 33}?per_page=1':
                    return '{"status": "behind"}'
                case path if path == f'repos/o/r/compare/baseba5{"0" * 33}...{HEAD}?per_page=1':
                    return '{"status": "ahead"}'
                case 'repos/o/r/issues/12':
                    return '{"created_at": "2026-11-19T04:57:52Z"}'
                case 'repos/o/r/issues/13' | 'repos/o/r/collaborators/stranger/permission':
                    raise GateError('Not Found (HTTP 404)')
                case 'repos/o/r/collaborators/maintainer/permission':
                    return '{"permission": "write"}'
                case _:
                    raise GateError(f'Unexpected {args[-1]} (HTTP 500)')

    github = Metadata('o/r', 7)
    assert github.own_commits(OLD_HEAD, HEAD, set()) == set()
    assert calls == []
    found = github.own_commits(OLD_HEAD, HEAD, {'1234abc', 'abcdef0', 'baseba5', '7777777'})
    assert found == ({'1234abc', 'abcdef0'} if listed == 250 else {'1234abc'})
    assert len(calls) == (1 if listed == 1 else 8)
    assert github.issue_created_at(12) == datetime(2026, 11, 19, 4, 57, 52, tzinfo=UTC)
    assert github.issue_created_at(13) is None
    with pytest.raises(GateError):
        github.issue_created_at(14)
    assert github.permission('maintainer') == 'write'
    assert github.permission('stranger') == 'none'
    # Fail closed: a permission that cannot be read never grants write access.
    assert github.permission('unreachable') == 'none'


@pytest.mark.parametrize(
    ('author', 'permission', 'files', 'allowed'),
    [
        (('maintainer', 'User'), 'write', ['.github/workflows/ci.yml'], True),
        (('owner', 'User'), 'admin', ['.github/scripts/pr_gates.py'], True),
        (('contributor', 'User'), 'read', ['.github/workflows/ci.yml'], False),
        (('contributor', 'User'), 'none', ['.github/workflows/ci.yml'], False),
        (('pydanty[bot]', 'Bot'), 'write', ['.github/scripts/pr_gates.py'], False),
        (('dependabot[bot]', 'Bot'), 'none', ['.github/workflows/ci.yml'], True),
        (('contributor', 'User'), 'read', ['docs/agents.md', 'pydantic_ai_slim/pydantic_ai/agent.py'], True),
    ],
    ids=['write', 'admin', 'read', 'none', 'app', 'dependabot', 'nothing-protected'],
)
def test_only_a_maintainer_may_change_the_github_directory(
    author: tuple[str, str], permission: str, files: list[str], allowed: bool
) -> None:
    """A fork can edit its own copy of `protect-github-dir.yml`, so the default-branch gate decides."""
    github = ReviewGitHub(
        author=author, author_permission=permission, files=[ChangedFile(filename=name) for name in files]
    )
    assert (not verdicts(github)['github-dir'].failures) is allowed


def test_a_rename_out_of_the_github_directory_needs_a_maintainer() -> None:
    github = ReviewGitHub(
        author=('contributor', 'User'),
        author_permission='read',
        files=[ChangedFile(filename='tools/ci.yml', previous_filename='.github/workflows/ci.yml')],
    )
    assert verdicts(github)['github-dir'].failures == [
        'Only a maintainer may change `.github/`; this PR changes .github/workflows/ci.yml.'
    ]


def test_publisher_binds_inline_comments_to_one_marked_review():
    result = ReviewResult(
        verdict='APPROVE', comments=[Finding(path='a.py', line=3, side='RIGHT', body='Optional: fix')], discussion=''
    )
    payload = json.loads(publication(result, HEAD, 20, engine=None))
    parent = review(body=payload['body'])
    comment = review_gates.ReviewComment(
        created_at=POSTED,
        id=30,
        pull_request_review_id=parent.id,
        user=BOT,
        original_commit_id=HEAD,
        body=payload['comments'][0]['body'],
    )
    assert payload['commit_id'] == HEAD
    assert payload['event'] == 'APPROVE'
    assert verdicts(ReviewGitHub(current_reviews=[parent], comments=[comment]))['douwebot'].failures
    assert not verdicts(ReviewGitHub(current_reviews=[parent], comments=[comment], resolved={30}))['douwebot'].failures
    assert (
        json.loads(publication(result.model_copy(update={'verdict': 'INCOMPLETE'}), HEAD, 20, engine=None))['event']
        == 'COMMENT'
    )


@pytest.mark.parametrize('engine', ['claude', 'opencode', 'opencode-zai', 'opencode-deepinfra'])
def test_publisher_engine_attribution_preserves_the_gate(engine: publish_douwebot_review.ReviewEngine):
    result = ReviewResult(verdict='APPROVE', comments=[], discussion='')
    body = json.loads(publication(result, HEAD, 20, engine=engine))['body']
    assert f'engine={engine}' in body
    assert not verdicts(ReviewGitHub(current_reviews=[review(body=body)]))['douwebot'].failures
    finding = result.model_copy(update={'discussion': 'Extra discussion'})
    body = json.loads(publication(finding, HEAD, 20, engine=engine))['body']
    assert verdicts(ReviewGitHub(current_reviews=[review(body=body)]))['douwebot'].failures
    duplicated = result.model_copy(update={'discussion': publish_douwebot_review.REVIEW_ENGINES[engine]})
    body = json.loads(publication(duplicated, HEAD, 20, engine=engine))['body']
    assert verdicts(ReviewGitHub(current_reviews=[review(body=body)]))['douwebot'].failures
    body = body.replace('\n\n' + publish_douwebot_review.REVIEW_ENGINES[engine], '')
    assert verdicts(ReviewGitHub(current_reviews=[review(body=body)]))['douwebot'].failures
    legacy = publication(
        result.model_copy(update={'discussion': publish_douwebot_review.REVIEW_ENGINES[engine]}), HEAD, 20, engine=None
    )
    assert verdicts(ReviewGitHub(current_reviews=[review(body=json.loads(legacy)['body'])]))['douwebot'].failures


@pytest.mark.parametrize(
    ('engine', 'attribution'),
    [(engine, text) for engine, texts in publish_douwebot_review.EARLIER_REVIEW_ATTRIBUTIONS.items() for text in texts],
)
def test_gate_reads_the_attribution_of_an_engine_earlier_model(
    engine: publish_douwebot_review.ReviewEngine, attribution: str
):
    """A clean approval an engine published on its earlier model stays clean on its head."""
    result = ReviewResult(verdict='APPROVE', comments=[], discussion='')
    current = json.loads(publication(result, HEAD, 20, engine=engine))['body']
    earlier = current.replace(publish_douwebot_review.REVIEW_ENGINES[engine], attribution)
    assert earlier != current
    assert not verdicts(ReviewGitHub(current_reviews=[review(body=earlier)]))['douwebot'].failures


@pytest.mark.parametrize('state', ['open', 'moved', 'closed'])
def test_publisher_checks_live_head_before_publication(monkeypatch: pytest.MonkeyPatch, state: str):
    for key, value in {
        **ENV,
        'GH_TOKEN': 'token',
        'PR_NUMBER': '7',
        'REVIEW_ENGINE': 'claude',
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
        assert kwargs['input'] == publication(
            ReviewResult(verdict='APPROVE', comments=[], discussion=''), HEAD, 100, engine='claude'
        )
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
        ('.github/workflows/bots.yml', 'pull_request', HEAD, 'success', 'success', True),
        ('.github/workflows/other.yml', 'pull_request', HEAD, 'success', 'success', False),
        ('.github/workflows/bots.yml', 'pull_request_target', HEAD, 'success', 'success', True),
        ('.github/workflows/bots.yml', 'workflow_dispatch', HEAD, 'success', 'success', False),
        ('.github/workflows/bots.yml', 'pull_request', OLD_HEAD, 'success', 'success', False),
        ('.github/workflows/bots.yml', 'pull_request', HEAD, 'failure', 'success', False),
        ('.github/workflows/bots.yml', 'pull_request', HEAD, 'success', 'failure', False),
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
                return '[[{"id":30,"pull_request_review_id":10,"user":{"id":1,"login":"github-actions[bot]","type":"Bot"},"original_commit_id":"old","created_at":"2026-12-01T00:00:00Z","body":"finding"}]]'
            if args[-1] == 'graphql':
                assert args[:2] == ('--paginate', '--slurp')
                assert (
                    ('-f', 'owner=o') == args[2:4] and ('-f', 'name=r') == args[4:6] and ('-F', 'number=7') == args[6:8]
                )

                def page(resolved: bool, comment: int) -> object:
                    thread = {'isResolved': resolved, 'comments': {'nodes': [{'databaseId': comment}]}}
                    return {'data': {'repository': {'pullRequest': {'reviewThreads': {'nodes': [thread]}}}}}

                return json.dumps([page(True, 30), page(False, 31)])
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
    assert github.resolved_comment_ids() == {30}
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
