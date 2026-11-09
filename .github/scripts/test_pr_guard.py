"""Regression tests for PR Guard's issue-assignment rule.

`.github/workflows/pr-guard.yml` closes an outside contributor's pull request unless it links an
open issue assigned to them. It triggers on `pull_request_target`, and GitHub runs such a workflow
from the **default branch of the base repository**, so the PR that edits the guard never exercises
its own change — these tests are the only place its logic runs before `main`.

As in `test_protect_github_dir.py`, the tests extract the guard's `run:` block straight from the
YAML and execute it against a stubbed `gh`, so they track the file that Actions actually runs.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

REPO = 'pydantic/pydantic-ai'
PR_NUMBER = 1234
ISSUE_NUMBER = 1200
CONTRIBUTOR = 'outside-contributor'
WORKFLOW = Path(__file__).parent.parent / 'workflows' / 'pr-guard.yml'

# Each login's `role_name` on the repo. A login missing here gets the 404 GitHub returns for
# bots and deleted accounts.
ROLES = {
    CONTRIBUTOR: 'read',
    'reporter': 'read',
    'drive-by': 'read',
    'triager': 'triage',
    'writer': 'write',
    'lead': 'maintain',
    'owner': 'admin',
}

# The guard parses timestamps with GNU `date -d`. macOS ships BSD `date`, and coreutils installs
# the GNU one as `gdate`.
GDATE = shutil.which('gdate')

requires_tools = pytest.mark.skipif(
    shutil.which('jq') is None or (sys.platform == 'darwin' and GDATE is None),
    reason='the gh stub filters payloads with jq, and the guard parses timestamps with GNU date',
)

# The stub serves the PR and its one linked issue, and records the comment and close it is
# asked to make. It pipes each payload through the real `jq`, so the guard's own `--jq`
# expressions are under test, not just the shell around them.
FAKE_GH = """#!/usr/bin/env python3
import json
import os
import subprocess
import sys

args = sys.argv[1:]
state = json.loads(os.environ['FAKE_GH_STATE'])
repo, pr, issue = state['repo'], state['pr'], state['issue']['number']


def respond(payload):
    text = json.dumps(payload)
    if '--jq' not in args:
        print(text)
        sys.exit(0)
    result = subprocess.run(['jq', '-r', args[args.index('--jq') + 1]], input=text, capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    sys.exit(result.returncode)


def not_found():
    sys.stderr.write('gh: Not Found (HTTP 404)\\n')
    sys.exit(1)


if args[:2] == ['api', 'graphql']:
    query = next(arg for arg in args if arg.startswith('query='))
    if 'closingIssuesReferences' in query:
        nodes = [{'number': issue, 'repository': {'nameWithOwner': repo}}]
        respond({'data': {'repository': {'pullRequest': {'closingIssuesReferences': {'nodes': nodes}}}}})
    if 'closedByPullRequestsReferences' in query:
        respond({'data': {'repository': {'issue': {'closedByPullRequestsReferences': {'nodes': []}}}}})
    sys.exit('stub gh: unhandled query')
elif args[:1] == ['api']:
    endpoint = args[1]
    if endpoint.endswith('/permission'):
        role = state['roles'].get(endpoint.split('/')[-2])
        if role is None:
            not_found()
        respond({'role_name': role})
    elif endpoint == f'repos/{repo}/issues/{pr}/timeline':
        respond([])
    elif endpoint == f'repos/{repo}/issues/{issue}/timeline':
        if state['issue_timeline'] is None:
            not_found()
        respond(state['issue_timeline'])
    elif endpoint == f'repos/{repo}/issues/{issue}':
        respond(state['issue'])
    elif endpoint.startswith(f'repos/{repo}/pulls/{pr}/files'):
        respond(state['files'])
    sys.exit('stub gh: unhandled endpoint ' + endpoint)
elif args[:2] in (['pr', 'comment'], ['pr', 'close']):
    body = args[args.index('--body') + 1] if '--body' in args else ''
    with open(os.environ['FAKE_GH_ACTIONS'], 'a') as handle:
        handle.write(json.dumps({'command': args[1], 'body': body}) + '\\n')
else:
    sys.exit('stub gh: unhandled command ' + repr(args))
"""


def _event(kind: str, actor: str) -> dict[str, object]:
    return {'event': kind, 'actor': {'login': actor}}


@dataclass
class Case:
    """One PR by an outside contributor that links open issue #1200 and changes code.

    Defaults describe the loophole: the contributor filed the issue themselves and nobody else
    has touched it. Each case overrides only what makes it different.
    """

    id: str
    closed: bool
    # `None` makes the timeline unreadable.
    issue_timeline: list[dict[str, object]] | None = field(default_factory=list[dict[str, object]])
    issue_author: str = CONTRIBUTOR
    assignees: list[str] = field(default_factory=list[str])
    # The job fails and leaves the PR untouched when the API can't answer (see `cannot_verify`).
    fails: bool = False
    # A closing reference to a pull request links no issue.
    issue_is_pull_request: bool = False


PYDANTY_TRIAGE = [
    _event('commented', 'pydanty[bot]'),
    _event('assigned', 'pydanty[bot]'),
    _event('labeled', 'pydanty[bot]'),
]

CASES: list[Case] = [
    # #12483 linked #12482, which its author filed one second earlier, and stayed open.
    Case(id='issue-author-with-no-maintainer-engagement', closed=True),
    # pydanty triages every new issue, commenting, labeling and assigning a maintainer before
    # any maintainer has read it.
    Case(id='issue-author-triaged-only-by-pydanty', closed=True, issue_timeline=PYDANTY_TRIAGE),
    Case(
        id='issue-author-labeled-by-github-actions',
        closed=True,
        issue_timeline=[_event('labeled', 'github-actions[bot]')],
    ),
    # The REST timeline reports `github-actions[bot]`; GraphQL spells the same account without the suffix.
    Case(
        id='issue-author-commented-on-by-github-actions-without-bot-suffix',
        closed=True,
        issue_timeline=[_event('commented', 'github-actions')],
    ),
    Case(
        id='issue-author-commented-on-by-another-app',
        closed=True,
        issue_timeline=[_event('commented', 'renovate[bot]')],
    ),
    Case(
        id='issue-author-commented-on-by-a-read-only-user',
        closed=True,
        issue_timeline=[_event('commented', 'drive-by')],
    ),
    Case(id='issue-author-labeled-by-a-triager', closed=True, issue_timeline=[_event('labeled', 'triager')]),
    Case(
        id='issue-author-with-a-maintainer-comment',
        closed=False,
        issue_timeline=[*PYDANTY_TRIAGE, _event('commented', 'writer')],
    ),
    Case(id='issue-author-labeled-by-a-maintainer', closed=False, issue_timeline=[_event('labeled', 'lead')]),
    Case(
        id='issue-author-whose-issue-an-admin-assigned-to-someone-else',
        closed=False,
        issue_timeline=[_event('assigned', 'owner')],
        assignees=['owner'],
    ),
    Case(id='unreadable-issue-timeline', closed=False, fails=True, issue_timeline=None),
    Case(
        id='unreadable-role-of-an-issue-commenter',
        closed=False,
        fails=True,
        issue_timeline=[_event('commented', 'deleted-account')],
    ),
    Case(
        id='contributor-assigned-to-someone-elses-issue', closed=False, issue_author='reporter', assignees=[CONTRIBUTOR]
    ),
    # Maintainer engagement exempts only the issue's author.
    Case(
        id='contributor-on-someone-elses-unassigned-issue',
        closed=True,
        issue_author='reporter',
        issue_timeline=[_event('commented', 'writer')],
    ),
]


def _run_guard(case: Case, tmp_path: Path) -> tuple[subprocess.CompletedProcess[str], list[dict[str, str]]]:
    """Execute the guard's `run:` block for one case; return the result and the PR actions it took."""
    script = yaml.safe_load(WORKFLOW.read_text(encoding='utf-8'))['jobs']['guard']['steps'][0]['run']
    assert isinstance(script, str)
    script_path = tmp_path / 'guard.sh'
    script_path.write_text(script, encoding='utf-8')

    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    gh = bin_dir / 'gh'
    gh.write_text(FAKE_GH, encoding='utf-8')
    gh.chmod(0o755)
    if GDATE is not None:
        (bin_dir / 'date').symlink_to(GDATE)

    state = {
        'repo': REPO,
        'pr': PR_NUMBER,
        'issue': {
            'number': ISSUE_NUMBER,
            'state': 'open',
            'user': {'login': case.issue_author},
            'assignees': [{'login': login} for login in case.assignees],
            **({'pull_request': {}} if case.issue_is_pull_request else {}),
        },
        'issue_timeline': case.issue_timeline,
        'roles': ROLES,
        'files': [{'filename': 'pydantic_ai_slim/pydantic_ai/agent.py'}],
    }
    actions_path = tmp_path / 'actions.jsonl'
    result = subprocess.run(
        ['bash', str(script_path)],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            'PATH': f'{bin_dir}{os.pathsep}{os.environ["PATH"]}',
            'GH_TOKEN': 'stub-token',
            'REPO': REPO,
            'DEFAULT_BRANCH': 'main',
            'GITHUB_EVENT_SENDER_LOGIN': CONTRIBUTOR,
            'GITHUB_EVENT_PULL_REQUEST_NUMBER': str(PR_NUMBER),
            'GITHUB_EVENT_PULL_REQUEST_USER_LOGIN': CONTRIBUTOR,
            'GITHUB_EVENT_PULL_REQUEST_AUTHOR_ASSOCIATION': 'CONTRIBUTOR',
            'GITHUB_EVENT_PULL_REQUEST_CREATED_AT': '2026-11-04T04:30:55Z',
            'GITHUB_EVENT_PULL_REQUEST_DRAFT': 'false',
            'GITHUB_EVENT_PULL_REQUEST_STATE': 'open',
            'GITHUB_EVENT_PULL_REQUEST_BASE_REF': 'main',
            'FAKE_GH_STATE': json.dumps(state),
            'FAKE_GH_ACTIONS': str(actions_path),
        },
    )
    actions: list[dict[str, str]] = (
        [json.loads(line) for line in actions_path.read_text(encoding='utf-8').splitlines()]
        if actions_path.exists()
        else []
    )
    return result, actions


@requires_tools
@pytest.mark.parametrize('case', CASES, ids=lambda case: case.id)
def test_guard_rules_on_the_pull_request(case: Case, tmp_path: Path):
    result, actions = _run_guard(case, tmp_path)

    assert result.returncode == (1 if case.fails else 0), f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}'
    assert [action['command'] for action in actions] == (['comment', 'close'] if case.closed else [])


@requires_tools
def test_a_self_filed_issue_gets_the_assignment_explanation(tmp_path: Path):
    _, actions = _run_guard(Case(id='self-filed', closed=True), tmp_path)

    issue_link = f'[issue #{ISSUE_NUMBER}](https://github.com/{REPO}/issues/{ISSUE_NUMBER})'
    assert f'{issue_link} is not assigned to you' in actions[0]['body']


@requires_tools
def test_the_missing_issue_comment_does_not_offer_a_new_issue_as_the_way_back(tmp_path: Path):
    _, actions = _run_guard(Case(id='links-a-pull-request', closed=True, issue_is_pull_request=True), tmp_path)

    assert (
        "Opening an issue isn't enough on its own: a PR that links it is closed the same way until a maintainer"
        ' has commented on the issue, labeled it or assigned it to you.'
    ) in actions[0]['body']
