"""Regression tests for the `.github/` protection guard.

`.github/workflows/protect-github-dir.yml` decides whether a pull request may change
`.github/`. It triggers on `pull_request_target`, and GitHub runs the **base branch's**
copy of such a workflow, so the guard can never be exercised by the PR that edits it —
these tests are the only place its logic is checked before it reaches `main`. That
matters more than for an ordinary workflow: a bug in one direction blocks every
contributor, and in the other it lets fork-authored changes into a directory that
executes with repository credentials.

The tests extract the guard's `run:` block straight from the YAML and execute it against
a stubbed `gh`, so they track the file that Actions actually runs rather than a copy.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

REPO = 'pydantic/pydantic-ai'
PR_NUMBER = '1234'
MARKER = '<!-- protect-github-dir-guard -->'
WORKFLOW = Path(__file__).parent.parent / 'workflows' / 'protect-github-dir.yml'

# The stub answers the three endpoints the guard reads and records the comment it posts.
# It pipes the canned payload through the real `jq` so the guard's own `--jq` expressions
# are under test, not just the shell around them.
FAKE_GH = '''#!/usr/bin/env python3
import os
import subprocess
import sys

args = sys.argv[1:]


def jq(payload):
    expr = args[args.index('--jq') + 1]
    result = subprocess.run(['jq', '-r', expr], input=payload, capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    sys.exit(result.returncode)


if args[:1] == ['api']:
    endpoint = args[1]
    if endpoint.endswith('/files'):
        jq(os.environ['FAKE_FILES_JSON'])
    elif endpoint.endswith('/comments'):
        jq(os.environ['FAKE_COMMENTS_JSON'])
    elif endpoint.endswith('/permission'):
        role = os.environ['FAKE_ROLE_JSON']
        if role == 'FAIL':
            sys.stderr.write('gh: HTTP 403: Must have push access\\n')
            sys.exit(1)
        jq(role)
    else:
        sys.exit('stub gh: unhandled endpoint ' + endpoint)
elif args[:2] == ['pr', 'comment']:
    with open(os.environ['FAKE_COMMENT_OUT'], 'w') as handle:
        handle.write(args[args.index('--body') + 1])
else:
    sys.exit('stub gh: unhandled command ' + repr(args))
'''


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding='utf-8'))


@dataclass
class Case:
    """One PR the guard has to rule on.

    Defaults describe the case the guard exists for: an outside contributor, on a fork,
    with no elevated role. Each case overrides only what makes it different.
    """

    id: str
    files: list[dict[str, str]]
    blocked: bool
    comments: bool
    author: str = 'outside-contributor'
    association: str = 'CONTRIBUTOR'
    head_repo: str = 'outside-contributor/pydantic-ai'
    role: str = 'read'
    existing_comments: list[dict[str, object]] = field(default_factory=list)


UNRELATED = [{'filename': 'docs/agents.md'}, {'filename': 'pydantic_ai_slim/pydantic_ai/agent.py'}]
PROTECTED = [{'filename': 'docs/agents.md'}, {'filename': '.github/workflows/ci.yml'}]
RENAMED_OUT = [{'filename': 'tools/ci.yml', 'previous_filename': '.github/workflows/ci.yml'}]

CASES: list[Case] = [
    Case(id='external-pr-touching-nothing-protected', files=UNRELATED, blocked=False, comments=False),
    Case(id='external-pr-touching-dot-github', files=PROTECTED, blocked=True, comments=True),
    # A rename *out of* `.github/` changes the directory as surely as an edit in place,
    # and the file only appears under its new path unless `previous_filename` is read.
    Case(id='external-pr-renaming-a-file-out-of-dot-github', files=RENAMED_OUT, blocked=True, comments=True),
    # Dependabot owns the action-pin bumps in `.github/workflows/`; blocking it would
    # freeze them. It is the only allowlisted bot.
    Case(id='dependabot-bumping-an-action-pin', files=PROTECTED, blocked=False, comments=False, author='dependabot[bot]'),
    # A branch in the base repository can only exist if its author has push access.
    Case(id='maintainer-on-a-same-repo-branch', files=PROTECTED, blocked=False, comments=False, head_repo=REPO),
    # `author_association` can report CONTRIBUTOR for a genuine collaborator (#6359), so
    # the repo-role lookup is what keeps a maintainer's fork PR from being blocked.
    Case(id='misreported-collaborator-with-a-write-role', files=PROTECTED, blocked=False, comments=False, role='maintain'),
    # Fails closed: unlike `pr-guard.yml`'s courtesy gate, an unreadable role blocks.
    Case(id='unreadable-repo-role', files=PROTECTED, blocked=True, comments=True, role='FAIL'),
    Case(id='deleted-fork-with-no-head-repo', files=PROTECTED, blocked=True, comments=True, head_repo=''),
    # `synchronize` re-runs the guard on every push; the explanation is posted once.
    Case(
        id='already-explained-on-an-earlier-push',
        files=PROTECTED,
        blocked=True,
        comments=False,
        existing_comments=[{'id': 9, 'body': f'Thanks for the PR! {MARKER}'}],
    ),
] + [
    Case(id=f'{association.lower()}-editing-dot-github', files=PROTECTED, blocked=False, comments=False, association=association)
    for association in ('OWNER', 'MEMBER', 'COLLABORATOR')
]


@pytest.mark.skipif(shutil.which('jq') is None, reason='the gh stub filters payloads with jq')
@pytest.mark.parametrize('case', CASES, ids=lambda case: case.id)
def test_guard_rules_on_the_pull_request(case: Case, tmp_path: Path):
    script = tmp_path / 'guard.sh'
    script.write_text(_workflow()['jobs']['guard']['steps'][0]['run'], encoding='utf-8')

    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    gh = bin_dir / 'gh'
    gh.write_text(FAKE_GH, encoding='utf-8')
    gh.chmod(0o755)

    comment = tmp_path / 'comment.md'
    role = case.role if case.role == 'FAIL' else json.dumps({'role_name': case.role})
    result = subprocess.run(
        ['bash', str(script)],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            'PATH': f'{bin_dir}{os.pathsep}{os.environ["PATH"]}',
            'GH_TOKEN': 'stub-token',
            'REPO': REPO,
            'PR_NUMBER': PR_NUMBER,
            'PR_AUTHOR': case.author,
            'AUTHOR_ASSOCIATION': case.association,
            'HEAD_REPO': case.head_repo,
            'FAKE_FILES_JSON': json.dumps(case.files),
            'FAKE_COMMENTS_JSON': json.dumps(case.existing_comments),
            'FAKE_ROLE_JSON': role,
            'FAKE_COMMENT_OUT': str(comment),
        },
    )

    assert result.returncode == (1 if case.blocked else 0), f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}'
    assert comment.exists() is case.comments


@pytest.mark.skipif(shutil.which('jq') is None, reason='the gh stub filters payloads with jq')
def test_the_comment_names_the_offending_files_and_carries_the_dedup_marker(tmp_path: Path):
    case = Case(id='external-pr-touching-dot-github', files=PROTECTED, blocked=True, comments=True)
    test_guard_rules_on_the_pull_request(case, tmp_path)

    body = (tmp_path / 'comment.md').read_text(encoding='utf-8')
    assert '- `.github/workflows/ci.yml`' in body
    # The unrelated file is the contributor's actual work — naming it would read as if it
    # were part of the problem.
    assert 'docs/agents.md' not in body
    assert MARKER in body


def test_the_guard_never_checks_out_pull_request_code():
    """`pull_request_target` hands the job a write token in base-repository context.

    Checking out or running the PR's own code under that trigger is precisely the
    supply-chain hole this workflow exists to close, so the guard reads PR metadata over
    the API and runs no action at all.
    """
    steps = _workflow()['jobs']['guard']['steps']
    assert [step for step in steps if 'uses' in step] == []

    source = WORKFLOW.read_text(encoding='utf-8')
    assert 'actions/checkout' not in source
    # `head.repo.full_name` is read to tell a fork from a same-repo branch; the head
    # *ref* and *sha* are the handles you'd need to fetch the contributor's code, and
    # nothing here has any business knowing them.
    assert 'head.sha' not in source
    assert 'head.ref' not in source


def test_the_trigger_has_no_paths_filter():
    """A `paths:` filter here would deadlock the repository.

    A `pull_request_target` filtered to `.github/**` does not run at all on the PRs that
    don't match it, and a *required* check that never runs stays pending forever. The job
    runs on every PR and exits 0 when nothing protected changed.
    """
    # YAML 1.1 resolves the bare `on:` key to the boolean True.
    trigger = _workflow()[True]['pull_request_target']
    assert 'paths' not in trigger
    assert 'paths-ignore' not in trigger


def test_the_guard_job_holds_only_the_permission_it_uses():
    """No `contents:` scope — the job never reads the repository's code."""
    workflow = _workflow()
    assert workflow['permissions'] == {}
    assert workflow['jobs']['guard']['permissions'] == {'pull-requests': 'write'}
