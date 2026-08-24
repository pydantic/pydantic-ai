"""Regression tests for the `.github/` protection guard.

`.github/workflows/protect-github-dir.yml` decides whether a pull request may change
`.github/`. It triggers on `pull_request_target`, and GitHub runs such a workflow from the
**default branch of the base repository**, so the guard can never be exercised by the PR
that edits it — these tests are the only place its logic is checked before `main`. That
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
BOT = 'github-actions[bot]'
WORKFLOW = Path(__file__).parent.parent / 'workflows' / 'protect-github-dir.yml'

# The stub answers the four endpoints the guard reads and records the comment it posts. It
# pipes the canned payload through the real `jq` so the guard's own `--jq` expressions are
# under test, not just the shell around them. It ignores `--paginate` and serves every
# payload as one page: the guard's page-boundary behaviour is GitHub's to get right, while
# the truncation case that IS the guard's problem is driven through FAKE_TOTAL_FILES.
FAKE_GH = """#!/usr/bin/env python3
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
    if '/files' in endpoint:
        jq(os.environ['FAKE_FILES_JSON'])
    elif '/comments' in endpoint:
        jq(os.environ['FAKE_COMMENTS_JSON'])
    elif '/permission' in endpoint:
        permission = os.environ['FAKE_PERMISSION']
        if permission == 'FAIL':
            sys.stderr.write('gh: HTTP 403\\n')
            sys.exit(1)
        jq('{"permission": "' + permission + '"}')
    elif '/pulls/' in endpoint:
        jq('{"changed_files": ' + os.environ['FAKE_TOTAL_FILES'] + '}')
    else:
        sys.exit('stub gh: unhandled endpoint ' + endpoint)
elif args[:2] == ['pr', 'comment']:
    with open(os.environ['FAKE_COMMENT_OUT'], 'w') as handle:
        handle.write(args[args.index('--body') + 1])
else:
    sys.exit('stub gh: unhandled command ' + repr(args))
"""


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding='utf-8'))


@dataclass
class Case:
    """One PR the guard has to rule on.

    Defaults describe the case the guard exists for: an outside contributor with no
    elevated permission. Each case overrides only what makes it different.
    """

    id: str
    files: list[dict[str, str]]
    blocked: bool
    comments: bool
    author: str = 'outside-contributor'
    permission: str = 'read'
    state: str = 'open'
    existing_comments: list[dict[str, object]] = field(default_factory=list)
    # Defaults to len(files); set higher to model the PR-files API truncating.
    total_files: int | None = None


UNRELATED = [{'filename': 'docs/agents.md'}, {'filename': 'pydantic_ai_slim/pydantic_ai/agent.py'}]
PROTECTED = [{'filename': 'docs/agents.md'}, {'filename': '.github/workflows/ci.yml'}]
RENAMED_OUT = [{'filename': 'tools/ci.yml', 'previous_filename': '.github/workflows/ci.yml'}]

CASES: list[Case] = [
    Case(id='external-pr-touching-nothing-protected', files=UNRELATED, blocked=False, comments=False),
    Case(id='external-pr-touching-dot-github', files=PROTECTED, blocked=True, comments=True),
    # A rename *out of* `.github/` changes the directory as surely as an edit in place, and
    # the file only appears under its new path unless `previous_filename` is read.
    Case(id='external-pr-renaming-a-file-out-of-dot-github', files=RENAMED_OUT, blocked=True, comments=True),
    # Dependabot owns the action-pin bumps in `.github/workflows/`; blocking it would freeze
    # them. It holds no repo permission, so the allowlist is what lets it through.
    Case(
        id='dependabot-bumping-an-action-pin',
        files=PROTECTED,
        blocked=False,
        comments=False,
        author='dependabot[bot]',
        permission='none',
    ),
    # pydanty builds its branches inside this repo from externally-authored issue text, so a
    # base-repo head branch must not read as trust. It holds no repo permission and is not
    # allowlisted, so it is blocked like any other untrusted author.
    Case(
        id='pydanty-on-a-base-repo-branch',
        files=PROTECTED,
        blocked=True,
        comments=True,
        author='pydanty[bot]',
        permission='none',
    ),
    # `.permission` maps `maintain` and custom repo roles onto a stable base level, so a
    # maintainer clears the check whatever their role is named (#6797).
    Case(id='maintainer-with-write-permission', files=PROTECTED, blocked=False, comments=False, permission='write'),
    Case(id='admin-editing-dot-github', files=PROTECTED, blocked=False, comments=False, permission='admin'),
    # Fails closed: unlike `pr-guard.yml`'s courtesy gate, an unreadable permission blocks.
    Case(id='unreadable-repo-permission', files=PROTECTED, blocked=True, comments=True, permission='FAIL'),
    # The PR-files API caps at 3000 and truncates silently, so a padded PR could push a
    # `.github/` edit out of the response. A short list must never read as "nothing changed".
    Case(
        id='truncated-file-list-hiding-a-dot-github-edit',
        files=UNRELATED,
        blocked=True,
        comments=False,
        total_files=3001,
    ),
    # `synchronize` re-runs the guard on every push; the explanation is posted once.
    Case(
        id='already-explained-on-an-earlier-push',
        files=PROTECTED,
        blocked=True,
        comments=False,
        existing_comments=[{'id': 9, 'user': {'login': BOT}, 'body': f'Thanks for the PR! {MARKER}'}],
    ),
    # ...but anyone can paste the marker into a comment, and that must not suppress the
    # explanation for good.
    Case(
        id='marker-forged-by-another-commenter',
        files=PROTECTED,
        blocked=True,
        comments=True,
        existing_comments=[{'id': 9, 'user': {'login': 'outside-contributor'}, 'body': f'see {MARKER}'}],
    ),
    # `edited` fires on closed and merged PRs too, where there is no merge left to guard.
    Case(id='closed-pr-being-edited', files=PROTECTED, blocked=False, comments=False, state='closed'),
]


def _run_guard(case: Case, tmp_path: Path) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Execute the guard's `run:` block for one case; return the result and the comment path."""
    script = tmp_path / 'guard.sh'
    script.write_text(_workflow()['jobs']['guard']['steps'][0]['run'], encoding='utf-8')

    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    gh = bin_dir / 'gh'
    gh.write_text(FAKE_GH, encoding='utf-8')
    gh.chmod(0o755)

    comment = tmp_path / 'comment.md'
    total_files = len(case.files) if case.total_files is None else case.total_files
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
            'PR_STATE': case.state,
            'FAKE_FILES_JSON': json.dumps(case.files),
            'FAKE_COMMENTS_JSON': json.dumps(case.existing_comments),
            'FAKE_PERMISSION': case.permission,
            'FAKE_TOTAL_FILES': str(total_files),
            'FAKE_COMMENT_OUT': str(comment),
        },
    )
    return result, comment


@pytest.mark.skipif(shutil.which('jq') is None, reason='the gh stub filters payloads with jq')
@pytest.mark.parametrize('case', CASES, ids=lambda case: case.id)
def test_guard_rules_on_the_pull_request(case: Case, tmp_path: Path):
    result, comment = _run_guard(case, tmp_path)

    assert result.returncode == (1 if case.blocked else 0), f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}'
    assert comment.exists() is case.comments


@pytest.mark.skipif(shutil.which('jq') is None, reason='the gh stub filters payloads with jq')
def test_the_comment_names_the_offending_files_and_carries_the_dedup_marker(tmp_path: Path):
    case = Case(id='external-pr-touching-dot-github', files=PROTECTED, blocked=True, comments=True)
    _, comment = _run_guard(case, tmp_path)

    body = comment.read_text(encoding='utf-8')
    assert '- `.github/workflows/ci.yml`' in body
    # The unrelated file is the contributor's actual work — naming it would read as if it
    # were part of the problem.
    assert 'docs/agents.md' not in body
    assert MARKER in body


@pytest.mark.skipif(shutil.which('jq') is None, reason='the gh stub filters payloads with jq')
def test_a_hostile_filename_cannot_reach_the_shell(tmp_path: Path):
    """Every PR-controlled value arrives through `env:` and is quoted at each use.

    A filename is attacker-chosen, so if any of it were interpolated into the script it
    would execute with the write token this trigger hands the job.
    """
    hostile = '.github/workflows/$(touch {})`touch {}`.yml'.format(tmp_path / 'pwned-sub', tmp_path / 'pwned-tick')
    case = Case(id='hostile', files=[{'filename': hostile}], blocked=True, comments=True)
    _, comment = _run_guard(case, tmp_path)

    assert not (tmp_path / 'pwned-sub').exists()
    assert not (tmp_path / 'pwned-tick').exists()
    assert hostile in comment.read_text(encoding='utf-8')


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
    # The head ref and sha are the handles you'd need to fetch the contributor's code, and
    # nothing here has any business knowing them.
    assert 'head.sha' not in source
    assert 'head.ref' not in source


def test_the_guard_does_not_trust_a_base_repo_head_branch():
    """GitHub Apps push branches straight into this repo, so a same-repo head proves nothing.

    `pydanty[bot]` builds those branches out of externally-authored issue text — the very
    input this guard exists to keep out of `.github/` — so reading `head.repo` as trust
    would hand it the bypass. Resolved repo permission is the only signal.
    """
    # Match the payload expressions and the jq selector rather than the bare field names,
    # which also appear in the comments explaining why they aren't used.
    source = WORKFLOW.read_text(encoding='utf-8')
    assert 'github.event.pull_request.head.repo' not in source
    assert 'github.event.pull_request.author_association' not in source
    # `.role_name` can be an arbitrary custom role name, which fails a hardcoded match and
    # blocks a genuine maintainer; `.permission` is the stable base access level (#6797).
    assert "--jq '.role_name'" not in source
    assert "--jq '.permission'" in source


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
    # `edited` is the only event fired when a PR's base branch changes, and the verdict is a
    # diff against that base — without it a green check survives a base switch.
    assert 'edited' in trigger['types']


def test_the_guard_job_holds_only_the_permission_it_uses():
    """No `contents:` scope — the job never reads the repository's code.

    The collaborator-permission endpoint needs only metadata access, which every token
    carries and no `permissions:` key can withhold, so `pull-requests: write` covers every
    call the guard makes.
    """
    workflow = _workflow()
    assert workflow['permissions'] == {}
    assert workflow['jobs']['guard']['permissions'] == {'pull-requests': 'write'}
