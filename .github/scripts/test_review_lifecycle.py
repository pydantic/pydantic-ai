"""Execute review eligibility, publication checks and close-out with a recording gh fake."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import NotRequired
from urllib.parse import quote

import pytest
import yaml
from pydantic import TypeAdapter
from typing_extensions import TypedDict

ROOT = Path(__file__).parents[2]
REPO = 'pydantic/pydantic-ai'
PULL = f'repos/{REPO}/pulls/123'
HEAD = '1' * 40
RUN_URL = f'https://github.com/{REPO}/actions/runs/456'
WORKFLOWS: list[str] = ['pydantic-ai-pr-review', 'pydantic-ai-ui-security-review']


class Step(TypedDict, total=False):
    name: str
    run: str
    env: dict[str, str]
    uses: str


class Job(TypedDict):
    steps: list[Step]


Workflow = TypedDict('Workflow', {'jobs': dict[str, Job], 'safe-outputs': NotRequired[dict[str, object]]})


def workflow_frontmatter(workflow: str) -> Workflow:
    source = (ROOT / '.github/workflows' / f'{workflow}.md').read_text()
    frontmatter = source.removeprefix('---\n').split('\n---\n', 1)[0]
    return TypeAdapter(Workflow).validate_python(yaml.safe_load(frontmatter))


def workflow_script(workflow: str, job: str, step_name: str | None = None) -> str:
    data = workflow_frontmatter('shared/review-publication' if job == 'flag_failed_review' else workflow)
    step = next(step for step in data['jobs'][job]['steps'] if step_name is None or step.get('name') == step_name)
    script = step.get('run')
    assert script is not None
    return script


GH_STUB = r"""
gh() {
  local endpoint='' filter='' payload='' method=GET paginate=false slurp=false previous=''
  for arg in "$@"; do
    case "$arg" in repos/*) endpoint=$arg ;; --paginate) paginate=true ;; --slurp) slurp=true ;; esac
    [ "$previous" != '--jq' ] || filter=$arg
    if [ "$previous" = '--method' ] || [ "$previous" = '-X' ]; then method=$arg; fi
    previous=$arg
  done
  printf '%s %s paginate=%s slurp=%s\n' "$method" "${endpoint:-graphql}" "$paginate" "$slurp" >> "$CALLS"
  [ "${FAIL_ENDPOINT:-none}" != "${endpoint:-graphql}" ] || return 1
  if [ "$2" = graphql ]; then
    payload=$GRAPHQL_PAGES
    if [ "$paginate" != true ] || [ "$slurp" != true ]; then
      payload=$(jq -c '.[0]' <<<"$payload")
    fi
  elif [ "$endpoint" = "$PULL" ]; then
    payload=$PULL_JSON
  elif [ "$endpoint" = "$PULL/reviews" ]; then
    payload=$REVIEW_PAGES
    if [ "$paginate" != true ] || [ "$slurp" != true ]; then
      payload=$(jq -c '.[0]' <<<"$payload")
    fi
  elif [ "$endpoint" = "$PULL/reviews/101" ]; then
    payload=$REVIEW_JSON
  elif [ "$endpoint" = "$PULL/reviews/101/dismissals" ] && [ "$method" = PUT ]; then
    cat > "$DISMISS_BODY"
    payload='{"state":"DISMISSED"}'
  elif [[ "$endpoint" = "repos/$REPO/issues/123/labels"* ]]; then
    payload='{}'
  else
    echo "unexpected gh request: $*" >&2
    return 91
  fi
  if [ -n "$filter" ]; then jq -r "$filter" <<<"$payload"; else printf '%s\n' "$payload"; fi
}
"""


def run_shell(script: str, tmp_path: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
    env: dict[str, str] = {
        **os.environ,
        'REPO': REPO,
        'PULL': PULL,
        'PR': PULL,
        'LABELS': f'repos/{REPO}/issues/123/labels',
        'SHA': HEAD,
        'RUN_URL': RUN_URL,
        'FAILED': 'false',
        'FAILURE_LABEL': 'ci-review-failed',
        'REVIEW_MARKER': f'<!-- gh-aw-workflow-call-id: {REPO}/pydantic-ai-pr-review -->',
        'WORKFLOW_CALL_ID': f'{REPO}/pydantic-ai-pr-review',
        'EVENT_NAME': 'pull_request',
        'LABEL_NAME': 'final-review',
        'EVENT_HEAD_REPO': REPO,
        'EVENT_PR_NUMBER': '123',
        'REVIEW_PAGES': '[[]]',
        'PULL_JSON': json.dumps({'state': 'open', 'head': {'sha': HEAD}}),
        'GITHUB_OUTPUT': str(tmp_path / 'outputs'),
        'GITHUB_STEP_SUMMARY': str(tmp_path / 'summary'),
        'CALLS': str(tmp_path / 'calls'),
        'DISMISS_BODY': str(tmp_path / 'dismiss-body'),
        **overrides,
    }
    return subprocess.run(['bash', '-c', GH_STUB + '\n' + script], text=True, capture_output=True, env=env)


def check(name: str = 'check', conclusion: str = 'SUCCESS', required: bool = True) -> dict[str, object]:
    return {'name': name, 'conclusion': conclusion, 'isRequired': required}


def pr_page(checks: list[dict[str, object]], *, head: str = HEAD) -> dict[str, object]:
    return {
        'data': {
            'repository': {
                'pullRequest': {
                    'state': 'OPEN',
                    'isDraft': False,
                    'reviewDecision': 'REVIEW_REQUIRED',
                    'isCrossRepository': False,
                    'baseRefName': 'main',
                    'headRefName': 'feature',
                    'headRefOid': head,
                    'commits': {'nodes': [{'commit': {'statusCheckRollup': {'contexts': {'nodes': checks}}}}]},
                }
            }
        }
    }


@pytest.mark.parametrize(
    ('checks', 'eligible'),
    [
        ([], False),
        ([check(conclusion='FAILURE')], False),
        ([check(conclusion='NEUTRAL')], False),
        ([check(conclusion='SKIPPED')], False),
        ([check(required=False)], False),
        ([check()], True),
        ([check(), check('optional', 'FAILURE', False)], True),
        ([check(), check('another', 'SKIPPED')], True),
        ([check(), check('another', 'NEUTRAL')], True),
        ([check(), check('another', 'IN_PROGRESS')], False),
    ],
    ids=[
        'missing',
        'failed',
        'neutral',
        'skipped',
        'not-required',
        'passed',
        'optional-failure',
        'other-skipped',
        'other-neutral',
        'other-required',
    ],
)
def test_eligibility_requires_positive_required_check(tmp_path: Path, checks: list[dict[str, object]], eligible: bool):
    result = run_shell(
        workflow_script(WORKFLOWS[0], 'eligibility', 'Decide whether to review'),
        tmp_path,
        GRAPHQL_PAGES=json.dumps([pr_page(checks)]),
    )
    assert result.returncode == 0, result.stderr
    assert f'eligible={str(eligible).lower()}' in (tmp_path / 'outputs').read_text()


@pytest.mark.parametrize('second_head', [HEAD, '2' * 40])
def test_eligibility_checks_later_pages_and_head_consistency(tmp_path: Path, second_head: str):
    pages = [pr_page([check()]), pr_page([check('required-on-page-2', 'FAILURE')], head=second_head)]
    result = run_shell(
        workflow_script(WORKFLOWS[0], 'eligibility', 'Decide whether to review'),
        tmp_path,
        GRAPHQL_PAGES=json.dumps(pages),
    )
    assert result.returncode == 0, result.stderr
    outputs = (tmp_path / 'outputs').read_text()
    assert 'eligible=false' in outputs
    if second_head == HEAD:
        assert 'required-on-page-2' in (tmp_path / 'outputs').read_text()
    else:
        assert 'reason=PR #123 moved while reading required checks' in outputs
    assert 'graphql paginate=true slurp=true' in (tmp_path / 'calls').read_text()


def test_required_check_after_first_hundred_contexts_is_seen(tmp_path: Path):
    pages = [pr_page([check(f'optional-{i}', required=False) for i in range(100)]), pr_page([check()])]
    result = run_shell(
        workflow_script(WORKFLOWS[0], 'eligibility', 'Decide whether to review'),
        tmp_path,
        GRAPHQL_PAGES=json.dumps(pages),
    )
    assert result.returncode == 0, result.stderr
    assert 'eligible=true' in (tmp_path / 'outputs').read_text()


@pytest.mark.parametrize(
    'pages', ['[]', '[{"errors":[{"message":"denied"}]}]', '[{"data":{"repository":{"pullRequest":null}}}]']
)
def test_eligibility_refuses_incomplete_api_data(tmp_path: Path, pages: str):
    result = run_shell(
        workflow_script(WORKFLOWS[0], 'eligibility', 'Decide whether to review'),
        tmp_path,
        GRAPHQL_PAGES=pages,
    )
    assert result.returncode != 0


@pytest.mark.parametrize('workflow', WORKFLOWS)
@pytest.mark.parametrize('reference', [False, True], ids=['html', 'reference'])
@pytest.mark.parametrize(
    'receipt', ['current', 'none', 'old-head', 'old-run', 'missing-run', 'other-workflow', 'human', 'dismissed']
)
def test_publication_requires_this_runs_review(tmp_path: Path, workflow: str, reference: bool, receipt: str):
    # GitHub REST read-back examples for these two marker forms:
    # https://api.github.com/repos/pydantic/pydantic-ai/pulls/11151/reviews/5631437278
    # https://api.github.com/repos/pydantic/pydantic-ai/pulls/10347/reviews/5540079394
    caller_id = f'{REPO}/{workflow}'
    marker = (
        f'[gh-aw-workflow-call-id]: # "{quote(caller_id, safe="")}"'
        if reference
        else f'<!-- gh-aw-workflow-call-id: {caller_id} -->'
    )
    body = f'<!-- gh-aw-agentic-workflow: Test Review, workflow_id: {workflow}, run: {RUN_URL} -->\n\n{marker}'
    review: dict[str, object] = {
        'user': {'type': 'Bot'},
        'commit_id': HEAD,
        'state': 'COMMENTED',
        'body': body,
    }
    if receipt == 'old-head':
        review['commit_id'] = '2' * 40
    elif receipt == 'old-run':
        review['body'] = body.replace(RUN_URL, f'{RUN_URL}0')
    elif receipt == 'missing-run':
        review['body'] = marker
    elif receipt == 'other-workflow':
        review['body'] = body.replace(marker, marker.replace(workflow, 'other'))
    elif receipt == 'human':
        review['user'] = {'type': 'User'}
    elif receipt == 'dismissed':
        review['state'] = 'DISMISSED'
    result = run_shell(
        workflow_script(workflow, 'flag_failed_review'),
        tmp_path,
        WORKFLOW_CALL_ID=caller_id,
        REVIEW_PAGES=json.dumps([[] if receipt == 'none' else [review]]),
    )
    assert result.returncode == (0 if receipt == 'current' else 1), result.stderr
    calls = (tmp_path / 'calls').read_text()
    assert '/labels' in calls
    assert ('DELETE' in calls) == (receipt == 'current')


@pytest.mark.parametrize('workflow', WORKFLOWS)
def test_compiled_publication_job_binds_each_workflow(workflow: str):
    compiled = TypeAdapter(dict[str | bool, object]).validate_python(
        yaml.safe_load((ROOT / '.github/workflows' / f'{workflow}.lock.yml').read_text())
    )
    jobs = TypeAdapter(dict[str, dict[str, object]]).validate_python(compiled['jobs'])
    job = jobs['flag_failed_review']
    steps = TypeAdapter(list[Step]).validate_python(job['steps'])
    step = next(step for step in steps if 'FAILED' in step.get('env', {}))
    env = step.get('env')
    assert env is not None
    is_ci = workflow == 'pydantic-ai-pr-review'
    pr_number = '${{ needs.eligibility.outputs.pr_number }}' if is_ci else '${{ github.event.pull_request.number }}'
    head_sha = '${{ needs.eligibility.outputs.head_sha }}' if is_ci else '${{ github.event.pull_request.head.sha }}'
    assert env['PR'] == 'repos/${{ github.repository }}/pulls/' + pr_number
    assert env['LABELS'] == 'repos/${{ github.repository }}/issues/' + pr_number + '/labels'
    assert env['SHA'] == head_sha
    assert env['WORKFLOW_CALL_ID'] == '${{ github.repository }}/' + workflow
    assert env['FAILURE_LABEL'] == ('ci-review-failed' if is_ci else 'ui-security-review-failed')
    assert env['GH_TOKEN'] == '${{ github.token }}'
    assert env['RUN_URL'] == '${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}'
    expected_needs: list[str] = ['agent', 'eligibility', 'safe_outputs'] if is_ci else ['agent', 'safe_outputs']
    assert job['needs'] == expected_needs
    condition = "(!cancelled()) && needs.agent.result != 'skipped'"
    assert job['if'] == condition + ('' if is_ci else ' && github.event.pull_request.number')
    assert job['permissions'] == {'pull-requests': 'write'}
    assert job['runs-on'] == 'ubuntu-slim'
    assert job['timeout-minutes'] == 5
    assert not any('actions/checkout' in step.get('uses', '') for step in steps)
    script = step.get('run')
    assert script is not None
    assert script.rstrip('\n') == workflow_script(workflow, 'flag_failed_review').rstrip('\n')
    assert 'flag_failed_review' not in workflow_frontmatter(workflow)['jobs']


@pytest.mark.parametrize(
    'live', [{'state': 'closed', 'head': {'sha': HEAD}}, {'state': 'open', 'head': {'sha': '2' * 40}}]
)
def test_publication_preserves_host_verified_skips(tmp_path: Path, live: dict[str, object]):
    result = run_shell(
        workflow_script('shared/review-publication', 'flag_failed_review'),
        tmp_path,
        FAILED='true',
        PULL_JSON=json.dumps(live),
    )
    assert result.returncode == 0, result.stderr
    assert 'Review no longer required' in (tmp_path / 'summary').read_text()
    assert '/labels' not in (tmp_path / 'calls').read_text()


def test_missing_pr_data_is_not_a_legitimate_skip(tmp_path: Path):
    result = run_shell(workflow_script('shared/review-publication', 'flag_failed_review'), tmp_path, PULL_JSON='{}')
    assert result.returncode != 0


def evaluate_expression(expression: str, values: dict[str, str | bool | int]) -> object:
    expression = expression.strip().removeprefix('${{').removesuffix('}}')
    for name in sorted(values, key=len, reverse=True):
        expression = expression.replace(name, repr(values[name]))
    return eval(expression.replace('&&', ' and ').replace('||', ' or ').strip(), {'__builtins__': {}})


@pytest.mark.parametrize(
    ('action', 'label', 'isolated', 'cancels'),
    [
        ('labeled', 'douwebot', False, False),
        ('labeled', 'douwebot:review-agent-config', False, False),
        ('labeled', 'size: M', True, False),
        ('labeled', 'docs', True, False),
        ('unlabeled', 'douwebot', True, False),
        ('unlabeled', 'douwebot:review-agent-config', False, True),
        ('synchronize', '', False, True),
    ],
)
def test_unrelated_label_events_cannot_evict_pending_reviews(action: str, label: str, isolated: bool, cancels: bool):
    workflow = TypeAdapter(dict[str | bool, object]).validate_python(
        yaml.safe_load((ROOT / '.github/workflows/bots.yml').read_text())
    )
    concurrency = TypeAdapter(dict[str, str]).validate_python(workflow['concurrency'])
    values: dict[str, str | bool | int] = {
        'github.event.action': action,
        'github.event.label.name': label,
        "format('-noop-{0}', github.run_id)": '-noop-456',
    }
    suffix = concurrency['group'].split('}}', 1)[1]
    assert evaluate_expression(suffix, values) == ('-noop-456' if isolated else '')
    assert evaluate_expression(concurrency['cancel-in-progress'], values) == cancels


@pytest.mark.parametrize(
    'agent,safe,failed,expected',
    [
        ('success', 'success', '0', False),
        ('failure', 'success', '0', True),
        ('success', 'failure', '0', True),
        ('success', 'success', '1', True),
        ('success', 'skipped', '', True),
    ],
)
def test_failed_stage_cannot_be_reported_as_success(agent: str, safe: str, failed: str, expected: bool):
    data = workflow_frontmatter('shared/review-publication')
    env = data['jobs']['flag_failed_review']['steps'][0].get('env')
    assert env is not None
    expression = env['FAILED']
    assert (
        evaluate_expression(
            expression,
            {
                'needs.agent.result': agent,
                'needs.safe_outputs.result': safe,
                'needs.safe_outputs.outputs.process_safe_outputs_items_failed': failed,
            },
        )
        == expected
    )


def threads_page(
    *,
    resolved: bool = True,
    head: str = HEAD,
    author: str = 'github-actions',
    reply_author: str | None = None,
    comments_complete: bool = True,
) -> dict[str, object]:
    comments: list[dict[str, dict[str, str]]] = [{'author': {'login': author}}]
    if reply_author is not None:
        comments.append({'author': {'login': reply_author}})
    return {
        'data': {
            'repository': {
                'pullRequest': {
                    'state': 'OPEN',
                    'headRefOid': head,
                    'reviewThreads': {
                        'nodes': [
                            {
                                'isResolved': resolved,
                                'comments': {
                                    'totalCount': len(comments) + (0 if comments_complete else 1),
                                    'nodes': comments,
                                },
                            }
                        ]
                    },
                }
            }
        }
    }


@pytest.mark.parametrize(
    'case',
    [
        'addressed',
        'human',
        'approved',
        'unresolved',
        'later-unresolved',
        'human-thread-bot-reply',
        'incomplete-comments',
        'unrelated-human-thread',
        'changed-page',
        'changed-final',
        'api-error',
    ],
)
def test_closeout_dismisses_only_verified_addressed_bot_verdicts(tmp_path: Path, case: str):
    reason = tmp_path / 'reason'
    reason.write_text('> Agent attribution\n\nVerified the fixes and resolved every blocking finding.\n')
    review: dict[str, object] = {
        'id': 101,
        'state': 'CHANGES_REQUESTED',
        'user': {'type': 'Bot', 'login': 'github-actions[bot]'},
    }
    pages = [threads_page()]
    live: dict[str, object] = {'state': 'open', 'head': {'sha': HEAD}}
    if case == 'human':
        review['user'] = {'type': 'User', 'login': 'maintainer'}
    elif case == 'approved':
        review['state'] = 'APPROVED'
    elif case == 'unresolved':
        pages = [threads_page(resolved=False)]
    elif case == 'later-unresolved':
        pages.append(threads_page(resolved=False))
    elif case == 'human-thread-bot-reply':
        pages = [threads_page(resolved=False, author='maintainer', reply_author='github-actions')]
    elif case == 'incomplete-comments':
        pages = [threads_page(resolved=False, author='maintainer', comments_complete=False)]
    elif case == 'unrelated-human-thread':
        pages = [threads_page(resolved=False, author='maintainer')]
    elif case == 'changed-page':
        pages.append(threads_page(head='2' * 40))
    elif case == 'changed-final':
        live['head'] = {'sha': '2' * 40}
    script = 'source "$HELPER" "$REPO" 123 101 "$SHA" "$REASON_FILE" --all-blockers-addressed'
    result = run_shell(
        script,
        tmp_path,
        HELPER=str(ROOT / '.agents/skills/pushing-commits-to-the-repo/dismiss-addressed-bot-review'),
        REASON_FILE=str(reason),
        REVIEW_JSON=json.dumps(review),
        GRAPHQL_PAGES=json.dumps(pages),
        PULL_JSON=json.dumps(live),
        FAIL_ENDPOINT=f'{PULL}/reviews/101' if case == 'api-error' else 'none',
    )
    dismissed = case in {'addressed', 'unrelated-human-thread'}
    assert result.returncode == (0 if dismissed else 1), result.stderr
    assert (tmp_path / 'dismiss-body').exists() == dismissed
    if dismissed:
        body = TypeAdapter(dict[str, str]).validate_json((tmp_path / 'dismiss-body').read_text())
        assert body['event'] == 'DISMISS'
        assert body['message'].startswith('> Agent attribution')
        assert f'Verified at `{HEAD}`' in body['message']


def test_closeout_requires_explicit_full_assessment(tmp_path: Path):
    result = run_shell(
        'source "$HELPER" "$REPO" 123 101 "$SHA" missing-reason',
        tmp_path,
        HELPER=str(ROOT / '.agents/skills/pushing-commits-to-the-repo/dismiss-addressed-bot-review'),
    )
    assert result.returncode == 2
    assert result.stderr.strip()
    assert not (tmp_path / 'calls').exists()


@pytest.mark.parametrize('case', ['repo', 'number', 'review-id', 'head', 'missing-reason', 'blank-reason'])
def test_invalid_closeout_inputs_cannot_dismiss_a_review(tmp_path: Path, case: str):
    reason = tmp_path / 'reason'
    reason.write_text('   ' if case == 'blank-reason' else 'Verified all blocking findings.')
    arguments: dict[str, str] = {
        'CLOSE_REPO': '../other/repo' if case == 'repo' else REPO,
        'CLOSE_PR': '0' if case == 'number' else '123',
        'CLOSE_REVIEW': 'invalid' if case == 'review-id' else '101',
        'CLOSE_HEAD': 'main' if case == 'head' else HEAD,
        'CLOSE_REASON': str(tmp_path / 'missing') if case == 'missing-reason' else str(reason),
    }
    result = run_shell(
        'source "$HELPER" "$CLOSE_REPO" "$CLOSE_PR" "$CLOSE_REVIEW" "$CLOSE_HEAD" "$CLOSE_REASON" --all-blockers-addressed',
        tmp_path,
        HELPER=str(ROOT / '.agents/skills/pushing-commits-to-the-repo/dismiss-addressed-bot-review'),
        **arguments,
    )
    assert result.returncode == 2
    assert result.stderr.strip()
    assert not (tmp_path / 'calls').exists()


@pytest.mark.parametrize('workflow', WORKFLOWS)
def test_review_receipt_metadata_is_retained(workflow: str):
    safe_outputs = workflow_frontmatter(workflow).get('safe-outputs')
    assert safe_outputs is not None
    review = TypeAdapter(dict[str, object]).validate_python(safe_outputs['submit-pull-request-review'])
    assert review['footer'] == 'always'
    if workflow == 'pydantic-ai-ui-security-review':
        assert review['allowed-events'] == ['COMMENT']
