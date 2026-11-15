"""Exercise the douwebot approval and config guard from the workflow YAML."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, get_args

import pytest
import yaml
from category_classify_opencode import TOOL_GUIDANCE, Classification
from douwebot_opencode_run import OPENCODE_GO, ZAI, ZAI_BACKUP
from publish_douwebot_review import REVIEW_ENGINES, Finding, ReviewEngine, ReviewResult, publication

WORKFLOW = Path(__file__).parent.parent / 'workflows' / 'bots.yml'
APPROVAL_LABEL = 'douwebot:review-agent-config'
REPOSITORY = 'pydantic/pydantic-ai'
PR_NUMBER = '1234'
EVENT_HEAD = 'head-sha'

FAKE_GH = """#!/usr/bin/env python3
import json
import os
import subprocess
import sys

args = sys.argv[1:]

if args[:2] in (['pr', 'view'], ['pr', 'diff']):
    with open(os.environ['FAKE_CALLS_OUT'], 'a') as handle:
        handle.write(' '.join(args) + '\\n')
    if args[1] == 'view':
        sys.stdout.write(os.environ['FAKE_PR_VIEW'])
    else:
        sys.stdout.write(os.environ['FAKE_PR_FILES' if '--name-only' in args else 'FAKE_PR_DIFF'])
    sys.exit(0)

if args[:1] != ['api']:
    sys.exit('stub gh: unhandled command ' + repr(args))

endpoint = next(arg for arg in args[1:] if not arg.startswith('-'))
with open(os.environ['FAKE_CALLS_OUT'], 'a') as handle:
    handle.write(endpoint + '\\n')

if '/collaborators/' in endpoint:
    if os.environ.get('FAKE_PERMISSION_ERROR') == '1':
        sys.exit('stub gh: permission API error')
    payload = json.dumps({'permission': os.environ['FAKE_PERMISSION']})
elif '/reviews' in endpoint:
    payload = os.environ['FAKE_REVIEWS']
elif '/comments' in endpoint:
    payload = os.environ['FAKE_COMMENTS']
elif '/pulls/' in endpoint:
    if os.environ.get('FAKE_PULL_ERROR') == '1':
        sys.exit('stub gh: pull request API error')
    payload = os.environ['FAKE_PULL_JSON']
elif '/compare/' in endpoint:
    payload = os.environ['FAKE_DIFF']
else:
    sys.exit('stub gh: unhandled endpoint ' + endpoint)

if '--jq' in args:
    result = subprocess.run(
        ['jq', '-r', args[args.index('--jq') + 1]],
        input=payload,
        capture_output=True,
        text=True,
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    sys.exit(result.returncode)

sys.stdout.write(payload)
"""


def _workflow() -> dict[Any, Any]:
    return yaml.safe_load(WORKFLOW.read_text(encoding='utf-8'))


def _step(name: str) -> dict[str, Any]:
    steps = _workflow()['jobs']['review']['steps']
    return next(step for step in steps if step.get('name') == name)


def _run_step(
    step: dict[str, Any], tmp_path: Path, overrides: dict[str, str]
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    script = tmp_path / 'workflow-step.sh'
    source = step['run']
    for expression, value in {'${{ github.repository }}': REPOSITORY}.items():
        source = source.replace(expression, value)
    script.write_text(source, encoding='utf-8')

    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    gh = bin_dir / 'gh'
    gh.write_text(FAKE_GH, encoding='utf-8')
    gh.chmod(0o755)

    calls = tmp_path / 'calls.txt'
    result = subprocess.run(
        ['bash', str(script)],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            'PATH': f'{bin_dir}{os.pathsep}{os.environ["PATH"]}',
            'GH_TOKEN': 'stub-token',
            'FAKE_CALLS_OUT': str(calls),
            'FAKE_PERMISSION': 'write',
            'FAKE_PULL_JSON': json.dumps({'head': {'sha': EVENT_HEAD}, 'labels': [{'name': APPROVAL_LABEL}]}),
            'FAKE_DIFF': 'diff --git a/.claude/settings.json b/.claude/settings.json\n',
            'REPOSITORY': REPOSITORY,
            'PR_NUMBER': PR_NUMBER,
            'PR_AUTHOR': 'contributor',
            'BASE_SHA': 'base-sha',
            'HEAD_SHA': EVENT_HEAD,
            **overrides,
        },
    )
    return result, calls.read_text(encoding='utf-8').splitlines() if calls.exists() else []


def _approval_env(**overrides: str) -> dict[str, str]:
    return {
        'APPROVAL_LABEL': APPROVAL_LABEL,
        'EVENT_ACTION': 'labeled',
        'EVENT_LABEL': APPROVAL_LABEL,
        'EVENT_SENDER_TYPE': 'User',
        'EVENT_SENDER': 'maintainer',
        'EVENT_HEAD_SHA': EVENT_HEAD,
        'RUN_ATTEMPT': '1',
        **overrides,
    }


@pytest.mark.skipif(shutil.which('jq') is None, reason='the gh stub filters payloads with jq')
def test_ordinary_douwebot_keeps_config_denial_and_author_exemption(tmp_path: Path):
    """The ordinary label keeps both protected-path denial and its author exemption."""
    guard = _step('Check for modified config files')
    assert guard['if'] == "github.event.action == 'labeled' && github.event.label.name == 'douwebot'"

    denied, _ = _run_step(guard, tmp_path / 'external', {'FAKE_PERMISSION': 'read'})
    assert denied.returncode == 1
    assert 'agent config the reviewer loads' in denied.stdout

    exempt, _ = _run_step(guard, tmp_path / 'maintainer', {'FAKE_PERMISSION': 'write'})
    assert exempt.returncode == 0
    assert 'skipping agent-config guard' in exempt.stdout


@pytest.mark.skipif(shutil.which('jq') is None, reason='the gh stub filters payloads with jq')
@pytest.mark.parametrize('permission', ['write', 'admin'])
def test_first_attempt_user_with_permission_can_review_current_labeled_head(tmp_path: Path, permission: str):
    """A qualifying user can authorize the one live head on the first attempt."""
    step = _step('Validate one-head maintainer request')
    result, calls = _run_step(step, tmp_path, _approval_env(FAKE_PERMISSION=permission))

    assert result.returncode == 0, f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}'
    assert any('/collaborators/maintainer/permission' in call for call in calls)
    assert any(f'/pulls/{PR_NUMBER}' in call for call in calls)


@pytest.mark.skipif(shutil.which('jq') is None, reason='the gh stub filters payloads with jq')
@pytest.mark.parametrize(
    ('overrides', 'error'),
    [
        ({'EVENT_SENDER_TYPE': 'Bot'}, 'Only a GitHub User'),
        ({'EVENT_SENDER': 'unknown-user', 'FAKE_PERMISSION': 'read'}, 'current write or admin'),
        ({'EVENT_SENDER': 'unknown-user', 'FAKE_PERMISSION': 'none'}, 'current write or admin'),
        ({'FAKE_PERMISSION_ERROR': '1'}, 'Could not verify the request sender permission'),
        ({'EVENT_HEAD_SHA': 'old-head'}, 'head changed'),
        ({'FAKE_PULL_JSON': json.dumps({'head': {'sha': EVENT_HEAD}, 'labels': []})}, 'label is no longer present'),
        ({'RUN_ATTEMPT': '2'}, 'cannot be rerun'),
        ({'EVENT_ACTION': 'synchronize'}, 'original labeled event'),
        ({'EVENT_LABEL': 'douwebot'}, 'original labeled event'),
        ({'FAKE_PULL_ERROR': '1'}, 'Could not verify the live pull request'),
    ],
    ids=[
        'bot',
        'read-user',
        'unknown-user',
        'permission-api-error',
        'stale-head',
        'missing-label',
        'rerun',
        'synchronize',
        'ordinary-label',
        'pull-api-error',
    ],
)
def test_invalid_approval_events_fail_closed(tmp_path: Path, overrides: dict[str, str], error: str):
    """Reject untrusted, stale, replayed, removed-label, and errored requests."""
    step = _step('Validate one-head maintainer request')
    result, _ = _run_step(step, tmp_path, _approval_env(**overrides))

    assert result.returncode == 1
    assert error in result.stdout


def test_workflow_routes_events_and_cancellation_without_relying_on_label_presence():
    """Review, cancellation, cleanup, and label jobs follow their intended events."""
    workflow = _workflow()
    trigger = workflow[True]['pull_request_target']
    assert trigger['types'] == ['opened', 'synchronize', 'edited', 'labeled', 'unlabeled']

    jobs = workflow['jobs']
    review = jobs['review']
    review_requested = (
        "github.event.action == 'labeled' && (github.event.label.name == 'douwebot' "
        "|| github.event.label.name == 'douwebot:review-agent-config')"
    )
    assert jobs['prior-findings']['if'] == review_requested
    assert review['needs'] == 'prior-findings'
    assert review['if'] == f"!cancelled() && needs.prior-findings.outputs.review_url == '' && {review_requested}"
    assert jobs['size-label']['if'] == (
        "github.event.action == 'opened' || github.event.action == 'synchronize' || github.event.action == 'edited'"
    )
    assert jobs['category-classify']['if'] == "github.event.action == 'opened' || github.event.action == 'synchronize'"
    assert workflow['concurrency']['group'] == (
        "pr-bots-${{ github.event.pull_request.number }}${{ github.event.action == 'edited' && '-size' "
        "|| ((github.event.action == 'labeled' "
        "&& github.event.label.name != 'douwebot' && github.event.label.name != 'douwebot:review-agent-config') "
        "|| (github.event.action == 'unlabeled' && github.event.label.name != 'douwebot:review-agent-config')) "
        "&& format('-noop-{0}', github.run_id) || '' }}"
    )
    assert "github.event.action == 'unlabeled'" in workflow['concurrency']['cancel-in-progress']
    assert APPROVAL_LABEL in workflow['concurrency']['cancel-in-progress']

    steps = review['steps']
    publish_review = jobs['publish-review']
    assert review['permissions']['pull-requests'] == 'read'
    assert publish_review['needs'] == ['prior-findings', 'review', 'douwebot-opencode']
    assert publish_review['if'] == (
        "always() && (needs.review.result != 'skipped' || needs.prior-findings.outputs.review_url != '')"
    )
    skipped = next(step for step in publish_review['steps'] if step.get('name') == 'Report skipped review')
    assert skipped['if'] == "needs.prior-findings.outputs.review_url != ''"
    assert publish_review['permissions']['pull-requests'] == 'write'
    cleanup = next(step for step in publish_review['steps'] if step.get('name') == 'Remove douwebot label')
    assert cleanup['if'] == "always() && github.event.action == 'labeled' && github.event.label.name == 'douwebot'"
    assert APPROVAL_LABEL not in cleanup['run']

    claude_index = next(
        index for index, step in enumerate(steps) if step.get('uses', '').startswith('anthropics/claude-code-action@')
    )
    approval_index = next(
        index for index, step in enumerate(steps) if step.get('name') == 'Validate one-head maintainer request'
    )
    assert approval_index + 1 == claude_index

    source = WORKFLOW.read_text(encoding='utf-8')
    assert re.search(r'labels/douwebot\" --method DELETE', source)
    assert not re.search(r'labels/douwebot:review-agent-config\" --method DELETE', source)


@pytest.mark.skipif(shutil.which('jq') is None, reason='the gh stub filters payloads with jq')
def test_reapplication_is_a_new_first_attempt_request(tmp_path: Path):
    """A fresh label event is eligible, while replaying the old run is denied."""
    step = _step('Validate one-head maintainer request')
    first, _ = _run_step(step, tmp_path / 'first', _approval_env())
    reapplied, _ = _run_step(step, tmp_path / 'reapplied', _approval_env())
    replayed, _ = _run_step(step, tmp_path / 'replayed', _approval_env(RUN_ATTEMPT='2'))

    assert first.returncode == 0
    assert reapplied.returncode == 0
    assert replayed.returncode == 1


def test_workflow_only_publishes_the_successful_review() -> None:
    """A fallback failure cannot publish the primary's missing result as an approval."""
    jobs = _workflow()['jobs']
    review = jobs['review']
    fallback = jobs['douwebot-opencode']
    publish = jobs['publish-review']
    detector = next(step for step in review['steps'] if step.get('id') == 'usage-limit')
    assert detector['if'] == "steps.review.outcome == 'failure'"
    assert 'contents/.github/scripts/douwebot_usage_limit.py?ref=${TRUSTED_REF}' in detector['run']
    assert 'pip install' not in detector['run']
    assert 'python3 -I "$RUNNER_TEMP/douwebot_usage_limit.py"' in detector['run']
    assert detector['env']['TRUSTED_REF'] == '${{ github.sha }}'
    assert fallback['needs'] == 'review'
    assert fallback['if'] == "needs.review.result == 'success' && needs.review.outputs.fallback == 'true'"
    assert 'douwebot-opencode' in publish['needs']
    publish_step = next(step for step in publish['steps'] if step.get('id') == 'publish')
    assert "needs.douwebot-opencode.result == 'success'" in publish_step['if']
    assert 'needs.douwebot-opencode.outputs.result' in publish_step['env']['REVIEW_RESULT']
    assert (
        publish_step['env']['REVIEW_ENGINE']
        == "${{ needs.review.outputs.fallback != 'true' && 'claude' || needs.douwebot-opencode.outputs.engine }}"
    )
    assert fallback['outputs']['engine'] == '${{ steps.opencode.outputs.engine }}'
    opencode = next(step for step in fallback['steps'] if step.get('id') == 'opencode')
    assert opencode['run'].endswith('python3 .github/scripts/douwebot_opencode_run.py\n')
    claude = next(step for step in review['steps'] if step.get('id') == 'review')
    assert '--model claude-opus-5-5[1m]' in claude['with']['claude_args']
    assert 'claude-opus-5-5' in REVIEW_ENGINES['claude']
    assert 'DEEPINFRA_API_KEY' not in opencode['env']
    for provider in (OPENCODE_GO, ZAI, ZAI_BACKUP):
        assert opencode['env'][provider.secret_variable] == f'${{{{ secrets.{provider.secret_variable} }}}}'
        assert REVIEW_ENGINES[provider.review_engine] == f'Reviewer: OpenCode ({provider.model})'


def test_category_classify_falls_back_to_opencode_only_on_a_usage_limit() -> None:
    """Only a subscription-limit rejection reruns the classification, and either job's category is applied."""
    jobs = _workflow()['jobs']
    classify = jobs['category-classify']
    fallback = jobs['category-classify-opencode']
    apply = jobs['category-apply']
    steps = {step.get('id'): step for step in classify['steps']}
    review_detector = next(step for step in jobs['review']['steps'] if step.get('id') == 'usage-limit')
    assert steps['classify']['continue-on-error'] is True
    assert steps['usage-limit']['if'] == "steps.classify.outcome == 'failure'"
    assert steps['usage-limit']['run'] == review_detector['run']
    assert steps['usage-limit']['env'] == {
        **review_detector['env'],
        'EXECUTION_FILE': '${{ steps.classify.outputs.execution_file }}',
    }
    assert steps['extract']['if'] == "steps.classify.outcome == 'success'"
    assert classify['outputs']['fallback'] == '${{ steps.usage-limit.outputs.fallback }}'

    assert fallback['needs'] == 'category-classify'
    assert fallback['if'] == (
        "needs.category-classify.result == 'success' && needs.category-classify.outputs.fallback == 'true'"
    )
    assert fallback['permissions'] == {'contents': 'read', 'pull-requests': 'read'}
    assert fallback['outputs']['category'] == '${{ steps.classify.outputs.category }}'
    opencode = next(step for step in fallback['steps'] if step.get('id') == 'classify')
    assert opencode['run'].endswith('python3 .github/scripts/category_classify_opencode.py\n')
    assert opencode['env']['CLASSIFY_PROMPT'] == steps['classify']['with']['prompt']
    assert 'GH_TOKEN' not in opencode['env']
    assert 'DEEPINFRA_API_KEY' not in opencode['env']
    for provider in (OPENCODE_GO, ZAI, ZAI_BACKUP):
        assert opencode['env'][provider.secret_variable] == f'${{{{ secrets.{provider.secret_variable} }}}}'

    assert apply['needs'] == ['category-classify', 'category-classify-opencode']
    assert apply['if'].startswith('!cancelled() && ')
    [apply_step] = apply['steps']
    assert apply_step['env']['CLASSIFIED_CATEGORY'] == (
        '${{ needs.category-classify.outputs.category || needs.category-classify-opencode.outputs.category }}'
    )
    allowed = re.search(r'ALLOWED="([^"]*)"', apply_step['run'])
    assert allowed is not None
    assert tuple(allowed[1].split()) == get_args(Classification.model_fields['category'].annotation)


@pytest.mark.parametrize(
    ('changed', 'fetches_diff'),
    [
        pytest.param('README.md\n', False, id='no-pyproject'),
        pytest.param('README.md\npydantic_ai_slim/pyproject.toml\n', True, id='pyproject'),
    ],
)
def test_category_prefetch_saves_the_files_the_classifier_reads(tmp_path: Path, changed: str, fetches_diff: bool):
    """The prefetch writes each `gh` command's output under the name the OpenCode prompt gives it."""
    step = next(
        step
        for step in _workflow()['jobs']['category-classify-opencode']['steps']
        if step['name'] == 'Prefetch PR context'
    )
    runner_temp = tmp_path / 'runner-temp'
    runner_temp.mkdir()
    diff = 'diff --git a/pydantic_ai_slim/pyproject.toml b/pydantic_ai_slim/pyproject.toml\n'
    result, calls = _run_step(
        step,
        tmp_path / 'step',
        {
            'RUNNER_TEMP': str(runner_temp),
            'REPO': REPOSITORY,
            'FAKE_PR_VIEW': 'title:\tFix a crash\n',
            'FAKE_PR_FILES': changed,
            'FAKE_PR_DIFF': diff,
        },
    )
    assert result.returncode == 0, result.stderr
    expected = {'pr-view.txt': 'title:\tFix a crash\n', 'changed-files.txt': changed}
    if fetches_diff:
        expected['pr.diff'] = diff
    context = runner_temp / 'category-context'
    assert {path.name: path.read_text(encoding='utf-8') for path in context.iterdir()} == expected
    for name in [*expected, 'pr.diff']:
        assert f'`{name}`' in TOOL_GUIDANCE
    assert calls == [
        f'pr view {PR_NUMBER} --repo {REPOSITORY}',
        f'pr diff {PR_NUMBER} --repo {REPOSITORY} --name-only',
        *([f'pr diff {PR_NUMBER} --repo {REPOSITORY}'] if fetches_diff else []),
    ]


REVIEWED_HEAD = 'a' * 40
FINDING = Finding(path='a.py', line=1, side='RIGHT', body='Rename this.')


def _douwebot_review(
    review_id: int,
    state: str,
    result: ReviewResult,
    *,
    head: str = REVIEWED_HEAD,
    inline: bool = True,
    engine: ReviewEngine | None = None,
) -> dict[str, Any]:
    """A review as the trusted publisher posts it, so the skip filter follows the publisher's wording."""
    return {
        'id': review_id,
        'user': {'login': 'github-actions[bot]', 'type': 'Bot'},
        'commit_id': REVIEWED_HEAD,
        'state': state,
        'html_url': f'review-{review_id}',
        'body': json.loads(publication(result, head, 7, inline=inline, engine=engine))['body'],
    }


@pytest.mark.skipif(shutil.which('jq') is None, reason='the prior-findings step filters reviews with jq')
@pytest.mark.parametrize(
    ('review', 'commented', 'expected'),
    [
        pytest.param(
            _douwebot_review(1, 'APPROVED', ReviewResult(verdict='APPROVE', comments=[], discussion='')),
            False,
            '',
            id='clean-approval',
        ),
        pytest.param(
            _douwebot_review(
                10, 'APPROVED', ReviewResult(verdict='APPROVE', comments=[], discussion=''), engine='claude'
            ),
            False,
            '',
            id='clean-claude',
        ),
        pytest.param(
            _douwebot_review(
                11, 'APPROVED', ReviewResult(verdict='APPROVE', comments=[], discussion=''), engine='opencode'
            ),
            False,
            '',
            id='clean-opencode',
        ),
        pytest.param(
            _douwebot_review(
                14, 'APPROVED', ReviewResult(verdict='APPROVE', comments=[], discussion=''), engine='opencode-deepinfra'
            ),
            False,
            '',
            id='clean-opencode-deepinfra',
        ),
        pytest.param(
            _douwebot_review(
                15, 'APPROVED', ReviewResult(verdict='APPROVE', comments=[], discussion=''), engine='opencode-zai'
            ),
            False,
            '',
            id='clean-opencode-zai',
        ),
        pytest.param(
            _douwebot_review(
                12,
                'APPROVED',
                ReviewResult(verdict='APPROVE', comments=[], discussion='Reviewer: Claude Code (claude-opus-5-5)'),
            ),
            False,
            'review-12',
            id='legacy-discussion-is-not-attribution',
        ),
        pytest.param(
            _douwebot_review(
                13,
                'APPROVED',
                ReviewResult(verdict='APPROVE', comments=[], discussion='Reviewer: Claude Code (claude-opus-5-5)'),
                engine='claude',
            ),
            False,
            'review-13',
            id='engine-line-in-discussion-is-still-a-finding',
        ),
        pytest.param(
            _douwebot_review(2, 'APPROVED', ReviewResult(verdict='APPROVE', comments=[FINDING], discussion='')),
            True,
            'review-2',
            id='optional-inline',
        ),
        pytest.param(
            _douwebot_review(3, 'APPROVED', ReviewResult(verdict='APPROVE', comments=[], discussion='Consider X.')),
            False,
            'review-3',
            id='discussion',
        ),
        pytest.param(
            _douwebot_review(
                4, 'CHANGES_REQUESTED', ReviewResult(verdict='REQUEST_CHANGES', comments=[], discussion='')
            ),
            False,
            '',
            id='request-changes-without-findings-on-head',
        ),
        pytest.param(
            _douwebot_review(
                5, 'COMMENTED', ReviewResult(verdict='INCOMPLETE', comments=[FINDING], discussion='Could not read X.')
            ),
            True,
            '',
            id='incomplete',
        ),
        pytest.param(
            _douwebot_review(
                6, 'DISMISSED', ReviewResult(verdict='REQUEST_CHANGES', comments=[FINDING], discussion='')
            ),
            True,
            'review-6',
            id='dismissed',
        ),
        pytest.param(
            _douwebot_review(
                7,
                'CHANGES_REQUESTED',
                ReviewResult(verdict='REQUEST_CHANGES', comments=[FINDING], discussion=''),
                inline=False,
            ),
            False,
            'review-7',
            id='body-findings-after-422',
        ),
        pytest.param(
            _douwebot_review(
                8,
                'CHANGES_REQUESTED',
                ReviewResult(verdict='REQUEST_CHANGES', comments=[FINDING], discussion=''),
                head='b' * 40,
            ),
            True,
            '',
            id='another-head',
        ),
        pytest.param(
            {
                'id': 9,
                'user': {'login': 'maintainer', 'type': 'User'},
                'commit_id': REVIEWED_HEAD,
                'state': 'CHANGES_REQUESTED',
                'html_url': 'review-9',
                'body': 'Please rename this.',
            },
            True,
            '',
            id='human-review',
        ),
    ],
)
def test_prior_findings_skip_only_heads_that_already_have_douwebot_findings(
    tmp_path: Path, review: dict[str, Any], commented: bool, expected: str
):
    """A head with douwebot findings cannot pass the gate, so only those heads skip a re-review."""
    step = _workflow()['jobs']['prior-findings']['steps'][0]
    comments = [{'pull_request_review_id': review['id']}] if commented else []
    output = tmp_path / 'github-output'

    result, _ = _run_step(
        step,
        tmp_path,
        {
            'FAKE_REVIEWS': json.dumps([[review]]),
            'FAKE_COMMENTS': json.dumps([comments]),
            'HEAD_SHA': REVIEWED_HEAD,
            'REPO': REPOSITORY,
            'RUNNER_TEMP': str(tmp_path),
            'GITHUB_OUTPUT': str(output),
        },
    )

    assert result.returncode == 0, result.stderr
    assert output.read_text(encoding='utf-8') == f'review_url={expected}\n'
