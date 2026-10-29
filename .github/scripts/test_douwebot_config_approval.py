"""Exercise the douwebot approval and config guard from the workflow YAML."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

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

if args[:1] != ['api']:
    sys.exit('stub gh: unhandled command ' + repr(args))

endpoint = args[1]
with open(os.environ['FAKE_CALLS_OUT'], 'a') as handle:
    handle.write(endpoint + '\\n')

if '/collaborators/' in endpoint:
    if os.environ.get('FAKE_PERMISSION_ERROR') == '1':
        sys.exit('stub gh: permission API error')
    payload = json.dumps({'permission': os.environ['FAKE_PERMISSION']})
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
    assert trigger['types'] == ['opened', 'synchronize', 'labeled', 'unlabeled']

    jobs = workflow['jobs']
    review = jobs['review']
    assert review['if'] == (
        "github.event.action == 'labeled' && (github.event.label.name == 'douwebot' "
        "|| github.event.label.name == 'douwebot:review-agent-config')"
    )
    assert jobs['size-label']['if'] == "github.event.action == 'opened' || github.event.action == 'synchronize'"
    assert jobs['category-classify']['if'] == "github.event.action == 'opened' || github.event.action == 'synchronize'"
    assert workflow['concurrency']['group'] == (
        "pr-bots-${{ github.event.pull_request.number }}${{ ((github.event.action == 'labeled' "
        "&& github.event.label.name != 'douwebot' && github.event.label.name != 'douwebot:review-agent-config') "
        "|| (github.event.action == 'unlabeled' && github.event.label.name != 'douwebot:review-agent-config')) "
        "&& format('-noop-{0}', github.run_id) || '' }}"
    )
    assert "github.event.action == 'unlabeled'" in workflow['concurrency']['cancel-in-progress']
    assert APPROVAL_LABEL in workflow['concurrency']['cancel-in-progress']

    steps = review['steps']
    cleanup = next(step for step in steps if step.get('name') == 'Remove douwebot label')
    assert cleanup['if'] == "always() && github.event.action == 'labeled' && github.event.label.name == 'douwebot'"
    assert all(APPROVAL_LABEL not in step.get('run', '') for step in steps if step is cleanup)

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
