"""Tests for the pinned Test Quality Review controller."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from pydantic import ValidationError
from review_test_quality import (
    Candidate,
    GitHub,
    PinnedReview,
    ReviewContext,
    _ci_jobs,
    _matches_pinned_pr,
    build_candidates,
    check_payload,
    has_completed_report,
    inconclusive_report,
    summarize,
    validate_report,
)


def _entry(path: str, outcome: str = 'preserved_or_strengthened') -> dict[str, str]:
    return {
        'path': path,
        'guarantee': 'rejects an observable failure',
        'outcome': outcome,
        'evidence': 'pinned assertion and active CI selection',
        'action': 'No change',
    }


def _context(candidates: list[Candidate]) -> ReviewContext:
    return ReviewContext(
        version=1,
        complete=True,
        reason='candidate inventory and pinned context are complete',
        repository='pydantic/pydantic-ai',
        pr_number=1,
        base_sha='base-sha',
        head_sha='pinned-head-sha',
        merge_base_sha='merge-base-sha',
        workflow_version='workflow-sha',
        ci_run_url='https://github.com/pydantic/pydantic-ai/actions/runs/10',
        review_run_url='https://github.com/pydantic/pydantic-ai/actions/runs/11',
        candidates=candidates,
    )


def test_removed_and_renamed_test_paths_remain_candidates() -> None:
    files: list[Mapping[str, object]] = [
        {'filename': 'src/legacy.py', 'previous_filename': 'tests/test_legacy.py', 'status': 'renamed'},
        {'filename': 'tests/test_removed.py', 'status': 'removed'},
    ]

    candidates, complete, _ = build_candidates(files)

    assert complete
    assert [item.relevant_paths for item in candidates] == [
        ['tests/test_legacy.py'],
        ['tests/test_removed.py'],
    ]


def test_new_test_is_candidate_and_unrelated_source_is_not() -> None:
    candidate, complete, _ = build_candidates([{'filename': 'pkg/test_new.py', 'status': 'added'}])
    unrelated, unrelated_complete, _ = build_candidates([{'filename': 'pkg/runtime.py', 'status': 'modified'}])

    assert complete and [item.path for item in candidate] == ['pkg/test_new.py']
    assert unrelated_complete and unrelated == []


def test_current_ci_test_workflows_are_candidates() -> None:
    workflow_paths = [
        '.github/workflows/benchmark.yml',
        '.github/workflows/latest-versions-canary.yml',
        '.github/workflows/sandbox-live.yml',
        '.github/workflows/gateway-model-health.yml',
    ]
    repo_root = Path(__file__).parents[2]

    for path in workflow_paths:
        source = (repo_root / path).read_text(encoding='utf-8')
        candidates, complete, reason = build_candidates(
            [{'filename': path, 'status': 'modified'}], workflow_contents={path: (source, source)}
        )

        assert complete, reason
        assert [candidate.path for candidate in candidates] == [path]


@pytest.mark.parametrize('status', ['modified', 'renamed'])
def test_sandbox_nightly_local_caller_removal_or_rename_is_a_candidate(status: str) -> None:
    old_path = '.github/workflows/sandbox-live-nightly.yml'
    new_path = '.github/workflows/sandbox-live-nightly-renamed.yml'
    call = 'uses: ./.github/workflows/sandbox-live.yml'
    old_source = (Path(__file__).parents[2] / old_path).read_text(encoding='utf-8')
    assert call in old_source
    new_source = 'name: Sandbox live nightly\non: workflow_dispatch\njobs: {}\n'

    if status == 'modified':
        filename = old_path
        files: list[Mapping[str, object]] = [{'filename': filename, 'status': status}]
        workflow_contents = {filename: (old_source, new_source)}
    else:
        filename = new_path
        files = [{'filename': filename, 'previous_filename': old_path, 'status': status}]
        workflow_contents = {old_path: (old_source, None), new_path: (None, new_source)}

    candidates, complete, reason = build_candidates(files, workflow_contents=workflow_contents)

    assert complete, reason
    assert [candidate.path for candidate in candidates] == [filename]
    assert candidates[0].relevant_paths == [old_path]


@pytest.mark.parametrize(
    'command',
    [
        'pytest tests/test_example.py',
        'tox -e py313',
        'nox -s tests',
        'unittest',
        'python -m pytest',
        'python3.13 -m unittest',
        'coverage run -m pytest',
        'uv run coverage run -m pytest',
        'uv run --no-sync coverage run -m pytest',
        'uv run --with=pytest pytest',
        'uv run --with pytest pytest tests/test_example.py',
        'uv run --directory=tests pytest',
        'make test',
        'make testcov',
        'make test-integration',
        'make integration-ci',
    ],
)
def test_known_workflow_test_commands_are_candidates(command: str) -> None:
    path = '.github/workflows/coverage-tests.yml'
    source = f'jobs:\n  tests:\n    steps:\n      - run: {command}\n'

    candidates, complete, reason = build_candidates(
        [{'filename': path, 'status': 'modified'}], workflow_contents={path: (source, source)}
    )

    assert complete, reason
    assert [candidate.path for candidate in candidates] == [path]


def test_long_workflow_command_options_are_recognized_and_rejected_deterministically() -> None:
    path = '.github/workflows/long-tests.yml'
    long_selector = '--with=! ' * 1000
    long_valid = '--with=test-dependency ' * 1000

    candidate, complete, reason = build_candidates(
        [{'filename': path, 'status': 'modified'}],
        workflow_contents={
            path: (
                f'jobs:\n  tests:\n    steps:\n      - run: uv run {long_selector}not-a-test-command\n',
                f'jobs:\n  tests:\n    steps:\n      - run: uv run {long_valid}pytest\n',
            )
        },
    )

    assert complete, reason
    assert [item.path for item in candidate] == [path]
    assert candidate[0].relevant_paths == [path]


def test_irrelevant_long_workflow_command_is_not_a_candidate() -> None:
    path = '.github/workflows/long-tests.yml'
    source = f'jobs:\n  docs:\n    steps:\n      - run: uv run {"--with=! " * 1000}not-a-test-command\n'

    candidates, complete, reason = build_candidates(
        [{'filename': path, 'status': 'modified'}], workflow_contents={path: (source, source)}
    )

    assert complete, reason
    assert candidates == []


def test_uv_option_without_value_does_not_raise_or_select_a_test() -> None:
    path = '.github/workflows/malformed-test-command.yml'
    source = 'jobs:\n  docs:\n    steps:\n      - run: uv run --with\n'

    candidates, complete, reason = build_candidates(
        [{'filename': path, 'status': 'modified'}], workflow_contents={path: (source, source)}
    )

    assert complete, reason
    assert candidates == []


def test_changed_workflow_commands_are_detected_in_both_revisions() -> None:
    old = 'jobs:\n  test:\n    steps:\n      - run: uv run pytest tests/old.py\n'
    new = 'jobs:\n  test:\n    steps:\n      - run: |\n          uv run --no-sync \\\n            python -m unittest\n'
    files: list[Mapping[str, object]] = [
        {
            'filename': '.github/workflows/new-tests.yml',
            'previous_filename': '.github/workflows/old-tests.yaml',
            'status': 'renamed',
        },
        {'filename': '.github/workflows/removed-tests.yml', 'status': 'removed'},
    ]

    candidates, complete, reason = build_candidates(
        files,
        workflow_contents={
            '.github/workflows/old-tests.yaml': (old, None),
            '.github/workflows/new-tests.yml': (None, new),
            '.github/workflows/removed-tests.yml': (old, None),
        },
    )

    assert complete, reason
    assert [candidate.relevant_paths for candidate in candidates] == [
        ['.github/workflows/old-tests.yaml', '.github/workflows/new-tests.yml'],
        ['.github/workflows/removed-tests.yml'],
    ]


def test_workflow_comments_prompts_and_non_test_commands_are_not_candidates() -> None:
    source = """# run: pytest tests/commented.py
# uses: ./.github/workflows/sandbox-live.yml
name: Documentation only
prompt: |
  run: pytest tests/prompt.py
  uses: ./.github/workflows/sandbox-live.yml
  pytest tests/prompt.py
jobs:
  docs:
    steps:
      - run: echo "pytest is mentioned, but not invoked"
"""
    candidates, complete, reason = build_candidates(
        [{'filename': '.github/workflows/docs.yml', 'status': 'modified'}],
        workflow_contents={'.github/workflows/docs.yml': (source, source)},
    )

    assert complete, reason
    assert candidates == []


def test_changed_workflow_without_pinned_content_fails_closed() -> None:
    candidates, complete, reason = build_candidates([{'filename': '.github/workflows/tests.yml', 'status': 'modified'}])

    assert candidates == []
    assert not complete
    assert reason


def test_pyproject_candidate_requires_pytest_or_coverage_configuration_change() -> None:
    before = '[project]\nversion = "1"\n\n[tool.pytest.ini_options]\naddopts = "-q"\n'
    unrelated_after = '[project]\nversion = "2"\n\n[tool.pytest.ini_options]\naddopts = "-q"\n'
    relevant_after = '[project]\nversion = "1"\n\n[tool.pytest.ini_options]\naddopts = "-q -ra"\n'

    unrelated, unrelated_complete, _ = build_candidates(
        [{'filename': 'pyproject.toml', 'status': 'modified'}],
        pyproject_before=before,
        pyproject_after=unrelated_after,
    )
    relevant, relevant_complete, _ = build_candidates(
        [{'filename': 'pyproject.toml', 'status': 'modified'}],
        pyproject_before=before,
        pyproject_after=relevant_after,
    )

    assert unrelated_complete and unrelated == []
    assert relevant_complete and [item.path for item in relevant] == ['pyproject.toml']


@pytest.mark.parametrize(
    'files,before,after',
    [
        ([{'filename': 'tests/test_a.py'}], None, None),
        ([{'filename': 'pyproject.toml', 'status': 'modified'}], '[broken', '[broken'),
        ([{'filename': 'tests/test_a.py', 'status': 3}], None, None),
    ],
)
def test_incomplete_candidate_metadata_fails_closed(
    files: list[Mapping[str, object]], before: str | None, after: str | None
) -> None:
    candidates, complete, reason = build_candidates(files, pyproject_before=before, pyproject_after=after)

    assert candidates == []
    assert not complete
    assert reason


def test_report_must_cover_every_candidate_and_no_others() -> None:
    with pytest.raises(ValueError, match='omits candidate'):
        validate_report(json.dumps({'entries': [_entry('tests/test_a.py')]}), ['tests/test_a.py', 'tests/test_b.py'])
    with pytest.raises(ValueError, match='unknown candidate'):
        validate_report(json.dumps({'entries': [_entry('tests/test_other.py')]}), ['tests/test_a.py'])
    with pytest.raises((ValidationError, ValueError)):
        validate_report('{broken', ['tests/test_a.py'])
    with pytest.raises(ValueError, match='missing'):
        validate_report(None, ['tests/test_a.py'])

    candidate = Candidate(path='tests/test_a.py', status='modified', relevant_paths=['tests/test_a.py'])
    title, summary = inconclusive_report(_context([candidate]), 'report is missing')
    assert title == 'Inconclusive'
    assert 'tests/test_a.py' in summary
    assert 'report is missing' in summary


def test_outcomes_keep_a_mixed_report_actionable_and_deduplicate_same_identity() -> None:
    report = validate_report(
        json.dumps({'entries': [_entry('tests/test_a.py'), _entry('tests/test_b.py', 'changes_needed')]}),
        ['tests/test_a.py', 'tests/test_b.py'],
    )
    title, _ = summarize(report)
    check = check_payload(
        head_sha='pinned-head-sha',
        title=title,
        summary='summary',
        details_url='https://github.com/pydantic/pydantic-ai/actions/runs/11',
        base_sha='base-sha',
        workflow_version='workflow-sha',
    )

    assert title == 'Changes needed'
    assert check['head_sha'] == 'pinned-head-sha'
    assert has_completed_report(
        [check], base_sha='base-sha', head_sha='pinned-head-sha', workflow_version='workflow-sha'
    )
    assert not has_completed_report(
        [check], base_sha='base-sha', head_sha='newer-head', workflow_version='workflow-sha'
    )


@pytest.mark.parametrize(
    'title,conclusion,status,expected',
    [
        ('Protection accounted for', 'neutral', 'completed', True),
        ('Changes needed', 'neutral', 'completed', True),
        ('Inconclusive', 'neutral', 'completed', False),
        ('Skipped', 'neutral', 'completed', False),
        ('No candidate changes', 'neutral', 'completed', True),
        ('Unknown title', 'neutral', 'completed', False),
        ('Protection accounted for', 'failure', 'completed', False),
        ('Changes needed', 'neutral', 'in_progress', False),
    ],
)
def test_deduplication_only_accepts_completed_report_or_discovery_results(
    title: str, conclusion: str, status: str, expected: bool
) -> None:
    payload = check_payload(
        head_sha='pinned-head-sha',
        title=title,
        summary='result',
        details_url='https://github.com/pydantic/pydantic-ai/actions/runs/11',
        base_sha='base-sha',
        workflow_version='workflow-sha',
    )
    payload['conclusion'] = conclusion
    payload['status'] = status

    assert (
        has_completed_report(
            [payload], base_sha='base-sha', head_sha='pinned-head-sha', workflow_version='workflow-sha'
        )
        is expected
    )


def test_check_summary_byte_boundary_and_overflow_are_explicitly_inconclusive() -> None:
    head_sha = 'pinned-head-sha'
    details_url = 'https://github.com/pydantic/pydantic-ai/actions/runs/11'
    marker = f'<!-- test-quality-review:v1 base=base-sha head={head_sha} workflow=workflow-sha -->'
    overhead = len(f'{marker}\n\n'.encode())
    budget = 65_535 - overhead
    boundary_summary = 'é' * (budget // 2) + ('x' if budget % 2 else '')

    boundary_payload = check_payload(
        head_sha=head_sha,
        title='Protection accounted for',
        summary=boundary_summary,
        details_url=details_url,
        base_sha='base-sha',
        workflow_version='workflow-sha',
    )
    rendered_boundary = boundary_payload['output']['summary']
    assert len(rendered_boundary.encode('utf-8')) == 65_535
    assert boundary_payload['output']['title'] == 'Protection accounted for'

    overflow_payload = check_payload(
        head_sha=head_sha,
        title='Protection accounted for',
        summary=f'{boundary_summary}💡',
        details_url=details_url,
        base_sha='base-sha',
        workflow_version='workflow-sha',
    )
    overflow_summary = overflow_payload['output']['summary']
    assert overflow_payload['output']['title'] == 'Inconclusive'
    assert len(overflow_summary.encode('utf-8')) <= 65_535
    assert marker in overflow_summary
    assert '[review run and its artifact]' in overflow_summary
    assert details_url in overflow_summary
    assert overflow_payload['head_sha'] == head_sha


def test_pinned_pull_request_recheck_rejects_head_or_base_changes() -> None:
    pinned = PinnedReview(
        pr_number=1,
        pr={},
        head_sha='head-sha',
        base_sha='base-sha',
        base_ref='main',
        run_id=10,
        ci_run={},
        ci_run_url='https://github.com/pydantic/pydantic-ai/actions/runs/10',
        review_run_url='https://github.com/pydantic/pydantic-ai/actions/runs/11',
        workflow_version='workflow-sha',
    )
    current: dict[str, object] = {
        'state': 'open',
        'draft': False,
        'head': {'sha': 'head-sha', 'repo': {'full_name': 'pydantic/pydantic-ai'}},
        'base': {'sha': 'base-sha', 'ref': 'main', 'repo': {'full_name': 'pydantic/pydantic-ai'}},
    }

    assert _matches_pinned_pr(current, pinned, 'pydantic/pydantic-ai')
    retargeted = {
        **current,
        'base': {'sha': 'base-sha', 'ref': 'release', 'repo': {'full_name': 'pydantic/pydantic-ai'}},
    }
    advanced = {**current, 'head': {'sha': 'new-head', 'repo': {'full_name': 'pydantic/pydantic-ai'}}}
    assert not _matches_pinned_pr(retargeted, pinned, 'pydantic/pydantic-ai')
    assert not _matches_pinned_pr(advanced, pinned, 'pydantic/pydantic-ai')


def test_ci_evidence_preserves_skipped_job_and_step_conclusions() -> None:
    class FakeGitHub(GitHub):
        def __init__(self) -> None:
            super().__init__('unused', 'pydantic/pydantic-ai')

        def request(self, path: str, **kwargs: object) -> object:
            assert path.startswith('actions/runs/10/jobs?')
            return {
                'jobs': [
                    {
                        'name': 'optional test matrix',
                        'conclusion': 'skipped',
                        'html_url': 'https://github.com/pydantic/pydantic-ai/actions/runs/10/job/20',
                        'steps': [{'name': 'classify', 'conclusion': 'success'}],
                    }
                ]
            }

    assert _ci_jobs(FakeGitHub(), 10) == [
        {
            'name': 'optional test matrix',
            'conclusion': 'skipped',
            'html_url': 'https://github.com/pydantic/pydantic-ai/actions/runs/10/job/20',
            'steps': [{'name': 'classify', 'conclusion': 'success'}],
        }
    ]


def test_ci_jobs_paginate_wrapped_job_collections() -> None:
    class FakeGitHub(GitHub):
        def __init__(self) -> None:
            super().__init__('unused', 'pydantic/pydantic-ai')
            self.calls: list[str] = []

        def request(self, path: str, **kwargs: object) -> object:
            self.calls.append(path)
            page = int(parse_qs(urlsplit(path).query)['page'][0])
            jobs = [
                {
                    'name': f'job {index}',
                    'conclusion': 'success',
                    'html_url': f'https://github.com/pydantic/pydantic-ai/actions/runs/10/job/{index}',
                    'steps': [],
                }
                for index in range((page - 1) * 100, 101 if page == 2 else 100)
            ]
            return {'total_count': 101, 'jobs': jobs}

    github = FakeGitHub()

    jobs = _ci_jobs(github, 10)

    assert len(jobs) == 101
    assert jobs[0]['name'] == 'job 0'
    assert jobs[-1]['name'] == 'job 100'
    assert github.calls == [
        'actions/runs/10/jobs?per_page=100&page=1',
        'actions/runs/10/jobs?per_page=100&page=2',
    ]


@pytest.mark.parametrize('response', [{'total_count': 0}, {'jobs': None}, {'jobs': {}}])
def test_ci_jobs_malformed_wrapped_response_fails_closed(response: Mapping[str, object]) -> None:
    class FakeGitHub(GitHub):
        def __init__(self) -> None:
            super().__init__('unused', 'pydantic/pydantic-ai')

        def request(self, path: str, **kwargs: object) -> object:
            return response

    with pytest.raises(ValueError, match='expected a jobs list'):
        _ci_jobs(FakeGitHub(), 10)
