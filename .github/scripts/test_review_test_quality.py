"""Tests for the pinned Test Quality Review controller."""

from __future__ import annotations

import json
from collections.abc import Mapping

import pytest
from pydantic import ValidationError
from review_test_quality import (
    Candidate,
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
    payload = check_payload(
        head_sha='pinned-head-sha',
        title=title,
        summary='summary',
        details_url='https://github.com/pydantic/pydantic-ai/actions/runs/11',
        base_sha='base-sha',
        workflow_version='workflow-sha',
    )
    check = {
        **payload,
        'status': 'completed',
        'conclusion': 'neutral',
        'output': {
            'summary': '<!-- test-quality-review:v1 base=base-sha head=pinned-head-sha workflow=workflow-sha -->'
        },
    }

    assert title == 'Changes needed'
    assert check['head_sha'] == 'pinned-head-sha'
    assert has_completed_report(
        [check], base_sha='base-sha', head_sha='pinned-head-sha', workflow_version='workflow-sha'
    )
    assert not has_completed_report(
        [check], base_sha='base-sha', head_sha='newer-head', workflow_version='workflow-sha'
    )


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
    class FakeGitHub:
        def paginated(self, path: str) -> list[object]:
            assert path == 'actions/runs/10/jobs'
            return [
                {
                    'name': 'optional test matrix',
                    'conclusion': 'skipped',
                    'html_url': 'https://github.com/pydantic/pydantic-ai/actions/runs/10/job/20',
                    'steps': [{'name': 'classify', 'conclusion': 'success'}],
                }
            ]

    assert _ci_jobs(FakeGitHub(), 10) == [
        {
            'name': 'optional test matrix',
            'conclusion': 'skipped',
            'html_url': 'https://github.com/pydantic/pydantic-ai/actions/runs/10/job/20',
            'steps': [{'name': 'classify', 'conclusion': 'success'}],
        }
    ]
