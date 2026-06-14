from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import ci_duration


def test_normalize_matrix_job_signature():
    job = ci_duration.normalize_job(
        {
            'id': 123,
            'name': 'test on 3.10 (all-extras)',
            'status': 'completed',
            'conclusion': 'success',
            'started_at': '2026-06-13T17:15:03Z',
            'completed_at': '2026-06-13T17:24:05Z',
            'runner_name': 'GitHub Actions 1001364942',
            'runner_group_name': 'GitHub Actions',
            'html_url': 'https://github.com/pydantic/pydantic-ai/actions/runs/1/job/123',
            'steps': [],
        }
    )

    assert job.job_family == 'test'
    assert job.matrix_python == '3.10'
    assert job.matrix_extra == 'all-extras'
    assert job.runner_class == 'github-hosted'
    assert job.job_signature == 'job=test / runner=github-hosted / py=3.10 / extra=all-extras'
    assert job.duration_seconds == 542
    assert ci_duration.is_tracked_test_job(job)


def test_non_test_jobs_are_not_tracked():
    jobs = [
        ci_duration.normalize_job(
            {
                'id': 123,
                'name': name,
                'status': 'completed',
                'conclusion': 'success',
                'started_at': '2026-06-13T17:15:03Z',
                'completed_at': '2026-06-13T17:16:03Z',
                'runner_name': 'GitHub Actions 1001364942',
                'runner_group_name': 'GitHub Actions',
                'html_url': 'https://github.com/pydantic/pydantic-ai/actions/runs/1/job/123',
                'steps': [],
            }
        )
        for name in ['lint', 'test examples on 3.13']
    ]

    assert [ci_duration.is_tracked_test_job(job) for job in jobs] == [False, False]


def test_classify_slow_job_requires_relative_and_absolute_delta():
    baseline = ci_duration.compute_baseline([360, 370, 380, 390, 400, 410, 420, 430, 440, 450])
    job = ci_duration.JobRecord(
        job_id=123,
        raw_name='test on 3.10 (all-extras)',
        job_family='test',
        job_signature='job=test / runner=github-hosted / py=3.10 / extra=all-extras',
        matrix_python='3.10',
        matrix_extra='all-extras',
        conclusion='success',
        status='completed',
        started_at='2026-06-13T17:15:03Z',
        completed_at='2026-06-13T17:24:05Z',
        duration_seconds=600,
        runner_name='GitHub Actions 1001364942',
        runner_group_name='GitHub Actions',
        runner_class='github-hosted',
        html_url='https://github.com/pydantic/pydantic-ai/actions/runs/1/job/123',
        steps=[],
    )

    row = ci_duration.classify_job(job, baseline)

    assert row.status == 'slow'
    assert row.delta_seconds == 195


def test_render_report_uses_sticky_marker_and_threshold_context():
    workflow: ci_duration.JsonObject = {
        'duration_seconds': 840,
        'html_url': 'https://github.com/pydantic/pydantic-ai/actions/runs/1',
    }
    row = ci_duration.ReportRow(
        job_name='test on 3.10 (all-extras)',
        job_signature='job=test / runner=github-hosted / py=3.10 / extra=all-extras',
        duration_seconds=600,
        baseline=ci_duration.compute_baseline([360, 370, 380, 390, 400, 410, 420, 430, 440, 450]),
        delta_seconds=195,
        delta_percent=48,
        status='slow',
    )

    report = ci_duration.render_report(123, 'abcdef1234567890', workflow, [row])

    assert report.startswith('<!-- ci-duration-report -->\n## CI Duration Report')
    assert 'Tracked test jobs: 1' in report
    assert 'Total tracked test job duration: 10m 00s' in report
    assert 'Baseline: up to 30 successful `main` CI runs and 60 successful PR CI runs' in report
    assert 'Minimum baseline sample: 10 successful matching jobs' in report
    assert '| test on 3.10 (all-extras) | 10m 00s | 6m 45s | 7m 08s | +3m 15s (+48%) | slow |' in report
    assert 'trigger:ci-duration-report' in report
