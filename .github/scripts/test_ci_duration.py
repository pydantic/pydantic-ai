from __future__ import annotations

import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import ci_duration


@pytest.mark.parametrize(
    'job,expected',
    [
        pytest.param(
            {
                'name': 'test on 3.10 (all-extras)',
                'completed_at': '2026-06-13T17:24:05Z',
                'runner_name': 'GitHub Actions 1001364942',
                'runner_group_name': 'GitHub Actions',
                'labels': ['ubuntu-latest'],
            },
            (
                'test',
                '3.10',
                'all-extras',
                'github-hosted',
                'job=test / runner=github-hosted / py=3.10 / extra=all-extras',
                542,
            ),
            id='main-matrix',
        ),
        pytest.param(
            {
                'name': 'test durable-exec on 3.11 (locked)',
                'completed_at': '2026-06-13T17:21:43Z',
                'runner_name': 'gr6b98dxdphe5q821s6ezttz97',
                'runner_group_name': 'Default',
                'labels': ['ubicloud-premium-4'],
            },
            (
                'test-durable-exec',
                '3.11',
                'locked',
                'ubicloud-premium-4',
                'job=test-durable-exec / runner=ubicloud-premium-4 / py=3.11 / extra=locked',
                400,
            ),
            id='durable-exec',
        ),
    ],
)
def test_normalize_tracked_job(job: ci_duration.JsonObject, expected: tuple[str, str, str, str, str, float]):
    record = ci_duration.normalize_job(
        {
            'id': 123,
            'status': 'completed',
            'conclusion': 'success',
            'started_at': '2026-06-13T17:15:03Z',
            'html_url': 'https://github.com/pydantic/pydantic-ai/actions/runs/1/job/123',
            'steps': [],
            **job,
        }
    )

    assert (
        record.job_family,
        record.matrix_python,
        record.matrix_extra,
        record.runner_class,
        record.job_signature,
        record.duration_seconds,
    ) == expected
    assert ci_duration.is_tracked_test_job(record)


# The `main` job names closest to a tracked one, none of which may enter the tracked set.
@pytest.mark.parametrize(
    'name',
    [
        'quality checks',
        'mypy',
        'docs-assets',
        'coverage',
        'check',
        'test examples on 3.13',
        'test Temporal latest on Python 3.10',
        'test FastMCP 4 compatibility',
    ],
)
def test_non_test_jobs_are_not_tracked(name: str):
    job = ci_duration.normalize_job(
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

    assert not ci_duration.is_tracked_test_job(job)


@pytest.mark.parametrize(
    'runner_group_name,runner_name,labels,expected',
    [
        ('Default', 'gr6b98dxdphe5q821s6ezttz97', ['ubicloud-premium-4'], 'ubicloud-premium-4'),
        ('Default', 'grftmf0e4q520g401db7md40ge', ['ubicloud-premium-8'], 'ubicloud-premium-8'),
        ('Default', 'gr3re7hexr6y2pfmr2413vkzv9', ['ubicloud'], 'ubicloud'),
        # `runner_group_name` and `runner_name` are Ubicloud's to change; only the label is ours.
        ('Ubicloud', 'gr6b98dxdphe5q821s6ezttz97', ['ubicloud-premium-8'], 'ubicloud-premium-8'),
        ('Default', 'ubicloud-runner-abc123', ['ubicloud-premium-8'], 'ubicloud-premium-8'),
        ('Ubicloud', 'ubicloud-runner-abc123', [], 'ubicloud'),
        ('GitHub Actions', 'GitHub Actions 1001364942', ['ubuntu-latest'], 'github-hosted'),
        ('Default', 'depot-runner', ['depot-ubuntu-24.04'], 'depot'),
        (None, None, ['self-hosted', 'linux'], 'self-hosted'),
        (None, None, None, 'unknown'),
    ],
)
def test_parse_runner_class(
    runner_group_name: str | None, runner_name: str | None, labels: list[ci_duration.JsonValue] | None, expected: str
):
    assert ci_duration.parse_runner_class(runner_group_name, runner_name, labels) == expected


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


@pytest.mark.parametrize(
    'row_count,expect_omission',
    [(ci_duration.REPORT_ROW_LIMIT, False), (ci_duration.REPORT_ROW_LIMIT + 1, True)],
)
def test_render_report_truncates_only_past_the_row_limit(row_count: int, expect_omission: bool):
    workflow: ci_duration.JsonObject = {
        'duration_seconds': 840,
        'html_url': 'https://github.com/pydantic/pydantic-ai/actions/runs/1',
    }
    # A freshly-minted signature has no baseline, and `no_baseline` sorts with `normal` into the
    # truncated tail -- so the limit has to clear the whole tracked matrix, not just the slow rows.
    rows = [
        ci_duration.ReportRow(
            job_name=f'test on 3.10 (extra-{index})',
            job_signature=f'job=test / runner=github-hosted / py=3.10 / extra=extra-{index}',
            duration_seconds=600,
            baseline=None,
            delta_seconds=None,
            delta_percent=None,
            status='no_baseline',
        )
        for index in range(row_count)
    ]

    report = ci_duration.render_report(123, 'abcdef1234567890', workflow, rows)

    assert ('more jobs omitted' in report) == expect_omission
    assert f'Tracked test jobs: {row_count}' in report


def test_collect_baselines_skips_unavailable_historical_run():
    class StubGitHubClient(ci_duration.GitHubClient):
        def request_paginated(self, path: str, *, max_items: int | None = None) -> list[ci_duration.JsonObject]:
            if path == 'actions/workflows/ci.yml/runs?branch=main&event=push&status=success':
                return [
                    {
                        'id': run_id,
                        'run_attempt': 1,
                        'head_sha': f'baseline-{run_id}',
                    }
                    for run_id in range(11)
                ]
            if path == 'actions/workflows/ci.yml/runs?event=pull_request&status=success':
                return []
            if path == 'actions/runs/0/attempts/1/jobs':
                raise urllib.error.URLError('timed out')
            if path.startswith('actions/runs/') and path.endswith('/attempts/1/jobs'):
                return [
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
                ]
            raise RuntimeError(f'Unexpected path: {path}')

    baselines = ci_duration.collect_baselines(StubGitHubClient('pydantic/pydantic-ai', 'token'), 'current-sha')

    assert baselines['job=test / runner=github-hosted / py=3.10 / extra=all-extras'].sample_size == 10


def test_collect_baselines_keeps_families_and_runner_sizes_apart():
    jobs: list[ci_duration.JsonObject] = [
        {
            'id': 1,
            'name': 'test on 3.10 (all-extras)',
            'completed_at': '2026-06-13T17:25:03Z',
            'runner_name': 'gregszg8jvj21pxs0s0v75zn66',
            'labels': ['ubicloud-premium-4'],
        },
        {
            'id': 2,
            'name': 'test on 3.10 (all-extras)',
            'completed_at': '2026-06-13T17:20:03Z',
            'runner_name': 'graj7n4s025a7krr8mmch51ab9',
            'labels': ['ubicloud-premium-8'],
        },
        {
            'id': 3,
            'name': 'test durable-exec on 3.10 (locked)',
            'completed_at': '2026-06-13T17:18:23Z',
            'runner_name': 'grd4b5dgk1fj4jaf6kk5wsna3t',
            'labels': ['ubicloud-premium-4'],
        },
    ]

    class StubGitHubClient(ci_duration.GitHubClient):
        def request_paginated(self, path: str, *, max_items: int | None = None) -> list[ci_duration.JsonObject]:
            if path == 'actions/workflows/ci.yml/runs?branch=main&event=push&status=success':
                return [
                    {
                        'id': run_id,
                        'run_attempt': 1,
                        'head_sha': f'baseline-{run_id}',
                    }
                    for run_id in range(ci_duration.MIN_BASELINE_SAMPLES)
                ]
            if path == 'actions/workflows/ci.yml/runs?event=pull_request&status=success':
                return []
            if path.startswith('actions/runs/') and path.endswith('/attempts/1/jobs'):
                return [
                    {
                        **job,
                        'status': 'completed',
                        'conclusion': 'success',
                        'started_at': '2026-06-13T17:15:03Z',
                        'runner_group_name': 'Default',
                        'html_url': 'https://github.com/pydantic/pydantic-ai/actions/runs/1/job/1',
                        'steps': [],
                    }
                    for job in jobs
                ]
            raise RuntimeError(f'Unexpected path: {path}')

    baselines = ci_duration.collect_baselines(StubGitHubClient('pydantic/pydantic-ai', 'token'), 'current-sha')

    # One bucket per (family, runner size). Pooling any two of these would average a 4-core
    # sample into an 8-core baseline, or the durable-exec suite into the main matrix.
    assert {signature: baseline.median_seconds for signature, baseline in baselines.items()} == {
        'job=test / runner=ubicloud-premium-4 / py=3.10 / extra=all-extras': 600,
        'job=test / runner=ubicloud-premium-8 / py=3.10 / extra=all-extras': 300,
        'job=test-durable-exec / runner=ubicloud-premium-4 / py=3.10 / extra=locked': 200,
    }
    assert [baseline.sample_size for baseline in baselines.values()] == [ci_duration.MIN_BASELINE_SAMPLES] * 3


def test_collect_baselines_stops_after_time_budget(monkeypatch: pytest.MonkeyPatch):
    class StubGitHubClient(ci_duration.GitHubClient):
        def request_paginated(self, path: str, *, max_items: int | None = None) -> list[ci_duration.JsonObject]:
            if path == 'actions/workflows/ci.yml/runs?branch=main&event=push&status=success':
                return [
                    {
                        'id': 1,
                        'run_attempt': 1,
                        'head_sha': 'baseline-1',
                    }
                ]
            if path == 'actions/workflows/ci.yml/runs?event=pull_request&status=success':
                return []
            raise RuntimeError(f'Unexpected path: {path}')

    # Force the deadline into the past so the first loop guard always trips, rather than patching the
    # process-global time.monotonic (which any concurrent caller in the worker can desync).
    monkeypatch.setattr(ci_duration, 'BASELINE_COLLECTION_MAX_SECONDS', -1.0)

    baselines = ci_duration.collect_baselines(StubGitHubClient('pydantic/pydantic-ai', 'token'), 'current-sha')

    assert baselines == {}
