from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

SCHEMA_VERSION = 1
CI_WORKFLOW_FILE = 'ci.yml'
REPORT_MARKER = '<!-- ci-duration-report -->'
REPORT_LABEL = 'trigger:ci-duration-report'
BASELINE_MAIN_RUN_LIMIT = 30
BASELINE_PR_RUN_LIMIT = 60
MIN_BASELINE_SAMPLES = 10
WARNING_MIN_SECONDS = 60
SLOW_THRESHOLD_MULTIPLIER = 1.25
FAST_THRESHOLD_MULTIPLIER = 0.75
VERY_SLOW_MIN_SECONDS = 300
VERY_SLOW_THRESHOLD_MULTIPLIER = 1.5

JsonValue = None | bool | int | float | str | list['JsonValue'] | dict[str, 'JsonValue']
JsonObject = dict[str, JsonValue]


@dataclass(frozen=True)
class WorkflowRunRecord:
    repo: str
    workflow_id: int
    workflow_name: str
    workflow_path: str
    run_id: int
    run_attempt: int
    run_number: int
    event: str
    status: str
    conclusion: str | None
    head_branch: str | None
    base_branch: str | None
    head_sha: str
    pr_numbers: list[int]
    run_started_at: str | None
    updated_at: str | None
    duration_seconds: float | None
    html_url: str
    actor: str | None


@dataclass(frozen=True)
class StepRecord:
    number: int
    name: str
    status: str
    conclusion: str | None
    started_at: str | None
    completed_at: str | None
    duration_seconds: float | None


@dataclass(frozen=True)
class JobRecord:
    job_id: int
    raw_name: str
    job_family: str
    job_signature: str
    matrix_python: str | None
    matrix_extra: str | None
    conclusion: str | None
    status: str
    started_at: str | None
    completed_at: str | None
    duration_seconds: float | None
    runner_name: str | None
    runner_group_name: str | None
    runner_class: str
    html_url: str
    steps: list[StepRecord]


@dataclass(frozen=True)
class Baseline:
    sample_size: int
    median_seconds: float
    p75_seconds: float
    p90_seconds: float
    mad_seconds: float


@dataclass(frozen=True)
class ReportRow:
    job_name: str
    job_signature: str
    duration_seconds: float | None
    baseline: Baseline | None
    delta_seconds: float | None
    delta_percent: float | None
    status: Literal['normal', 'fast', 'slow', 'very_slow', 'no_baseline', 'not_completed']


class GitHubClient:
    def __init__(self, repo: str, token: str):
        self.repo = repo
        self.token = token
        self.ssl_context = _ssl_context()

    def request_json(self, path: str, *, method: str = 'GET', body: JsonObject | None = None) -> JsonValue:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            self._url(path),
            data=data,
            method=method,
            headers={
                'Accept': 'application/vnd.github+json',
                'Authorization': f'Bearer {self.token}',
                'Content-Type': 'application/json',
                'X-GitHub-Api-Version': '2022-11-28',
            },
        )
        with urllib.request.urlopen(request, timeout=30, context=self.ssl_context) as response:
            response_body = response.read()
        if not response_body:
            return None
        return json.loads(response_body)

    def request_paginated(self, path: str, *, max_items: int | None = None) -> list[JsonObject]:
        parsed = urllib.parse.urlsplit(path)
        query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        query = [item for item in query if item[0] not in {'page', 'per_page'}]
        query.append(('per_page', '100'))

        page = 1
        results: list[JsonObject] = []
        while True:
            page_query = [*query, ('page', str(page))]
            page_path = urllib.parse.urlunsplit(
                (parsed.scheme, parsed.netloc, parsed.path, urllib.parse.urlencode(page_query), '')
            )
            value = self.request_json(page_path)
            page_items = _extract_page_items(value)
            if not page_items:
                return results
            results.extend(page_items)
            if max_items is not None and len(results) >= max_items:
                return results[:max_items]
            if len(page_items) < 100:
                return results
            page += 1

    def _url(self, path: str) -> str:
        if path.startswith('http://') or path.startswith('https://'):
            return path
        if not path.startswith('/'):
            path = f'/repos/{self.repo}/{path}'
        return f'https://api.github.com{path}'


def main() -> None:
    parser = argparse.ArgumentParser(description='Collect and report GitHub Actions CI duration telemetry.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    collect_parser = subparsers.add_parser('collect', help='Collect one CI workflow run and emit it to Logfire.')
    collect_parser.add_argument('--run-id', default=os.getenv('CI_RUN_ID') or os.getenv('GITHUB_EVENT_WORKFLOW_RUN_ID'))
    collect_parser.add_argument(
        '--run-attempt', default=os.getenv('CI_RUN_ATTEMPT') or os.getenv('GITHUB_EVENT_WORKFLOW_RUN_ATTEMPT')
    )
    collect_parser.add_argument('--output', default='ci-duration-record.json')
    collect_parser.add_argument('--skip-logfire', action='store_true')

    report_parser = subparsers.add_parser('report', help='Post or update a PR CI duration report.')
    report_parser.add_argument('--pr-number', default=os.getenv('PR_NUMBER'))
    report_parser.add_argument('--head-sha', default=os.getenv('HEAD_SHA'))
    report_parser.add_argument('--poll-seconds', type=int, default=0)
    report_parser.add_argument('--dry-run', action='store_true')

    args = parser.parse_args()
    client = _github_client_from_env()

    if args.command == 'collect':
        if args.run_id is None:
            raise SystemExit('CI_RUN_ID is required')
        run_attempt = int(args.run_attempt) if args.run_attempt else None
        record = collect_run(client, int(args.run_id), run_attempt)
        Path(args.output).write_text(json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        print(f'Wrote CI duration record to {args.output}')
        if not args.skip_logfire:
            emit_logfire(record)
    else:
        if args.pr_number is None:
            raise SystemExit('PR_NUMBER is required')
        if args.head_sha is None:
            raise SystemExit('HEAD_SHA is required')
        body = build_pr_report(client, int(args.pr_number), args.head_sha, args.poll_seconds)
        if args.dry_run:
            print(body)
        else:
            upsert_pr_comment(client, int(args.pr_number), body)


def collect_run(client: GitHubClient, run_id: int, run_attempt: int | None) -> JsonObject:
    run = _expect_object(client.request_json(f'actions/runs/{run_id}'), 'workflow run')
    attempt = run_attempt or _expect_int(run.get('run_attempt'), 'run_attempt')
    jobs = client.request_paginated(f'actions/runs/{run_id}/attempts/{attempt}/jobs')
    workflow = normalize_workflow_run(client.repo, run, attempt)
    job_records: list[JobRecord] = []
    for job_object in jobs:
        job = normalize_job(job_object)
        if is_tracked_test_job(job):
            job_records.append(job)
    return {
        'schema_version': SCHEMA_VERSION,
        'fetched_at': _now(),
        'workflow_run': workflow_to_json(workflow),
        'jobs': [job_to_json(job) for job in job_records],
    }


def build_pr_report(client: GitHubClient, pr_number: int, head_sha: str, poll_seconds: int) -> str:
    run = wait_for_completed_ci_run(client, head_sha, poll_seconds)
    if run is None:
        return render_waiting_report(head_sha)

    run_id = _expect_int(run.get('id'), 'run id')
    run_attempt = _expect_int(run.get('run_attempt'), 'run_attempt')
    current_record = collect_run(client, run_id, run_attempt)
    current_jobs = [_job_from_json(job) for job in _expect_list(current_record['jobs'], 'jobs')]
    baselines = collect_baselines(client, head_sha)
    rows = [classify_job(job, baselines.get(job.job_signature)) for job in current_jobs]
    workflow = _expect_object(current_record['workflow_run'], 'workflow_run')
    return render_report(pr_number, head_sha, workflow, rows)


def wait_for_completed_ci_run(client: GitHubClient, head_sha: str, poll_seconds: int) -> JsonObject | None:
    deadline = time.monotonic() + poll_seconds
    while True:
        run = find_latest_ci_run(client, head_sha)
        if run is not None and run.get('status') == 'completed':
            return run
        if poll_seconds <= 0 or time.monotonic() >= deadline:
            return None
        time.sleep(20)


def find_latest_ci_run(client: GitHubClient, head_sha: str) -> JsonObject | None:
    runs = client.request_paginated(f'actions/workflows/{CI_WORKFLOW_FILE}/runs?head_sha={head_sha}', max_items=10)
    matching = [run for run in runs if run.get('head_sha') == head_sha]
    if not matching:
        return None
    matching.sort(key=lambda run: str(run.get('created_at') or ''), reverse=True)
    return matching[0]


def collect_baselines(client: GitHubClient, current_head_sha: str) -> dict[str, Baseline]:
    main_runs = client.request_paginated(
        f'actions/workflows/{CI_WORKFLOW_FILE}/runs?branch=main&event=push&status=success',
        max_items=BASELINE_MAIN_RUN_LIMIT,
    )
    pr_runs = client.request_paginated(
        f'actions/workflows/{CI_WORKFLOW_FILE}/runs?event=pull_request&status=success',
        max_items=BASELINE_PR_RUN_LIMIT,
    )
    durations: dict[str, list[float]] = {}
    seen_run_ids: set[int] = set()
    for run in [*main_runs, *pr_runs]:
        run_id = _expect_int(run.get('id'), 'baseline run id')
        if run_id in seen_run_ids or run.get('head_sha') == current_head_sha:
            continue
        seen_run_ids.add(run_id)
        run_attempt = _expect_int(run.get('run_attempt'), 'baseline run_attempt')
        jobs = client.request_paginated(f'actions/runs/{run_id}/attempts/{run_attempt}/jobs')
        for job_object in jobs:
            job = normalize_job(job_object)
            if job.conclusion == 'success' and job.duration_seconds is not None and is_tracked_test_job(job):
                durations.setdefault(job.job_signature, []).append(job.duration_seconds)
    return {
        signature: compute_baseline(values)
        for signature, values in durations.items()
        if len(values) >= MIN_BASELINE_SAMPLES
    }


def normalize_workflow_run(repo: str, run: JsonObject, run_attempt: int) -> WorkflowRunRecord:
    run_started_at = _expect_optional_str(run.get('run_started_at'), 'run_started_at')
    updated_at = _expect_optional_str(run.get('updated_at'), 'updated_at')
    return WorkflowRunRecord(
        repo=repo,
        workflow_id=_expect_int(run.get('workflow_id'), 'workflow_id'),
        workflow_name=_expect_str(run.get('name'), 'workflow name'),
        workflow_path=_expect_str(run.get('path'), 'workflow path'),
        run_id=_expect_int(run.get('id'), 'run id'),
        run_attempt=run_attempt,
        run_number=_expect_int(run.get('run_number'), 'run_number'),
        event=_expect_str(run.get('event'), 'event'),
        status=_expect_str(run.get('status'), 'status'),
        conclusion=_expect_optional_str(run.get('conclusion'), 'conclusion'),
        head_branch=_expect_optional_str(run.get('head_branch'), 'head_branch'),
        base_branch=_expect_optional_str(run.get('base_ref'), 'base_ref'),
        head_sha=_expect_str(run.get('head_sha'), 'head_sha'),
        pr_numbers=_pull_request_numbers(run.get('pull_requests')),
        run_started_at=run_started_at,
        updated_at=updated_at,
        duration_seconds=_duration_seconds(run_started_at, updated_at),
        html_url=_expect_str(run.get('html_url'), 'html_url'),
        actor=_actor_login(run.get('actor')),
    )


def normalize_job(job: JsonObject) -> JobRecord:
    raw_name = _expect_str(job.get('name'), 'job name')
    started_at = _expect_optional_str(job.get('started_at'), 'job started_at')
    completed_at = _expect_optional_str(job.get('completed_at'), 'job completed_at')
    matrix_python, matrix_extra = parse_job_matrix(raw_name)
    job_family = parse_job_family(raw_name)
    runner_class = parse_runner_class(
        _expect_optional_str(job.get('runner_group_name'), 'runner_group_name'),
        _expect_optional_str(job.get('runner_name'), 'runner_name'),
        _expect_list_or_none(job.get('labels'), 'labels'),
    )
    return JobRecord(
        job_id=_expect_int(job.get('id'), 'job id'),
        raw_name=raw_name,
        job_family=job_family,
        job_signature=job_signature(job_family, matrix_python, matrix_extra, runner_class),
        matrix_python=matrix_python,
        matrix_extra=matrix_extra,
        conclusion=_expect_optional_str(job.get('conclusion'), 'job conclusion'),
        status=_expect_str(job.get('status'), 'job status'),
        started_at=started_at,
        completed_at=completed_at,
        duration_seconds=_duration_seconds(started_at, completed_at),
        runner_name=_expect_optional_str(job.get('runner_name'), 'runner_name'),
        runner_group_name=_expect_optional_str(job.get('runner_group_name'), 'runner_group_name'),
        runner_class=runner_class,
        html_url=_expect_str(job.get('html_url'), 'job html_url'),
        steps=[normalize_step(step) for step in _expect_list_or_none(job.get('steps'), 'steps') or []],
    )


def normalize_step(step: JsonValue) -> StepRecord:
    step_object = _expect_object(step, 'step')
    started_at = _expect_optional_str(step_object.get('started_at'), 'step started_at')
    completed_at = _expect_optional_str(step_object.get('completed_at'), 'step completed_at')
    return StepRecord(
        number=_expect_int(step_object.get('number'), 'step number'),
        name=_expect_str(step_object.get('name'), 'step name'),
        status=_expect_str(step_object.get('status'), 'step status'),
        conclusion=_expect_optional_str(step_object.get('conclusion'), 'step conclusion'),
        started_at=started_at,
        completed_at=completed_at,
        duration_seconds=_duration_seconds(started_at, completed_at),
    )


def workflow_to_json(workflow: WorkflowRunRecord) -> JsonObject:
    pr_numbers: list[JsonValue] = [number for number in workflow.pr_numbers]
    return {
        'repo': workflow.repo,
        'workflow_id': workflow.workflow_id,
        'workflow_name': workflow.workflow_name,
        'workflow_path': workflow.workflow_path,
        'run_id': workflow.run_id,
        'run_attempt': workflow.run_attempt,
        'run_number': workflow.run_number,
        'event': workflow.event,
        'status': workflow.status,
        'conclusion': workflow.conclusion,
        'head_branch': workflow.head_branch,
        'base_branch': workflow.base_branch,
        'head_sha': workflow.head_sha,
        'pr_numbers': pr_numbers,
        'run_started_at': workflow.run_started_at,
        'updated_at': workflow.updated_at,
        'duration_seconds': workflow.duration_seconds,
        'html_url': workflow.html_url,
        'actor': workflow.actor,
    }


def job_to_json(job: JobRecord) -> JsonObject:
    return {
        'job_id': job.job_id,
        'raw_name': job.raw_name,
        'job_family': job.job_family,
        'job_signature': job.job_signature,
        'matrix_python': job.matrix_python,
        'matrix_extra': job.matrix_extra,
        'conclusion': job.conclusion,
        'status': job.status,
        'started_at': job.started_at,
        'completed_at': job.completed_at,
        'duration_seconds': job.duration_seconds,
        'runner_name': job.runner_name,
        'runner_group_name': job.runner_group_name,
        'runner_class': job.runner_class,
        'html_url': job.html_url,
        'steps': [step_to_json(step) for step in job.steps],
    }


def step_to_json(step: StepRecord) -> JsonObject:
    return {
        'number': step.number,
        'name': step.name,
        'status': step.status,
        'conclusion': step.conclusion,
        'started_at': step.started_at,
        'completed_at': step.completed_at,
        'duration_seconds': step.duration_seconds,
    }


def parse_job_matrix(job_name: str) -> tuple[str | None, str | None]:
    match = re.fullmatch(r'test on (?P<python>[^ ]+) \((?P<extra>[^)]+)\)', job_name)
    if match:
        return match.group('python'), match.group('extra')
    match = re.fullmatch(r'test examples on (?P<python>[^ ]+)', job_name)
    if match:
        return match.group('python'), 'examples'
    return None, None


def parse_job_family(job_name: str) -> str:
    if job_name.startswith('test on '):
        return 'test'
    if job_name.startswith('test examples on '):
        return 'test-examples'
    if job_name in {'lint', 'mypy', 'docs', 'coverage', 'check'}:
        return job_name
    return job_name


def is_tracked_test_job(job: JobRecord) -> bool:
    return job.job_family == 'test'


def parse_runner_class(runner_group_name: str | None, runner_name: str | None, labels: list[JsonValue] | None) -> str:
    label_values = [value for value in labels or [] if isinstance(value, str)]
    lower_values = ' '.join([runner_group_name or '', runner_name or '', *label_values]).lower()
    if 'depot' in lower_values:
        return 'depot'
    if 'github actions' in lower_values or 'ubuntu' in lower_values:
        return 'github-hosted'
    if 'self-hosted' in lower_values:
        return 'self-hosted'
    return 'unknown'


def job_signature(job_family: str, matrix_python: str | None, matrix_extra: str | None, runner_class: str) -> str:
    parts = [f'job={job_family}', f'runner={runner_class}']
    if matrix_python is not None:
        parts.append(f'py={matrix_python}')
    if matrix_extra is not None:
        parts.append(f'extra={matrix_extra}')
    return ' / '.join(parts)


def compute_baseline(values: list[float]) -> Baseline:
    sorted_values = sorted(values)
    median = statistics.median(sorted_values)
    deviations = [abs(value - median) for value in sorted_values]
    return Baseline(
        sample_size=len(sorted_values),
        median_seconds=median,
        p75_seconds=percentile(sorted_values, 75),
        p90_seconds=percentile(sorted_values, 90),
        mad_seconds=statistics.median(deviations),
    )


def percentile(sorted_values: list[float], percentile_value: int) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = (len(sorted_values) - 1) * percentile_value / 100
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def classify_job(job: JobRecord, baseline: Baseline | None) -> ReportRow:
    if job.conclusion != 'success' or job.duration_seconds is None:
        return ReportRow(job.raw_name, job.job_signature, job.duration_seconds, baseline, None, None, 'not_completed')
    if baseline is None:
        return ReportRow(job.raw_name, job.job_signature, job.duration_seconds, None, None, None, 'no_baseline')

    delta = job.duration_seconds - baseline.median_seconds
    delta_percent = delta / baseline.median_seconds * 100 if baseline.median_seconds else None
    slow_threshold = max(
        baseline.p75_seconds * SLOW_THRESHOLD_MULTIPLIER, baseline.median_seconds + 2 * baseline.mad_seconds
    )
    very_slow_threshold = max(
        baseline.p90_seconds * VERY_SLOW_THRESHOLD_MULTIPLIER,
        baseline.median_seconds + 4 * baseline.mad_seconds,
    )
    if job.duration_seconds > very_slow_threshold or delta >= VERY_SLOW_MIN_SECONDS:
        status: Literal['normal', 'fast', 'slow', 'very_slow', 'no_baseline', 'not_completed'] = 'very_slow'
    elif job.duration_seconds > slow_threshold and delta >= WARNING_MIN_SECONDS:
        status = 'slow'
    elif job.duration_seconds < baseline.median_seconds * FAST_THRESHOLD_MULTIPLIER and delta <= -WARNING_MIN_SECONDS:
        status = 'fast'
    else:
        status = 'normal'
    return ReportRow(job.raw_name, job.job_signature, job.duration_seconds, baseline, delta, delta_percent, status)


def render_report(pr_number: int, head_sha: str, workflow: JsonObject, rows: list[ReportRow]) -> str:
    slow_rows = [row for row in rows if row.status in {'slow', 'very_slow'}]
    fast_rows = [row for row in rows if row.status == 'fast']
    failed_rows = [row for row in rows if row.status == 'not_completed']
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            row.status not in {'very_slow', 'slow', 'fast', 'not_completed'},
            -(row.delta_seconds or 0),
            row.job_name,
        ),
    )
    tracked_duration = sum(row.duration_seconds or 0 for row in rows)
    run_url = _expect_str(workflow.get('html_url'), 'workflow html_url')
    sha7 = head_sha[:7]
    lines = [
        REPORT_MARKER,
        '## CI Duration Report',
        '',
        f'PR #{pr_number}, commit `{sha7}`: [CI run]({run_url})',
        '',
        '**Summary**',
        f'- Tracked test jobs: {len(rows)}',
        f'- Total tracked test job duration: {_format_seconds(tracked_duration)}',
        f'- Slow jobs: {len(slow_rows)}',
        f'- Fast jobs: {len(fast_rows)}',
        f'- Failed/cancelled jobs: {len(failed_rows)}',
        f'- Baseline: up to {BASELINE_MAIN_RUN_LIMIT} successful `main` CI runs and {BASELINE_PR_RUN_LIMIT} successful PR CI runs, matched by job signature and runner class',
        f'- Minimum baseline sample: {MIN_BASELINE_SAMPLES} successful matching jobs',
        f'- Slow threshold: duration > max(p75 * {SLOW_THRESHOLD_MULTIPLIER}, median + 2 * MAD), with at least {WARNING_MIN_SECONDS}s increase',
        '',
        '| Job | Duration | Baseline median | p75 | Delta | Status |',
        '|---|---:|---:|---:|---:|---|',
    ]
    for row in sorted_rows[:20]:
        lines.append(
            '| '
            + ' | '.join(
                [
                    row.job_name,
                    _format_seconds(row.duration_seconds),
                    _format_seconds(row.baseline.median_seconds if row.baseline else None),
                    _format_seconds(row.baseline.p75_seconds if row.baseline else None),
                    _format_delta(row.delta_seconds, row.delta_percent),
                    row.status.replace('_', ' '),
                ]
            )
            + ' |'
        )
    if len(sorted_rows) > 20:
        lines.append(f'| ... | {len(sorted_rows) - 20} more jobs omitted |  |  |  |  |')
    lines.extend(
        [
            '',
            '<sub>Re-add the `trigger:ci-duration-report` label to refresh this report.</sub>',
        ]
    )
    return '\n'.join(lines)


def render_waiting_report(head_sha: str) -> str:
    return '\n'.join(
        [
            REPORT_MARKER,
            '## CI Duration Report — waiting for CI',
            '',
            f'No completed `CI` run was found for commit `{head_sha[:7]}` yet.',
            '',
            '<sub>Re-add the `trigger:ci-duration-report` label after CI completes, or rerun with a longer poll window.</sub>',
        ]
    )


def upsert_pr_comment(client: GitHubClient, pr_number: int, body: str) -> None:
    comments = client.request_paginated(f'issues/{pr_number}/comments')
    existing_url: str | None = None
    for comment in comments:
        user = _expect_object(comment.get('user'), 'comment user')
        if user.get('login') == 'github-actions[bot]' and str(comment.get('body') or '').startswith(REPORT_MARKER):
            existing_url = _expect_str(comment.get('url'), 'comment url')
            break
    if existing_url:
        client.request_json(existing_url, method='PATCH', body={'body': body})
        print('Updated existing CI duration report comment')
    else:
        client.request_json(f'issues/{pr_number}/comments', method='POST', body={'body': body})
        print('Created CI duration report comment')


def emit_logfire(record: JsonObject) -> None:
    token = os.getenv('LOGFIRE_WRITE_TOKEN') or os.getenv('LOGFIRE_TOKEN')
    if not token:
        print('LOGFIRE_WRITE_TOKEN is not set; skipping Logfire emission')
        return
    try:
        import logfire
    except ImportError:
        print('logfire is not installed; skipping Logfire emission')
        return

    logfire_base_url = os.getenv('LOGFIRE_URL')
    advanced_options = logfire.AdvancedOptions(base_url=logfire_base_url) if logfire_base_url else None
    logfire.configure(
        token=token,
        service_name='pydantic-ai-ci',
        environment='github-actions',
        console=False,
        advanced=advanced_options,
    )
    workflow = _expect_object(record['workflow_run'], 'workflow_run')
    jobs = _expect_list(record['jobs'], 'jobs')
    with logfire.span(
        'ci.duration.test_run',
        _tags=['ci-duration'],
        schema_version=SCHEMA_VERSION,
        repo=workflow.get('repo'),
        workflow_name=workflow.get('workflow_name'),
        run_id=workflow.get('run_id'),
        run_attempt=workflow.get('run_attempt'),
        event=workflow.get('event'),
        conclusion=workflow.get('conclusion'),
        head_branch=workflow.get('head_branch'),
        base_branch=workflow.get('base_branch'),
        head_sha=workflow.get('head_sha'),
        pr_numbers=workflow.get('pr_numbers'),
        duration_seconds=workflow.get('duration_seconds'),
        tracked_test_jobs=len(jobs),
        tracked_test_duration_seconds=sum(
            _expect_optional_float(_expect_object(job, 'job').get('duration_seconds'), 'duration_seconds') or 0
            for job in jobs
        ),
        html_url=workflow.get('html_url'),
    ):
        for job in jobs:
            job_object = _expect_object(job, 'job')
            logfire.info(
                'ci.duration.test_job',
                _tags=['ci-duration'],
                schema_version=SCHEMA_VERSION,
                repo=workflow.get('repo'),
                run_id=workflow.get('run_id'),
                run_attempt=workflow.get('run_attempt'),
                event=workflow.get('event'),
                head_branch=workflow.get('head_branch'),
                base_branch=workflow.get('base_branch'),
                head_sha=workflow.get('head_sha'),
                pr_numbers=workflow.get('pr_numbers'),
                job_id=job_object.get('job_id'),
                job_name=job_object.get('raw_name'),
                job_family=job_object.get('job_family'),
                job_signature=job_object.get('job_signature'),
                matrix_python=job_object.get('matrix_python'),
                matrix_extra=job_object.get('matrix_extra'),
                runner_class=job_object.get('runner_class'),
                conclusion=job_object.get('conclusion'),
                duration_seconds=job_object.get('duration_seconds'),
                html_url=job_object.get('html_url'),
            )
    logfire.force_flush()


def _github_client_from_env() -> GitHubClient:
    repo = os.getenv('GITHUB_REPOSITORY')
    token = os.getenv('GITHUB_TOKEN')
    if not repo:
        raise SystemExit('GITHUB_REPOSITORY is required')
    if not token:
        raise SystemExit('GITHUB_TOKEN is required')
    return GitHubClient(repo, token)


def _ssl_context() -> ssl.SSLContext | None:
    try:
        import certifi
    except ImportError:
        return None
    return ssl.create_default_context(cafile=certifi.where())


def _extract_page_items(value: JsonValue) -> list[JsonObject]:
    if isinstance(value, list):
        return [_expect_object(item, 'paginated item') for item in value]
    if isinstance(value, dict):
        for key in ('jobs', 'workflow_runs', 'comments'):
            items = value.get(key)
            if isinstance(items, list):
                return [_expect_object(item, key) for item in items]
    raise RuntimeError(f'Unexpected paginated response shape: {value!r}')


def _job_from_json(value: JsonValue) -> JobRecord:
    job = _expect_object(value, 'job')
    return JobRecord(
        job_id=_expect_int(job.get('job_id'), 'job_id'),
        raw_name=_expect_str(job.get('raw_name'), 'raw_name'),
        job_family=_expect_str(job.get('job_family'), 'job_family'),
        job_signature=_expect_str(job.get('job_signature'), 'job_signature'),
        matrix_python=_expect_optional_str(job.get('matrix_python'), 'matrix_python'),
        matrix_extra=_expect_optional_str(job.get('matrix_extra'), 'matrix_extra'),
        conclusion=_expect_optional_str(job.get('conclusion'), 'conclusion'),
        status=_expect_str(job.get('status'), 'status'),
        started_at=_expect_optional_str(job.get('started_at'), 'started_at'),
        completed_at=_expect_optional_str(job.get('completed_at'), 'completed_at'),
        duration_seconds=_expect_optional_float(job.get('duration_seconds'), 'duration_seconds'),
        runner_name=_expect_optional_str(job.get('runner_name'), 'runner_name'),
        runner_group_name=_expect_optional_str(job.get('runner_group_name'), 'runner_group_name'),
        runner_class=_expect_str(job.get('runner_class'), 'runner_class'),
        html_url=_expect_str(job.get('html_url'), 'html_url'),
        steps=[],
    )


def _pull_request_numbers(value: JsonValue) -> list[int]:
    if not isinstance(value, list):
        return []
    numbers: list[int] = []
    for item in value:
        if isinstance(item, dict):
            number = item.get('number')
            if isinstance(number, int):
                numbers.append(number)
    return numbers


def _actor_login(value: JsonValue) -> str | None:
    if isinstance(value, dict):
        login = value.get('login')
        if isinstance(login, str):
            return login
    return None


def _duration_seconds(start: str | None, end: str | None) -> float | None:
    if start is None or end is None:
        return None
    return (_parse_timestamp(end) - _parse_timestamp(start)).total_seconds()


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _format_seconds(value: float | None) -> str:
    if value is None:
        return 'n/a'
    seconds = int(round(value))
    minutes, remainder = divmod(seconds, 60)
    if minutes:
        return f'{minutes}m {remainder:02d}s'
    return f'{remainder}s'


def _format_delta(delta_seconds: float | None, delta_percent: float | None) -> str:
    if delta_seconds is None or delta_percent is None:
        return 'n/a'
    sign = '+' if delta_seconds >= 0 else '-'
    return f'{sign}{_format_seconds(abs(delta_seconds))} ({delta_percent:+.0f}%)'


def _expect_object(value: JsonValue, label: str) -> JsonObject:
    if isinstance(value, dict):
        return value
    raise RuntimeError(f'Expected {label} to be an object, got {type(value).__name__}')


def _expect_list(value: JsonValue, label: str) -> list[JsonValue]:
    if isinstance(value, list):
        return value
    raise RuntimeError(f'Expected {label} to be a list, got {type(value).__name__}')


def _expect_list_or_none(value: JsonValue, label: str) -> list[JsonValue] | None:
    if value is None:
        return None
    return _expect_list(value, label)


def _expect_str(value: JsonValue, label: str) -> str:
    if isinstance(value, str):
        return value
    raise RuntimeError(f'Expected {label} to be a string, got {type(value).__name__}')


def _expect_optional_str(value: JsonValue, label: str) -> str | None:
    if value is None:
        return None
    return _expect_str(value, label)


def _expect_int(value: JsonValue, label: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise RuntimeError(f'Expected {label} to be an integer, got {type(value).__name__}')


def _expect_optional_float(value: JsonValue, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    raise RuntimeError(f'Expected {label} to be a number, got {type(value).__name__}')


if __name__ == '__main__':
    try:
        main()
    except urllib.error.HTTPError as exc:
        print(f'GitHub API request failed: HTTP {exc.code}\n{exc.read().decode()}', file=sys.stderr)
        raise
