"""Tests for the agentic workflow spend report.

The alert cases are calibrated against the real #6766 measurements: a workflow
that never starts its agent, and one whose runs mostly deliver nothing.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import urllib.error
import zipfile
from http.client import HTTPMessage
from pathlib import Path
from typing import IO, Any

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from agent_spend_report import (
    AGENT_OUTPUT_FILE,
    AGENT_STDIO_LOG,
    AGENT_USAGE_FILE,
    ARCHIVE_DOWNLOAD_LIMIT,
    LOG_READ_LIMIT,
    SLACK_SECTION_LIMIT,
    ArtifactMetrics,
    GitHubClient,
    RunRecord,
    WorkflowSummary,
    _runs_in_window,  # pyright: ignore[reportPrivateUsage]
    _split_for_slack,  # pyright: ignore[reportPrivateUsage]
    build_slack_payload,
    collect_run,
    detect_alerts,
    format_report,
    gather,
    parse_agent_artifact,
    summarize,
)


def _artifact(usage: dict[str, int] | None = None, items: list[str] | None = None, log: str = '') -> IO[bytes]:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as bundle:
        if usage is not None:
            bundle.writestr(AGENT_USAGE_FILE, json.dumps(usage))
        if items is not None:
            bundle.writestr(AGENT_OUTPUT_FILE, json.dumps({'items': items}))
        if log:
            bundle.writestr(AGENT_STDIO_LOG, log)
    return buffer


def _record(
    workflow: str, *, conclusion: str = 'success', items: int = 1, tokens: int = 1000, **kwargs: Any
) -> RunRecord:
    return RunRecord(
        workflow,
        run_id=1,
        conclusion=conclusion,
        agent_invoked=True,
        output_tokens=tokens,
        item_count=items,
        **kwargs,
    )


# --- artifact parsing ---------------------------------------------------------


def test_parse_agent_artifact_reads_tokens_items_and_retries():
    archive = _artifact(
        usage={'output_tokens': 18375},
        items=['review'],
        log=(
            'attempt 1 failed: exitCode=1\nAttempt 1 failed — will retry with fallback model\n'
            'attempt 2 failed: exitCode=1\nAttempt 2 failed — will retry with fallback model\n'
            '429 Too Many Requests\n'
        ),
    )

    assert parse_agent_artifact(archive) == ArtifactMetrics(18375, 1, 2, True)


def test_parse_agent_artifact_counts_retries_not_failed_attempts():
    """gh-aw's harness logs `attempt N failed` on any non-zero exit, before deciding.

    Most of its branches then decline to retry, so counting that line would bill a
    one-shot failure as the full re-spend the alert text promises.
    """
    metrics = parse_agent_artifact(_artifact(log='attempt 1 failed: exitCode=2\nNo retry configured, giving up\n'))

    assert metrics.retries == 0


def test_parse_agent_artifact_detects_the_stream_json_rate_limit():
    """Behind gh-aw's api-proxy the limit surfaces as `"api_error_status": 429`.

    The transport-style `429 Too Many Requests` string never appears in that form, so
    matching only it meant the rate-limit alert could not fire for these workflows.
    """
    log = '{"type":"error","api_error_status": 429,"message":"overloaded"}\n'

    assert parse_agent_artifact(_artifact(log=log)).rate_limited is True


def test_parse_agent_artifact_treats_empty_items_as_no_output():
    """`{"items": []}` is the signature of a run that cost full price and delivered nothing."""
    metrics = parse_agent_artifact(_artifact(usage={'output_tokens': 81924}, items=[]))

    assert metrics == ArtifactMetrics(81924, 0, 0, False)


def test_parse_agent_artifact_reports_a_missing_output_file_as_unknown():
    """A killed run may upload the zip before every file lands.

    `item_count is None` keeps that distinct from a present-but-empty `{"items": []}`,
    so a truncated upload is not counted as a wasted run.
    """
    assert parse_agent_artifact(_artifact()).item_count is None


def test_parse_agent_artifact_survives_the_v0834_usage_schema():
    """gh-aw v0.83.4 dropped `effective_tokens` for `ai_credits`; `output_tokens` stayed."""
    archive = _artifact(usage={'output_tokens': 6979, 'ai_credits': 0}, items=['review'])

    assert parse_agent_artifact(archive).output_tokens == 6979


def test_parse_agent_artifact_refuses_to_decompress_an_oversized_member():
    """A runaway run's log compresses to nothing and expands to more than the job has.

    The read is bounded through the member stream, so the limit holds however small the
    archive claiming to hold it is.
    """
    archive = _artifact(usage={'output_tokens': 100}, items=[], log='x' * (LOG_READ_LIMIT + 1))

    with pytest.raises(ValueError, match=AGENT_STDIO_LOG):
        parse_agent_artifact(archive)


# --- aggregation --------------------------------------------------------------


def test_summarize_counts_only_agent_runs_towards_waste():
    """Runs that skip the agent cost nothing and must not dilute the rate."""
    records = [
        _record('a.lock.yml', items=0, tokens=100),
        _record('a.lock.yml', items=2, tokens=300),
        RunRecord('a.lock.yml', run_id=3, conclusion='success', agent_invoked=False),
    ]

    (summary,) = summarize(records)

    assert (summary.total_runs, summary.agent_runs) == (3, 2)
    assert summary.zero_output_runs == 1
    assert summary.zero_output_rate == 0.5
    assert summary.output_tokens == 400
    # Only the empty run's own 100 tokens are waste, not half of the 300-token run's.
    assert summary.wasted_tokens == 100


def test_wasted_tokens_is_exact_when_run_costs_differ():
    """Waste is summed from the runs that delivered nothing, never apportioned by rate.

    Per-run cost spans more than an order of magnitude, so an estimate scaled by the
    zero-output rate would bill successful runs' spend as waste and materially
    misstate the number.
    """
    records = [_record('a.lock.yml', items=0, tokens=100), _record('a.lock.yml', items=1, tokens=900)]

    (summary,) = summarize(records)

    assert summary.output_tokens == 1000
    assert summary.zero_output_rate == 0.5
    assert summary.wasted_tokens == 100  # not 500


def test_summarize_orders_by_spend():
    records = [_record('cheap.lock.yml', tokens=10), _record('costly.lock.yml', tokens=999)]

    assert [s.workflow for s in summarize(records)] == ['costly.lock.yml', 'cheap.lock.yml']


def test_wasted_tokens_is_zero_without_agent_runs():
    assert WorkflowSummary('w.lock.yml', total_runs=4).wasted_tokens == 0


# --- alerts -------------------------------------------------------------------


def test_alerts_flag_a_scheduled_workflow_whose_agent_never_starts():
    """The `ui-security-review` failure mode: green on every run, doing nothing.

    A scheduled run has no path filter to legitimately skip on, so this is a broken
    job graph rather than a design choice.
    """
    records = [
        RunRecord('sweep.lock.yml', run_id=i, conclusion='success', agent_invoked=False, event='schedule')
        for i in range(20)
    ]

    (alert,) = detect_alerts(summarize(records))

    assert 'the agent never started' in alert
    assert 'reports success' in alert


def test_alerts_still_fire_when_a_pr_run_starts_the_agent():
    """A PR run that starts the agent must not mask scheduled runs that never do.

    The two counts have to be compared like with like, or one incidental PR review
    hides exactly the broken scheduled job graph this alert exists to catch.
    """
    records = [
        RunRecord('sweep.lock.yml', run_id=i, conclusion='success', agent_invoked=False, event='schedule')
        for i in range(6)
    ] + [_record('sweep.lock.yml', items=1)]

    assert any('the agent never started' in alert for alert in detect_alerts(summarize(records)))


def test_alerts_ignore_a_manual_dispatch_that_skips_by_design():
    """A `workflow_dispatch` run can legitimately gate the agent on its inputs."""
    records = [
        RunRecord('w.lock.yml', run_id=i, conclusion='success', agent_invoked=False, event='workflow_dispatch')
        for i in range(20)
    ]

    assert detect_alerts(summarize(records)) == []


def test_alerts_ignore_a_pr_workflow_that_skips_by_design():
    """`ui-security-review` only reviews UI-touching PRs; skipping the rest is correct.

    The static guard covers the broken-job-graph case at review time, so alerting here
    would be weekly false noise.
    """
    records = [
        RunRecord('ui.lock.yml', run_id=i, conclusion='success', agent_invoked=False, event='pull_request')
        for i in range(20)
    ]

    assert detect_alerts(summarize(records)) == []


def test_alerts_flag_a_high_zero_output_rate():
    records = [_record('r.lock.yml', items=0) for _ in range(7)] + [_record('r.lock.yml', items=1) for _ in range(3)]

    alerts = detect_alerts(summarize(records))

    assert any('70%' in alert and 'produced no output' in alert for alert in alerts)


def test_alerts_stay_quiet_below_the_sample_threshold():
    """Two bad runs out of two is noise, not a regression."""
    assert detect_alerts(summarize([_record('r.lock.yml', items=0) for _ in range(2)])) == []


def test_alerts_stay_quiet_for_a_healthy_workflow():
    records = [_record('r.lock.yml', items=1) for _ in range(10)]

    assert detect_alerts(summarize(records)) == []


def test_alerts_flag_a_wholly_failing_workflow():
    """The `roundtrip-sweep` failure mode: daily, failing, filing nothing."""
    records = [_record('rt.lock.yml', conclusion='failure', items=1) for _ in range(6)]

    assert any('all 6 runs failed' in alert for alert in detect_alerts(summarize(records)))


def test_alerts_flag_rate_limited_retries():
    """Each whole-run retry is a full re-spend, so surface them even when runs succeed."""
    records = [_record('r.lock.yml', items=1, retries=4, rate_limited=True) for _ in range(3)]

    assert any('rate limits' in alert and 'full re-spend' in alert for alert in detect_alerts(summarize(records)))


# --- rendering ----------------------------------------------------------------


def test_format_report_leads_with_alerts_and_totals():
    records = [_record('r.lock.yml', items=0, tokens=1000) for _ in range(6)]

    report = format_report(summarize(records), days=7, sampled=6, total=6)

    assert 'Needs attention' in report
    assert 'Agentic workflow spend — last 7d' in report
    assert '6,000' in report


def test_format_report_discloses_partial_sampling():
    """Never imply full coverage: not every run yields a readable artifact."""
    report = format_report(summarize([_record('r.lock.yml')]), days=7, sampled=1, total=50)

    assert 'Measured 1 of 50 runs' in report


def test_format_report_omits_the_sampling_note_at_full_coverage():
    report = format_report(summarize([_record('r.lock.yml')]), days=7, sampled=1, total=1)

    assert 'Measured' not in report


def test_format_report_handles_a_window_with_no_spend():
    report = format_report([], days=7, sampled=0, total=0)

    assert '*0* output tokens' in report


def test_build_slack_payload_carries_the_text_in_both_fields():
    payload = build_slack_payload('hello')

    assert payload['text'] == 'hello'
    assert payload['blocks'][0]['text']['text'] == 'hello'


def test_build_slack_payload_chunks_past_slacks_section_cap():
    """Slack rejects the whole delivery when one section exceeds 3,000 characters."""
    text = '\n'.join(f'• line {i} ' + 'x' * 80 for i in range(100))
    assert len(text) > SLACK_SECTION_LIMIT

    payload = build_slack_payload(text)

    assert len(payload['blocks']) > 1
    assert all(len(block['text']['text']) <= SLACK_SECTION_LIMIT for block in payload['blocks'])
    assert payload['text'] == text, 'the fallback text stays whole'


def test_split_for_slack_hard_wraps_an_unsplittable_line():
    """A single line over the cap has no newline to break on, so it must be wrapped."""
    chunks = _split_for_slack('y' * 250, limit=100)

    assert [len(chunk) for chunk in chunks] == [100, 100, 50]


def test_split_for_slack_keeps_whole_lines_together_when_it_can():
    chunks = _split_for_slack('aaa\nbbb\nccc', limit=8)

    assert chunks == ['aaa\nbbb', 'ccc']


# --- collect_run: an unreadable artifact is not a missing agent -----------------


class _FakeClient(GitHubClient):
    """A `GitHubClient` serving canned artifact listings and archives."""

    def __init__(self, artifacts: list[dict[str, Any]], archive: IO[bytes] | Exception) -> None:
        super().__init__('owner/repo', 'token')
        self._artifacts = artifacts
        self._archive = archive

    def get_json(self, path: str) -> dict[str, Any]:
        return {'artifacts': self._artifacts}

    def download(self, url: str) -> IO[bytes]:
        if isinstance(self._archive, Exception):
            raise self._archive
        return self._archive


_RUN = {'id': 1, 'conclusion': 'success', 'event': 'schedule'}


def test_collect_run_treats_a_missing_artifact_as_agent_never_started():
    client = _FakeClient([{'name': 'activation', 'expired': False}], io.BytesIO())

    record = collect_run(client, 'w.lock.yml', _RUN)

    assert record.agent_invoked is False


class _UnreachableListingClient(_FakeClient):
    """Fails the artifact *listing*, before any download is attempted."""

    def get_json(self, path: str) -> dict[str, Any]:
        raise urllib.error.HTTPError('https://api.github.com', 502, 'Bad Gateway', HTTPMessage(), None)


def test_collect_run_survives_an_artifact_listing_failure():
    """A transient 502 on the listing must not abort the whole report.

    Nor may it read as `agent_invoked=False`: an unanswered listing is no evidence the
    agent skipped, and treating it as such would fire a false broken-job-graph alert.
    """
    record = collect_run(_UnreachableListingClient([], io.BytesIO()), 'w.lock.yml', _RUN)

    assert (record.agent_invoked, record.artifact_read) == (True, False)


def test_collect_run_treats_an_expired_artifact_as_unmeasured_not_missing():
    """The agent demonstrably ran; only the evidence aged out.

    Reporting this as `agent_invoked=False` would fire a false "agent never started"
    alert whenever the window exceeds artifact retention.
    """
    client = _FakeClient([{'name': 'agent', 'expired': True}], io.BytesIO())

    record = collect_run(client, 'w.lock.yml', _RUN)

    assert (record.agent_invoked, record.artifact_read) == (True, False)


def test_collect_run_survives_a_corrupt_artifact():
    """One bad zip must not abort the whole report."""
    client = _FakeClient([{'name': 'agent', 'expired': False, 'archive_download_url': 'u'}], io.BytesIO(b'not a zip'))

    record = collect_run(client, 'w.lock.yml', _RUN)

    assert (record.agent_invoked, record.artifact_read) == (True, False)


def test_collect_run_records_an_oversized_artifact_member_as_unmeasured():
    """One runaway log must not take the whole report down with it.

    Unmeasured rather than partially scanned: retries counted off a prefix would print
    as a measurement the report never made.
    """
    archive = _artifact(usage={'output_tokens': 700}, items=[], log='x' * (LOG_READ_LIMIT + 1))
    client = _FakeClient([{'name': 'agent', 'expired': False, 'archive_download_url': 'u'}], archive)

    record = collect_run(client, 'w.lock.yml', _RUN)

    assert (record.agent_invoked, record.artifact_read) == (True, False)


def test_collect_run_keeps_known_spend_when_the_output_file_is_missing():
    """A missing `agent_output.json` must not discard the tokens we do know about."""
    archive = _artifact(usage={'output_tokens': 700})
    client = _FakeClient([{'name': 'agent', 'expired': False, 'archive_download_url': 'u'}], archive)

    record = collect_run(client, 'w.lock.yml', _RUN)

    assert (record.artifact_read, record.output_tokens, record.item_count) == (True, 700, None)

    (summary,) = summarize([record])
    assert summary.output_tokens == 700, 'known spend is still counted'
    assert summary.output_measured_runs == 0, 'but the run cannot be judged for waste'
    assert summary.zero_output_runs == 0


def test_summarize_does_not_treat_a_missing_usage_file_as_a_free_run():
    """Output known, spend unknown: the run is judged for waste but adds no tokens."""
    record = RunRecord('w.lock.yml', run_id=1, conclusion='success', agent_invoked=True, item_count=0)

    (summary,) = summarize([record])

    assert (summary.spend_measured_runs, summary.output_measured_runs) == (0, 1)
    assert summary.zero_output_runs == 1
    assert summary.output_tokens == 0


def test_collect_run_measures_a_complete_artifact():
    archive = _artifact(usage={'output_tokens': 500}, items=[])
    client = _FakeClient([{'name': 'agent', 'expired': False, 'archive_download_url': 'u'}], archive)

    record = collect_run(client, 'w.lock.yml', _RUN)

    assert (record.agent_invoked, record.artifact_read, record.output_tokens, record.item_count) == (True, True, 500, 0)


def test_sampled_count_excludes_runs_that_yielded_no_numbers():
    """A readable but empty artifact must not inflate the coverage figure."""
    empty = RunRecord('w.lock.yml', run_id=1, conclusion='success', agent_invoked=True)
    useful = _record('w.lock.yml', items=1, tokens=10)

    assert empty.contributed_measurement is False
    assert useful.contributed_measurement is True

    report = format_report(summarize([empty, useful]), days=7, sampled=1, total=2)
    assert 'Measured 1 of 2 runs' in report


# --- pagination and alert arithmetic ------------------------------------------


class _PagingClient(GitHubClient):
    """Serves paged run listings and a single agent artifact per run."""

    def __init__(self, total_runs: int) -> None:
        super().__init__('owner/repo', 'token')
        self.total_runs = total_runs
        self.artifact_queries: list[str] = []

    def get_json(self, path: str) -> dict[str, Any]:
        if '/runs?' in path:
            page = int(path.split('page=')[-1])
            start = (page - 1) * 100
            batch = [
                {'id': i, 'conclusion': 'success', 'event': 'schedule'}
                for i in range(start, min(start + 100, self.total_runs))
            ]
            return {'workflow_runs': batch}
        self.artifact_queries.append(path)
        return {'artifacts': []}

    def download(self, url: str) -> IO[bytes]:
        return io.BytesIO()


def test_gather_pages_past_the_100_run_api_cap():
    """A busy workflow sees ~500 runs a week; one page would silently measure 100."""
    client = _PagingClient(total_runs=250)

    records, truncated = gather(client, ['w.lock.yml'], days=7, per_workflow_limit=500)

    assert len(records) == 250
    assert truncated == {}


def test_runs_in_window_does_not_claim_truncation_at_exactly_the_limit():
    """A window that fits the limit exactly is complete, not capped.

    Flagging it would attach an undercount caveat — and a `*` on the row — to a number
    that is in fact the whole total.
    """
    client = _PagingClient(total_runs=100)

    runs, more = _runs_in_window(client, 'w.lock.yml', since='2026-07-01T00:00:00Z', limit=100)

    assert (len(runs), more) == (100, False)


def test_gather_reports_when_the_limit_truncates():
    """Never silently cap: the report has to say the number is an undercount."""
    client = _PagingClient(total_runs=250)

    records, truncated = gather(client, ['w.lock.yml'], days=7, per_workflow_limit=100)

    assert len(records) == 100
    assert truncated == {'w.lock.yml': 100}

    report = format_report(summarize(records), days=7, sampled=0, total=100, truncated=truncated)
    assert 'Run list truncated' in report and 'undercount' in report


@pytest.mark.parametrize(('total_runs', 'expected'), [(99, False), (100, False), (101, True), (150, True)])
def test_gather_decides_truncation_from_what_it_actually_saw(total_runs: int, expected: bool):
    """Both sides of the cap are lies the report would print.

    Stopping the moment the cap is reached cannot distinguish a window holding exactly
    `limit` runs from one holding more. Calling it truncated attaches an undercount
    caveat to a complete measurement; calling it complete presents a sample as a total —
    and `150` runs capped to `100` is exactly that second case.
    """
    client = _PagingClient(total_runs=total_runs)

    records, truncated = gather(client, ['w.lock.yml'], days=7, per_workflow_limit=100)

    assert len(records) == min(total_runs, 100)
    assert bool(truncated) is expected


def test_format_report_marks_a_sampled_workflows_own_row():
    """A footnote caveats the report; the `*` caveats the number the reader is looking at."""
    records = [_record('busy.lock.yml', tokens=10), _record('quiet.lock.yml', tokens=5)]

    report = format_report(summarize(records), days=7, sampled=2, total=2, truncated={'busy.lock.yml': 500})

    assert 'busy.lock.yml*' in report
    assert 'quiet.lock.yml*' not in report


def test_collect_run_requests_the_agent_artifact_by_name():
    """Unfiltered, the default 30-artifact page can hide it and read as 'never started'."""
    client = _PagingClient(total_runs=1)

    collect_run(client, 'w.lock.yml', {'id': 1, 'conclusion': 'success', 'event': 'schedule'})

    assert any('name=agent' in q and 'per_page=100' in q for q in client.artifact_queries)


def test_rate_limit_alert_denominator_counts_inspected_logs():
    """`rate_limited` comes from the log, so a missing usage file must not zero the denominator."""
    records = [
        RunRecord('w.lock.yml', run_id=i, conclusion='success', agent_invoked=True, item_count=1, rate_limited=True)
        for i in range(3)
    ]

    (alert,) = detect_alerts(summarize(records))

    assert '3/3 runs hit provider rate limits' in alert


def test_parse_agent_artifact_treats_null_output_tokens_as_unknown():
    """A present-but-null field is unknown, not free — coercing to 0 understates spend."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as bundle:
        bundle.writestr(AGENT_USAGE_FILE, json.dumps({'output_tokens': None}))
        bundle.writestr(AGENT_OUTPUT_FILE, json.dumps({'items': []}))

    metrics = parse_agent_artifact(buffer)

    assert metrics.output_tokens is None
    (summary,) = summarize([RunRecord('w.lock.yml', 1, 'success', True, item_count=0)])
    assert summary.spend_measured_runs == 0


def test_strip_auth_drops_the_bearer_on_an_https_to_http_downgrade():
    """Same host, weaker scheme, still cross-origin — the token must not go in plaintext."""
    import urllib.request

    from agent_spend_report import _StripAuthOnRedirect  # pyright: ignore[reportPrivateUsage]

    req = urllib.request.Request('https://example.com/a')
    req.add_header('Authorization', 'Bearer secret')
    handler = _StripAuthOnRedirect()

    downgraded = handler.redirect_request(req, io.BytesIO(), 302, 'Found', HTTPMessage(), 'http://example.com/a')
    assert downgraded is not None and downgraded.get_header('Authorization') is None

    same_origin = handler.redirect_request(req, io.BytesIO(), 302, 'Found', HTTPMessage(), 'https://example.com/b')
    assert same_origin is not None and same_origin.get_header('Authorization') == 'Bearer secret'


class _CannedResponse(io.BytesIO):
    """A `_open` result carrying headers, as `urlopen` returns."""

    def __init__(self, payload: bytes, declared: str | None) -> None:
        super().__init__(payload)
        self.headers = HTTPMessage()
        if declared is not None:
            self.headers['Content-Length'] = declared

    def __enter__(self) -> _CannedResponse:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class _CannedClient(GitHubClient):
    def __init__(self, payload: bytes, declared: str | None = None) -> None:
        super().__init__('owner/repo', 'token')
        self._payload = payload
        self._declared = declared

    def _open(self, url: str) -> IO[bytes]:
        return _CannedResponse(self._payload, self._declared)


def test_download_stops_copying_an_oversized_artifact():
    """The per-member caps only apply once the zip is on disk.

    An incompressible artifact would otherwise cost the runner's disk and the job's
    `timeout-minutes` before `zipfile` opened it, so the copy loop is the real bound —
    `Content-Length` is absent under chunked encoding and is the server's claim anyway.
    """
    client = _CannedClient(b'x' * (ARCHIVE_DOWNLOAD_LIMIT + 1))

    with pytest.raises(ValueError, match='download limit'):
        client.download('https://example.com/a')


def test_download_rejects_an_oversized_content_length_before_transferring():
    client = _CannedClient(b'x', declared=str(ARCHIVE_DOWNLOAD_LIMIT + 1))

    with pytest.raises(ValueError, match='over the download limit'):
        client.download('https://example.com/a')


def test_download_returns_an_artifact_within_the_limit_at_offset_zero():
    client = _CannedClient(b'payload', declared='7')

    with client.download('https://example.com/a') as spooled:
        assert spooled.read() == b'payload'


def test_report_imports_with_stdlib_only():
    # Production invokes the script with the runner's bare `python` (no venv,
    # no third-party packages); `-S` blocks site-packages to reproduce that.
    result = subprocess.run(
        [sys.executable, '-S', '-c', 'import agent_spend_report'],
        env={**os.environ, 'PYTHONPATH': str(Path(__file__).parent)},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
