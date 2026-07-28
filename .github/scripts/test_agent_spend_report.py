"""Tests for the agentic workflow spend report.

The alert cases are calibrated against the real #6766 measurements: a workflow
that never starts its agent, and one whose runs mostly deliver nothing.
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from agent_spend_report import (
    RunRecord,
    WorkflowSummary,
    build_slack_payload,
    detect_alerts,
    format_report,
    parse_agent_artifact,
    summarize,
)


def _artifact(usage: dict[str, int] | None = None, items: list[str] | None = None, log: str = '') -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as bundle:
        if usage is not None:
            bundle.writestr('agent_usage.json', json.dumps(usage))
        if items is not None:
            bundle.writestr('agent_output.json', json.dumps({'items': items}))
        if log:
            bundle.writestr('agent-stdio.log', log)
    return buffer.getvalue()


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
        log='attempt 1 failed: exitCode=1\nattempt 2 failed: exitCode=1\n429 Too Many Requests\n',
    )

    assert parse_agent_artifact(archive) == (18375, 1, 2, True)


def test_parse_agent_artifact_treats_empty_items_as_no_output():
    """`{"items": []}` is the signature of a run that cost full price and delivered nothing."""
    assert parse_agent_artifact(_artifact(usage={'output_tokens': 81924}, items=[])) == (81924, 0, 0, False)


def test_parse_agent_artifact_tolerates_a_partial_bundle():
    """A killed run may upload the zip before every file lands."""
    assert parse_agent_artifact(_artifact()) == (0, 0, 0, False)


def test_parse_agent_artifact_survives_the_v0834_usage_schema():
    """gh-aw v0.83.4 dropped `effective_tokens` for `ai_credits`; `output_tokens` stayed."""
    archive = _artifact(usage={'output_tokens': 6979, 'ai_credits': 0}, items=['review'])

    assert parse_agent_artifact(archive)[0] == 6979


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
    assert summary.wasted_tokens == 200


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
    """Never imply full coverage: artifacts expire after ~7 days."""
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
