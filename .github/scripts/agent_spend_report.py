"""Weekly waste report for the `gh-aw` agentic workflows, delivered to Slack.

The static guard in `agentic_workflow_guard.py` catches known anti-patterns at
review time. This catches the other half: quantitative drift, and workflows that
die quietly. Both failure modes from #6766 were invisible on the Actions tab —
`ui-security-review` reported green for a month while never running its agent,
and `pr-review` spent 72% of its budget on runs that produced nothing.

The signal lives in each run's `agent` artifact, not in the OTel spans:

- `agent_usage.json` — token counts
- `agent_output.json` — `{"items": []}` means the run delivered nothing
- `agent-stdio.log` — whole-run retries, each one a full re-spend

Output goes to Slack rather than a public issue: it is operational cost data,
and a public comment on every regression would be noise.

Artifacts expire after ~7 days, so this only ever sees a recent window and says
so explicitly in the report rather than implying full coverage.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, cast
from urllib.parse import urlparse

API_ROOT = 'https://api.github.com'
AGENT_ARTIFACT = 'agent'
RETRY_MARKER = re.compile(r'attempt \d+ failed')
RATE_LIMIT_MARKER = '429 Too Many Requests'
# Slack rejects the whole delivery if any `section` block's text exceeds this.
SLACK_SECTION_LIMIT = 3000

# A workflow whose agent produces nothing this often is malfunctioning, not unlucky:
# the pre-fix `pr-review` sat at 72% and the post-fix baseline is ~47%.
ZERO_OUTPUT_ALERT_RATE = 0.5
# Below this many agent runs the rate is too noisy to alert on.
MIN_RUNS_FOR_RATE_ALERT = 5
# A scheduled run has no path filter to legitimately skip on, so an agent that never
# starts is a broken job graph. Every other trigger can skip by design — PR workflows
# (`ui-security-review` only reviews UI-touching PRs) and manual dispatches gated on
# their inputs — so alerting on those would be false noise. The static guard in
# `agentic_workflow_guard.py` covers that failure mode at review time instead.
UNCONDITIONAL_EVENTS = frozenset({'schedule'})


@dataclass(frozen=True)
class RunRecord:
    """One workflow run's measured cost and delivered output."""

    workflow: str
    run_id: int
    conclusion: str
    agent_invoked: bool
    event: str = ''
    measured: bool = True
    output_tokens: int = 0
    item_count: int | None = None
    retries: int = 0
    rate_limited: bool = False


@dataclass
class WorkflowSummary:
    """Aggregated cost and waste for one workflow over the sampled window."""

    workflow: str
    total_runs: int = 0
    agent_runs: int = 0
    zero_output_runs: int = 0
    zero_output_tokens: int = 0
    output_tokens: int = 0
    retries: int = 0
    rate_limited_runs: int = 0
    unconditional_runs: int = 0
    unconditional_agent_runs: int = 0
    measured_runs: int = 0
    unmeasured_runs: int = 0
    conclusions: Counter[str] = field(default_factory=lambda: Counter())

    @property
    def zero_output_rate(self) -> float:
        return self.zero_output_runs / self.measured_runs if self.measured_runs else 0.0

    @property
    def wasted_tokens(self) -> int:
        """Output tokens actually spent by runs that delivered nothing.

        Summed from those runs rather than derived from the rate: apportioning total
        spend by the zero-output *rate* would bill successful runs' tokens as waste,
        and per-run cost varies by more than an order of magnitude.
        """
        return self.zero_output_tokens


class _StripAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """Drop `Authorization` when a redirect crosses hosts.

    Artifact downloads redirect from `api.github.com` to Azure blob storage, which
    rejects a forwarded GitHub bearer token with `401 Server failed to authenticate`.
    """

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and urlparse(newurl).netloc != urlparse(req.full_url).netloc:
            redirected.remove_header('Authorization')
        return redirected


class GitHubClient:
    """Minimal GitHub REST client over `urllib` (no third-party deps in CI)."""

    def __init__(self, repo: str, token: str) -> None:
        self.repo = repo
        self.token = token
        self._opener = urllib.request.build_opener(_StripAuthOnRedirect())

    def _request(self, url: str) -> bytes:
        request = urllib.request.Request(url)
        request.add_header('Authorization', f'Bearer {self.token}')
        request.add_header('Accept', 'application/vnd.github+json')
        with self._opener.open(request, timeout=60) as response:
            return response.read()

    def get_json(self, path: str) -> dict[str, Any]:
        return _as_mapping(json.loads(self._request(f'{API_ROOT}/repos/{self.repo}/{path}')))

    def get_zip(self, url: str) -> bytes:
        return self._request(url)


def _as_mapping(value: object) -> dict[str, Any]:
    """Coerce a parsed-JSON value to a string-keyed mapping."""
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in cast(dict[Any, Any], value).items()}


def _as_list(value: object) -> list[Any]:
    return cast(list[Any], value) if isinstance(value, list) else []


@dataclass(frozen=True)
class ArtifactMetrics:
    """What one `agent` artifact reveals about its run.

    `item_count` is `None` when `agent_output.json` is absent: that means the delivered
    count is *unknown*, which is not the same as a present-but-empty `{"items": []}`.
    Conflating them would count a truncated upload as a wasted run and fire a false alert.
    """

    output_tokens: int = 0
    item_count: int | None = None
    retries: int = 0
    rate_limited: bool = False


def parse_agent_artifact(archive: bytes) -> ArtifactMetrics:
    """Extract cost and delivery signals from an agent artifact zip."""
    output_tokens = retries = 0
    item_count: int | None = None
    rate_limited = False
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        names = set(bundle.namelist())
        if 'agent_usage.json' in names:
            usage = _as_mapping(json.loads(bundle.read('agent_usage.json')))
            output_tokens = int(usage.get('output_tokens') or 0)
        if 'agent_output.json' in names:
            output = _as_mapping(json.loads(bundle.read('agent_output.json')))
            item_count = len(_as_list(output.get('items')))
        if 'agent-stdio.log' in names:
            log = bundle.read('agent-stdio.log').decode('utf-8', errors='ignore')
            retries = len(RETRY_MARKER.findall(log))
            rate_limited = RATE_LIMIT_MARKER in log
    return ArtifactMetrics(output_tokens, item_count, retries, rate_limited)


def collect_run(client: GitHubClient, workflow: str, run: dict[str, Any]) -> RunRecord:
    """Measure one run.

    Only a genuinely absent artifact means "the agent never started" — the signal for a
    workflow whose job graph silently skips. An artifact that exists but cannot be read
    (expired, undownloadable, corrupt) proves the agent *did* run, so it is recorded as
    unmeasured. Conflating the two would fire a false broken-job-graph alert.
    """
    run_id = int(run.get('id') or 0)
    conclusion = str(run.get('conclusion') or 'in_progress')
    event = str(run.get('event') or '')

    listing = client.get_json(f'actions/runs/{run_id}/artifacts')
    artifacts = [_as_mapping(entry) for entry in _as_list(listing.get('artifacts'))]
    agent = next((a for a in artifacts if a.get('name') == AGENT_ARTIFACT), None)
    if agent is None:
        return RunRecord(workflow, run_id, conclusion, agent_invoked=False, event=event)
    if agent.get('expired'):
        return RunRecord(workflow, run_id, conclusion, agent_invoked=True, event=event, measured=False)

    # One unreadable artifact must not abort the whole report, so parsing is guarded too.
    try:
        metrics = parse_agent_artifact(client.get_zip(str(agent['archive_download_url'])))
    except (urllib.error.URLError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        print(f'warning: could not process agent artifact for run {run_id}: {exc}', file=sys.stderr)
        return RunRecord(workflow, run_id, conclusion, agent_invoked=True, event=event, measured=False)

    return RunRecord(
        workflow,
        run_id,
        conclusion,
        agent_invoked=True,
        event=event,
        measured=metrics.item_count is not None,
        output_tokens=metrics.output_tokens,
        item_count=metrics.item_count,
        retries=metrics.retries,
        rate_limited=metrics.rate_limited,
    )


def summarize(records: list[RunRecord]) -> list[WorkflowSummary]:
    """Aggregate per-workflow, sorted by output tokens spent (descending)."""
    summaries: dict[str, WorkflowSummary] = {}
    for record in records:
        summary = summaries.setdefault(record.workflow, WorkflowSummary(record.workflow))
        summary.total_runs += 1
        summary.conclusions[record.conclusion] += 1
        unconditional = record.event in UNCONDITIONAL_EVENTS
        summary.unconditional_runs += int(unconditional)
        if not record.agent_invoked:
            continue
        summary.agent_runs += 1
        summary.unconditional_agent_runs += int(unconditional)
        if not record.measured:
            summary.unmeasured_runs += 1
            continue
        summary.measured_runs += 1
        summary.output_tokens += record.output_tokens
        summary.retries += record.retries
        summary.rate_limited_runs += int(record.rate_limited)
        if record.item_count == 0:
            summary.zero_output_runs += 1
            summary.zero_output_tokens += record.output_tokens
    return sorted(summaries.values(), key=lambda s: -s.output_tokens)


def detect_alerts(summaries: list[WorkflowSummary]) -> list[str]:
    """Return the regression signals worth waking someone for."""
    alerts: list[str] = []
    for summary in summaries:
        name = summary.workflow
        if summary.unconditional_runs >= MIN_RUNS_FOR_RATE_ALERT and not summary.unconditional_agent_runs:
            alerts.append(
                f'*{name}*: {summary.unconditional_runs} scheduled runs but the agent never started. '
                'A job skipped by `if:` reports success, so this shows green while doing nothing.'
            )
            continue
        if summary.measured_runs >= MIN_RUNS_FOR_RATE_ALERT and summary.zero_output_rate > ZERO_OUTPUT_ALERT_RATE:
            alerts.append(
                f'*{name}*: {summary.zero_output_runs}/{summary.measured_runs} measured runs '
                f'({summary.zero_output_rate:.0%}) produced no output, '
                f'~{summary.wasted_tokens:,} output tokens wasted.'
            )
        failures = summary.conclusions.get('failure', 0)
        if summary.total_runs >= MIN_RUNS_FOR_RATE_ALERT and failures == summary.total_runs:
            alerts.append(f'*{name}*: all {summary.total_runs} runs failed.')
        if summary.rate_limited_runs:
            alerts.append(
                f'*{name}*: {summary.rate_limited_runs}/{summary.measured_runs} runs hit provider rate limits '
                f'({summary.retries} whole-run retries, each a full re-spend).'
            )
    return alerts


def format_report(summaries: list[WorkflowSummary], days: int, sampled: int, total: int) -> str:
    """Render the Slack message body as mrkdwn."""
    lines = [f'*Agentic workflow spend — last {days}d*', '']

    alerts = detect_alerts(summaries)
    if alerts:
        lines.append(':rotating_light: *Needs attention*')
        lines += [f'• {alert}' for alert in alerts]
        lines.append('')

    lines.append('```')
    lines.append(f'{"workflow":<34}{"runs":>6}{"agent":>7}{"empty":>7}{"out tok":>10}')
    for summary in summaries:
        empty = f'{summary.zero_output_rate:.0%}' if summary.measured_runs else '-'
        lines.append(
            f'{summary.workflow[:33]:<34}{summary.total_runs:>6}{summary.agent_runs:>7}'
            f'{empty:>7}{summary.output_tokens:>10,}'
        )
    lines.append('```')

    total_out = sum(s.output_tokens for s in summaries)
    wasted = sum(s.wasted_tokens for s in summaries)
    share = f' ({wasted / total_out:.0%})' if total_out else ''
    lines.append(f'*{total_out:,}* output tokens, *~{wasted:,}*{share} on runs that delivered nothing.')

    if sampled < total:
        lines.append(
            f'_Measured {sampled} of {total} runs; the rest had no agent artifact '
            f'(expired after ~7d, or the agent never started)._'
        )
    return '\n'.join(lines)


def _split_for_slack(text: str, limit: int = SLACK_SECTION_LIMIT) -> list[str]:
    """Split `text` into chunks under Slack's per-section character cap, on line breaks."""
    chunks: list[str] = []
    current = ''
    for line in text.split('\n'):
        # A single line over the cap cannot be split further on newlines; hard-wrap it.
        while len(line) > limit:
            chunks.append(line[:limit])
            line = line[limit:]
        candidate = f'{current}\n{line}' if current else line
        if len(candidate) > limit:
            chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def build_slack_payload(text: str) -> dict[str, Any]:
    """Wrap the report in an incoming-webhook payload.

    Slack caps a `section` block's text at 3,000 characters and rejects the whole
    delivery when one exceeds it, so the report is chunked across sections. The
    top-level `text` stays whole as the notification fallback.
    """
    return {
        'text': text,
        'blocks': [{'type': 'section', 'text': {'type': 'mrkdwn', 'text': chunk}} for chunk in _split_for_slack(text)],
    }


def gather(client: GitHubClient, workflows: list[str], days: int, per_workflow_limit: int) -> list[RunRecord]:
    """Collect run records for each workflow within the window."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime('%Y-%m-%dT%H:%M:%SZ')
    records: list[RunRecord] = []
    for workflow in workflows:
        payload = client.get_json(f'actions/workflows/{workflow}/runs?created=>{since}&per_page={per_workflow_limit}')
        for run in _as_list(payload.get('workflow_runs')):
            records.append(collect_run(client, workflow, _as_mapping(run)))
    return records


def main(argv: list[str] | None = None) -> int:
    """Emit the Slack payload as a GitHub Actions output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--days', type=int, default=7)
    parser.add_argument('--per-workflow-limit', type=int, default=100)
    args = parser.parse_args(argv)

    repo = os.environ.get('GITHUB_REPOSITORY', '')
    token = os.environ.get('GITHUB_TOKEN', '')
    if not repo or not token:
        print('GITHUB_REPOSITORY and GITHUB_TOKEN are required', file=sys.stderr)
        return 1

    client = GitHubClient(repo, token)
    workflows = [
        path.removeprefix('.github/workflows/')
        for path in (
            str(_as_mapping(entry).get('path', ''))
            for entry in _as_list(client.get_json('actions/workflows?per_page=100').get('workflows'))
        )
        if path.endswith('.lock.yml')
    ]

    records = gather(client, workflows, args.days, args.per_workflow_limit)
    summaries = summarize(records)
    report = format_report(
        summaries,
        args.days,
        sampled=sum(1 for r in records if r.agent_invoked and r.measured),
        total=len(records),
    )
    print(report)

    if output_path := os.environ.get('GITHUB_OUTPUT'):
        payload = json.dumps(build_slack_payload(report), separators=(',', ':'))
        with open(output_path, 'a', encoding='utf-8') as handle:
            handle.write(f'slack_payload={payload}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
