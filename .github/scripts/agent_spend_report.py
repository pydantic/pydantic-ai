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

Coverage is bounded by the `--days` window and the per-workflow run cap, not by
artifact retention — the `agent` upload sets no `retention-days`, so it inherits the
repository default. The report states what it actually measured rather than implying
full coverage.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from http.client import HTTPMessage
from typing import IO, Any, cast
from urllib.parse import urlparse

API_ROOT = 'https://api.github.com'
# The runs endpoint caps `per_page` at 100.
API_PAGE_SIZE = 100
AGENT_ARTIFACT = 'agent'
AGENT_USAGE_FILE = 'agent_usage.json'
AGENT_OUTPUT_FILE = 'agent_output.json'
AGENT_STDIO_LOG = 'agent-stdio.log'
# A runaway run's log is unbounded, and decompressing one whole into memory would kill
# the report job — a monitor that dies on the week it has the most to say. Not a
# hypothetical: run 30511919760's `agent` artifact is 149,253,753 bytes *compressed*,
# ~750x a routine one. These caps are sized off a measured routine artifact
# (`agent_usage.json` 156 B, `agent_output.json` 24 B, `agent-stdio.log` 241,221 B), so
# they only ever fire on the pathological case.
JSON_READ_LIMIT = 1024 * 1024
LOG_READ_LIMIT = 32 * 1024 * 1024
# The per-member caps above only apply once the archive is on disk, so the transfer
# itself needs its own bound: an incompressible artifact costs runner disk and job
# minutes before `zipfile` ever opens it. 512 MiB is ~3.4x the largest artifact
# observed (the 149 MB one above), so it leaves the real outliers measurable.
ARCHIVE_DOWNLOAD_LIMIT = 512 * 1024 * 1024
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
# gh-aw's harness logs `attempt N failed` on *any* non-zero exit, before it decides
# whether to retry, and most of its branches then decline to. Only the later
# `— will retry with <mode>` line means a whole run was actually re-spent.
RETRY_MARKER = re.compile(r'will retry with')
# Mirrors the detector in gh-aw's own harness (`actions/setup/js/claude_harness.cjs`),
# which is the source of truth for what a rate-limited run looks like.
# `429 Too Many Requests` is only the transport-style form; behind gh-aw's api-proxy
# these workflows emit the stream-json `"api_error_status": 429` instead, so matching
# that one substring meant the alert could never fire.
RATE_LIMIT_MARKER = re.compile(
    r'rate_limit_error|429 Too Many Requests|"api_error_status"\s*:\s*429|request rejected \(429\)|rate limit',
    re.IGNORECASE,
)
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
    artifact_read: bool = True
    output_tokens: int | None = None
    item_count: int | None = None
    retries: int = 0
    rate_limited: bool = False

    @property
    def contributed_measurement(self) -> bool:
        """Whether this run yielded any usable number.

        A readable but truncated artifact carrying neither file contributes nothing, so
        counting it as sampled would overstate the report's coverage.
        """
        return self.output_tokens is not None or self.item_count is not None


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
    spend_measured_runs: int = 0
    output_measured_runs: int = 0
    unread_artifact_runs: int = 0
    conclusions: Counter[str] = field(default_factory=lambda: Counter())

    @property
    def zero_output_rate(self) -> float:
        return self.zero_output_runs / self.output_measured_runs if self.output_measured_runs else 0.0

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
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None
        old, new = urlparse(req.full_url), urlparse(newurl)
        # Compare the full origin, not just the host: an https->http redirect on the
        # same host would otherwise keep the bearer and send it in plaintext.
        if (new.scheme, new.netloc) != (old.scheme, old.netloc):
            redirected.remove_header('Authorization')
        return redirected


class GitHubClient:
    """Minimal GitHub REST client over `urllib` (no third-party deps in CI)."""

    def __init__(self, repo: str, token: str) -> None:
        self.repo = repo
        self.token = token
        self._opener = urllib.request.build_opener(_StripAuthOnRedirect())

    def _open(self, url: str) -> IO[bytes]:
        request = urllib.request.Request(url)
        request.add_header('Authorization', f'Bearer {self.token}')
        request.add_header('Accept', 'application/vnd.github+json')
        return self._opener.open(request, timeout=60)

    def get_json(self, path: str) -> dict[str, Any]:
        with self._open(f'{API_ROOT}/repos/{self.repo}/{path}') as response:
            return _as_mapping(json.loads(response.read()))

    def download(self, url: str) -> IO[bytes]:
        """Stream an artifact to a temp file, seeked back to the start.

        An `agent` artifact reaches ~150 MB on the busiest workflows, and buffering that
        in memory alongside the files unpacked from it is more than the job can afford.
        `zipfile` only needs a seekable file, so spilling to disk costs nothing.

        The copy is bounded rather than run to EOF: an incompressible artifact would
        otherwise be paid for in full — runner disk, and the job's `timeout-minutes` —
        before the per-member caps ever get to look at it. `Content-Length` is only an
        early exit; the loop is what actually enforces the limit, since the header is
        the server's claim and may be absent under chunked encoding.
        """
        spool = tempfile.TemporaryFile()
        try:
            with self._open(url) as response:
                headers = getattr(response, 'headers', None)
                declared: str | None = headers.get('Content-Length') if isinstance(headers, HTTPMessage) else None
                if declared is not None and int(declared) > ARCHIVE_DOWNLOAD_LIMIT:
                    raise ValueError(f'artifact declares {declared} bytes, over the download limit')
                copied = 0
                while chunk := response.read(DOWNLOAD_CHUNK_SIZE):
                    copied += len(chunk)
                    if copied > ARCHIVE_DOWNLOAD_LIMIT:
                        raise ValueError(f'artifact exceeds the {ARCHIVE_DOWNLOAD_LIMIT}-byte download limit')
                    spool.write(chunk)
        except Exception:
            spool.close()
            raise
        spool.seek(0)
        return spool


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

    Both counts are independently optional, because a truncated upload can drop either
    file. `None` means *unknown*, which is never the same as zero: a missing
    `agent_output.json` is not an empty `{"items": []}`, and a missing `agent_usage.json`
    is not a free run. Conflating them would fire false alerts and understate spend.
    """

    output_tokens: int | None = None
    item_count: int | None = None
    retries: int = 0
    rate_limited: bool = False


def _read_capped(bundle: zipfile.ZipFile, name: str, limit: int) -> bytes:
    """Read one zip member, refusing to decompress past `limit`.

    Bounded through the member stream rather than checked against `ZipInfo.file_size`:
    that field is a declaration in the archive, while the read is the ground truth.
    """
    with bundle.open(name) as member:
        data = member.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f'{name} exceeds the {limit}-byte read limit')
    return data


def parse_agent_artifact(archive: IO[bytes]) -> ArtifactMetrics:
    """Extract cost and delivery signals from an agent artifact zip.

    Raises `ValueError` when a member is too large to read, which `collect_run` records
    as an unmeasured run — the same state as an unreadable artifact, and deliberately
    not a partial scan: counting retries off a truncated prefix would report a number
    that looks measured and is not.
    """
    retries = 0
    output_tokens: int | None = None
    item_count: int | None = None
    rate_limited = False
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        if AGENT_USAGE_FILE in names:
            usage = _as_mapping(json.loads(_read_capped(bundle, AGENT_USAGE_FILE, JSON_READ_LIMIT)))
            # A present-but-null `output_tokens` is unknown, not free: coercing it to 0
            # would count the run as spend-measured and understate the workflow total.
            raw_tokens = usage.get('output_tokens')
            output_tokens = int(raw_tokens) if raw_tokens is not None else None
        if AGENT_OUTPUT_FILE in names:
            output = _as_mapping(json.loads(_read_capped(bundle, AGENT_OUTPUT_FILE, JSON_READ_LIMIT)))
            item_count = len(_as_list(output.get('items')))
        if AGENT_STDIO_LOG in names:
            log = _read_capped(bundle, AGENT_STDIO_LOG, LOG_READ_LIMIT).decode('utf-8', errors='ignore')
            retries = len(RETRY_MARKER.findall(log))
            rate_limited = RATE_LIMIT_MARKER.search(log) is not None
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

    # One unreachable run must not abort the whole report, so the listing is guarded
    # alongside the download: a transient 502 there is not evidence of anything, and
    # recording it as `agent_invoked=False` would fire a false broken-job-graph alert.
    try:
        # `name=` filters server-side and `per_page` lifts the default 30, so a run with
        # many artifacts cannot push the agent one off the first page and read as
        # 'agent never started'.
        listing = client.get_json(f'actions/runs/{run_id}/artifacts?name={AGENT_ARTIFACT}&per_page={API_PAGE_SIZE}')
        artifacts = [_as_mapping(entry) for entry in _as_list(listing.get('artifacts'))]
        agent = next((a for a in artifacts if a.get('name') == AGENT_ARTIFACT), None)
        if agent is None:
            return RunRecord(workflow, run_id, conclusion, agent_invoked=False, event=event)
        if agent.get('expired'):
            return RunRecord(workflow, run_id, conclusion, agent_invoked=True, event=event, artifact_read=False)
        with client.download(str(agent['archive_download_url'])) as archive:
            metrics = parse_agent_artifact(archive)
    # `URLError` does not wrap a read timeout raised after `open()` returns, so `OSError`
    # is needed too — one slow download must not abort the whole report.
    except (urllib.error.URLError, OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        print(f'warning: could not process agent artifact for run {run_id}: {exc}', file=sys.stderr)
        return RunRecord(workflow, run_id, conclusion, agent_invoked=True, event=event, artifact_read=False)

    return RunRecord(
        workflow,
        run_id,
        conclusion,
        agent_invoked=True,
        event=event,
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
        if not record.artifact_read:
            summary.unread_artifact_runs += 1
            continue
        summary.retries += record.retries
        summary.rate_limited_runs += int(record.rate_limited)
        # Known spend counts even when the output file is missing, and a known output
        # count is usable even when the usage file is missing. Dropping either because
        # its sibling is absent would silently understate the report.
        if record.output_tokens is not None:
            summary.spend_measured_runs += 1
            summary.output_tokens += record.output_tokens
        if record.item_count is not None:
            summary.output_measured_runs += 1
            if record.item_count == 0:
                summary.zero_output_runs += 1
                summary.zero_output_tokens += record.output_tokens or 0
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
        if (
            summary.output_measured_runs >= MIN_RUNS_FOR_RATE_ALERT
            and summary.zero_output_rate > ZERO_OUTPUT_ALERT_RATE
        ):
            alerts.append(
                f'*{name}*: {summary.zero_output_runs}/{summary.output_measured_runs} measured runs '
                f'({summary.zero_output_rate:.0%}) produced no output, '
                f'~{summary.wasted_tokens:,} output tokens wasted.'
            )
        failures = summary.conclusions.get('failure', 0)
        if summary.total_runs >= MIN_RUNS_FOR_RATE_ALERT and failures == summary.total_runs:
            alerts.append(f'*{name}*: all {summary.total_runs} runs failed.')
        if summary.rate_limited_runs:
            # `rate_limited_runs` comes from the stdio log, so the denominator must be the
            # runs whose logs were read — not those that happened to carry a usage file.
            inspected = summary.agent_runs - summary.unread_artifact_runs
            alerts.append(
                f'*{name}*: {summary.rate_limited_runs}/{inspected} runs hit provider rate limits '
                f'({summary.retries} whole-run retries, each a full re-spend).'
            )
    return alerts


def format_report(
    summaries: list[WorkflowSummary],
    days: int,
    sampled: int,
    total: int,
    truncated: dict[str, int] | None = None,
) -> str:
    """Render the Slack message body as mrkdwn."""
    capped = truncated or {}
    lines = [f'*Agentic workflow spend — last {days}d*', '']

    alerts = detect_alerts(summaries)
    if alerts:
        lines.append(':rotating_light: *Needs attention*')
        lines += [f'• {alert}' for alert in alerts]
        lines.append('')

    lines.append('```')
    lines.append(f'{"workflow":<34}{"runs":>6}{"agent":>7}{"empty":>7}{"out tok":>10}')
    for summary in summaries:
        empty = f'{summary.zero_output_rate:.0%}' if summary.output_measured_runs else '-'
        # Mark the row itself, not only the footnote: an unmarked number reads as the
        # workflow's total when it is really a sample of it.
        name = f'{summary.workflow}*' if summary.workflow in capped else summary.workflow
        lines.append(
            f'{name[:33]:<34}{summary.total_runs:>6}{summary.agent_runs:>7}{empty:>7}{summary.output_tokens:>10,}'
        )
    lines.append('```')

    total_out = sum(s.output_tokens for s in summaries)
    wasted = sum(s.wasted_tokens for s in summaries)
    share = f' ({wasted / total_out:.0%})' if total_out else ''
    lines.append(f'*{total_out:,}* output tokens, *~{wasted:,}*{share} on runs that delivered nothing.')

    if capped:
        detail = ', '.join(f'{name} (capped at {cap})' for name, cap in capped.items())
        lines.append(
            f'_Run list truncated for: {detail} — rows marked `*` are sampled, so their spend is an undercount._'
        )
    if sampled < total:
        lines.append(
            f'_Measured {sampled} of {total} runs; the rest had no readable agent artifact '
            f'(expired, undownloadable, or the agent never started)._'
        )
    return '\n'.join(lines)


def _split_for_slack(text: str, limit: int = SLACK_SECTION_LIMIT) -> list[str]:
    """Split `text` into chunks under Slack's per-section character cap, on line breaks."""
    chunks: list[str] = []
    current = ''
    for line in text.split('\n'):
        # A single line over the cap cannot be split further on newlines, so hard-wrap it —
        # but flush what came before first, or its segments would jump the preceding lines.
        if len(line) > limit:
            if current:
                chunks.append(current)
                current = ''
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


def _runs_in_window(client: GitHubClient, workflow: str, since: str, limit: int) -> tuple[list[Any], bool]:
    """Return up to `limit` runs for `workflow`, and whether more were available.

    The runs endpoint caps `per_page` at 100, so a busy workflow — `CI Review` alone
    sees ~500 a week — needs paging or the report silently measures the first page and
    reports it as the whole window.
    """
    runs: list[Any] = []
    page = 1
    # Keep reading one page past `limit` before deciding, then answer from what was
    # actually seen. Stopping at `len(runs) >= limit` cannot tell a window that holds
    # exactly `limit` runs from one that holds more, and guessing either way is a lie
    # the report then prints: claim truncation and a complete measurement carries a
    # spurious undercount caveat; claim completeness and a sampled one is presented as
    # a total. The extra page costs one request per workflow.
    while len(runs) <= limit:
        payload = client.get_json(
            f'actions/workflows/{workflow}/runs?created=>{since}&per_page={API_PAGE_SIZE}&page={page}'
        )
        batch = _as_list(payload.get('workflow_runs'))
        runs.extend(batch)
        if len(batch) < API_PAGE_SIZE:
            break
        page += 1
    return runs[:limit], len(runs) > limit


def gather(
    client: GitHubClient, workflows: list[str], days: int, per_workflow_limit: int
) -> tuple[list[RunRecord], dict[str, int]]:
    """Collect run records for each workflow within the window, plus any truncation caps."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime('%Y-%m-%dT%H:%M:%SZ')
    records: list[RunRecord] = []
    truncated: dict[str, int] = {}
    for workflow in workflows:
        runs, more = _runs_in_window(client, workflow, since, per_workflow_limit)
        if more:
            truncated[workflow] = per_workflow_limit
        for run in runs:
            records.append(collect_run(client, workflow, _as_mapping(run)))
    return records, truncated


def main(argv: list[str] | None = None) -> int:
    """Emit the Slack payload as a GitHub Actions output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--days', type=int, default=7)
    # Deliberately far below weekly volume: measured over one window, `pr-review` saw 889
    # runs and `ui-security-review` 815. Collection costs at least one API call per run,
    # and `GITHUB_TOKEN` is capped at 1,000 requests per hour per repository — so raising
    # this to cover a busy workflow in full exhausts the budget mid-report and the job
    # dies with nothing delivered. The report samples and says so: capped workflows are
    # marked `*` in the table and named in the footnote, rather than presenting a
    # fraction as a total.
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

    records, truncated = gather(client, workflows, args.days, args.per_workflow_limit)
    summaries = summarize(records)
    report = format_report(
        summaries,
        args.days,
        sampled=sum(1 for r in records if r.contributed_measurement),
        total=len(records),
        truncated=truncated,
    )
    print(report)

    if output_path := os.environ.get('GITHUB_OUTPUT'):
        payload = json.dumps(build_slack_payload(report), separators=(',', ':'))
        with open(output_path, 'a', encoding='utf-8') as handle:
            handle.write(f'slack_payload={payload}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
