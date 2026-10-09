"""Per-test durations for balancing `pytest-split` shards, and a report of what a change made slower.

Every sharded test job stores the durations of the tests it ran (`pytest --store-durations
--clean-durations`) and uploads them as an artifact named `<prefix><shard>`. Before running, each
shard fetches those artifacts from the newest `main` run that had uploaded all of them when this run
was created, so the split follows the suite as it is now instead of a committed file that drifts. Without them (the first run,
expired artifacts, an API failure), `pytest-split` falls back to splitting by test count.

`report` compares a shard's measured durations with that baseline and lists, in the job summary, new
tests that take a while and tests that got much slower.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Durations = dict[str, float]
GhApi = Callable[[str], bytes]

WORKFLOW_FILE = 'ci.yml'
# How many `main` runs back to look for one that uploaded every shard's durations.
MAX_CANDIDATE_RUNS = 20
NEW_TEST_MIN_SECONDS = 1.0
SLOWDOWN_FACTOR = 2.0
SLOWDOWN_MIN_SECONDS = 1.0
REPORT_ROW_LIMIT = 30


def gh_api(path: str) -> bytes:
    """Call the GitHub REST API through the `gh` CLI, which authenticates with `GH_TOKEN`."""
    return subprocess.run(['gh', 'api', path], check=True, capture_output=True).stdout


def _read_durations_zip(content: bytes) -> Durations:
    durations: Durations = {}
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for member in archive.namelist():
            if member.endswith('.json'):
                durations.update(json.loads(archive.read(member)))
    return durations


def fetch(repo: str, prefix: str, shards: int, before: str, api: GhApi = gh_api) -> tuple[int, Durations] | None:
    """The merged durations of every shard of the newest `main` push run created before `before` that uploaded all of them.

    Every shard of one run must split identically, or a test could run in two shards or in none. A run
    created before this one has uploaded its artifacts, or is still running, by the time any shard of this
    run asks; requiring all `shards` of them skips the latter, so every shard (and every re-run attempt,
    which keeps the run's creation time) settles on the same baseline.
    """
    runs: dict[str, list[dict[str, Any]]] = json.loads(
        api(
            f'repos/{repo}/actions/workflows/{WORKFLOW_FILE}/runs'
            f'?branch=main&event=push&created=%3C{before}&per_page={MAX_CANDIDATE_RUNS}'
        )
    )
    wanted = {f'{prefix}{shard}' for shard in range(1, shards + 1)}
    for run in runs.get('workflow_runs', []):
        data: dict[str, list[dict[str, Any]]] = json.loads(
            api(f'repos/{repo}/actions/runs/{run["id"]}/artifacts?per_page=100')
        )
        artifacts = [
            artifact
            for artifact in data.get('artifacts', [])
            if artifact['name'].startswith(prefix) and not artifact.get('expired')
        ]
        if not wanted <= {artifact['name'] for artifact in artifacts}:
            continue  # still running, a shard failed before uploading, or tests were skipped
        durations: Durations = {}
        for artifact in sorted(artifacts, key=lambda artifact: artifact['name']):
            durations.update(_read_durations_zip(api(f'repos/{repo}/actions/artifacts/{artifact["id"]}/zip')))
        return run['id'], durations
    return None


@dataclass(frozen=True)
class Report:
    new: list[tuple[str, float]]
    slower: list[tuple[str, float, float]]


def compare(baseline: Mapping[str, float], current: Mapping[str, float]) -> Report:
    new = sorted(
        (
            (test, seconds)
            for test, seconds in current.items()
            if test not in baseline and seconds >= NEW_TEST_MIN_SECONDS
        ),
        key=lambda row: -row[1],
    )
    slower = sorted(
        (
            (test, baseline[test], seconds)
            for test, seconds in current.items()
            if test in baseline
            and seconds >= SLOWDOWN_FACTOR * baseline[test]
            and seconds - baseline[test] >= SLOWDOWN_MIN_SECONDS
        ),
        key=lambda row: row[1] - row[2],
    )
    return Report(new=new, slower=slower)


def render(report: Report, title: str) -> str:
    if not report.new and not report.slower:
        return ''
    lines = [f'### Test durations: {title}', '']
    if report.new:
        lines += [
            f'New tests taking at least {NEW_TEST_MIN_SECONDS:g}s:',
            '',
            '| Test | Seconds |',
            '| --- | ---: |',
            *(f'| `{test}` | {seconds:.1f} |' for test, seconds in report.new[:REPORT_ROW_LIMIT]),
            '',
        ]
        if len(report.new) > REPORT_ROW_LIMIT:
            lines += [f'...and {len(report.new) - REPORT_ROW_LIMIT} more.', '']
    if report.slower:
        lines += [
            f'Tests at least {SLOWDOWN_FACTOR:g}x and {SLOWDOWN_MIN_SECONDS:g}s slower than on `main`:',
            '',
            '| Test | `main` | Now |',
            '| --- | ---: | ---: |',
            *(f'| `{test}` | {before:.1f} | {after:.1f} |' for test, before, after in report.slower[:REPORT_ROW_LIMIT]),
            '',
        ]
        if len(report.slower) > REPORT_ROW_LIMIT:
            lines += [f'...and {len(report.slower) - REPORT_ROW_LIMIT} more.', '']
    lines += ['Durations include setup and teardown; one-off slowdowns can be runner noise.', '']
    return '\n'.join(lines)


def _load(path: Path) -> Durations:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)

    fetch_parser = subparsers.add_parser('fetch', help="Fetch the latest `main` run's per-test durations.")
    fetch_parser.add_argument('--prefix', required=True, help='Artifact name prefix; the shard number follows it.')
    fetch_parser.add_argument('--shards', type=int, required=True)
    fetch_parser.add_argument('--output', type=Path, required=True)

    report_parser = subparsers.add_parser('report', help='Summarize new slow tests and slowdowns against a baseline.')
    report_parser.add_argument('--baseline', type=Path, required=True)
    report_parser.add_argument('--current', type=Path, required=True)
    report_parser.add_argument('--title', required=True)

    args = parser.parse_args(argv)
    if args.command == 'fetch':
        # Balancing is an optimization: never fail the job over it.
        try:
            repo = os.environ['GITHUB_REPOSITORY']
            this_run = json.loads(gh_api(f'repos/{repo}/actions/runs/{os.environ["GITHUB_RUN_ID"]}'))
            found = fetch(repo, args.prefix, args.shards, before=this_run['created_at'])
        except (KeyError, OSError, subprocess.CalledProcessError, ValueError, zipfile.BadZipFile) as exc:
            print(f'::warning::Could not fetch test durations, splitting shards by test count: {exc!r}')
            return 0
        if found is None:
            print(f'No `main` run has `{args.prefix}*` durations yet; splitting shards by test count.')
            return 0
        run_id, durations = found
        args.output.write_text(json.dumps(durations, sort_keys=True), encoding='utf-8')
        print(f'Fetched durations of {len(durations)} tests from run {run_id}.')
        return 0

    baseline = _load(args.baseline)
    if not baseline:
        print('No baseline durations to compare with.')
        return 0
    summary = render(compare(baseline, _load(args.current)), args.title)
    print(summary or 'No new slow tests or slowdowns.')
    if summary and (summary_path := os.environ.get('GITHUB_STEP_SUMMARY')):
        with open(summary_path, 'a', encoding='utf-8') as f:
            f.write(summary + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
