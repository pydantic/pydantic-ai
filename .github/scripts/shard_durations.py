"""Split sharded test jobs by measured durations, and report the tests a change made slow.

Every sharded test job stores the durations of the tests it ran (`pytest --store-durations
--clean-durations`) and uploads them as an artifact named `<prefix><shard>`.

- `fetch`, run once per workflow run by `test-durations-baseline`, merges those artifacts from the
  newest `main` run that had uploaded all of them when this run was created, so the split follows the
  suite as it is now instead of a committed file that drifts. Every shard then reads that one file, so
  they all agree on the split even when the API fails (the file is then empty).
- `select` assigns whole test modules to shards by those durations (by source size without them) and
  prints the `--ignore` options that leave pytest only this shard's modules.
- `report` compares a shard's measured durations with the baseline and lists, in the job summary, new
  tests that take a while and tests that got much slower.
"""

from __future__ import annotations

import argparse
import heapq
import io
import json
import os
import statistics
import subprocess
import sys
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
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
    """The merged durations of the newest `main` push run created before `before` that uploaded all `shards`.

    A run created before this one has either uploaded its artifacts or is still running; requiring all
    `shards` of them skips the latter, so a re-run (which keeps the run's creation time) picks the same
    baseline.
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


def find_test_modules(root: Path, exclude: Sequence[str]) -> list[str]:
    """Every tracked `tests/**/test_*.py` module outside `exclude`, as a repo-relative POSIX path."""
    tracked = subprocess.run(
        ['git', 'ls-files', '-z', 'tests'], cwd=root, check=True, capture_output=True, text=True
    ).stdout.split('\0')
    excluded = tuple(path.rstrip('/') + '/' for path in exclude)
    return sorted(
        path
        for path in tracked
        if PurePosixPath(path).name.startswith('test_') and path.endswith('.py') and not path.startswith(excluded)
    )


def module_weights(modules: Sequence[str], durations: Mapping[str, float], root: Path) -> dict[str, float]:
    """Each module's measured seconds; a module the baseline has not seen counts as the median one.

    Without a baseline, every module weighs its source size.
    """
    if not durations:
        return {path: float((root / path).stat().st_size) for path in modules}
    measured: dict[str, float] = {}
    for test, seconds in durations.items():
        path = test.split('::', 1)[0]
        measured[path] = measured.get(path, 0.0) + seconds
    known = [measured[path] for path in modules if path in measured]
    fallback = statistics.median(known) if known else 1.0
    return {path: measured.get(path, fallback) for path in modules}


def assign(weights: Mapping[str, float], shards: int) -> list[list[str]]:
    """Spread whole modules across `shards`, heaviest first onto the lightest shard.

    Splitting by module rather than by test (as `pytest-split` does) means a shard never imports the
    other shards' modules: with `pytest-split` every worker of every shard collects the whole suite and
    then deselects most of it. Ties break on the path, so every shard computes the same assignment from
    the same inputs and each module runs exactly once.
    """
    totals = [(0.0, shard) for shard in range(shards)]
    assigned: list[list[str]] = [[] for _ in range(shards)]
    for path in sorted(weights, key=lambda path: (-weights[path], path)):
        total, shard = heapq.heappop(totals)
        assigned[shard].append(path)
        heapq.heappush(totals, (total + weights[path], shard))
    return assigned


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

    select_parser = subparsers.add_parser(
        'select', help="Print the `--ignore` options that leave pytest only this shard's test modules."
    )
    select_parser.add_argument('--durations', type=Path, required=True)
    select_parser.add_argument('--shards', type=int, required=True)
    select_parser.add_argument('--shard', type=int, required=True, help='1-based.')
    select_parser.add_argument('--exclude', action='append', default=[], help='A directory pytest ignores anyway.')

    report_parser = subparsers.add_parser('report', help='Summarize new slow tests and slowdowns against a baseline.')
    report_parser.add_argument('--baseline', type=Path, required=True)
    report_parser.add_argument('--current', type=Path, required=True)
    report_parser.add_argument('--title', required=True)

    args = parser.parse_args(argv)
    if args.command == 'fetch':
        # Balancing is an optimization: never fail the run over it. An empty baseline splits by size.
        durations: Durations = {}
        try:
            repo = os.environ['GITHUB_REPOSITORY']
            this_run = json.loads(gh_api(f'repos/{repo}/actions/runs/{os.environ["GITHUB_RUN_ID"]}'))
            found = fetch(repo, args.prefix, args.shards, before=this_run['created_at'])
        except (KeyError, OSError, subprocess.CalledProcessError, ValueError, zipfile.BadZipFile) as exc:
            print(f'::warning::Could not fetch `{args.prefix}*` test durations, splitting by module size: {exc!r}')
        else:
            if found is None:
                print(f'No `main` run has `{args.prefix}*` durations yet; splitting by module size.')
            else:
                run_id, durations = found
                print(f'Fetched `{args.prefix}*` durations of {len(durations)} tests from run {run_id}.')
        args.output.write_text(json.dumps(durations, sort_keys=True), encoding='utf-8')
        return 0

    if args.command == 'select':
        root = Path.cwd()
        durations = _load(args.durations)
        weights = module_weights(find_test_modules(root, args.exclude), durations, root)
        assigned = assign(weights, args.shards)
        unit = 's' if durations else ' bytes of source (no durations baseline)'
        for shard, paths in enumerate(assigned, start=1):
            marker = '*' if shard == args.shard else ' '
            total = sum(weights[path] for path in paths)
            print(f'{marker} shard {shard}: {len(paths)} modules, estimated {total:.0f}{unit}', file=sys.stderr)
        for shard, paths in enumerate(assigned, start=1):
            if shard != args.shard:
                print('\n'.join(f'--ignore={path}' for path in paths))
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
