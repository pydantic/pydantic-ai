"""Report whether Pyright got much slower on `main`.

CI type-checks a `main` push that changes a file Pyright checks, in the `Type-check with Pyright` step of its
`quality checks` job, and no run fails on how long that step took: one shared runner can take twice as long as
another on the same commit. This compares the step's median duration over the newest `WINDOW` `main` push runs
that ran it with its median over the `WINDOW` runs before those, and reports a slowdown when the newer median is at
least `SLOWDOWN_FACTOR` times the older one.

A median only moves once most of its runs are slow, so one slow runner never trips this. A slowdown is reported
from the run that makes it most of the newer window until it is also most of the older one: about `WINDOW`
consecutive `main` runs, which `pyright-trend.yml` covers by running this hourly. A slowdown fixed within about
half a window of `main` runs is never reported.

A check judges only when the newest `main` run finished within `FRESH_FOR`, so a window stops being judged `FRESH_FOR`
after its newest run. `FRESH_FOR` is longer than the hourly schedule because GitHub delays scheduled runs, so more than
one check can judge the same window.

Usage:
    python .github/scripts/pyright_trend.py

It reads the repository from `GITHUB_REPOSITORY`, calls the GitHub API through the `gh` CLI, writes the medians to
`$GITHUB_STEP_SUMMARY`, and writes `slower=true` or `slower=false` to `$GITHUB_OUTPUT` when those are set.
"""

from __future__ import annotations

import os
import statistics
import subprocess
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta

GhApi = Callable[[Sequence[str]], list[str]]
"""Runs `gh api` with these arguments, and returns the lines its `--jq` filter printed."""

WORKFLOW_FILE = 'ci.yml'
JOB_NAME = 'quality checks'
STEP_NAME = 'Type-check with Pyright'
WINDOW = 20
SLOWDOWN_FACTOR = 1.5
FRESH_FOR = timedelta(minutes=90)
# A push that changes no file Pyright checks skips the step, so this many `main` runs are listed to find
# `2 * WINDOW` that ran it.
RUNS_LISTED = 100


def gh_api(arguments: Sequence[str]) -> list[str]:
    """Run `gh api` with `arguments`, which authenticates with `GH_TOKEN`, and return its output lines."""
    return subprocess.run(['gh', 'api', *arguments], check=True, capture_output=True, text=True).stdout.splitlines()


def main(api: GhApi = gh_api, now: datetime | None = None) -> int:
    """Judge the newest `main` runs as the module docstring says, and return `0`; a failed API call raises."""
    repo = os.environ['GITHUB_REPOSITORY']
    runs = [
        line.split('\t')
        for line in api(
            [
                f'repos/{repo}/actions/workflows/{WORKFLOW_FILE}/runs'
                f'?branch=main&event=push&status=completed&per_page={RUNS_LISTED}',
                '--jq',
                '.workflow_runs[] | [.id, .updated_at] | @tsv',
            ]
        )
    ]
    slower = False
    if not runs or (now or datetime.now(UTC)) - datetime.fromisoformat(runs[0][1]) > FRESH_FOR:
        report = (
            f'No `main` run finished in the last {FRESH_FOR.total_seconds() / 60:g} minutes, so nothing new to judge.'
        )
    else:
        durations: list[tuple[str, float]] = []
        for run_id, _ in runs:
            step = api(
                [
                    '--paginate',
                    f'repos/{repo}/actions/runs/{run_id}/jobs?filter=latest&per_page=100',
                    '--jq',
                    f'.jobs[] | select(.name == "{JOB_NAME}") | .steps[]'
                    f' | select(.name == "{STEP_NAME}" and .conclusion == "success")'
                    ' | [.started_at, .completed_at] | @tsv',
                ]
            )
            if step:
                started, completed = step[0].split('\t')
                durations.append(
                    (run_id, (datetime.fromisoformat(completed) - datetime.fromisoformat(started)).total_seconds())
                )
                if len(durations) == 2 * WINDOW:
                    break
        if len(durations) < 2 * WINDOW:
            report = (
                f'Only {len(durations)} of the newest {len(runs)} `main` runs ran `{STEP_NAME}`; '
                f'{2 * WINDOW} are needed.'
            )
        else:
            newer = statistics.median(seconds for _, seconds in durations[:WINDOW])
            older = statistics.median(seconds for _, seconds in durations[WINDOW:])
            slower = newer >= SLOWDOWN_FACTOR * older
            runs_url = f'https://github.com/{repo}/actions/runs'
            report = (
                f'`{STEP_NAME}` took a median {newer:.0f}s over the newest {WINDOW} `main` runs that ran it, '
                f'from {runs_url}/{durations[WINDOW - 1][0]} to {runs_url}/{durations[0][0]}, '
                f'and {older:.0f}s over the {WINDOW} before those, from {runs_url}/{durations[-1][0]}: '
                f'{newer / older:.2f} times as long, '
                f'{"at or over" if slower else "under"} the {SLOWDOWN_FACTOR:g} times that reports a slowdown.'
            )
    print(report)
    if summary := os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(summary, 'a', encoding='utf-8') as file:
            file.write(f'{report}\n')
    if output := os.environ.get('GITHUB_OUTPUT'):
        with open(output, 'a', encoding='utf-8') as file:
            file.write(f'slower={"true" if slower else "false"}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
