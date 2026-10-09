"""List tests over the per-test time budget, from the `--durations` output of a CI run.

`tests/cost_guards.py` fails a test whose setup and call take longer than `PYTEST_TEST_BUDGET_SECONDS` on the CI leg
that sets it. Run this against a CI run to see which tests that leg would fail, before turning the budget on or
lowering it, and mark the ones whose cost is inherent with `@pytest.mark.slow(reason=...)`:

    uv run scripts/list_over_budget_tests.py <run-id> [--budget 5] [--jobs 'all-extras']

It reads the jobs' logs with the `gh` CLI. The durations pytest reports include setup of fixtures shared beyond one
test, which the guard does not count, so a test listed here only for its setup may be within budget; and pytest
only prints the slowest `--durations=N` phases per job, so the list is complete only down to the Nth slowest.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import defaultdict

ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*m')
DURATION = re.compile(r'\s(?P<seconds>\d+\.\d+)s (?P<phase>setup|call|teardown)\s+(?P<nodeid>\S+::\S+)')


def job_log(job_id: int) -> str:
    """One job's log, without the terminal colors pytest adds."""
    endpoint = f'repos/{{owner}}/{{repo}}/actions/jobs/{job_id}/logs'
    log = subprocess.check_output(['gh', 'api', '--allow-escape-sequences', endpoint], text=True)
    return ANSI_ESCAPE.sub('', log)


def job_logs(run_id: str, job_filter: str) -> dict[str, str]:
    """The logs of the run's jobs whose names match `job_filter`, by job name."""
    jobs = json.loads(subprocess.check_output(['gh', 'run', 'view', run_id, '--json', 'jobs'], text=True))['jobs']
    pattern = re.compile(job_filter)
    return {job['name']: job_log(job['databaseId']) for job in jobs if pattern.search(job['name'])}


def over_budget(log: str, budget: float) -> dict[str, tuple[float, float]]:
    """`nodeid -> (setup, call)` seconds for each test whose setup and call together exceed `budget`."""
    phases: dict[str, dict[str, float]] = defaultdict(dict)
    for match in DURATION.finditer(log):
        phases[match['nodeid']][match['phase']] = float(match['seconds'])
    return {
        nodeid: (times.get('setup', 0.0), times.get('call', 0.0))
        for nodeid, times in phases.items()
        if times.get('setup', 0.0) + times.get('call', 0.0) > budget
    }


def main() -> None:
    """Print each test over the budget, slowest first."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_id', help='a GitHub Actions run of the CI workflow')
    parser.add_argument('--budget', type=float, default=5.0, help='seconds of setup and call per test (default: 5)')
    parser.add_argument('--jobs', default=r'3\.13 \(all-extras', help='regex for the job names to read')
    args = parser.parse_args()

    found: dict[str, tuple[float, float]] = {}
    for name, log in job_logs(args.run_id, args.jobs).items():
        tests = over_budget(log, args.budget)
        print(f'{name}: {len(tests)} over {args.budget:g}s')
        found.update(tests)
    for nodeid, (setup, call) in sorted(found.items(), key=lambda item: -sum(item[1])):
        print(f'{setup + call:7.2f}s  (setup {setup:.2f}s, call {call:.2f}s)  {nodeid}')


if __name__ == '__main__':
    main()
