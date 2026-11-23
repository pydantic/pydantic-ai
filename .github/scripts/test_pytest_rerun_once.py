"""Run `pytest_rerun_once.sh` over a fake pytest that records its arguments and exits with the statuses it is given.

On its first call, the fake also records `SESSION_FAILURE`, when set, as `fail_session` in `tests/cost_guards.py` does.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent / 'pytest_rerun_once.sh'
FAKE_PYTEST = r"""#!/usr/bin/env bash
echo "$*" >> "$CALLS"
call=$(wc -l < "$CALLS")
if (( call == 1 )) && [[ -n "$SESSION_FAILURE" ]]; then
  echo "$SESSION_FAILURE" >> "$PYTEST_SESSION_FAILURES_FILE"
fi
read -ra statuses <<< "$STATUSES"
exit "${statuses[$((call - 1))]}"
"""
ARGS = 'tests/test_a.py -n logical'
RERUN = f'{ARGS} --last-failed --last-failed-no-failures none'


def run(tmp_path: Path, event: str, *statuses: int, session_failure: str = '') -> tuple[int, list[str]]:
    fake = tmp_path / 'pytest'
    fake.write_text(FAKE_PYTEST, encoding='utf-8')
    fake.chmod(0o755)
    calls = tmp_path / 'calls'
    env = {
        **os.environ,
        'GITHUB_EVENT_NAME': event,
        'STATUSES': ' '.join(map(str, statuses)),
        'CALLS': str(calls),
        'SESSION_FAILURE': session_failure,
    }
    result = subprocess.run(['bash', str(SCRIPT), str(fake), *ARGS.split()], env=env, check=False, capture_output=True)
    return result.returncode, calls.read_text(encoding='utf-8').splitlines()


@pytest.mark.parametrize(('statuses', 'expected'), [((1, 0), 0), ((1, 1), 1), ((1, 5), 5)])
def test_the_merge_queue_reruns_failed_tests_once_and_takes_the_reruns_status(
    tmp_path: Path, statuses: tuple[int, int], expected: int
):
    assert run(tmp_path, 'merge_group', *statuses) == (expected, [ARGS, RERUN])


@pytest.mark.parametrize(
    ('session_failure', 'expected'),
    [
        pytest.param('', (0, [ARGS, RERUN]), id='flaky-test-only'),
        pytest.param('stale subprocess marker: tests/test_a.py::test_x', (1, [ARGS]), id='stale-marker'),
        pytest.param('test over the per-test budget: tests/test_a.py::test_x', (1, [ARGS]), id='over-budget'),
    ],
)
def test_a_first_run_that_failed_a_check_the_rerun_cannot_repeat_fails_without_a_rerun(
    tmp_path: Path, session_failure: str, expected: tuple[int, list[str]]
):
    """A rerun of only the failed tests would pass, but cannot repeat what `fail_session` recorded."""
    assert run(tmp_path, 'merge_group', 1, 0, session_failure=session_failure) == expected


@pytest.mark.parametrize(
    ('event', 'status'), [('merge_group', 0), ('merge_group', 2), ('merge_group', 5), ('pull_request', 1), ('push', 1)]
)
def test_any_other_status_or_event_passes_through_without_a_rerun(tmp_path: Path, event: str, status: int):
    assert run(tmp_path, event, status) == (status, [ARGS])
