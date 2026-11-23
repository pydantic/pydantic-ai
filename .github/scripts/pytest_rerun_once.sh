#!/usr/bin/env bash
# Run the pytest command given as arguments. In the merge queue, when pytest exits 1 (tests failed and the session
# finished), run the same command once more on the failed tests only, and exit with that run's status. Every other
# exit status passes through, and so does every status outside the merge queue.
#
# A test that fails and then passes here is flaky: `file-flakes` in `ci.yml` files it from the job's JUnit reports.
# The rerun runs only the failed tests, inside the job's own `timeout-minutes`, so the job still fits the merge
# queue's 60-minute check timeout. Under `coverage run`, the `multiprocessing` concurrency in `[tool.coverage.run]`
# puts coverage in parallel mode, so the rerun writes data files of its own, which `coverage combine` merges with the
# first run's.
#
# A check of the whole session can also fail it, such as a stale `subprocess` marker or a module over the collection
# budget in `tests/cost_guards.py`. A rerun of the failed tests selects too little to judge such a check again, so its
# pass would hide the failure. `fail_session` in `tests/cost_guards.py` appends each such failure to the file
# `PYTEST_SESSION_FAILURES_FILE` names, and a first run that leaves the file non-empty fails without a rerun, which
# could not change the outcome.
set -uo pipefail

if [[ "${GITHUB_EVENT_NAME:-}" != merge_group ]]; then
  exec "$@"
fi
session_failures=$(mktemp)
trap 'rm -f "$session_failures"' EXIT
PYTEST_SESSION_FAILURES_FILE=$session_failures "$@"
status=$?
if [[ $status -ne 1 ]]; then
  exit "$status"
fi
if [[ -s $session_failures ]]; then
  echo '::error::Tests failed a check of the whole session, which a rerun of the failed tests cannot repeat, so they are not rerun:'
  cat "$session_failures"
  exit "$status"
fi
echo '::warning::Tests failed. Rerunning the failed tests once: a test that then passes is filed as a flake.'
"$@" --last-failed --last-failed-no-failures none
