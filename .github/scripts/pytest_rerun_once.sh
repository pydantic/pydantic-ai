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
set -uo pipefail

"$@"
status=$?
if [[ $status -ne 1 || "${GITHUB_EVENT_NAME:-}" != merge_group ]]; then
  exit "$status"
fi
echo '::warning::Tests failed. Rerunning the failed tests once: a test that then passes is filed as a flake.'
"$@" --last-failed --last-failed-no-failures none
