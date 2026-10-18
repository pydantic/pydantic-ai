# PostgreSQL CI pilot

This experiment compares the official Docker PostgreSQL 16 image with the preinstalled PostgreSQL 16
on fresh Ubuntu 24.04 x64 GitHub-hosted runners. Production CI continues using its existing service.

The workflow runs 16 fresh trials per strategy: every current all-extras shard on Python 3.11–3.15,
plus the harness-only selection on each version. Pairs share one fetched production duration baseline.
The trial matrix is interleaved by strategy, with at most eight trials active.

## Measurements and evidence

Each artifact retains the source and lock/baseline hashes, exact runner image and server versions,
CPU and memory inventory, commands with output and timings, PostgreSQL packages before/after,
selected test IDs, phase outcomes and skip reasons, shard arguments, and complete pytest output.

Authenticated-ready time starts immediately before Docker pull or native service start and ends after
a successful SQL query using the production host/user/password/database. Docker uses the production
health configuration. Both servers reject a deliberately wrong password. SQL probes check UTF-8,
collation ordering, rollback, concurrent increments and schema cleanup. The actual harness tests
exercise their normal Absurd/pgtask migrations and queue/workflow behavior.

The network counter is interface ingress measured around the Docker pull. Background traffic is
included; this counter is neither image-only transfer nor billing. Registry manifests are retained
separately. Native startup executes no image pull or package installer and checks unchanged
PostgreSQL package inventory. Dependency downloads outside the measured startup are not PostgreSQL downloads.

The opt-in pytest plugin records collected IDs and controller outcomes. It is loaded only by the
pilot. Production selection, live-server flags, xdist grouping, coverage versions, cost budgets and
missing-database-URL smoke checks are preserved. Coverage artifacts from this experiment do not feed
production coverage.

## Gates and interpretation

Adoption requires a complete paired matrix; successful SQL probes and tests; identical paired IDs,
outcomes and skips; matching locale/encoding/order/timezone semantics; unchanged native packages;
at least ten fresh successful starts per strategy; and native median and nearest-rank p95 startup
no slower than Docker. A negative adoption result retains production Docker. A successful evidence
summary is not adoption approval; failed trial jobs remain visible failures.

Nearest-rank p95 is the sorted observation at index ceil(0.95 × sample count), using one-based indexing.
Every trial is retained, including failures. Missing evidence fails the summary.

## Run and reproduce

The feature-branch push workflow runs when its workflow or pilot scripts change. A report-only commit
does not repeat the batch. After merge, workflow_dispatch can run another explicit batch.

The workflow invokes postgres_pilot.py matrix, start, finish and summarize. Its shell steps show the
complete production pytest invocation and baseline selection. Each trial artifact includes the exact
expanded shard arguments and PostgreSQL commands.

Hosted results will be recorded here after the first completed batch.

## Deliberate scope

The production CI matrix, PostgreSQL 17/MongoDB/Redis suites, release workflows and product code are
unchanged. Image digest refresh, mirrors and cache adoption are separate decisions. No native package
installation or fallback pull conceals runner image drift.
