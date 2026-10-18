# PostgreSQL CI pilot

**Recommendation: retain the production Docker service.** Native startup was faster in this batch,
but database locale/order equivalence failed in every pair. Test-outcome equivalence and the
immediate post-stop acceptance gate also failed. This experiment does not authorize production adoption.

The [completed baseline run](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886)
executed all 32 trials on 2026-10-18 UTC at source
`fdd5c64a0cbb90dfbeac886cbafef7b6c8bc4c58`. The workflow concluded **failure** because one pytest
trial failed; its [comparison job](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886/job/117582661783)
successfully retained the complete evidence and reported failed adoption gates.

## Timing and transfer results

| Strategy | Fresh starts | Ready median s | Ready p95 s | Health median s | Health p95 s |
|---|---:|---:|---:|---:|---:|
| docker | 16 | 9.115 | 17.664 | 13.291 | 21.597 |
| native | 16 | 3.357 | 4.391 | 3.393 | 4.429 |

Ready time runs from immediately before image pull or native service start through authenticated
`SELECT 1` using the production host, user, password and database. Health time uses the same origin:
Docker waits for its configured periodic health check; native runs `pg_isready` after authenticated
readiness. The adoption timing gates compare **ready** time. Nearest-rank p95 selects observation
`ceil(0.95 × 16) = 16`, so p95 is the maximum in this small sample. All 16 observations per strategy
are included, including the trial with a later pytest failure and native post-stop failures.

Docker pull median was 7.909 s; p95 was
15.843 s. Measured default-interface ingress during the 16 pulls
totaled **2,581,886,787 bytes**; per-pull values ranged
from 161,046,420 to
163,291,583 bytes. These counters include background
traffic and protocol overhead. They are neither image-only transfer nor billing measurements.

The immutable linux/amd64 manifest separately describes **160,287,941 compressed layer bytes**
per image, or 2,564,607,056 bytes across 16 cold pulls. That is a registry-metadata estimate,
not the measured ingress counter. Every Docker trial observed the same image revision.

Native trials executed no PostgreSQL image pull or OS server-package installer, and their
PostgreSQL package inventories were unchanged. There is no native pull interval, so its pull
columns are N/A; this is not a claim of zero total runner network traffic. Shared Python/client
dependency downloads occur outside the measured startup interval.

## Equivalence and failed gates

| Check | Result |
|---|---|
| Complete unique matrix; same source, runner image and Docker revision | PASS |
| Authenticated startup, wrong-password rejection and SQL probes | PASS in every trial |
| Exact paired collected IDs; execution covers collected IDs | PASS |
| Full-shard PostgreSQL ID union equals harness-only IDs for Absurd and pgtask | PASS on every Python version and strategy |
| Paired outcomes and skip reasons | FAIL: only harness/Python 3.14 differs |
| Missing-URL smoke checks | PASS with matching Absurd/pgtask skip reasons |
| Locale, ctype and sample ordering | FAIL in every pair |
| Native PostgreSQL package inventory and no server download | PASS |
| Immediate post-stop acceptance | FAIL in eight native trials |
| Native ready median and p95 no slower than Docker | PASS |

Both servers used UTF8 and `Etc/UTC`, but their database defaults differ:

| Default / probe | Docker | Native |
|---|---|---|
| `datcollate` / `datctype` | `en_US.utf8` / `en_US.utf8` | `C.UTF-8` / `C.UTF-8` |
| `ORDER BY` sample | `['😀', 'a', 'A', 'é', 'z', 'Z']` | `['A', 'Z', 'a', 'z', 'é', '😀']` |

The Docker harness/Python 3.14 trial failed
`TestBridgeFailureModes.test_an_unwind_that_outlives_the_cancel_timeout_closes_the_retired_loop`
in [the AWS Lambda bridge tests](https://github.com/pydantic/pydantic-ai/blob/fdd5c64a0cbb90dfbeac886cbafef7b6c8bc4c58/tests/harness/aws_lambda/test_aws_lambda.py).
Its `cleanup_reached.is_set()` assertion was false after a 0.01 s cancellation window. This test
uses an in-memory fake context and does not call PostgreSQL. Its PostgreSQL probes and missing-URL
smoke checks passed. The other 31 pytest trials passed. The failure remains part of the outcome
comparison; no rerun or replacement establishes a flake diagnosis.

After native `systemctl stop postgresql.service` returned successfully, the single immediate
`pg_isready` probe reported accepting in eight trials, rejecting in six, and no response in two.
Docker reported no response in all 16. The eight accepting observations retain the declared failed
cleanup gate. There is no later poll, child-unit status, PID snapshot or shutdown journal, so
eventual shutdown and a persistent leak are **not established**.

Static inspection of the [exact installed `postgresql-common` package](https://apt.postgresql.org/pub/repos/apt/pool/main/p/postgresql-common/postgresql-common_293.pgdg24.04+1_all.deb)
shows a oneshot parent unit and a cluster template with `PartOf=postgresql.service` and
`Before=postgresql.service`. Systemd propagates stops through `PartOf` and reverses ordering at
shutdown ([unit documentation](https://www.freedesktop.org/software/systemd/man/latest/systemd.unit.html#Before=)).
The parent completing before the cluster stop is therefore a plausible explanation for the mixed
immediate probes. This remains an inference: the trials did not capture the loaded child-unit graph.
Both startup and shutdown commands target the same parent system unit; a manager mismatch is not proven.

## Exact versions and raw evidence

Every baseline trial used `ubuntu-24.04` x64, runner image `ubuntu24` version `20261012.334.1`,
and `Linux-6.17.0-1022-azure-x86_64-with-glibc2.39`. CPU models varied between hosted runners;
each artifact retains its own CPU/memory inventory and runner name. These are descriptive startup
results for this batch, not a measurement of whole-CI wall time or billing savings.

Every Docker server reported:

```text
PostgreSQL 16.15 (Debian 16.15-1.pgdg13+2) on x86_64-pc-linux-gnu, compiled by gcc (Debian 14.2.0-19) 14.2.0, 64-bit
```

Every native server reported:

```text
PostgreSQL 16.15 (Ubuntu 16.15-1.pgdg24.04+2) on x86_64-pc-linux-gnu, compiled by gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0, 64-bit
```

Docker engine was `28.0.4`; host `psql` was `16.15 (Ubuntu 16.15-1.pgdg24.04+2)`.
The observed image index was
`postgres@sha256:ca0bd484cb98bf4b24eb1010e73fb3fcbd6714d240fbc1a10eea5b7dbecb641d`.
Native packages included `postgresql-16` / `postgresql-client-16` `16.15-1.pgdg24.04+2` and
`postgresql-common` / `postgresql-client-common` `293.pgdg24.04+1`.

The [comparison artifact](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886/artifacts/12124319455) contains `comparison.json`,
`trials.json` and the generated summary. The [shared baselines](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886/artifacts/12124300806)
retain the production duration input fetched once per Python version. Each paired trial has matching
lock/baseline hashes. Every raw trial below includes commands with stdout/stderr/timing, exact
versions, package and image inventories, selected IDs, phase outcomes, skip reasons, expanded shard
arguments, test durations and full pytest output. Artifacts have a 30-day retention period.

| Trial / raw artifact | Ready s | Health s | Pull s | Pull interface RX bytes | Tests | Post-stop `pg_isready` |
|---|---:|---:|---:|---:|---|---|
| [docker-full-3.11-1](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886/artifacts/12123488061) | 10.083 | 14.032 | 8.711 | 163291583 | PASS | no response |
| [native-full-3.11-1](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886/artifacts/12123773400) | 3.635 | 3.673 | N/A | N/A | PASS | accepting |
| [docker-full-3.11-2](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886/artifacts/12124530296) | 9.060 | 13.110 | 7.625 | 161086582 | PASS | no response |
| [native-full-3.11-2](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886/artifacts/12124406369) | 4.391 | 4.429 | N/A | N/A | PASS | accepting |
| [docker-full-3.11-3](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886/artifacts/12123502890) | 16.331 | 20.252 | 15.023 | 161479415 | PASS | no response |
| [native-full-3.11-3](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886/artifacts/12124515788) | 3.181 | 3.221 | N/A | N/A | PASS | accepting |
| [docker-harness-3.11-1](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886/artifacts/12123549538) | 9.450 | 13.682 | 8.297 | 161046420 | PASS | no response |
| [native-harness-3.11-1](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886/artifacts/12124540608) | 3.270 | 3.311 | N/A | N/A | PASS | accepting |
| [docker-full-3.12-1](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886/artifacts/12123902244) | 8.929 | 13.160 | 7.813 | 161267569 | PASS | no response |
| [native-full-3.12-1](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886/artifacts/12124585181) | 3.774 | 3.804 | N/A | N/A | PASS | rejecting |
| [docker-full-3.12-2](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886/artifacts/12124530674) | 9.170 | 13.422 | 8.005 | 161064688 | PASS | no response |
| [native-full-3.12-2](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886/artifacts/12124171729) | 3.036 | 3.074 | N/A | N/A | PASS | accepting |
| [docker-harness-3.12-1](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886/artifacts/12123918614) | 17.664 | 21.597 | 15.843 | 161198055 | PASS | no response |
| [native-harness-3.12-1](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886/artifacts/12123898658) | 3.406 | 3.431 | N/A | N/A | PASS | rejecting |
| [docker-full-3.13-1](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886/artifacts/12123648193) | 8.495 | 12.748 | 7.378 | 161097173 | PASS | no response |
| [native-full-3.13-1](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886/artifacts/12123624714) | 3.319 | 3.348 | N/A | N/A | PASS | accepting |
| [docker-full-3.13-2](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886/artifacts/12124302678) | 12.650 | 16.580 | 11.344 | 161530388 | PASS | no response |
| [native-full-3.13-2](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886/artifacts/12124695164) | 3.529 | 3.560 | N/A | N/A | PASS | no response |
| [docker-harness-3.13-1](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886/artifacts/12124497139) | 14.113 | 18.064 | 12.799 | 162179871 | PASS | no response |
| [native-harness-3.13-1](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886/artifacts/12124536817) | 3.117 | 3.156 | N/A | N/A | PASS | rejecting |
| [docker-full-3.14-1](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886/artifacts/12123484205) | 8.351 | 12.578 | 7.229 | 161093052 | PASS | no response |
| [native-full-3.14-1](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886/artifacts/12123424423) | 3.334 | 3.370 | N/A | N/A | PASS | no response |
| [docker-full-3.14-2](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886/artifacts/12124517352) | 8.409 | 12.634 | 7.236 | 161063043 | PASS | no response |
| [native-full-3.14-2](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886/artifacts/12124077408) | 3.125 | 3.163 | N/A | N/A | PASS | rejecting |
| [docker-harness-3.14-1](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886/artifacts/12124680762) | 9.294 | 13.515 | 8.209 | 161220549 | FAIL | no response |
| [native-harness-3.14-1](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886/artifacts/12124213759) | 3.380 | 3.416 | N/A | N/A | PASS | rejecting |
| [docker-full-3.15-1](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886/artifacts/12124586990) | 7.577 | 11.572 | 6.167 | 161078398 | PASS | no response |
| [native-full-3.15-1](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886/artifacts/12124945328) | 3.823 | 3.854 | N/A | N/A | PASS | rejecting |
| [docker-full-3.15-2](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886/artifacts/12124073218) | 6.791 | 11.027 | 5.599 | 161069301 | PASS | no response |
| [native-full-3.15-2](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886/artifacts/12124397486) | 3.441 | 3.467 | N/A | N/A | PASS | accepting |
| [docker-harness-3.15-1](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886/artifacts/12124935260) | 6.792 | 11.028 | 5.674 | 161120700 | PASS | no response |
| [native-harness-3.15-1](https://github.com/pydantic/pydantic-ai/actions/runs/39093970886/artifacts/12124781269) | 3.278 | 3.317 | N/A | N/A | PASS | accepting |

## Protocol and reproduction

The [workflow at the measured source](https://github.com/pydantic/pydantic-ai/blob/fdd5c64a0cbb90dfbeac886cbafef7b6c8bc4c58/.github/workflows/postgres-pilot.yml)
defines every command and environment flag. It runs 16 fresh trials per strategy: all 11 production
all-extras shards across Python 3.11–3.15, plus five harness-only selections. Strategies are
interleaved, with at most eight trial jobs active. Every runner must initially have no live port 5432
and no cached `postgres:16` image. Native has no package-install or Docker fallback.

The collector commands are:

```bash
uv run --no-sync python .github/scripts/postgres_pilot.py matrix
uv run --no-sync python .github/scripts/postgres_pilot.py start \
  --cell "$CELL" --directory "$POSTGRES_PILOT_EVIDENCE" --baseline "baseline/$PYTHON_VERSION.json"
uv run --no-sync python .github/scripts/postgres_pilot.py finish \
  --directory "$POSTGRES_PILOT_EVIDENCE" --test-exit "$TEST_EXIT" --smoke-exit "$SMOKE_EXIT"
uv run --no-sync python .github/scripts/postgres_pilot.py summarize --directory evidence
```

Docker executes `docker pull postgres:16`, then `docker run --detach --name postgres-pilot
-e POSTGRES_PASSWORD=postgres -p 127.0.0.1:5432:5432 --health-cmd 'pg_isready -U postgres'
--health-interval 5s --health-timeout 5s --health-retries 10 postgres:16`. Native executes
`sudo systemctl start postgresql.service` and peer-authenticated `ALTER ROLE postgres PASSWORD 'postgres'`.
Both test URLs are `postgresql://postgres:postgres@127.0.0.1:5432/postgres`, with
`ABSURD_REQUIRE_LIVE=1` and `PGTASK_REQUIRE_LIVE=1`.

The workflow preserves production selection, shard arguments, xdist grouping, coverage versions,
example type-checking flags and budgets. Its opt-in plugin captures collection and controller
reports, including skipped modules. SQL probes check version/authentication, `uuid-ossp`, UTF8
roundtrip, transaction rollback, two concurrent increments and probe-schema removal. Actual
Absurd/pgtask tests exercise their normal migrations and workflow/queue paths. The Python 3.14
shard 1 smoke steps unset both URLs and retain the two normal skip reasons. Pilot coverage does not
feed production coverage.

A push changing pilot scripts/workflow or `shard_durations.py` triggers the feature-branch batch;
a report-only commit does not repeat it. After merge, `workflow_dispatch` can request another batch.

## Excluded setup attempts

No observations from these attempts enter the baseline statistics above:

- [Workflow validation attempt](https://github.com/pydantic/pydantic-ai/actions/runs/39092211074):
  rejected before jobs because job-level environment used unavailable `runner.temp` context.
- [Aborted startup-probe attempt](https://github.com/pydantic/pydantic-ai/actions/runs/39092496858):
  PostgreSQL rejected `current_setting('lc_collate')`. The protocol was repaired to read
  `pg_database.datcollate/datctype`; 26 trial jobs failed and six were cancelled. All available raw
  artifacts remain linked below. Timings are excluded for the entire attempt, rather than pooling
  partial observations with the repaired protocol.
- Local ARM64 collector verification exposed and repaired a separate SQL transaction-boundary bug,
  then passed the complete Docker startup/SQL/cleanup path. Its timings and transfer are excluded
  from the hosted x64 comparison.

<details>
<summary>Raw artifacts from the aborted startup-probe attempt</summary>

- [postgres-pilot-docker-full-3.11-1](https://github.com/pydantic/pydantic-ai/actions/runs/39092496858/artifacts/12123580762)
- [postgres-pilot-docker-full-3.11-2](https://github.com/pydantic/pydantic-ai/actions/runs/39092496858/artifacts/12123660601)
- [postgres-pilot-docker-full-3.11-3](https://github.com/pydantic/pydantic-ai/actions/runs/39092496858/artifacts/12123048848)
- [postgres-pilot-docker-full-3.12-1](https://github.com/pydantic/pydantic-ai/actions/runs/39092496858/artifacts/12123053868)
- [postgres-pilot-docker-full-3.12-2](https://github.com/pydantic/pydantic-ai/actions/runs/39092496858/artifacts/12123456355)
- [postgres-pilot-docker-full-3.13-1](https://github.com/pydantic/pydantic-ai/actions/runs/39092496858/artifacts/12123148447)
- [postgres-pilot-docker-full-3.13-2](https://github.com/pydantic/pydantic-ai/actions/runs/39092496858/artifacts/12123595772)
- [postgres-pilot-docker-full-3.14-1](https://github.com/pydantic/pydantic-ai/actions/runs/39092496858/artifacts/12123097471)
- [postgres-pilot-docker-full-3.14-2](https://github.com/pydantic/pydantic-ai/actions/runs/39092496858/artifacts/12123600855)
- [postgres-pilot-docker-harness-3.11-1](https://github.com/pydantic/pydantic-ai/actions/runs/39092496858/artifacts/12123470701)
- [postgres-pilot-docker-harness-3.12-1](https://github.com/pydantic/pydantic-ai/actions/runs/39092496858/artifacts/12123740072)
- [postgres-pilot-docker-harness-3.13-1](https://github.com/pydantic/pydantic-ai/actions/runs/39092496858/artifacts/12123024178)
- [postgres-pilot-docker-harness-3.14-1](https://github.com/pydantic/pydantic-ai/actions/runs/39092496858/artifacts/12123158423)
- [postgres-pilot-native-full-3.11-1](https://github.com/pydantic/pydantic-ai/actions/runs/39092496858/artifacts/12123640701)
- [postgres-pilot-native-full-3.11-2](https://github.com/pydantic/pydantic-ai/actions/runs/39092496858/artifacts/12123143404)
- [postgres-pilot-native-full-3.11-3](https://github.com/pydantic/pydantic-ai/actions/runs/39092496858/artifacts/12123073677)
- [postgres-pilot-native-full-3.12-1](https://github.com/pydantic/pydantic-ai/actions/runs/39092496858/artifacts/12123213081)
- [postgres-pilot-native-full-3.12-2](https://github.com/pydantic/pydantic-ai/actions/runs/39092496858/artifacts/12122853092)
- [postgres-pilot-native-full-3.13-1](https://github.com/pydantic/pydantic-ai/actions/runs/39092496858/artifacts/12122718611)
- [postgres-pilot-native-full-3.13-2](https://github.com/pydantic/pydantic-ai/actions/runs/39092496858/artifacts/12122909594)
- [postgres-pilot-native-full-3.14-1](https://github.com/pydantic/pydantic-ai/actions/runs/39092496858/artifacts/12122344715)
- [postgres-pilot-native-full-3.14-2](https://github.com/pydantic/pydantic-ai/actions/runs/39092496858/artifacts/12123685650)
- [postgres-pilot-native-harness-3.11-1](https://github.com/pydantic/pydantic-ai/actions/runs/39092496858/artifacts/12123545947)
- [postgres-pilot-native-harness-3.12-1](https://github.com/pydantic/pydantic-ai/actions/runs/39092496858/artifacts/12123810113)
- [postgres-pilot-native-harness-3.13-1](https://github.com/pydantic/pydantic-ai/actions/runs/39092496858/artifacts/12123550583)
- [postgres-pilot-native-harness-3.14-1](https://github.com/pydantic/pydantic-ai/actions/runs/39092496858/artifacts/12122648833)

</details>

## Deliberate scope and next decision

Production CI, the PostgreSQL 17/MongoDB/Redis suites, release workflows and product code are
unchanged. There is no image digest updater, mirror or cache adoption. The experiment keeps the
native runner defaults and the failed gates; it does not reconfigure locale or replace failed trials.

The concrete follow-up recommendation is a separately authorized, locale-matched native cluster
pilot with explicit cluster-unit evidence and a bounded post-stop check. It must re-verify the same
test/skip/shard and database semantics before production adoption is reconsidered. The AWS Lambda
assertion needs independent investigation before it can be classified as a flake. Maintainers decide
whether those follow-ups are worth pursuing; the current evidence supports retaining Docker.
