# Reduce repeated CI service image downloads

## Problem and evidence

CI repeatedly downloads PostgreSQL 16 for the all-extras matrix. In the 2026-10-16 21:00–23:59:59 UTC window, 190 CI runs were listed. Available job records from 149 runs showed 1,251 successful PostgreSQL 16 service setups and six cancelled starts. Setup took 11–28 seconds, median 15 seconds. The other 41 job-list reads failed and are excluded.

The current amd64 image contains 160,287,941 compressed layer bytes. Multiplying by the successful setups gives approximately 200.5 GB of fresh layer demand. This is an estimate from manifests and job records, not measured network billing.

The [CI workflow](https://github.com/pydantic/pydantic-ai/blob/52ff60dc01441a078c987b030df86201357dc22f/.github/workflows/ci.yml) starts PostgreSQL 16 on all 11 full-run shards, or five harness-only cells. Absurd and pgtask use this server. The separate PostgreSQL 17 storage/media suite, MongoDB suite and Redis suite provide distinct live-server coverage.

[Release run 39053926124](https://github.com/pydantic/pydantic-ai/actions/runs/39053926124) failed twice pulling PostgreSQL 17 from ECR Public before checkout. Redis also hit ECR's request-rate limit before recovering. The separate [source-registry repair #10580](https://github.com/pydantic/pydantic-ai/pull/10580) owns that immediate failure.

## Proposed decision

Validate the preinstalled PostgreSQL 16 on `ubuntu-24.04` for `test-all-extras`. Adopt it only after a hosted pilot proves equivalent tests and acceptable startup behavior.

The [actual Ubuntu 24.04 runner image](https://github.com/actions/runner-images/blob/ubuntu24/20261012.334/images/ubuntu/Ubuntu2404-Readme.md) lists PostgreSQL 16.15 and documents starting its system service. This offers a way to eliminate image transfer for the highest-fanout service. Native startup and test compatibility have not been benchmarked here.

Pin the OS label for this pilot. The [announced migration](https://github.com/actions/runner-images/issues/14748) moves `ubuntu-latest` to Ubuntu 26.04 beginning October 19. The explicit OS label still receives package/image updates.

## Implementation after agreement

1. Run a hosted comparison on fresh `ubuntu-24.04` x64 runners: current Docker PostgreSQL 16 versus the preinstalled server. Record runner image, server version, startup/health time and image bytes transferred. Use repeated trials with both full and harness-only shard selection.
2. Configure native PostgreSQL for the existing localhost URL, database, user and password. Assert major version 16 and successful authenticated SQL before tests. Preserve `ABSURD_REQUIRE_LIVE`, `PGTASK_REQUIRE_LIVE`, all Python versions, shards, test selections and coverage collection.
3. Exercise the actual Absurd/pgtask suites and complete all-extras jobs on Python 3.11–3.15. Compare selected test identities and pass/skip outcomes with the container control. Check migrations, schema cleanup, concurrency, locale/encoding and authentication defaults.
4. If correctness and timing gates pass, replace only `test-all-extras.services.postgres` with a native startup step before the tests. Keep shard selection unchanged. Fail clearly if the expected server is missing; do not hide image drift with an unmeasured install/pull fallback.
5. Keep PostgreSQL 17, MongoDB and Redis integration jobs, their path filters and tag coverage. Incorporate the separately merged registry repair before implementing.
6. In a separately reviewable change, pin remaining service images by digest and add a verified weekly refresh mechanism that opens update PRs. Keep the supported major/minor policy explicit, test refreshes before promotion and report stale/failed updates. Do not add pins without an updater.

## Acceptance gates

- Native and container controls execute the same selected PostgreSQL tests across the existing Python matrix, with no new skips.
- Each native trial makes zero PostgreSQL image pulls or package downloads.
- Startup timing does not regress against contemporaneous container controls; publish raw trials and median/tail comparisons.
- PRs from forks and Dependabot need no database/registry secret.
- The dedicated PostgreSQL 17/MongoDB/Redis suites still run on tags and relevant changes.
- Runner/server version drift fails visibly. Rollback restores the existing PostgreSQL 16 service declaration.
- No product runtime code or release workflow is changed by this plan.

## Alternatives and constraints

- **Maintained upstream images:** retain them for low-fanout jobs. [GitHub documents](https://docs.github.com/en/actions/reference/limits) that public Docker Hub pulls on hosted runners are exempt from Docker Hub pull-rate limits. Observed ECR data-limit errors do not prove this repository exhausted a shared source-IP allowance. [AWS pricing](https://aws.amazon.com/ecr/pricing/)
- **Ordinary image caching:** hosted jobs start on fresh VMs. Service initialization precedes checkout/cache-restoration steps, so adding `actions/cache` cannot preload a declared `services:` image. [Service lifecycle](https://docs.github.com/en/actions/tutorials/use-containerized-services/use-docker-service-containers)
- **Save/load archive:** requires manual startup after restoration and still transfers an archive per runner. If the native pilot fails, benchmark restore/load/start against direct pulls before adopting it. Use digest/OS/architecture keys and trusted writers. [Cache scopes/limits](https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching)
- **Public GHCR mirror:** anonymous pulls work for forks; a mirror can isolate upstream outages but still sends the same layers to fresh runners. It needs a trusted publisher, freshness monitoring and architecture preservation. No throughput win has been measured. [GHCR](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
- **Pull-through registry:** reduces upstream requests, not runner downloads. AWS's implementation requires private ECR/IAM access and retains an upstream refresh dependency; that adds an unnecessary authentication dependency for public fork CI. [AWS pull-through behavior](https://docs.aws.amazon.com/AmazonECR/latest/userguide/pull-through-cache.html)

## Deliberate scope

This PR contains a plan only. It does not implement the registry repair, change product code, remove coverage, change compiler-managed gh-aw images, retry/retag a release or merge anything. Archive caches and mirrors remain alternatives requiring evidence if the native pilot fails.
