#!/usr/bin/env bash
# Report the REST and GraphQL rate limits the Actions `GITHUB_TOKEN` currently holds.
#
# Every workflow in this repository draws on one REST bucket, and several of them now fail with
# "API rate limit exceeded for installation" while nothing records what drained it: no workflow
# logs the limit, step debugging is off, and the agentic workflows' containers upload no tool log.
# Reading the limit before and after a job's own API work separates a bucket that arrives already
# empty -- drained by something outside this run -- from one the run spends itself, and `reset`
# says whether the exhaustion is bursty or sustained.
#
# `GET /rate_limit` does not count against the limits it reports, so the probe costs no quota.
#
# Temporary. Delete this script and its two steps once the drain is identified.
set -uo pipefail

phase=${1:-unknown}

if ! json=$(gh api rate_limit 2>&1); then
  # Never fail the job over diagnostics -- including when the probe itself is rate limited, which
  # is a reading in its own right.
  echo "rate-limit-probe ${phase}: unavailable: ${json}"
  exit 0
fi

if ! line=$(jq -er --arg phase "${phase}" '
  "rate-limit-probe \($phase)"
  + " rest=\(.resources.core.remaining)/\(.resources.core.limit)"
  + " rest_reset=\(.resources.core.reset)"
  + " graphql=\(.resources.graphql.remaining)/\(.resources.graphql.limit)"
  + " graphql_reset=\(.resources.graphql.reset)"
' <<<"${json}"); then
  echo "rate-limit-probe ${phase}: unparseable response"
  exit 0
fi

echo "${line}"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  printf '%s\n\n' "${line}" >>"${GITHUB_STEP_SUMMARY}"
fi
