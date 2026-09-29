#!/usr/bin/env bash
# Prefetch the open issue and PR indexes before the AWF firewall starts.
#
# Issue-list requests made through gh-proxy have repeatedly stalled until the
# agent's wall-clock limit. Use the token scoped to this pre-agent step without
# printing it, and give the agent a local deduplication corpus instead.
set -uo pipefail

agent_dir="${GH_AW_AGENT_DIR:-/tmp/gh-aw/agent}"
context_dir="${agent_dir}/github-context"
corpus_limit=1000
issues_file="${context_dir}/open-issues.json"
prs_file="${context_dir}/open-pull-requests.json"
issues_tmp="${issues_file}.tmp.$$"
prs_tmp="${prs_file}.tmp.$$"
mkdir -p "${context_dir}"
rm -f "${issues_file}" "${prs_file}" "${issues_tmp}" "${prs_tmp}"
trap 'rm -f "${issues_tmp}" "${prs_tmp}"' EXIT

token="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
if [ -z "${token}" ]; then
  echo "::warning::No GitHub credential available for context prefetch"
  exit 0
fi

repo="${GITHUB_REPOSITORY:-}"
if [ -z "${repo}" ]; then
  echo "::warning::GITHUB_REPOSITORY is unavailable for context prefetch"
  exit 0
fi

export GH_TOKEN="${token}"
run_gh() {
  if command -v timeout >/dev/null 2>&1; then
    timeout 60s gh "$@"
  elif command -v gtimeout >/dev/null 2>&1; then
    gtimeout 60s gh "$@"
  else
    echo "::warning::No bounded timeout command available for context prefetch"
    return 1
  fi
}

if ! run_gh issue list \
  --repo "${repo}" \
  --state open \
  --limit "${corpus_limit}" \
  --json number,title,url,labels,updatedAt \
  > "${issues_tmp}" \
  || ! jq -e --argjson limit "${corpus_limit}" 'type == "array" and length < $limit' "${issues_tmp}" >/dev/null; then
  echo "::warning::Could not prefetch open issues"
  rm -f "${issues_tmp}"
else
  mv "${issues_tmp}" "${issues_file}"
  echo "Prefetched $(jq 'length' "${issues_file}") open issues"
fi

if ! run_gh pr list \
  --repo "${repo}" \
  --state open \
  --limit "${corpus_limit}" \
  --json number,title,url,labels,updatedAt \
  > "${prs_tmp}" \
  || ! jq -e --argjson limit "${corpus_limit}" 'type == "array" and length < $limit' "${prs_tmp}" >/dev/null; then
  echo "::warning::Could not prefetch open pull requests"
  rm -f "${prs_tmp}"
else
  mv "${prs_tmp}" "${prs_file}"
  echo "Prefetched $(jq 'length' "${prs_file}") open pull requests"
fi
