---
emoji: "🛡️"
name: "Pydantic AI UI Security Review"
description: "Security review of UI-adapter PRs (Vercel AI + AG-UI): audits the client/server trust boundary for outbound leakage and inbound abuse. Inline comments + a non-voting COMMENT-type review summary (pydantic-ai-pr-review owns the merge-gate verdict until gh-aw check-runs land). Prompt iterable from a Logfire managed variable; read-only via gh-aw safe-outputs."
on:
  # Runs on EVERY PR (no `paths:` filter) so the review's check is always
  # reported on the head commit. The UI-path selection moved into the `detect`
  # job below: non-UI PRs skip the agent via `if:` (a job skipped by `if:`
  # reports as success, so it never blocks merge), while UI PRs gate the agent
  # behind the `ui-security-review` Environment (required reviewers) — a
  # maintainer clicks "Approve" to start the AI run, and the pending approval
  # blocks merge until the review has run. `synchronize` is kept so a new
  # commit re-reports the check on the new head and re-pends the approval.
  pull_request:
    types: [opened, synchronize, ready_for_review]
  workflow_dispatch:
  # Fork-PR safety: only trigger when the actor has admin/maintainer/write
  # access. Without this, any established external contributor's PR would
  # consume the configured Anthropic key and a model run.
  roles: [admin, maintainer, write]
  # Pause for a maintainer's approval of the `ui-security-review` Environment
  # before the agent runs. This gates the `activation` job, but because
  # `activation` is itself gated by the `if:` below, the approval is only
  # requested on UI-touching PRs (a skipped job never requests Environment
  # approval). The pending approval keeps the agent's required check pending,
  # blocking merge until a maintainer clicks Approve to start the review.
  manual-approval: ui-security-review
# Only run the review (and request approval) when the PR touches the UI
# security surface. Non-UI PRs skip the activation chain, so the agent reports
# "skipped" (= success for required checks) and never blocks merge.
if: ${{ needs.detect.outputs.touched == 'true' }}
permissions:
  contents: read
  # safe-outputs perform the actual writes in a separate conclusion job; the
  # agent job stays read-only (gh-aw strict mode requires this).
  pull-requests: read
  issues: read
concurrency:
  # One security review per PR; newer pushes supersede in-flight reviews.
  group: ${{ github.workflow }}-ui-security-review-${{ github.event.pull_request.number || github.ref }}
  cancel-in-progress: true
tools:
  github:
    mode: gh-proxy
    # PR-scoped surface: read the PR, related issues, repo, and search.
    toolsets: [pull_requests, repos, search, issues]
safe-outputs:
  footer: false
  activation-comments: false
  noop:
  create-pull-request-review-comment:
    max: 30
  # Non-voting by design: the prompt restricts the event to COMMENT only,
  # because both this workflow and pydantic-ai-pr-review submit reviews as
  # `github-actions[bot]` and GitHub's merge-gate uses the latest verdict
  # per reviewer login — an APPROVE/REQUEST_CHANGES from here would
  # overwrite pr-review's. To be reconsidered when gh-aw supports check
  # runs (https://github.com/githubnext/gh-aw — Bill Easton's WIP).
  submit-pull-request-review:
    max: 1
timeout-minutes: 30
imports:
  - shared/network-vendor-domains.md
  - shared/otel-logfire.md
  - shared/tool-hints.md
  - shared/repo-context.md
  - shared/rigor.md
  - shared/review-context.md
  - shared/checkout.md
  - shared/engine-minimax.md
  - shared/pre-steps.md
  - shared/pre-agent-steps.md
pre-agent-steps:
  # Pre-fetch PR context into `/tmp/gh-aw/.review-context/` (pr-details, diffs,
  # comments, review threads, related issues, AGENTS.md excerpts). The agent
  # reads these files instead of calling the GitHub API at run time.
  #
  # The script lives at scripts/ (NOT .github/scripts/) because gh-aw's
  # "Save/Restore agent config folders from base branch" step snapshots and
  # restores `.github/` from the BASE branch, making any new file added under
  # it unreliable for steps that run after the restore. `scripts/` is outside
  # that set. Non-fatal: missing context just reduces signal.
  - name: Gather PR review context
    if: ${{ github.event.pull_request.number }}
    env:
      GH_TOKEN: ${{ github.token }}
      PR_NUMBER: ${{ github.event.pull_request.number }}
      REPO: ${{ github.repository }}
    run: |
      set -uo pipefail
      script=scripts/gather-pydantic-ai-review-context.sh
      if [ -x "$script" ]; then
        "$script" "$PR_NUMBER" "$REPO" \
          || echo "::warning::${script} failed; reviewer will run with less context"
      else
        echo "::warning::${script} not present; reviewer will run with less context"
      fi

jobs:
  detect:
    # Cheap, no-AI path filter: does this PR touch the UI security surface?
    # The agent job's top-level `if:` keys on this output. Replaces the old
    # trigger-level `paths:` filter so the workflow still runs (and reports a
    # check) on every PR — non-UI PRs skip the agent and merge freely.
    runs-on: ubuntu-latest
    timeout-minutes: 5
    permissions:
      contents: read
      pull-requests: read
    outputs:
      touched: ${{ steps.filter.outputs.touched }}
    steps:
      - name: Detect UI security paths in the PR
        id: filter
        if: ${{ github.event.pull_request.number }}
        env:
          GH_TOKEN: ${{ github.token }}
          PR_NUMBER: ${{ github.event.pull_request.number }}
          REPO: ${{ github.repository }}
        run: |
          set -euo pipefail
          files="$(gh api --paginate "repos/${REPO}/pulls/${PR_NUMBER}/files" --jq '.[].filename')"
          if printf '%s\n' "$files" | grep -qE '^(pydantic_ai_slim/pydantic_ai/ui/|pydantic_ai_slim/pydantic_ai/_ssrf\.py$|pydantic_ai_slim/pydantic_ai/messages\.py$|pydantic_ai_slim/pydantic_ai/common_tools/web_fetch\.py$|docs/ui/|docs/input\.md$|tests/test_vercel_ai\.py$|tests/test_ag_ui\.py$|tests/test_ui\.py$|tests/test_ui_web\.py$)'; then
            echo 'touched=true' >> "$GITHUB_OUTPUT"
          else
            echo 'touched=false' >> "$GITHUB_OUTPUT"
          fi
  fetch_dynamic_prompt:
    runs-on: ubuntu-latest
    timeout-minutes: 5
    permissions:
      contents: read
    outputs:
      dynamic_prompt: ${{ steps.resolve.outputs.dynamic_prompt }}
    steps:
      - name: Check out the prompt resolver action and default prompt
        uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2
        with:
          persist-credentials: false
          sparse-checkout: |
            .github/actions/fetch-dynamic-prompt
            .github/workflows/shared/prompts/pydantic-ai-ui-security-review.md
          sparse-checkout-cone-mode: false
      - name: Resolve agent prompt (Logfire managed variable, else committed default)
        id: resolve
        uses: ./.github/actions/fetch-dynamic-prompt
        with:
          logfire-variable-key: gh_aw_pydantic_ai_ui_security_review_prompt
          default-prompt-file: .github/workflows/shared/prompts/pydantic-ai-ui-security-review.md
          logfire-read-key: ${{ secrets.LOGFIRE_READ_EXTERNAL_VARIABLES }}
          logfire-base-url: ${{ secrets.LOGFIRE_URL || vars.LOGFIRE_URL || 'https://logfire-api.pydantic.dev' }}
---

${{ needs.fetch_dynamic_prompt.outputs.dynamic_prompt }}
