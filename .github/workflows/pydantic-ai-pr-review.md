---
emoji: "🔎"
name: "Pydantic AI PR Review"
description: "AI-driven PR review on the Pydantic AI harness: inline comments + a single review verdict. Prompt iterable from a Logfire managed variable; read-only via gh-aw safe-outputs."
on:
  pull_request:
    types: [opened, synchronize, ready_for_review]
  workflow_dispatch:
  # Fork-PR safety: only trigger when the actor has admin/maintainer/write
  # access. Without this, any established external contributor's PR would
  # consume the configured Anthropic key and a model run.
  roles: [admin, maintainer, write]
# Mutually exclusive with the legacy `auto-review`-label-triggered reviewer in
# `bots.yml`: if a PR carries that label, the user has opted into the old
# reviewer, so skip this one.
if: ${{ !contains(github.event.pull_request.labels.*.name, 'auto-review') }}
permissions:
  contents: read
  # safe-outputs perform the actual writes in a separate conclusion job; the
  # agent job stays read-only (gh-aw strict mode requires this).
  pull-requests: read
  issues: read
# Full git history: the reviewer reads `git log`/`git diff` for context and the
# gather-pydantic-ai-review-context script annotates per-file diffs against
# the base ref.
checkout:
  fetch-depth: 0
concurrency:
  # One review per PR; newer pushes supersede in-flight reviews.
  group: ${{ github.workflow }}-pr-review-${{ github.event.pull_request.number || github.ref }}
  cancel-in-progress: true
network:
  allowed:
    - defaults
    # Python/PyPI ecosystem — the harness installs its deps via `uv` at agent
    # time; allow them through the AWF firewall.
    - python
    # ANTHROPIC_BASE_URL is a compile-time literal (below) so gh-aw already
    # auto-allowlists the host; this explicit entry is a harmless safety net.
    - api.minimax.io
# We register as the built-in `claude` engine and only override `command`, so
# gh-aw runs its full Claude proxy + credential-injection machinery for us.
# ANTHROPIC_BASE_URL MUST be a compile-time literal (gh-aw parses its path at
# compile time). Only ANTHROPIC_API_KEY stays a secret (injected by the AWF
# api-proxy, excluded from the agent container).
runtimes:
  uv: {}
engine:
  id: claude
  # The checked-out workspace is mounted no-exec in the AWF sandbox, so a
  # pre-step stages a launcher in gh-aw's exec-able /tmp/gh-aw/bin that runs
  # `uv run --script` against the workspace harness.
  command: /tmp/gh-aw/bin/pydantic-ai-runner-launch
  env:
    ANTHROPIC_BASE_URL: https://api.minimax.io/anthropic
    ANTHROPIC_API_KEY: ${{ secrets.MINIMAX_API_KEY }}
    GH_AW_HARNESS_MODEL: ${{ vars.MODEL }}
tools:
  github:
    mode: gh-proxy
    # PR-scoped surface: read the PR, related issues, repo, and search.
    toolsets: [pull_requests, repos, search, issues]
safe-outputs:
  activation-comments: false
  noop:
  create-pull-request-review-comment:
    max: 30
  submit-pull-request-review:
    max: 1
timeout-minutes: 90
imports:
  - shared/otel-logfire.md
pre-steps:
  # Setting engine.command makes gh-aw skip ALL engine installation steps,
  # which also drops the bundled AWF firewall binary install. Re-run gh-aw's
  # own installer (the same call it makes for non-custom-command jobs).
  - name: Install AWF firewall binary (skipped by custom engine.command)
    run: bash "${RUNNER_TEMP}/gh-aw/actions/install_awf_binary.sh" v0.25.46

pre-agent-steps:
  # Stage the committed launcher script at gh-aw's exec-able
  # /tmp/gh-aw/bin/ path. Runs in pre-agent-steps (not pre-steps) because
  # gh-aw's repository checkout happens between pre-steps and
  # pre-agent-steps, and this step reads from .github/scripts/ in the
  # workspace.
  - name: Stage Pydantic AI harness launcher
    run: |
      mkdir -p /tmp/gh-aw/bin
      install -m 755 .github/scripts/pydantic-ai-runner-launch.sh /tmp/gh-aw/bin/pydantic-ai-runner-launch
  # Warm the harness's uv script environment on the OPEN network so the
  # firewalled agent reuses a warm cache (non-fatal on failure).
  - name: Pre-warm Pydantic AI harness uv environment
    run: bash .github/scripts/prewarm-pydantic-ai-runner.sh
  # Pre-fetch PR context into `/tmp/gh-aw/.review-context/`: pr-details, PR
  # comments, review threads (with annotated diff hunks + resolved/outdated
  # state), annotated per-file diffs, related issues, AGENTS.md excerpts for
  # changed dirs, file orderings for sub-agent fan-out, and a PR-size summary.
  # The agent reads these files instead of calling the GitHub API at run time.
  # Non-fatal: missing context just reduces signal.
  #
  # The script lives at scripts/ (NOT .github/scripts/) because gh-aw's
  # "Save/Restore agent config folders from base branch" step snapshots and
  # restores `.github/` (and other managed agent-config folders) from the
  # BASE branch — making any new file added under those folders unreliable
  # for steps that run after the restore. `scripts/` is outside that set,
  # matching where the legacy reviewer's gather-review-context.sh already
  # lives. The script is a fork of scripts/gather-review-context.sh — see
  # the TODO at the top of the fork.
  - name: Gather PR review context
    if: ${{ github.event.pull_request.number }}
    env:
      GH_TOKEN: ${{ github.token }}
      PR_NUMBER: ${{ github.event.pull_request.number }}
      REPO: ${{ github.repository }}
    run: |
      set -uo pipefail
      # Diagnostic: log the workspace state of both scripts/ and .github/scripts/
      # so we can verify the gh-aw restore-from-base step left them untouched.
      echo "::group::workspace inventory at gather time"
      ls -la scripts/ 2>&1 | head -20 || true
      echo "---"
      ls -la .github/scripts/ 2>&1 | head -20 || true
      echo "::endgroup::"
      script=scripts/gather-pydantic-ai-review-context.sh
      if [ -x "$script" ]; then
        "$script" "$PR_NUMBER" "$REPO" \
          || echo "::warning::${script} failed; reviewer will run with less context"
      else
        echo "::warning::${script} not present; reviewer will run with less context"
      fi

jobs:
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
            .github/workflows/shared/prompts/pydantic-ai-pr-review.md
          sparse-checkout-cone-mode: false
      - name: Resolve agent prompt (Logfire managed variable, else committed default)
        id: resolve
        uses: ./.github/actions/fetch-dynamic-prompt
        with:
          logfire-variable-key: gh_aw_pydantic_ai_pr_review_prompt
          default-prompt-file: .github/workflows/shared/prompts/pydantic-ai-pr-review.md
          logfire-read-key: ${{ secrets.LOGFIRE_READ_EXTERNAL_VARIABLES }}
          logfire-base-url: ${{ secrets.LOGFIRE_URL || vars.LOGFIRE_URL || 'https://logfire-api.pydantic.dev' }}
---

${{ needs.fetch_dynamic_prompt.outputs.dynamic_prompt }}
