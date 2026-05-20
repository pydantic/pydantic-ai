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
# gather-review-context script annotates per-file diffs against the base ref.
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
  # Stage (not install) a launcher at gh-aw's exec-able /tmp/gh-aw/bin path.
  # uv itself is installed by runtimes.uv; this only writes a wrapper file
  # that runs `uv run --script` on the workspace harness at agent time.
  - name: Stage Pydantic AI harness launcher
    run: |
      set -euo pipefail
      mkdir -p /tmp/gh-aw/bin
      cat > /tmp/gh-aw/bin/pydantic-ai-runner-launch <<'WRAP'
      #!/usr/bin/env bash
      set -euo pipefail
      export UV_CACHE_DIR=/tmp/gh-aw/uv/cache
      export UV_PYTHON_INSTALL_DIR=/tmp/gh-aw/uv/python
      export UV_TOOL_DIR=/tmp/gh-aw/uv/tool
      export XDG_DATA_HOME=/tmp/gh-aw/uv/data
      export XDG_CACHE_HOME=/tmp/gh-aw/uv/xdg-cache
      mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_TOOL_DIR" "$XDG_DATA_HOME" "$XDG_CACHE_HOME"
      runner="${GITHUB_WORKSPACE}/.github/scripts/pydantic-ai-runner"
      echo "[harness-launch] cwd=$(pwd) GITHUB_WORKSPACE=${GITHUB_WORKSPACE:-unset} UV_CACHE_DIR=${UV_CACHE_DIR}" >&2
      echo "[harness-launch] runner=${runner} exists=$([ -f "${runner}" ] && echo yes || echo no)" >&2
      uv_bin=""
      if command -v uv >/dev/null 2>&1; then
        uv_bin="$(command -v uv)"
      else
        for c in "${HOME}/.local/bin/uv" "${RUNNER_TOOL_CACHE:-/opt/hostedtoolcache}"/uv/*/*/uv /opt/hostedtoolcache/uv/*/*/uv /home/runner/work/_tool/uv/*/*/uv /usr/local/bin/uv; do
          [ -x "$c" ] && uv_bin="$c" && break
        done
      fi
      if [ -z "${uv_bin}" ]; then
        echo "[harness-launch] FATAL: uv not found; PATH=${PATH}" >&2
        exit 127
      fi
      echo "[harness-launch] using uv=${uv_bin}" >&2
      exec "${uv_bin}" run --script "${runner}" "$@"
      WRAP
      chmod +x /tmp/gh-aw/bin/pydantic-ai-runner-launch

pre-agent-steps:
  # Warm the harness's uv script environment on the OPEN network so the
  # firewalled agent reuses a warm cache.
  - name: Pre-warm Pydantic AI harness uv environment
    run: |
      set -uo pipefail
      export UV_CACHE_DIR=/tmp/gh-aw/uv/cache
      export UV_PYTHON_INSTALL_DIR=/tmp/gh-aw/uv/python
      export UV_TOOL_DIR=/tmp/gh-aw/uv/tool
      export XDG_DATA_HOME=/tmp/gh-aw/uv/data
      export XDG_CACHE_HOME=/tmp/gh-aw/uv/xdg-cache
      mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_TOOL_DIR" "$XDG_DATA_HOME" "$XDG_CACHE_HOME"
      runner="${GITHUB_WORKSPACE}/.github/scripts/pydantic-ai-runner"
      uv_bin=""
      if command -v uv >/dev/null 2>&1; then
        uv_bin="$(command -v uv)"
      else
        for c in "${HOME}/.local/bin/uv" "${RUNNER_TOOL_CACHE:-/opt/hostedtoolcache}"/uv/*/*/uv /opt/hostedtoolcache/uv/*/*/uv /home/runner/work/_tool/uv/*/*/uv /usr/local/bin/uv; do
          [ -x "$c" ] && uv_bin="$c" && break
        done
      fi
      if [ -z "${uv_bin}" ]; then
        echo "::warning::uv not found for pre-warm; agent will install under the firewall"
        exit 0
      fi
      echo "[harness-prewarm] using uv=${uv_bin} cache=${UV_CACHE_DIR}"
      "${uv_bin}" sync --script "${runner}" \
        || echo "::warning::harness uv pre-warm failed; agent will install under the firewall"
  # Pre-fetch PR context using the repo's own gather-review-context.sh —
  # writes `.github/.review-context/` with pr-details, comments, review
  # threads, annotated per-file diffs, related issues, and AGENTS.md
  # excerpts. The agent reads these files instead of calling the GitHub
  # API at run time. Non-fatal: missing context just reduces signal.
  - name: Gather PR review context
    if: ${{ github.event.pull_request.number }}
    env:
      GH_TOKEN: ${{ github.token }}
      PR_NUMBER: ${{ github.event.pull_request.number }}
      REPO: ${{ github.repository }}
    run: |
      set -uo pipefail
      if [ -x scripts/gather-review-context.sh ]; then
        scripts/gather-review-context.sh "$PR_NUMBER" "$REPO" \
          || echo "::warning::gather-review-context.sh failed; reviewer will run with less context"
      else
        echo "::warning::scripts/gather-review-context.sh not present; reviewer will run with less context"
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
