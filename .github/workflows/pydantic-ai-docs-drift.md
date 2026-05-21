---
emoji: "📚"
name: "Pydantic AI Docs Drift"
description: "Detect code changes that require documentation updates and file an issue. Runs on the Pydantic AI harness engine; the task prompt is iterable from a Logfire managed variable."
# Fuzzy schedule: docs drift accumulates slowly, so weekly (not daily).
# gh-aw scatters the run to a repo-stable time and auto-adds workflow_dispatch.
on: weekly on monday
permissions:
  contents: read
  issues: read
  pull-requests: read
# Full git history: the agent's repo checkout is shallow by default, so
# `git log --since=...` would see only the tip commit. fetch-depth: 0 gives
# the agent the real commit history it needs to detect documentation drift.
checkout:
  fetch-depth: 0
concurrency:
  group: ${{ github.workflow }}-docs-drift
  cancel-in-progress: true
network:
  allowed:
    - defaults
    # Python/PyPI ecosystem — the harness installs its deps via `uv` at agent
    # time; allow them through the AWF firewall. (`defaults` already covers most
    # PyPI mirrors here, but the explicit ecosystem id is robust + preferred.)
    - python
    # ANTHROPIC_BASE_URL is a compile-time literal (below) so gh-aw already
    # auto-allowlists the host; this explicit entry is a harmless safety net.
    - api.minimax.io
# We register as the built-in `claude` engine and only override `command`, so
# gh-aw runs its full Claude proxy + credential-injection machinery for us.
# ANTHROPIC_BASE_URL MUST be a compile-time literal (not a ${{ vars.* }}
# expression): gh-aw derives the api-proxy target host AND the
# `--anthropic-api-base-path` from its parsed URL path at compile time. Only
# ANTHROPIC_API_KEY stays a secret (injected by the AWF api-proxy, excluded
# from the agent container). MiniMax exposes an Anthropic-compatible API at
# https://api.minimax.io/anthropic.
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
    toolsets: [default]
safe-outputs:
  activation-comments: false
  noop:
  create-issue:
    max: 1
    title-prefix: "[docs-drift] "
    close-older-key: "[docs-drift]"
    close-older-issues: false
    expires: 7d
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
            .github/workflows/shared/prompts/pydantic-ai-docs-drift.md
          sparse-checkout-cone-mode: false
      - name: Resolve agent prompt (Logfire managed variable, else committed default)
        id: resolve
        uses: ./.github/actions/fetch-dynamic-prompt
        with:
          logfire-variable-key: gh_aw_pydantic_ai_docs_drift_prompt
          default-prompt-file: .github/workflows/shared/prompts/pydantic-ai-docs-drift.md
          logfire-read-key: ${{ secrets.LOGFIRE_READ_EXTERNAL_VARIABLES }}
          logfire-base-url: ${{ secrets.LOGFIRE_URL || vars.LOGFIRE_URL || 'https://logfire-api.pydantic.dev' }}
---

${{ needs.fetch_dynamic_prompt.outputs.dynamic_prompt }}
