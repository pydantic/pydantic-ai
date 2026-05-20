---
emoji: "🐛"
name: "Pydantic AI Bug Hunter"
description: "Find a reproducible, user-impacting bug in pydantic-ai and file a report issue. Runs on the Pydantic AI harness engine; the task prompt is iterable from a Logfire managed variable."
# Fuzzy schedule: gh-aw scatters the daily run to a repo-stable off-peak time
# (avoids the top-of-hour stampede) and auto-adds workflow_dispatch.
on: daily
permissions:
  contents: read
  issues: read
  pull-requests: read
# Full git history: the agent's repo checkout is shallow by default, so
# `git log --since=...` would see only the tip commit. fetch-depth: 0 gives
# the agent the real commit history it needs to find recent changes.
checkout:
  fetch-depth: 0
concurrency:
  group: ${{ github.workflow }}-bug-hunter
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
# `--anthropic-api-base-path` from its parsed URL path at compile time. With a
# vars expression the path can't be parsed, so the proxy drops the `/anthropic`
# prefix and the upstream returns 404. Only ANTHROPIC_API_KEY stays a secret
# (injected by the AWF api-proxy, excluded from the agent container). MiniMax
# exposes an Anthropic-compatible API at https://api.minimax.io/anthropic.
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
    title-prefix: "[bug-hunter] "
    close-older-key: "[bug-hunter]"
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
      # setup-uv points UV_CACHE_DIR at ${RUNNER_TEMP}/setup-uv-cache, which is
      # not writable by the chrooted sandbox user (UID 1001). Only /tmp/gh-aw
      # is owned by that user, so redirect every uv-writable dir there.
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
  # Warm the harness's uv script environment on the OPEN network. This hook
  # runs after checkout + Setup uv but before the firewalled agent step, into
  # the same uv dirs the in-sandbox launcher uses, so the agent run reuses the
  # warm cache instead of depending on PyPI access through the AWF firewall.
  # Strictly non-fatal: on any failure the sandboxed `uv run --script` (with
  # the `python` allowlist above) still installs.
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
            .github/workflows/shared/prompts/pydantic-ai-bug-hunter.md
          sparse-checkout-cone-mode: false
      - name: Resolve agent prompt (Logfire managed variable, else committed default)
        id: resolve
        uses: ./.github/actions/fetch-dynamic-prompt
        with:
          logfire-variable-key: gh_aw_pydantic_ai_bug_hunter_prompt
          default-prompt-file: .github/workflows/shared/prompts/pydantic-ai-bug-hunter.md
          logfire-read-key: ${{ secrets.LOGFIRE_READ_EXTERNAL_VARIABLES }}
          logfire-base-url: ${{ secrets.LOGFIRE_URL || vars.LOGFIRE_URL || 'https://logfire-api.pydantic.dev' }}
---

${{ needs.fetch_dynamic_prompt.outputs.dynamic_prompt }}
