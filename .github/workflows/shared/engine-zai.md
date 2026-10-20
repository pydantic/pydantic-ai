---
# Shared runtime + engine config for the Pydantic AI gh-aw shim (Z.AI Coding Plan backend).
#
# Registers as the built-in `claude` engine and only overrides `command`, so
# gh-aw runs its full Claude proxy + credential-injection machinery.
#
# ANTHROPIC_BASE_URL MUST be a compile-time literal (not a ${{ vars.* }}
# expression): gh-aw derives the api-proxy target host AND the
# `--anthropic-api-base-path` from its parsed URL path at compile time. With a
# vars expression the path can't be parsed, so the proxy drops the `/api/anthropic`
# prefix and the upstream returns 404. Only ANTHROPIC_API_KEY stays a secret
# (injected by the AWF api-proxy, excluded from the agent container).
# Z.AI Coding Plan exposes an Anthropic-compatible API at https://api.z.ai/api/anthropic.
#
# The checked-out workspace is mounted no-exec in the AWF sandbox, so a
# pre-step stages a launcher in gh-aw's exec-able /tmp/gh-aw/bin that runs
# `uv run --script` against the workspace harness.
#
# Required repo variable:
#   GH_AW_MODEL — model name forwarded as `--model <name>` to the harness.
# Required secret:
#   ZAI_API_KEY — API key injected by the AWF api-proxy.
#
# Usage:
#   imports:
#     - shared/engine-zai.md
network:
  allowed:
    - api.z.ai
runtimes:
  uv: {}
engine:
  id: claude
  model: ${{ vars.GH_AW_MODEL }}
  command: /tmp/gh-aw/bin/pydantic-ai-runner-launch
  env:
    # Match v2's generic Anthropic output default for review responses.
    GH_AW_MAX_OUTPUT_TOKENS: '16384'
    ANTHROPIC_BASE_URL: https://api.z.ai/api/anthropic
    ANTHROPIC_API_KEY: ${{ secrets.ZAI_API_KEY }}
pre-agent-steps:
  - name: Prepare reviewer runtime settings support
    # gh-aw restores .github from the base branch before these steps.
    # Bootstrap reviewed settings support only for the known v1 runtime;
    # an updated base runtime remains authoritative.
    run: |
      runtime=.github/scripts/pydantic_ai_gh_aw_shim/cli.py
      if [ "$(git hash-object "$runtime")" = aa6f9359ef548c2038c6a5eb7bd70e50a20cf321 ]; then
        git show 392ca4204325c70315c3e1d92f3930d0caee2c61:"$runtime" > "${RUNNER_TEMP}/review-cli.py"
        install -m 644 "${RUNNER_TEMP}/review-cli.py" "$runtime"
      fi
---
