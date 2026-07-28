---
# Shared runtime + engine config for the Pydantic AI gh-aw shim (MiniMax backend).
#
# Registers as the built-in `claude` engine and only overrides `command`, so
# gh-aw runs its full Claude proxy + credential-injection machinery.
#
# ANTHROPIC_BASE_URL MUST be a compile-time literal (not a ${{ vars.* }}
# expression): gh-aw derives the api-proxy target host AND the
# `--anthropic-api-base-path` from its parsed URL path at compile time. With a
# vars expression the path can't be parsed, so the proxy drops the `/anthropic`
# prefix and the upstream returns 404. Only ANTHROPIC_API_KEY stays a secret
# (injected by the AWF api-proxy, excluded from the agent container).
# MiniMax exposes an Anthropic-compatible API at https://api.minimax.io/anthropic.
#
# The checked-out workspace is mounted no-exec in the AWF sandbox, so a
# pre-step stages a launcher in gh-aw's exec-able /tmp/gh-aw/bin that runs
# `uv run --script` against the workspace harness.
#
# Required repo variable:
#   GH_AW_MODEL — model name forwarded as `--model <name>` to the harness.
# Required secret:
#   MINIMAX_API_KEY — API key injected by the AWF api-proxy.
#
# Usage:
#   imports:
#     - shared/engine-minimax.md
runtimes:
  uv: {}
# MiniMax pricing, for AI-credits accounting (gh-aw v0.83.4 / AWF v0.27.42).
#
# Units are dollars per 1M tokens, matching the models.dev catalog gh-aw
# resolves built-in pricing from. The compiler passes these through unscaled,
# so per-token figures here would under-count spend by a factor of 1e6 and
# silently defeat the `maxAiCredits` ceiling.
#
# This block only reaches `GH_AW_INFO_MODEL_COSTS` (run cost reporting). The
# api-proxy guardrail reads `models.default-ai-credits-pricing`, which gh-aw
# does NOT merge in from imports — so that key is repeated in each workflow
# source instead of living here. Keep the two sets of rates in step.
models:
  providers:
    anthropic:
      models:
        MiniMax-M3:
          cost:
            input: 0.6
            output: 2.4
            cache_read: 0.12
engine:
  id: claude
  model: ${{ vars.GH_AW_MODEL }}
  command: /tmp/gh-aw/bin/pydantic-ai-runner-launch
  env:
    ANTHROPIC_BASE_URL: https://api.minimax.io/anthropic
    ANTHROPIC_API_KEY: ${{ secrets.MINIMAX_API_KEY }}
---
