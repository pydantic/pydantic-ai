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
# so per-token figures here would under-count spend by a factor of 1e6.
#
# Today this block only reaches `GH_AW_INFO_MODEL_COSTS` (run cost reporting):
# gh-aw emits it as `apiProxy.providers` only for AWF >= v0.27.43, which is not
# released yet — see the `max-ai-credits` note below.
models:
  providers:
    anthropic:
      models:
        MiniMax-M3:
          cost:
            input: 0.6
            output: 2.4
            cache_read: 0.12
# Disable the api-proxy's AI-credits budget (and the token steering that rides
# on it). `MiniMax-M3` is absent from both the AWF built-in pricing table and
# its bundled models.dev catalog, so while the budget is active every request
# is rejected with HTTP 400 `unknown_model_ai_credits`.
#
# Neither documented pricing escape hatch reaches the proxy on AWF v0.27.42
# (the version gh-aw v0.83.4 pins, and the newest released):
#   - `models.default-ai-credits-pricing` compiles into
#     `apiProxy.defaultAiCreditsPricing`, but AWF parses that key and then
#     drops it before building the api-proxy environment, so the container
#     never receives `AWF_DEFAULT_AI_CREDITS_PRICING`.
#   - `models.providers` compiles into `apiProxy.providers` only for
#     AWF >= v0.27.43, which has not been released.
# Restore a budget here once AWF >= v0.27.43 ships.
max-ai-credits: -1
engine:
  id: claude
  model: ${{ vars.GH_AW_MODEL }}
  command: /tmp/gh-aw/bin/pydantic-ai-runner-launch
  env:
    ANTHROPIC_BASE_URL: https://api.minimax.io/anthropic
    ANTHROPIC_API_KEY: ${{ secrets.MINIMAX_API_KEY }}
safe-outputs:
  threat-detection:
    # The detection sub-agent runs the same MiniMax model behind its own budget
    # and deliberately does not inherit `engine.max-ai-credits`, so it needs the
    # same switch — otherwise detection 400s and every safe output is dropped
    # with an "agentic threat detected" banner.
    max-ai-credits: -1
---
