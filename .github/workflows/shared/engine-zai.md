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
# prefix and the upstream returns 404. The `ZAI_API_KEY` secret is injected as
# `ANTHROPIC_API_KEY` by the AWF api-proxy and excluded from the agent container.
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
runtimes:
  uv: {}
# Do not add a `models.providers` pricing overlay. gh-aw compiles that block into
# `apiProxy.providers`, which is the AWF API proxy's *AI-credits accounting* table rather
# than a reporting field: pricing a model there starts charging it credits. AWF then
# enforces a hard cap of 10,000 credits per run that nothing in this
# frontmatter can lift — `max-ai-credits` only omits gh-aw's own budget, and the schema
# types it `exclusiveMinimum: 0`. One ordinary request costs tens of thousands, so every
# agent job 403s on its first call with `ai_credits_limit_exceeded`.
#
# Nothing here needs that overlay: the weekly spend report reads token counts from each
# run's `agent_usage.json` artifact, never a pricing table.
max-ai-credits: -1
engine:
  id: claude
  model: ${{ vars.GH_AW_MODEL }}
  command: /tmp/gh-aw/bin/pydantic-ai-runner-launch
  env:
    ANTHROPIC_BASE_URL: https://api.z.ai/api/anthropic
    ANTHROPIC_API_KEY: ${{ secrets.ZAI_API_KEY }}
    GITHUB_WORKFLOW: ${{ github.workflow }}
    PYDANTIC_AI_TRIGGER_EVENT: ${{ github.event_name }}
    PYDANTIC_AI_RUN_ATTEMPT: ${{ github.run_attempt }}
    PYDANTIC_AI_TASK_KEY: ${{ github.workflow }}:${{ github.event_name }}:${{ github.event.pull_request.number || github.event.issue.number || github.event.workflow_run.head_branch || github.ref_name }}:${{ github.event.pull_request.head.sha || github.event.workflow_run.head_sha || github.sha }}:${{ github.event.comment.id || github.event.issue.id || (github.event_name == 'workflow_dispatch' && github.run_id) || '' }}
    # The custom shim is stateless, so an outer retry repeats the whole task.
    GH_AW_HARNESS_MAX_RETRIES: "0"
safe-outputs:
  threat-detection:
    # Detection has an independent budget and the same unknown-model constraint.
    max-ai-credits: -1
    # Detection uses the stateful Claude CLI, so it retains normal recovery.
    engine:
      id: claude
      model: ${{ vars.GH_AW_MODEL }}
      env:
        ANTHROPIC_BASE_URL: https://api.z.ai/api/anthropic
        ANTHROPIC_API_KEY: ${{ secrets.ZAI_API_KEY }}
        GH_AW_HARNESS_MAX_RETRIES: "3"
---
