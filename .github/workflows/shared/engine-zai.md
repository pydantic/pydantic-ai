---
# Shared public CLAI 2 engine with repository workflow policy and Z.ai routing.
# Each workflow declares only engine.model. Keep identity and credentials here:
# gh-aw validates root engine.env before registering imported engines.
#
# ANTHROPIC_BASE_URL MUST be a compile-time literal (not a ${{ vars.* }}
# expression): gh-aw derives the api-proxy target host AND the
# `--anthropic-api-base-path` from its parsed URL path at compile time. With a
# vars expression the path can't be parsed, so the proxy drops the `/api/anthropic`
# prefix and the upstream returns 404. The `ZAI_API_KEY` secret is injected as
# `ANTHROPIC_API_KEY` by the AWF api-proxy and excluded from the agent container.
# Z.AI Coding Plan exposes an Anthropic-compatible API at https://api.z.ai/api/anthropic.
#
# The public definition installs its published runtime. The repository policy is
# staged outside the checkout and selected through the public PAI_AGENT seam.
#
# Required repo variable:
#   GH_AW_MODEL — unchanged model ID, prefixed with anthropic/ by each workflow.
# Required secret:
#   ZAI_API_KEY — API key injected by the AWF api-proxy.
#
# Usage:
#   imports:
#     - shared/engine-zai.md
imports:
  - pydantic/pydantic-ai/src/pydantic_ai_harness/gh-aw/pydantic.md@29bc2db4d2e109740999d3572c626b8919153ddc
runtimes:
  uv: {}
# Do not add a `models.providers` pricing overlay. gh-aw compiles that block into
# `apiProxy.providers`, which is the AWF API proxy's *AI-credits accounting* table rather
# than a reporting field: pricing a model there starts charging it credits. AWF then
# enforces a hard cap of 10,000 credits per run that nothing in this
# frontmatter can lift — `max-ai-credits` only omits gh-aw's own budget, and the schema
# types it `exclusiveMinimum: 0`. Negative configurable caps do not remove that
# firewall cap. Leave provider pricing unchanged for this migration.
#
# Nothing here needs that overlay: the weekly spend report reads token counts from each
# run's `agent_usage.json` artifact, never a pricing table.
max-ai-credits: -1
sandbox:
  agent:
    token-steering: false
engine:
  id: pydantic-ai
  version: '0.56.0'
  env:
    ANTHROPIC_BASE_URL: https://api.z.ai/api/anthropic
    ANTHROPIC_API_KEY: ${{ secrets.ZAI_API_KEY }}
    PAI_AGENT: pydantic_ai_gh_aw_shim.agent:agent
    PYTHONPATH: /tmp/gh-aw/workflow-python
    XDG_CONFIG_HOME: /tmp/gh-aw/clai2-config
    GITHUB_WORKFLOW: ${{ github.workflow }}
    PYDANTIC_AI_TRIGGER_EVENT: ${{ github.event_name }}
    PYDANTIC_AI_RUN_ATTEMPT: ${{ github.run_attempt }}
    # A `_KEY` suffix is treated as a secret and removed by the custom-engine sandbox.
    PYDANTIC_AI_TASK_ID: ${{ github.workflow }}:${{ github.event_name }}:${{ github.event.pull_request.number || github.event.issue.number || github.event.workflow_run.head_branch || github.ref_name }}:${{ github.event.pull_request.head.sha || github.event.workflow_run.head_sha || github.sha }}:${{ github.event.comment.id || github.event.issue.id || (github.event_name == 'workflow_dispatch' && github.run_id) || '' }}
    # An outer retry repeats the task and can duplicate investigation or output.
    GH_AW_HARNESS_MAX_RETRIES: "0"
safe-outputs:
  threat-detection:
    # Detection has an independent budget and the same unknown-model constraint.
    max-ai-credits: -1
    # Detection uses the stateful Claude CLI, so it retains normal recovery.
    engine:
      id: claude
      version: '2.1.227'
      model: ${{ vars.GH_AW_MODEL }}
      env:
        ANTHROPIC_BASE_URL: https://api.z.ai/api/anthropic
        ANTHROPIC_API_KEY: ${{ secrets.ZAI_API_KEY }}
        GH_AW_HARNESS_MAX_RETRIES: "3"
        CLAUDE_CODE_ENABLE_TELEMETRY: "1"
        OTEL_LOGS_EXPORTER: otlp
        OTEL_METRICS_EXPORTER: none
        OTEL_EXPORTER_OTLP_PROTOCOL: http/protobuf
        OTEL_SERVICE_NAME: gh-aw.threat-detection
---
