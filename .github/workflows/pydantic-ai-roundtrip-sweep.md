---
emoji: "♻️"
name: "Pydantic AI Round-Trip Sweep"
description: "Find serialize/deserialize state-loss bugs across a message round-trip boundary and file a reproducible report. Runs on the Pydantic AI gh-aw shim; the prompt is iterable from a Logfire managed variable."
on: daily
permissions:
  contents: read
  issues: read
  pull-requests: read
concurrency:
  group: ${{ github.workflow }}-roundtrip-sweep
  cancel-in-progress: true
tools:
  github:
    mode: gh-proxy
    toolsets: [default]
safe-outputs:
  footer: false
  activation-comments: false
  # Engine/model failures are tracked as ERROR spans in Logfire (service_name
  # `gh-aw.pydantic-ai-roundtrip-sweep`) via the otel-logfire import + the shim's
  # `instrument_pydantic_ai`, so we don't also file an auto-generated failure issue.
  report-failure-as-issue: false
  noop:
  create-issue:
    max: 1
    title-prefix: "[roundtrip-sweep] "
    labels: [roundtrip-sweep]
    close-older-key: "[roundtrip-sweep]"
    close-older-issues: false
    expires: 7d
# 45 rather than the 30 the other sweeps use: this one writes and executes
# round-trip reproductions, so its turns are dominated by model latency
# (~130 tool calls in a run). At 30 it hit the wall every day from 2026-07-06
# and filed nothing (#6766 F6).
timeout-minutes: 45
env:
  # Must equal `timeout-minutes` above — the shim subtracts teardown headroom
  # from it so the agent stops itself and emits a result instead of being killed
  # mid-flight. gh-aw's own `GH_AW_TIMEOUT_MINUTES` is set only on the failure
  # handler step and never reaches the agent container, hence this duplicate.
  # `agentic_workflow_guard.py` fails the build if the two ever diverge.
  PYDANTIC_AI_JOB_TIMEOUT_MINUTES: "45"
imports:
  - shared/network-vendor-domains.md
  - shared/otel-logfire.md
  - shared/tool-hints.md
  - shared/repo-context.md
  - shared/rigor.md
  - shared/adversarial-review.md
  - shared/checkout.md
  - shared/engine-minimax.md
  - shared/pre-steps.md
  - shared/pre-agent-steps.md
  - shared/issue-filing-context.md

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
            .github/workflows/shared/prompts/pydantic-ai-roundtrip-sweep.md
          sparse-checkout-cone-mode: false
      - name: Resolve agent prompt (Logfire managed variable, else committed default)
        id: resolve
        uses: ./.github/actions/fetch-dynamic-prompt
        with:
          logfire-variable-key: gh_aw_pydantic_ai_roundtrip_sweep_prompt
          default-prompt-file: .github/workflows/shared/prompts/pydantic-ai-roundtrip-sweep.md
          logfire-read-key: ${{ secrets.LOGFIRE_READ_EXTERNAL_VARIABLES }}
          logfire-base-url: ${{ secrets.LOGFIRE_URL || vars.LOGFIRE_URL || 'https://logfire-api.pydantic.dev' }}
---

${{ needs.fetch_dynamic_prompt.outputs.dynamic_prompt }}
