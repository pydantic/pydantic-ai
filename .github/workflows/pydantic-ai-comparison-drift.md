---
emoji: "🔎"
name: "Pydantic AI Comparison Drift"
description: "Re-research Pydantic AI comparison pages against first-party competitor sources and file one issue with proposed edits. Runs on the Pydantic AI gh-aw shim; the task prompt is iterable from a Logfire managed variable."
on:
  schedule:
    - cron: '17 9 1 * *'
  workflow_dispatch:
    inputs:
      recovery_issue:
        type: string
        required: false
        description: 'Open workflow incident to verify with this run'
permissions:
  contents: read
  issues: read
  pull-requests: read
concurrency:
  group: ${{ github.workflow }}-comparison-drift
  cancel-in-progress: true
tools:
  github:
    mode: gh-proxy
    toolsets: [default]
network:
  # Competitor docs this workflow re-checks. Kept here, not in the shared allowlist,
  # so only this workflow can reach them.
  allowed:
    - docs.langchain.com
    - openai.github.io
    - adk.dev
    - docs.crewai.com
    - docs.agno.com
    - code.claude.com
    - mastra.ai
    - ai-sdk.dev
    - docs.livekit.io
    - docs.pipecat.ai
    - elevenlabs.io
safe-outputs:
  footer: false
  activation-comments: false
  report-failure-as-issue: false
  noop:
    report-as-issue: false
  missing-tool:
    create-issue: false
  missing-data:
    create-issue: false
  report-incomplete:
    create-issue: false
  create-issue:
    max: 1
    title-prefix: "[comparison-drift] "
    labels: [comparison-drift]
    close-older-key: "[comparison-drift]"
    close-older-issues: false
    expires: 30d
timeout-minutes: 60
env:
  # Must equal `timeout-minutes` above. The shim subtracts teardown headroom from it
  # so the agent stops itself and emits a result instead of being killed mid-flight.
  # gh-aw's own `GH_AW_TIMEOUT_MINUTES` is set only on the failure-handler step and
  # never reaches the agent container, hence this duplicate; `agentic_workflow_guard.py`
  # fails the build if the two ever diverge.
  PYDANTIC_AI_JOB_TIMEOUT_MINUTES: "60"
if: ${{ needs.provider_health.outputs.ready == 'true' }}
# Keep in this file, not an import: see `daily-aic-guardrail` in `.github/workflows/AGENTS.md`.
max-daily-ai-credits: -1
imports:
  - shared/network-vendor-domains.md
  - shared/otel-logfire.md
  - shared/tool-hints.md
  - shared/repo-context.md
  - shared/rigor.md
  - shared/checkout.md
  - shared/engine-zai.md
  - shared/provider-health.md
  - shared/pre-steps.md
  - shared/pre-agent-steps.md
  - shared/issue-filing-context.md
  - shared/security-findings.md

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
            .github/workflows/shared/prompts/pydantic-ai-comparison-drift.md
          sparse-checkout-cone-mode: false
      - name: Resolve agent prompt (Logfire managed variable, else committed default)
        id: resolve
        uses: ./.github/actions/fetch-dynamic-prompt
        with:
          logfire-variable-key: gh_aw_pydantic_ai_comparison_drift_prompt
          default-prompt-file: .github/workflows/shared/prompts/pydantic-ai-comparison-drift.md
          logfire-read-key: ${{ secrets.LOGFIRE_READ_EXTERNAL_VARIABLES }}
          logfire-base-url: ${{ secrets.LOGFIRE_URL || vars.LOGFIRE_URL || 'https://logfire-api.pydantic.dev' }}
---
<!-- provider_health must run before activation: ${{ needs.provider_health.outputs.ready }} -->


${{ needs.fetch_dynamic_prompt.outputs.dynamic_prompt }}
