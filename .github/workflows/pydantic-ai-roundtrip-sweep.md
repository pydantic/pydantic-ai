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
timeout-minutes: 30
# AI-credits pricing for MiniMax-M3, in dollars per 1M tokens. Required: the
# model is absent from the AWF api-proxy's built-in table, so without it
# v0.83.4 rejects every request with HTTP 400 `unknown_model_ai_credits`, and
# the guardrail cannot be switched off (`apiProxy.maxAiCredits` is emitted
# unconditionally and must be > 0). gh-aw does not merge this key in from
# imports, so it is declared per workflow rather than in
# shared/engine-minimax.md — keep the rates there in step with these.
models:
  default-ai-credits-pricing:
    input: 0.6
    output: 2.4
    cache_read: 0.12
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
