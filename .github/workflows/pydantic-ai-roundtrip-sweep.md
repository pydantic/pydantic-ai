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
  noop:
  create-issue:
    max: 1
    title-prefix: "[roundtrip-sweep] "
    close-older-key: "[roundtrip-sweep]"
    close-older-issues: false
    expires: 7d
timeout-minutes: 30
imports:
  - shared/network-vendor-domains.md
  - shared/otel-logfire.md
  - shared/tool-hints.md
  - shared/repo-context.md
  - shared/rigor.md
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
