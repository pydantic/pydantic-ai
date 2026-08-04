---
emoji: "🧭"
name: "Pydantic AI Provider Parity Explore"
description: "Explore one cross-cutting capability's support across all providers and file an issue for concrete parity gaps. Runs on the Pydantic AI gh-aw shim; the prompt is iterable from a Logfire managed variable."
on: weekly on tuesday
permissions:
  contents: read
  issues: read
  pull-requests: read
concurrency:
  group: ${{ github.workflow }}-provider-parity-explore
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
    title-prefix: "[provider-parity-explore] "
    labels: [provider-parity-explore]
    close-older-key: "[provider-parity-explore]"
    close-older-issues: false
    expires: 7d
timeout-minutes: 30
env:
  # Must equal `timeout-minutes` above. The shim subtracts teardown headroom from it
  # so the agent stops itself and emits a result instead of being killed mid-flight.
  # gh-aw's own `GH_AW_TIMEOUT_MINUTES` is set only on the failure-handler step and
  # never reaches the agent container, hence this duplicate; `agentic_workflow_guard.py`
  # fails the build if the two ever diverge.
  PYDANTIC_AI_JOB_TIMEOUT_MINUTES: "30"
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
            .github/workflows/shared/prompts/pydantic-ai-provider-parity-explore.md
          sparse-checkout-cone-mode: false
      - name: Resolve agent prompt (Logfire managed variable, else committed default)
        id: resolve
        uses: ./.github/actions/fetch-dynamic-prompt
        with:
          logfire-variable-key: gh_aw_pydantic_ai_provider_parity_explore_prompt
          default-prompt-file: .github/workflows/shared/prompts/pydantic-ai-provider-parity-explore.md
          logfire-read-key: ${{ secrets.LOGFIRE_READ_EXTERNAL_VARIABLES }}
          logfire-base-url: ${{ secrets.LOGFIRE_URL || vars.LOGFIRE_URL || 'https://logfire-api.pydantic.dev' }}
---

${{ needs.fetch_dynamic_prompt.outputs.dynamic_prompt }}
