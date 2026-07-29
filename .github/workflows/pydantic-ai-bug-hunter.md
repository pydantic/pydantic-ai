---
emoji: "🐛"
name: "Pydantic AI Bug Hunter"
description: "Find a reproducible, user-impacting bug in pydantic-ai and file a report issue. Runs on the Pydantic AI gh-aw shim; the task prompt is iterable from a Logfire managed variable."
on: weekly on thursday
permissions:
  contents: read
  issues: read
  pull-requests: read
concurrency:
  group: ${{ github.workflow }}-bug-hunter
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
    title-prefix: "[bug-hunter] "
    close-older-key: "[bug-hunter]"
    close-older-issues: false
    expires: 7d
timeout-minutes: 30
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
            .github/workflows/shared/prompts/pydantic-ai-bug-hunter.md
          sparse-checkout-cone-mode: false
      - name: Resolve agent prompt (Logfire managed variable, else committed default)
        id: resolve
        uses: ./.github/actions/fetch-dynamic-prompt
        with:
          logfire-variable-key: gh_aw_pydantic_ai_bug_hunter_prompt
          default-prompt-file: .github/workflows/shared/prompts/pydantic-ai-bug-hunter.md
          logfire-read-key: ${{ secrets.LOGFIRE_READ_EXTERNAL_VARIABLES }}
          logfire-base-url: ${{ secrets.LOGFIRE_URL || vars.LOGFIRE_URL || 'https://logfire-api.pydantic.dev' }}
---

${{ needs.fetch_dynamic_prompt.outputs.dynamic_prompt }}
