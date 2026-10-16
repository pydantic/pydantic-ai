---
# Trusted host-side Z.AI health check. Keep ZAI_API_KEY in this runner job.
jobs:
  provider_health:
    name: Check Z.AI provider health
    runs-on: ubuntu-latest
    timeout-minutes: 5
    permissions:
      contents: read
      issues: read
    outputs:
      ready: ${{ steps.health.outputs.ready }}
      reason: ${{ steps.health.outputs.reason }}
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2
        with:
          repository: ${{ github.repository }}
          ref: ${{ github.event.repository.default_branch }}
          persist-credentials: false
          sparse-checkout: .github/scripts/agent_provider_health.py
          sparse-checkout-cone-mode: false
      - name: Check quota and active incidents
        id: health
        env:
          GITHUB_TOKEN: ${{ github.token }}
          ZAI_API_KEY: ${{ secrets.ZAI_API_KEY }}
          GITHUB_WORKFLOW: ${{ github.workflow }}
          PYDANTIC_AI_TRIGGER_EVENT: ${{ github.event_name }}
          PYDANTIC_AI_RUN_ATTEMPT: ${{ github.run_attempt }}
          PYDANTIC_AI_TASK_KEY: ${{ github.workflow }}:${{ github.event_name }}:${{ github.event.pull_request.number || github.event.issue.number || github.event.workflow_run.head_branch || github.ref_name }}:${{ github.event.pull_request.head.sha || github.event.workflow_run.head_sha || github.sha }}:${{ github.event.comment.id || github.event.issue.id || (github.event_name == 'workflow_dispatch' && github.run_id) || '' }}
        run: |
          mkdir -p provider-health
          python3 .github/scripts/agent_provider_health.py check --output provider-health/provider-health.json
      - name: Preserve health decision for the incident monitor
        if: always()
        uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7.0.1
        with:
          name: provider-health
          path: provider-health/provider-health.json
          if-no-files-found: warn
          retention-days: 7
          overwrite: true
---

<!-- `provider_health` must precede activation. The top-level gate reads its output. -->
