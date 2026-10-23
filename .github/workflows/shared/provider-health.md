---
# Trusted host-side Z.AI health check. Keep ZAI_API_KEY in host steps: this job and the
# agent-start recheck below, never the agent container.
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
          PYDANTIC_AI_RECOVERY_ISSUE: ${{ inputs.recovery_issue || '' }}
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
# The agent job can wait hours for its engine concurrency slot after `provider_health`
# decided, and incidents opened in the meantime must still stop inference. Once the job
# holds the slot, these steps rerun the default-branch controller with the gate's identity
# and replace its artifact, so the monitor reconciles the decision the run acted on.
# They run on the host before any workspace script; the key never reaches the container.
# A blocked decision fails the job before the engine starts: a `noop` would still send
# the threat-detection pass to the blocked provider.
pre-agent-steps:
  - name: Recheck Z.AI provider health at agent start
    id: agent_start_health
    env:
      GITHUB_TOKEN: ${{ github.token }}
      ZAI_API_KEY: ${{ secrets.ZAI_API_KEY }}
      PYDANTIC_AI_RECOVERY_ISSUE: ${{ inputs.recovery_issue || '' }}
      GITHUB_WORKFLOW: ${{ github.workflow }}
      PYDANTIC_AI_TRIGGER_EVENT: ${{ github.event_name }}
      PYDANTIC_AI_RUN_ATTEMPT: ${{ github.run_attempt }}
      PYDANTIC_AI_TASK_KEY: ${{ github.workflow }}:${{ github.event_name }}:${{ github.event.pull_request.number || github.event.issue.number || github.event.workflow_run.head_branch || github.ref_name }}:${{ github.event.pull_request.head.sha || github.event.workflow_run.head_sha || github.sha }}:${{ github.event.comment.id || github.event.issue.id || (github.event_name == 'workflow_dispatch' && github.run_id) || '' }}
    run: |
      set -euo pipefail
      mkdir -p "$RUNNER_TEMP/provider-health"
      # Without `ref`, the contents API serves the default branch.
      gh api -H 'Accept: application/vnd.github.raw+json' \
        "repos/${GITHUB_REPOSITORY}/contents/.github/scripts/agent_provider_health.py" \
        > "$RUNNER_TEMP/provider-health/agent_provider_health.py"
      python3 "$RUNNER_TEMP/provider-health/agent_provider_health.py" check \
        --output "$RUNNER_TEMP/provider-health/provider-health.json"
  - name: Replace health decision for the incident monitor
    if: (!cancelled()) && steps.agent_start_health.outputs.ready != ''
    continue-on-error: true
    uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7.0.1
    with:
      name: provider-health
      path: ${{ runner.temp }}/provider-health/provider-health.json
      retention-days: 7
      overwrite: true
  - name: Stop before inference on a blocked provider-health decision
    if: steps.agent_start_health.outputs.ready != 'true'
    env:
      REASON: ${{ steps.agent_start_health.outputs.reason }}
    run: |
      echo "::error::Provider health blocked inference at agent start: ${REASON}"
      exit 1
---

<!-- `provider_health` must precede activation. The top-level gate reads its output. -->
