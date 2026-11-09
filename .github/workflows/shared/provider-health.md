---
# Trusted host-side Z.AI health check. Keep ZAI_API_KEY in the provider_health job, the agent-start recheck below
# and the record step of provider_health_record, never in the agent container.
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
  # The monitor trusts only a terminal result written outside the agent container. After the agent job, including a
  # failed one, this job reads the `agent` artifact's `agent-session.jsonl`, which gh-aw's parser wrote on the agent
  # host from the engine's verified recording. It confirms a provider-wide failure with its own Z.ai check and adds
  # the record to the `provider-health` artifact. It is a separate job because gh-aw allows no secret in post-steps.
  # gh-aw also runs it only after threat detection passes cleanly. `continue-on-error` keeps a failed record from
  # failing the run. Without a record, the monitor reads the `agent` artifact, which never confirms a provider failure.
  provider_health_record:
    name: Record the terminal agent result on the host
    needs: [agent]
    if: (!cancelled()) && needs.agent.result != 'skipped'
    continue-on-error: true
    runs-on: ubuntu-latest
    timeout-minutes: 5
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2
        with:
          repository: ${{ github.repository }}
          ref: ${{ github.event.repository.default_branch }}
          persist-credentials: false
          sparse-checkout: .github/scripts/agent_provider_health.py
          sparse-checkout-cone-mode: false
      - name: Download the agent's terminal session record
        id: agent_artifact
        continue-on-error: true
        uses: actions/download-artifact@9000827ccba6bdab643e8b6fd33ac0654aef8333 # v8.0.2
        with:
          name: agent
          path: agent
      - name: Download the health decision the run acted on
        id: health_artifact
        continue-on-error: true
        uses: actions/download-artifact@9000827ccba6bdab643e8b6fd33ac0654aef8333 # v8.0.2
        with:
          name: provider-health
          path: provider-health
      # `-I` ignores user site-packages and `PYTHON*` variables.
      - name: Record the terminal agent result
        if: steps.agent_artifact.outcome == 'success' && steps.health_artifact.outcome == 'success'
        env:
          ZAI_API_KEY: ${{ secrets.ZAI_API_KEY }}
          GITHUB_WORKFLOW: ${{ github.workflow }}
          PYDANTIC_AI_TRIGGER_EVENT: ${{ github.event_name }}
          PYDANTIC_AI_RUN_ATTEMPT: ${{ github.run_attempt }}
          PYDANTIC_AI_TASK_KEY: ${{ github.workflow }}:${{ github.event_name }}:${{ github.event.pull_request.number || github.event.issue.number || github.event.workflow_run.head_branch || github.ref_name }}:${{ github.event.pull_request.head.sha || github.event.workflow_run.head_sha || github.sha }}:${{ github.event.comment.id || github.event.issue.id || (github.event_name == 'workflow_dispatch' && github.run_id) || '' }}
        run: |
          python3 -I .github/scripts/agent_provider_health.py record \
            --agent-session agent/agent-session.jsonl \
            --output provider-health/run-result.json
      - name: Add the host-recorded agent result to the health decision artifact
        if: hashFiles('provider-health/run-result.json') != ''
        uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7.0.1
        with:
          name: provider-health
          path: |
            provider-health/provider-health.json
            provider-health/run-result.json
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
