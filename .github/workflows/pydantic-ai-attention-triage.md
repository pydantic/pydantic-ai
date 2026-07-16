---
emoji: "👀"
name: "Pydantic AI Attention Triage"
description: "Conservatively decide whether an issue or PR needs a maintainer to act next."
on:
  issues:
    types: [opened, edited, reopened]
  issue_comment:
    types: [created]
  pull_request:
    types: [opened, edited, reopened, synchronize, ready_for_review, converted_to_draft]
  pull_request_review:
    types: [submitted]
  pull_request_review_comment:
    types: [created]
  check_suite:
    types: [completed]
  roles: all
user-rate-limit:
  max-runs-per-window: 3
  window: 60
permissions:
  contents: read
  issues: read
  pull-requests: read
  checks: read
concurrency:
  group: attention-triage-${{ github.event.issue.number || github.event.pull_request.number || github.event.check_suite.pull_requests[0].number || github.run_id }}
  cancel-in-progress: true
env:
  PYDANTIC_AI_DYNAMIC_WORKFLOW: attention-triage
  ATTENTION_TRIAGE_STAGED: ${{ vars.ATTENTION_TRIAGE_STAGED || 'true' }}
network:
  allowed:
    - defaults
    - python
    - api.minimax.io
tools:
  github:
    mode: gh-proxy
    toolsets: [issues, pull_requests, repos]
    min-integrity: none
    allowed-repos: [pydantic/pydantic-ai]
safe-outputs:
  footer: false
  activation-comments: false
  report-failure-as-issue: false
  noop:
  jobs:
    record-attention-decision:
      description: "Record who must take the next meaningful action. Host policy validates every mutation."
      runs-on: ubuntu-latest
      if: needs.detection.result == 'success' && needs.detection.outputs.detection_success == 'true'
      permissions:
        contents: read
        issues: write
        pull-requests: write
      inputs:
        item_number:
          description: "Issue or pull request number"
          required: true
          type: string
        next_actor:
          description: "Who must take the next meaningful action"
          required: true
          type: choice
          options: [maintainer, contributor, automation, none, uncertain]
        confidence:
          description: "Use high only when the evidence is clear"
          required: true
          type: choice
          options: [high, medium, low]
      steps:
        - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2
          if: >-
            env.ATTENTION_TRIAGE_STAGED != 'true' &&
            (github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository)
          with:
            persist-credentials: false
            ref: ${{ github.event.repository.default_branch }}
        - name: Verify the default-branch policy installation
          id: policy
          if: >-
            env.ATTENTION_TRIAGE_STAGED != 'true' &&
            (github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository)
          env:
            GH_TOKEN: ${{ github.token }}
            DEFAULT_BRANCH: ${{ github.event.repository.default_branch }}
          run: |
            if [[ -f .github/scripts/issue_pr_attention_monitor.py ]]; then
              echo 'ready=true' >> "$GITHUB_OUTPUT"
            elif gh api "repos/${GITHUB_REPOSITORY}/contents/.github/workflows/pydantic-ai-attention-triage.lock.yml?ref=${DEFAULT_BRANCH}" >/dev/null 2>&1; then
              echo '::error::The installed attention workflow is missing its policy script.'
              exit 1
            else
              echo 'ready=false' >> "$GITHUB_OUTPUT"
              echo '::notice::Skipping until the workflow reaches the default branch.'
            fi
        - name: Apply decision through deterministic policy
          if: >-
            env.ATTENTION_TRIAGE_STAGED != 'true' &&
            (github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository) &&
            steps.policy.outputs.ready == 'true'
          env:
            GITHUB_TOKEN: ${{ github.token }}
          run: python .github/scripts/issue_pr_attention_monitor.py apply-decisions
        - name: Record shadow decision
          if: env.ATTENTION_TRIAGE_STAGED == 'true'
          run: |
            count=$(jq '[.items[] | select(.type == "record_attention_decision")] | length' "$GH_AW_AGENT_OUTPUT")
            echo "Recorded ${count} decision(s); no repository state was changed." >> "$GITHUB_STEP_SUMMARY"
timeout-minutes: 20
imports:
  - shared/tool-hints.md
  - shared/repo-context.md
  - shared/rigor.md
  - shared/engine-minimax.md
  - shared/pre-steps.md
  - shared/pre-agent-steps.md
---

# Decide who must act next

Inspect only the triggering open issue or PR. Fork PRs are intentionally picked up from the safe
base-repository `check_suite` event instead of a secretless fork event. For a completed check suite,
inspect only its associated open PRs, at most five. Ignore bot-authored events.

Decide whether the **next meaningful action must come from a maintainer**. Age, validity, importance,
or an unanswered conversation alone are not enough. Select:

- `maintainer` when a maintainer must now review, decide scope or architecture, merge or close, answer
  a blocked contributor, or otherwise make the next project decision;
- `contributor` when the author or reporter must provide information or revise code;
- `automation` when CI, Pydanty, or another automated process is the next actor;
- `none` when no concrete action is due;
- `uncertain` when evidence conflicts or is incomplete.

Use high confidence sparingly. Only a high-confidence decision can change the action label; medium and
low confidence abstain. Maintainer label actions are durable overrides enforced by host policy.

Make obvious classifications directly. If deeper issue validity, repository architecture, PR readiness,
or conflicting evidence is decisive, call `run_workflow` once with the matching `issue_evidence` or
`pr_evidence` specialist. The host bounds specialist calls.

For each eligible item, call `record_attention_decision` exactly once with its number, next actor, and
confidence. Do not add labels, assign users, comment, or include repository content in output text. If
there is no eligible item, use the noop safe output.
