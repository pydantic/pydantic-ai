---
emoji: "👀"
name: "Pydantic AI Attention Triage"
description: "Conservatively decide whether the next meaningful action on an issue or PR must come from a maintainer. Uses bounded DynamicWorkflow specialists only for genuinely ambiguous cases."
run-name: "Pydantic AI Attention Triage [item:${{ github.event.issue.number || github.event.pull_request.number || github.event.check_suite.pull_requests[0].number || 'none' }}]"
on:
  issues:
    types: [opened, edited, reopened, labeled, unlabeled, assigned, unassigned]
  issue_comment:
    types: [created]
  pull_request:
    types: [opened, edited, reopened, synchronize, ready_for_review, converted_to_draft, labeled, unlabeled, assigned, unassigned]
    forks: ["*"]
  pull_request_review:
    types: [submitted]
  pull_request_review_comment:
    types: [created]
  check_suite:
    types: [completed]
  schedule: every 6h
  workflow_dispatch: {}
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
  group: issue-pr-attention-agent-${{ github.event.issue.number || github.event.pull_request.number || github.event.check_suite.pull_requests[0].number || 'reconcile' }}
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
    toolsets: [issues, pull_requests, repos, search]
    min-integrity: none
    allowed-repos: [pydantic/pydantic-ai]
jobs:
  conclusion:
    pre-steps:
      - name: Check whether threat detection blocked an urgent item
        id: blocked_urgent
        if: >-
          env.ATTENTION_TRIAGE_STAGED != 'true' &&
          (github.event.issue.number != '' || github.event.pull_request.number != '') &&
          needs.detection.result == 'success' &&
          needs.detection.outputs.detection_success != 'true'
        env:
          GH_TOKEN: ${{ secrets.GH_AW_GITHUB_TOKEN || github.token }}
          ITEM_NUMBER: ${{ github.event.issue.number || github.event.pull_request.number }}
        run: |
          labels=$(gh api "repos/${GITHUB_REPOSITORY}/issues/${ITEM_NUMBER}" --jq '.labels[].name')
          if grep -Fxq 'p:1-highest' <<< "$labels"; then
            echo 'urgent=true' >> "$GITHUB_OUTPUT"
          fi
      - name: Report urgent triage blocked by threat detection
        if: steps.blocked_urgent.outputs.urgent == 'true'
        uses: slackapi/slack-github-action@45a88b9581bfab2566dc881e2cd66d334e621e2c # v3.0.3
        with:
          errors: true
          webhook: ${{ secrets.PYDANTIC_AI_TRIAGE_SLACK_WEBHOOK_URL }}
          webhook-type: incoming-webhook
          payload: |
            text: ":warning: Attention triage failed on urgent item <${{ github.server_url }}/${{ github.repository }}/issues/${{ github.event.issue.number || github.event.pull_request.number }}|#${{ github.event.issue.number || github.event.pull_request.number }}>. <${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}|View run>."
safe-outputs:
  footer: false
  activation-comments: false
  report-failure-as-issue: false
  noop:
  jobs:
    record-attention-decision:
      description: "Record one conservative next-actor decision. Deterministic host code validates overrides, labels, assignment, clocks, and any due reminder."
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
          description: "Use high only when the evidence clearly supports the decision"
          required: true
          type: choice
          options: [high, medium, low]
        recommended_action:
          description: "Concise next action, without mentions"
          required: true
          type: string
        context:
          description: "One concise evidence-based sentence suitable for a reminder comment"
          required: true
          type: string
        urgent:
          description: "True only for a p:1-highest or clearly time-critical item"
          required: true
          type: boolean
      steps:
        - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2
          if: env.ATTENTION_TRIAGE_STAGED != 'true'
          with:
            persist-credentials: false
            # This job has a write token. Never execute a PR's policy script.
            ref: ${{ github.event.repository.default_branch }}
        - name: Verify the default-branch policy installation
          id: policy
          if: env.ATTENTION_TRIAGE_STAGED != 'true'
          env:
            GH_TOKEN: ${{ github.token }}
            DEFAULT_BRANCH: ${{ github.event.repository.default_branch }}
          run: |
            if [[ -f .github/scripts/issue_pr_attention_monitor.py ]]; then
              echo 'ready=true' >> "$GITHUB_OUTPUT"
            elif gh api "repos/${GITHUB_REPOSITORY}/contents/.github/workflows/pydantic-ai-attention-triage.lock.yml?ref=${DEFAULT_BRANCH}" >/dev/null 2>&1; then
              echo '::error::The installed attention workflow is missing its deterministic policy script.'
              exit 1
            else
              echo 'ready=false' >> "$GITHUB_OUTPUT"
              echo '::notice::Skipping the one-time bootstrap run until the workflow and policy script reach the default branch.'
            fi
        - name: Apply decision through deterministic policy
          id: apply
          if: env.ATTENTION_TRIAGE_STAGED != 'true' && steps.policy.outputs.ready == 'true'
          env:
            GITHUB_TOKEN: ${{ github.token }}
          run: python .github/scripts/issue_pr_attention_monitor.py apply-decisions --github-output "$GITHUB_OUTPUT"
        - name: Record shadow decisions
          if: env.ATTENTION_TRIAGE_STAGED == 'true'
          run: |
            {
              echo '## Shadow attention decisions'
              count=$(jq '[.items[] | select(.type == "record_attention_decision")] | length' "$GH_AW_AGENT_OUTPUT")
              echo "- Recorded ${count} schema-validated decision(s); no repository state was changed."
            } >> "$GITHUB_STEP_SUMMARY"
timeout-minutes: 30
imports:
  - shared/network-vendor-domains.md
  - shared/tool-hints.md
  - shared/repo-context.md
  - shared/rigor.md
  - shared/engine-minimax.md
  - shared/pre-steps.md
  - shared/pre-agent-steps.md
---

# Decide who must act next

Classify conservatively. This workflow is not a validity detector, priority bot, or stale bot. Its
single question is: **must the next meaningful action on this item come from a maintainer?** A valid,
important, or old item is not enough. Ordinary backlog, contributor follow-up, waiting for CI, and
items that need no immediate decision must not receive maintainer attention.

## Select items

- For an issue, issue-comment, PR, review, or review-comment event, inspect only the triggering item.
- For a completed check suite, inspect only its associated open PRs, at most 5.
- For a manual run in shadow mode (`ATTENTION_TRIAGE_STAGED=true`), inspect at most 5 recently active
  open issues and 5 recently active open PRs as a one-off bounded calibration sample. Scheduled shadow
  runs inspect nothing; event runs already cover new activity, and repeatedly sampling the same recent
  items would add cost without evidence. Once live, scheduled and manual runs inspect only open items
  carrying `needs-maintainer-action`, oldest updated first, at most 10. Never bulk-classify the
  historical backlog.
- Ignore closed items and bot-authored events.

Read the title, body, labels, assignees, conversation, reviews, and CI state. For issues, treat a
recent Pydanty assessment as evidence and do not repeat its validation work. For PRs, use review
threads and CI state to decide whether the contributor, automation, or a maintainer owns the next
step.

## Overrides and classification

- `attention:skip` is an absolute maintainer override. Never recommend restoring attention. Record
  `none` with high confidence; deterministic policy preserves the maintainer-authored label.
- `attention:force` forces `maintainer`, unless `attention:skip` also exists; record `uncertain` for
  that conflict so host policy reports it without guessing.
- Maintainer statements such as "not a priority", "no action needed", or "leave in backlog" are strong
  evidence for `none`, but model output can never create a durable suppression override.
- Choose `maintainer` only when a maintainer must now review, decide scope/architecture, merge/close,
  answer a blocked contributor, or otherwise make the next project decision.
- Choose `contributor` when the author/reporter must provide information or revise code.
- Choose `automation` when CI, Pydanty, or another automated process is the next actor.
- Choose `none` when no concrete next action is due.
- Choose `uncertain` when evidence conflicts or is incomplete.

Use `high` confidence sparingly. Only high-confidence `maintainer` decisions may add the attention
label, and only high-confidence non-maintainer decisions may remove it. Medium/low confidence is an
abstention and leaves state unchanged.

## DynamicWorkflow, only when needed

Make obvious classifications directly. If the next actor depends on deeper issue validity, repository
architecture, PR readiness, or conflicting signals, call `run_workflow` once. Pass the gathered item
context to the matching `issue_evidence` or `pr_evidence` specialist; you may call both in parallel only
when an item genuinely crosses both concerns. The host caps the workflow at 2 sub-agent calls. If the
specialist does not resolve the uncertainty, abstain.

## Output

For each selected item, call `record_attention_decision` exactly once with:

- the item number;
- `next_actor` and calibrated confidence;
- a concise recommended action;
- one evidence-based context sentence, under 240 characters, natural enough to appear after
  `**Context:**` in a maintainer reminder;
- `urgent=true` only for `p:1-highest` or clearly time-critical impact;

Do not add labels, assign users, or comment directly. Do not emit a public classification comment.
If no eligible item exists, use the noop safe output.
