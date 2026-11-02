---
# The compiler embeds this host job in each caller; no runtime checkout is needed.
import-schema:
  dependencies:
    type: array
    items:
      type: string
    required: true
  condition:
    type: string
    required: true
  pr_number:
    type: string
    required: true
  head_sha:
    type: string
    required: true
  workflow_id:
    type: string
    required: true
  failure_label:
    type: string
    required: true
jobs:
  flag_failed_review:
    # Labels the PR when this run posted no review; a later run that does clears it.
    needs: ${{ github.aw.import-inputs.dependencies }}
    if: ${{ github.aw.import-inputs.condition }}
    runs-on: ubuntu-slim
    timeout-minutes: 5
    permissions:
      pull-requests: write
    steps:
      - env:
          GH_TOKEN: ${{ github.token }}
          PR: repos/${{ github.repository }}/pulls/${{ github.aw.import-inputs.pr_number }}
          LABELS: repos/${{ github.repository }}/issues/${{ github.aw.import-inputs.pr_number }}/labels
          SHA: ${{ github.aw.import-inputs.head_sha }}
          FAILED: ${{ needs.agent.result != 'success' || needs.safe_outputs.result != 'success' || needs.safe_outputs.outputs.process_safe_outputs_items_failed != '0' }}
          WORKFLOW_CALL_ID: ${{ github.repository }}/${{ github.aw.import-inputs.workflow_id }}
          FAILURE_LABEL: ${{ github.aw.import-inputs.failure_label }}
          RUN_URL: ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}
        run: |
          set -euo pipefail
          # Runs for different heads overlap, so only the run for the current head owns the label.
          live=$(gh api "$PR" --jq '
            if (.state == "open" or .state == "closed") and (.head.sha | type == "string" and length > 0)
            then "\(.state) at \(.head.sha)" else error("Invalid live PR state") end')
          if [ "$live" != "open at ${SHA}" ]; then
            echo "Review no longer required for ${SHA}: PR is ${live}." >> "$GITHUB_STEP_SUMMARY"
            exit 0
          fi
          # A queued safe-output call is not proof that GitHub received the review.
          reviews=$(gh api --paginate --slurp "$PR/reviews")
          if ! printf '%s' "$reviews" | jq -e --arg sha "$SHA" --arg workflow "$WORKFLOW_CALL_ID" --arg run "$RUN_URL" '
            any(.[][]; .user.type == "Bot" and .commit_id == $sha
              and (.state == "APPROVED" or .state == "CHANGES_REQUESTED" or .state == "COMMENTED")
              and ((.body // "") |
                (contains("<!-- gh-aw-workflow-call-id: " + $workflow + " -->")
                  or contains("[gh-aw-workflow-call-id]: # \"" + ($workflow | @uri) + "\""))
                and contains("run: " + $run + " -->")))
          ' >/dev/null; then
            FAILED=true
          fi
          if [ "$FAILED" = true ]; then
            gh api "$LABELS" -f "labels[]=$FAILURE_LABEL" --silent
            echo '::error::This run did not successfully publish its review on the current head.'
            exit 1
          else
            gh api -X DELETE "$LABELS/$FAILURE_LABEL" --silent || true
          fi
---
