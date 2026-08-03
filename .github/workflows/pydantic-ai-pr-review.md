---
emoji: "🔎"
name: "CI Review"
description: "AI-driven PR review on the Pydantic AI gh-aw shim, triggered once CI succeeds on the PR's current head: inline comments + a single review verdict. Prompt iterable from a Logfire managed variable; read-only via gh-aw safe-outputs."
on:
  # Review once, after CI has passed — not on every push. `synchronize` used to
  # start a review that the next push cancelled mid-flight after its tokens were
  # already spent (#6766 measured 4.1 runs per PR, 96% of cancelled runs emitting
  # nothing). CI completion is both a cheaper cadence and a better signal: there is
  # no point reviewing a diff that does not compile or pass its tests.
  #
  # No `branches:` filter — the head branch here is the PR's branch, not a release
  # branch, so restricting it would suppress every review.
  workflow_run:
    workflows: ["CI"]
    types: [completed]
  workflow_dispatch:
  # Restricts this reviewer to collaborators: `roles:` compiles to a
  # `check_membership` step on `github.actor`, which under `workflow_run` is
  # whoever's push started the triggering CI run.
  #
  # It gates the actor, not the code — a collaborator can still open a PR from a
  # fork, which `eligibility` rejects separately. gh-aw's own `workflow_run` guard
  # is no help there either: it asserts the triggering run belongs to this repo,
  # true of a fork PR, whose CI run the base repo owns. Forks are reviewed on
  # demand via the `douwebot` label path in bots.yml.
  roles: [admin, maintainer, write]
permissions:
  contents: read
  # safe-outputs perform the actual writes in a separate conclusion job; the
  # agent job stays read-only (gh-aw strict mode requires this).
  pull-requests: read
  issues: read
  # Needed by the eligibility job to read the triggering CI run.
  actions: read
concurrency:
  # One review per head commit. Workflow-level `concurrency` is evaluated before
  # any job runs, so it cannot use the PR number the `eligibility` job resolves —
  # only the `github` context is in scope there.
  #
  # Keyed on the head branch AND the head SHA. The branch alone made a CI
  # completion for an OLDER head cancel an in-flight review of a NEWER one, and
  # the older run then skipped itself under current-head authority below, so
  # nothing replaced the review it cancelled (#6904). The SHA alone would let two
  # branches sitting on the same commit, each with its own open PR, cancel each
  # other. Together they name exactly what must not be reviewed twice at once —
  # and a CI re-run of one head still supersedes an in-flight review of it, which
  # is the one cancellation worth keeping.
  group: ${{ github.workflow }}-${{ github.event.workflow_run.head_branch || github.ref }}-${{ github.event.workflow_run.head_sha }}
  cancel-in-progress: true
# Deterministic, pre-inference gate: unless `eligibility` says so, no model runs.
#
# `eligibility` MUST also be referenced from the prompt body below. gh-aw copies
# this `if:` onto the `activation` job, but only adds jobs referenced by the PROMPT
# to `activation.needs` — a job referenced only here resolves to empty there, so
# `activation` skips and takes the whole graph with it. That is the live bug in
# pydantic-ai-ui-security-review (#6766 item 7). Referencing the job in the prompt
# is what hoists it above `activation` and wires it into `activation.needs`.
if: ${{ needs.eligibility.outputs.eligible == 'true' }}
tools:
  github:
    mode: gh-proxy
    # PR-scoped surface: read the PR, related issues, repo, and search.
    toolsets: [pull_requests, repos, search, issues]
safe-outputs:
  # `workflow_run` carries no PR, so the PR-targeting outputs below default to a
  # triggering PR that does not exist and silently discard the review. `target:`
  # is the supported way to name one.
  #
  # `safe-outputs.needs:` is mandatory alongside it. The expression is emitted
  # verbatim into the `safe_outputs` job, which does not otherwise depend on
  # `eligibility`, and an out-of-scope `needs.*` expression evaluates to the empty
  # string with no compile-time error.
  needs: [eligibility]
  footer: false
  activation-comments: false
  noop:
  create-pull-request-review-comment:
    max: 30
    target: ${{ needs.eligibility.outputs.pr_number }}
  submit-pull-request-review:
    supersede-older-reviews: true
    max: 1
    target: ${{ needs.eligibility.outputs.pr_number }}
    # Overrides `footer: false` above for this output only. The footer carries
    # gh-aw's `gh-aw-workflow-call-id` marker, and two things match on it: the
    # rerun dedup in `eligibility`, and `supersede-older-reviews`, which filters
    # `review.body.includes(marker)` and is a silent no-op without it.
    footer: always
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
  - shared/review-context.md
  - shared/checkout.md
  - shared/engine-minimax.md
  - shared/pre-steps.md
  - shared/pre-agent-steps.md
pre-steps:
  # Setting engine.command makes gh-aw skip ALL engine installation steps,
  # which also drops the bundled AWF firewall binary install. Re-run gh-aw's
  # own installer (the same call it makes for non-custom-command jobs).
  - name: Install AWF firewall binary (skipped by custom engine.command)
    run: bash "${RUNNER_TEMP}/gh-aw/actions/install_awf_binary.sh" v0.27.42

pre-agent-steps:
  # Check out the PR head. `workflow_run` starts the job on the default branch, and
  # gh-aw's own "Checkout PR branch" step is gated on `github.event.pull_request` /
  # `aw_context`, so under this trigger it no-ops — without this the agent would
  # review `main`.
  #
  # The steps after this one run workspace scripts over contributor-authored code,
  # which is safe only because of the collaborator + same-repo gates above. gh-aw
  # does not backstop that: its "Restore agent config folders from base branch"
  # step is gated on *its* checkout having succeeded, so it never runs here, and
  # nothing restores `AGENTS.md`, `agent_docs/` or `scripts/` from base.
  - name: Check out the PR head
    env:
      HEAD_SHA: ${{ needs.eligibility.outputs.head_sha }}
      HEAD_REF: ${{ needs.eligibility.outputs.head_ref }}
    run: |
      set -euo pipefail
      git fetch --no-tags origin "+refs/heads/${HEAD_REF}:refs/remotes/origin/${HEAD_REF}"
      git checkout --detach "$HEAD_SHA"
  # Pre-fetch PR context into `$GITHUB_WORKSPACE/.review-context/`: pr-details, PR
  # comments, review threads (with annotated diff hunks + resolved/outdated
  # state), annotated per-file diffs, related issues, AGENTS.md excerpts for
  # changed dirs, file orderings for sub-agent fan-out, and a PR-size summary.
  # The agent reads these files instead of calling the GitHub API at run time.
  # Non-fatal: missing context just reduces signal.
  #
  # The script lives at scripts/ (NOT .github/scripts/) because gh-aw's
  # "Save/Restore agent config folders from base branch" step snapshots and
  # restores `.github/` (and other managed agent-config folders) from the
  # BASE branch — making any new file added under those folders unreliable
  # for steps that run after the restore. `scripts/` is outside that set,
  # matching where the legacy reviewer's gather-review-context.sh already
  # lives. The script is a fork of scripts/gather-review-context.sh — see
  # the TODO at the top of the fork.
  - name: Gather PR review context
    env:
      GH_TOKEN: ${{ github.token }}
      PR_NUMBER: ${{ needs.eligibility.outputs.pr_number }}
      REPO: ${{ github.repository }}
    run: |
      set -uo pipefail
      script=scripts/gather-pydantic-ai-review-context.sh
      if [ -x "$script" ]; then
        "$script" "$PR_NUMBER" "$REPO" \
          || echo "::warning::${script} failed; reviewer will run with less context"
      else
        echo "::warning::${script} not present; reviewer will run with less context"
      fi

jobs:
  eligibility:
    # Everything that decides whether this PR is worth a review happens here, in
    # shell, before a single token is spent. Every exit path writes `reason` so a
    # skipped run explains itself in the job summary.
    runs-on: ubuntu-latest
    timeout-minutes: 5
    permissions:
      contents: read
      pull-requests: read
      actions: read
      # Only so a skip can leave a neutral check run behind (last step). Scoped to
      # this job on purpose: the top-level `permissions:` above, which is what the
      # agent job runs with, stays read-only. This job checks out no contributor
      # code — it runs `gh api` calls against the event and the resolved PR.
      checks: write
    outputs:
      eligible: ${{ steps.gate.outputs.eligible }}
      pr_number: ${{ steps.gate.outputs.pr_number }}
      head_sha: ${{ steps.gate.outputs.head_sha }}
      head_ref: ${{ steps.gate.outputs.head_ref }}
      base_ref: ${{ steps.gate.outputs.base_ref }}
      reason: ${{ steps.gate.outputs.reason }}
      marker_sha: ${{ steps.gate.outputs.marker_sha }}
    steps:
      - name: Decide whether to review
        id: gate
        env:
          GH_TOKEN: ${{ github.token }}
          REPO: ${{ github.repository }}
          EVENT_NAME: ${{ github.event_name }}
          RUN_EVENT: ${{ github.event.workflow_run.event }}
          RUN_CONCLUSION: ${{ github.event.workflow_run.conclusion }}
          RUN_HEAD_SHA: ${{ github.event.workflow_run.head_sha }}
          RUN_HEAD_BRANCH: ${{ github.event.workflow_run.head_branch }}
          RUN_HEAD_REPO: ${{ github.event.workflow_run.head_repository.full_name }}
          AW_CONTEXT: ${{ github.event.inputs.aw_context }}
          # Must match gh-aw's GH_AW_CALLER_WORKFLOW_ID for this workflow.
          REVIEW_MARKER: "<!-- gh-aw-workflow-call-id: ${{ github.repository }}/pydantic-ai-pr-review -->"
        run: |
          set -euo pipefail

          # The commit this run is deciding about, once it is known to be one this
          # reviewer is responsible for. Left empty — and the step below then
          # no-ops — for the skips where a marker would claim a reviewer that was
          # never going to run: a CI run that is not a PR run or did not pass, and
          # a fork, which `douwebot` covers instead. A fork head is not unmarkable
          # (it resolves in this repository through `refs/pull/<n>/head`, so the
          # check run would post and show); it is deliberately unmarked.
          MARKER_SHA=''

          skip() {
            echo "eligible=false" >> "$GITHUB_OUTPUT"
            echo "reason=$1" >> "$GITHUB_OUTPUT"
            echo "marker_sha=${MARKER_SHA}" >> "$GITHUB_OUTPUT"
            echo "Not reviewing: $1" >> "$GITHUB_STEP_SUMMARY"
            exit 0
          }

          # --- Resolve the PR -------------------------------------------------
          # `workflow_run` carries no `github.event.pull_request`, so the PR, its
          # current head and its refs are resolved here and handed to every later
          # step and to the agent's safe outputs as explicit values.
          if [ "$EVENT_NAME" = 'workflow_dispatch' ]; then
            CONTEXT_JSON="${AW_CONTEXT:-}"
            [ -n "$CONTEXT_JSON" ] || CONTEXT_JSON='{}'
            PR_NUMBER=$(printf '%s' "$CONTEXT_JSON" | jq -r 'select(.item_type == "pull_request") | .item_number // empty')
            [ -n "$PR_NUMBER" ] || skip 'workflow_dispatch without a pull_request in aw_context'
            # A manual dispatch is an explicit ask to review whatever is on the PR now.
            TRIGGER_SHA=''
          else
            [ "$RUN_EVENT" = 'pull_request' ] || skip "triggering CI run was a ${RUN_EVENT} run, not a pull_request run"
            [ "$RUN_CONCLUSION" = 'success' ] || skip "triggering CI run concluded ${RUN_CONCLUSION}"
            # `roles:` proves the actor is a collaborator, not that the head lives
            # here — a collaborator can open a PR from their own fork. Checked on
            # the event, and again on the resolved PR below.
            [ "$RUN_HEAD_REPO" = "$REPO" ] || skip "CI ran on ${RUN_HEAD_REPO}, not ${REPO}"
            TRIGGER_SHA="$RUN_HEAD_SHA"
            MARKER_SHA="$TRIGGER_SHA"
            # `$ENV`, not string interpolation: `"` is legal in a branch name and
            # would otherwise splice into the jq program and abort the step.
            PRS=$(gh api "repos/${REPO}/commits/${TRIGGER_SHA}/pulls" \
              --jq '[.[] | select(.state == "open" and .head.ref == $ENV.RUN_HEAD_BRANCH) | .number]')
            # One branch can have several open PRs, each onto a different base.
            # Reviewing an arbitrary one of them is worse than not reviewing.
            MATCHES=$(printf '%s' "$PRS" | jq 'length')
            [ "$MATCHES" -ne 0 ] || skip "no open PR found for ${RUN_HEAD_BRANCH}@${TRIGGER_SHA}"
            [ "$MATCHES" -eq 1 ] || skip "${MATCHES} open PRs share ${RUN_HEAD_BRANCH}@${TRIGGER_SHA}; cannot tell which one CI ran for"
            PR_NUMBER=$(printf '%s' "$PRS" | jq -r '.[0]')
          fi

          # --- Read the PR's live state, in one round trip --------------------
          # `isRequired` is only exposed on the GraphQL rollup, and only when the
          # PR number is passed to it.
          PR_JSON=$(gh api graphql -f query='
            query($owner:String!, $name:String!, $pr:Int!) {
              repository(owner:$owner, name:$name) {
                pullRequest(number:$pr) {
                  state isDraft reviewDecision isCrossRepository
                  baseRefName headRefName headRefOid
                  commits(last:1) { nodes { commit { statusCheckRollup { contexts(first:100) { nodes {
                    __typename
                    ... on CheckRun { name conclusion isRequired(pullRequestNumber:$pr) }
                    ... on StatusContext { context state isRequired(pullRequestNumber:$pr) }
                  } } } } } }
                }
              }
            }' -F owner="${REPO%%/*}" -F name="${REPO##*/}" -F pr="$PR_NUMBER" --jq '.data.repository.pullRequest')

          HEAD_SHA=$(printf '%s' "$PR_JSON" | jq -r '.headRefOid')
          HEAD_REF=$(printf '%s' "$PR_JSON" | jq -r '.headRefName')
          BASE_REF=$(printf '%s' "$PR_JSON" | jq -r '.baseRefName')

          [ "$(printf '%s' "$PR_JSON" | jq -r '.state')" = 'OPEN' ] || skip "PR #${PR_NUMBER} is not open"
          [ "$(printf '%s' "$PR_JSON" | jq -r '.isDraft')" = 'false' ] || skip "PR #${PR_NUMBER} is a draft"
          # The checkout step fetches `refs/heads/<head_ref>` from origin, which a
          # fork head is not; forks go through the `douwebot` label path instead.
          [ "$(printf '%s' "$PR_JSON" | jq -r '.isCrossRepository')" = 'false' ] \
            || skip "PR #${PR_NUMBER} is from a fork"

          # A manual dispatch has no triggering CI run, so the PR's current head is
          # the commit it is deciding about — set only past the fork gate, so a
          # dispatch on a fork PR skips as silently as the `workflow_run` path does.
          [ -n "$MARKER_SHA" ] || MARKER_SHA="$HEAD_SHA"

          # --- Current-head authority ----------------------------------------
          # The PR moved on while CI was running: reviewing the CI run's head would
          # comment on code that is no longer there. The CI run for the new head
          # will trigger this workflow again.
          if [ -n "$TRIGGER_SHA" ] && [ "$TRIGGER_SHA" != "$HEAD_SHA" ]; then
            skip "CI ran on ${TRIGGER_SHA}, but PR #${PR_NUMBER} has since moved to ${HEAD_SHA}"
          fi

          # --- Required checks -------------------------------------------------
          # Note: this is currently vacuous — `main` has no checks marked required,
          # so the list comes back empty. It is here so that marking a check
          # required starts gating reviews on it without another workflow change.
          FAILING=$(printf '%s' "$PR_JSON" | jq -r '
            [ .commits.nodes[0].commit.statusCheckRollup.contexts.nodes[]
              | select(.isRequired)
              | select((.conclusion // .state) as $c | $c != "SUCCESS" and $c != "SKIPPED" and $c != "NEUTRAL")
              | (.name // .context) ] | join(", ")')
          [ -z "$FAILING" ] || skip "required checks not passing: ${FAILING}"

          # --- A maintainer already asked for changes --------------------------
          # Piling an AI verdict on top of a human REQUEST_CHANGES adds noise to a
          # PR whose next move is already known.
          [ "$(printf '%s' "$PR_JSON" | jq -r '.reviewDecision')" != 'CHANGES_REQUESTED' ] \
            || skip "PR #${PR_NUMBER} already has changes requested"

          # --- Deduplicate reruns ----------------------------------------------
          # A rerun of the same CI run re-fires this workflow. Match our own marker,
          # not the bot account: every gh-aw reviewer here posts as
          # `github-actions[bot]`, so an actor match would let a UI Security Review
          # on this head suppress this review entirely.
          ALREADY=$(gh api --paginate --slurp "repos/${REPO}/pulls/${PR_NUMBER}/reviews" \
            | jq --arg sha "$HEAD_SHA" --arg marker "$REVIEW_MARKER" \
              '[add[] | select(.commit_id == $sha and ((.body // "") | contains($marker)))] | length')
          [ "$ALREADY" -eq 0 ] || skip "already reviewed ${HEAD_SHA}"

          {
            echo "eligible=true"
            echo "pr_number=${PR_NUMBER}"
            echo "head_sha=${HEAD_SHA}"
            echo "head_ref=${HEAD_REF}"
            echo "base_ref=${BASE_REF}"
            echo "reason=eligible"
          } >> "$GITHUB_OUTPUT"
          echo "Reviewing PR #${PR_NUMBER} at ${HEAD_SHA}" >> "$GITHUB_STEP_SUMMARY"

      # Every skip above writes only to `$GITHUB_STEP_SUMMARY`, on a run that
      # `workflow_run` attributes to `main` — so from the PR, "skipped, this commit
      # will not be reviewed" and "queued, be patient" look identical (#6904). A
      # `neutral` check run on the commit the decision was about says which it is.
      # Neutral rather than a comment: it never gates a merge, it lands on the head
      # it describes instead of on the conversation, and a fix-up push moves the PR
      # to a new head rather than accumulating notices on the old one.
      - name: Record the skip on the commit
        if: steps.gate.outputs.eligible == 'false' && steps.gate.outputs.marker_sha != ''
        env:
          GH_TOKEN: ${{ github.token }}
          REPO: ${{ github.repository }}
          MARKER_SHA: ${{ steps.gate.outputs.marker_sha }}
          REASON: ${{ steps.gate.outputs.reason }}
          RUN_URL: ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}
        run: |
          set -euo pipefail
          # `--input -`, not `-f output[title]=…`: `gh api` has no nesting syntax
          # for form fields, and `output` must be an object.
          jq -n --arg sha "$MARKER_SHA" --arg reason "$REASON" --arg url "$RUN_URL" '{
            name: "CI Review skipped",
            head_sha: $sha,
            status: "completed",
            conclusion: "neutral",
            details_url: $url,
            output: { title: $reason, summary: "This commit was not reviewed: \($reason)." }
          }' | gh api "repos/${REPO}/check-runs" --input -

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
            .github/workflows/shared/prompts/pydantic-ai-pr-review.md
          sparse-checkout-cone-mode: false
      - name: Resolve agent prompt (Logfire managed variable, else committed default)
        id: resolve
        uses: ./.github/actions/fetch-dynamic-prompt
        with:
          logfire-variable-key: gh_aw_pydantic_ai_pr_review_prompt
          default-prompt-file: .github/workflows/shared/prompts/pydantic-ai-pr-review.md
          logfire-read-key: ${{ secrets.LOGFIRE_READ_EXTERNAL_VARIABLES }}
          logfire-base-url: ${{ secrets.LOGFIRE_URL || vars.LOGFIRE_URL || 'https://logfire-api.pydantic.dev' }}
---

## The pull request under review

You were started by the `CI` workflow finishing successfully, not by a push, so the
event carries no pull request. These values were resolved before you started and are
authoritative — do not re-derive them:

- Pull request number: `${{ needs.eligibility.outputs.pr_number }}`
- Head commit: `${{ needs.eligibility.outputs.head_sha }}`
- Head branch: `${{ needs.eligibility.outputs.head_ref }}`
- Base branch: `${{ needs.eligibility.outputs.base_ref }}`

The head commit above is checked out for you, so files you `Read` are the PR's versions.

Your review comments are pinned to that pull request for you; you do not pass a pull
request number to `create_pull_request_review_comment` or `submit_pull_request_review`.

Begin the `submit_pull_request_review` body with this line, then a blank line:

Reviewed at `${{ needs.eligibility.outputs.head_sha }}`.

Someone can push while you are running. Your review is posted minutes after you read
the files, and GitHub attributes it to whatever the head commit is at that moment, not
to the commit you read. When those differ, a reader who checks your finding against the
commit GitHub named finds a quote that is not there and concludes you made it up — this
has already happened. That line is how they can tell.

${{ needs.fetch_dynamic_prompt.outputs.dynamic_prompt }}
