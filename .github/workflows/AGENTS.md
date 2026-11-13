# Naming

Check names are what humans and agents refer to when they talk about CI, so they are
descriptive rather than generic. Two reviewers perform the same role — the
maintainer-voice standards review, driven by the repo's `AGENTS.md` and
`agent_docs/*.md` — on different engines and different cadences:

| Name | Where | Runs when |
|------|-------|-----------|
| `CI Review` | `pydantic-ai-pr-review.md` | when a user with write access or higher **approves** the PR's current head, on applying the **`final-review` label**, or on dispatch, once `CI` has passed on that head. Never in the merge queue. Z.AI Coding Plan engine, submits formal `REQUEST_CHANGES` for verified HIGH or CRITICAL findings, `COMMENT` for incomplete coverage without those findings, and `APPROVE` after complete coverage without them. Same-repo PRs only. |
| `douwebot` | `bots.yml` | only on applying the **`douwebot` label** — the fork-capable path (`pull_request_target`). Claude Opus 5.5 reviews. A rejected Claude subscription window reruns the review with `opencode-go/deepseek-v4.1-flash` and its [Coding Plan fallback](#opencode-falls-back-from-opencode-go-to-the-glm-coding-plan). Deletes the label when it finishes. A trusted publisher posts structured findings as one formal `APPROVE`, `REQUEST_CHANGES`, or incomplete `COMMENT` review. |
| `douwebot:review-agent-config` | `bots.yml` | one explicit review request for one PR head; the labeling sender must be a `User` with current `write` or `admin` permission. The label is not standing approval. |

`douwebot` treats every finding as blocking unless its comment starts with `Optional:`. Repository-rule violations in `AGENTS.md` or `agent_docs/*.md` are blocking and cannot be marked `Optional:`. It returns `REQUEST_CHANGES` while any blocking finding remains outstanding, including unaddressed findings from earlier review rounds, without duplicating comments. It returns `APPROVE` after a complete review with no outstanding blocking findings; optional findings may remain. Incomplete coverage returns `INCOMPLETE`, which the publisher records as `COMMENT`. A host-verified closed PR or moved head needs no publication and ends successfully. A new review cannot clear douwebot findings already on a head, so a later request for that head skips the review and comments with a link to the review that left them; an approval's optional findings stop blocking once their threads are resolved; a head with a clean approval, an incomplete review or a failed run is reviewed again, an incomplete one to complete its coverage even when its findings already block the head. The reviewer assesses choices against issue guidance and repository standards; a missing issue link or separate human sign-off alone does not block review. Both review labels use the OpenCode fallback. It has read, grep and glob tools, but no live GitHub or Git commands. It uses the prefetched review context and returns `INCOMPLETE` if missing information prevents a complete review.

`.github/scripts/douwebot_opencode_result.py` unwraps a Markdown code fence that ends the OpenCode answer.
It then validates the JSON object that ends the answer, so prose before the object passes, even prose that holds a `{`.
Text after the object or the closing fence still fails.

If GitHub rejects inline publication with HTTP 422, retry once with the findings in the review body.
Recheck the live head before that retry. Preserve the verdict; body findings still prevent a clean approval.
Cancellation keeps ordinary-label cleanup and ends without a failure notice. Report actual review or publisher failures.

The `douwebot:review-agent-config` label requests one review only from its first `labeled` event. The workflow checks the sender's current repository permission, the event's immutable head against the live PR head, and that the label is still present before each reviewer starts. API errors, non-`User` senders, reruns, stale heads, and removed labels deny the request. An ordinary `douwebot` event never inherits approval from the label. To retry or approve a new head, a maintainer removes and reapplies the label; do not rerun the old event. The workflow never removes this label automatically because an old run could consume a newer request.

Removing the approval label cancels in-flight work on a best-effort basis; cancellation may not stop a reviewer that has already started. Both review labels share the per-PR concurrency group, so removing the approval label can also cancel an ordinary review, and reapplying it starts another workflow run that uses that shared slot.

Unrelated label additions and removals use unique concurrency groups so their no-op runs cannot replace queued review requests.

**They are independent.** Neither reads the other's state, and the label suppresses
nothing: `douwebot` is an on-demand deep pass on top of `CI Review`, requested when a
PR warrants a second opinion. Do not reintroduce a gate from one onto the other — the
previous label-as-switch design left a window in which a push got *neither* reviewer.

`eligibility` skips a request made before `CI` has passed on the head. An approval is
checked again when `CI` passes; a label or dispatch must be repeated. Nothing removes the
`final-review` label: to retry a skipped request, remove and reapply it. A head that
already has a `CI Review` is not reviewed again.

`CI Review` starts on `workflow_run`, `pull_request_review`, `pull_request` (the
`final-review` label) and `workflow_dispatch`. `workflow_run` and `workflow_dispatch`
carry no `github.event.pull_request`, so its `eligibility` job resolves the PR, its
current head and its refs, and every later step —
including the agent's safe outputs — consumes those as explicit values. Anything you add
that needs to know which PR this is must read them from `needs.eligibility.outputs`,
never from `github.event`. For the safe outputs specifically that means frontmatter
`target:` plus `safe-outputs.needs:` — an out-of-scope `needs.*` expression compiles
without error and evaluates to the empty string, silently discarding the review.

The `CI Review` agent job queues in the engine concurrency group that
`shared/engine-zai.md` sets (`queue: max`), so one review runs at a time across all PRs.
`eligibility` and `provider_health` can therefore decide hours before the model starts.
Provider health is checked again at agent start; see [Z.AI provider health](#zai-provider-health). The pre-agent step
`Skip a PR that closed or moved on while queued` writes a `noop` to safe outputs when the
PR has closed or moved to a new head. The engine's gh-aw harness must exit on a pre-written
`noop` before it starts the model. gh-aw's `claude_harness.cjs` does this
(`hasNoopInSafeOutputs`). Any other engine harness, including
`src/pydantic_ai_harness/gh-aw/pydantic.md`, must do the same, or every stale run
spends a full review. The `noop` does not skip the `detection` job, which still runs
one threat-detection pass outside the agent slot. The shim withholds `noop` from the
`CI Review` model, so this step writes the run's only `noop`. Keep `safe-outputs.noop`
enabled for it.

`CI Review` requires a formal review from this run on the reviewed head.
Its completion validator rejects placeholder noops and missing-data signals alone.
After safe outputs, the [host verifier](shared/review-publication.md) checks GitHub for the exact workflow marker, run URL and head.
A missing publication fails the workflow and marks the current PR head with its failure label.
A host-verified closed PR or changed head needs no review and cannot alter the newer head's label.

gh-aw drops its default agent group for a workflow with `pull_request*` triggers, so
`shared/engine-zai.md` sets `engine.concurrency` for every importer: one agent run at a
time per workflow. A workflow cannot override it. gh-aw allows one `engine` across a
workflow and its imports, and strict mode rejects secrets in a workflow's own
`engine.env`.

## What confines `CI Review` to same-repo PRs

Two different gates, guarding two different things. Neither is redundant, and the
distinction is the whole point: **`roles:` gates the actor, `eligibility` gates the
code.**

**`eligibility` is what rejects forks.** It compares the event's head repository
(`workflow_run.head_repository` or `pull_request.head.repo`) to `github.repository`, and
then the resolved PR's `isCrossRepository` — and skips on either. That is the check to look
at when you want to know whether fork code can reach this workflow.

**`roles:` gates who may trigger it, and is not a fork filter.** It compiles to a
`check_membership` step validating `github.actor`: the actor of the triggering CI run
under `workflow_run`, and the reviewer or labeler under an approval or label. It does
exclude external contributors, who have `read`.
But a collaborator with write access can open a PR from their own fork, and that PR
passes `roles:` — so `roles:` alone would leave fork code reaching the checkout. Forks
are reviewed on demand via the `douwebot` label path, which is built for untrusted code.

**gh-aw's own `workflow_run` guard is not a fork filter either.** It is emitted whether
or not `roles:` is set, and asserts the *triggering run* belongs to this repository —
true of a fork PR, whose `pull_request` CI run is owned by the base repo. It inspects
`workflow_run.repository`, not `head_repository`. It bounds which runs may start us,
nothing more.

All three matter because `workflow_run` puts the workflow in base-repository context
with full access to repository secrets, and the agent job checks out contributor-authored
code and runs workspace scripts over it. Weaken any one of them and that checkout starts
accepting code the remaining gates do not cover. gh-aw will not backstop you: its
"Restore agent config folders from base branch" step is gated on gh-aw's *own* PR
checkout succeeding, which never happens under `workflow_run`. Under an approval or
label, GitHub runs the PR's own workflow file and withholds secrets from a fork PR.

## A custom job named in `if:` must also appear in the prompt

When a workflow's top-level `if:` references a custom job's output
(`needs.<job>.outputs.<x>`), **that job must also be referenced somewhere in the prompt
body**, even if only inside an HTML comment.

`gh aw compile` copies the top-level `if:` onto the generated `activation` job, but it
only adds jobs referenced by the *prompt* to `activation.needs`. A job named only in the
`if:` therefore resolves to the empty string inside `activation`, so `activation` skips —
and because the compiler makes custom jobs depend on `activation`, the named job and the
agent skip with it. A job skipped by `if:` reports as success, so the whole thing goes
green while never having run.

That is not hypothetical: the former `pydantic-ai-ui-security-review` shipped this way and its
agent never fired once, reporting false-green on every PR (#6766). `pydantic-ai-pr-review`
carries the prompt-body reference and a comment saying why. After changing it, check the
recompiled lock: `activation.needs` must list the job,
and the job itself must **not** have `needs: activation`.

# A PR that edits `bots.yml` cannot test its own change

`bots.yml` triggers on `pull_request_target`, and GitHub runs that trigger's
workflow file from the **base branch**, never from the PR head — that is what makes
the trigger safe to give secrets to. So a PR that changes `bots.yml` sees the
*current `main`* version of every job in it, and its own edit takes effect only
after merge.

The practical consequence is that a `bots.yml` check can stay red on the very PR
that fixes it, and re-pushing will not help. Reason about the change by reading the
diff, and expect the first real execution to be on `main`. This is unlike
`.github/workflows/ci.yml`, which runs on `pull_request` from the PR's own merge
ref and therefore does test itself.

# `Protect .github` keeps external changes out of this directory

`protect-github-dir.yml` fails a PR that changes anything under `.github/` unless its
author is trusted. Everything here executes with repository credentials, so an edit from
outside the org is a supply-chain change rather than an ordinary code review — a
maintainer carries legitimate ones forward in their own PR (#7024 is the case that
prompted the guard).

**Its job, `.github Directory Guard`, is the merge gate for `.github/`.** It blocks only once the
`main` ruleset lists it as a required status check; until then it is advisory.

A required check must also report on every merge group, or the queue waits until the check times
out. The guard therefore runs on `merge_group` and passes there without reading anything: each PR
was judged on its head before it could be enqueued. Keep the job name. GitHub matches required
checks by job name across events. The concurrency group falls back to the merge group's head SHA,
because a merge group has no PR number and a shared group would cancel the other groups.

**The author's resolved repository permission is the only trust signal**, and the two
shortcuts it replaces are both unsound — the same reasoning `bots.yml`'s agent-config
guard spells out:

- **A base-repo head branch does not imply push access.** GitHub Apps push branches
  straight into this repository, so `pydanty[bot]` — which builds those branches out of
  externally-authored issue text — clears any same-repo check. Trusting the head repo
  would hand the guard's whole threat model a bypass.
- **`author_association` misreports** a genuine collaborator as `CONTRIBUTOR` when their
  org membership is private (#6359).

Read `.permission`, never `.role_name`: the latter can be an arbitrary custom role name
that fails a hardcoded match and blocks a real maintainer, while `.permission` maps
`maintain` and custom roles onto their stable base level (#6797, which fixed exactly this
in `bots.yml` and `at-claude.yml`). The endpoint needs only metadata access, which every
token carries and no `permissions:` key can withhold — so `pull-requests: write` alone
reaches it. It fails **closed**: an unreadable permission blocks, because this is a
security boundary, not `pr-guard.yml`'s courtesy gate.

`dependabot[bot]` is allowlisted because it owns the action-pin bumps here (the
`github-actions` ecosystem entry in `dependabot.yml`) and holds no repo permission of its
own. It is the only bot with a bypass: a blanket `*[bot]` glob would hand one to any bot
installed later, and `pydanty[bot]` must not have one for the reason above.

**Do not add a `paths:` filter to its trigger.** A `pull_request_target` filtered to
`.github/**` does not run at all on the PRs that don't touch it, and a required check
that never runs stays *pending* — blocking every merge. The job runs on every PR and
exits 0 when nothing protected changed. `edited` is in `types:` because it is the only
event fired when a PR's **base branch** changes, and the verdict is a diff against that
base — hence also the `state != open` early exit, since `edited` fires on closed PRs too.

Two consequences of the trigger, both easy to get backwards:

- On github.com `pull_request_target` runs in the context of the **default branch of the
  base repository** ([docs](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#pull_request_target)),
  *not* the PR's base branch. So `main`'s copy guards PRs targeting **every** branch,
  `v1` included — a maintenance branch needs no copy of its own. It also means a PR that
  changes the guard is judged by `main`'s copy, the same blind spot as `bots.yml` above.
- Branches whose names look like SHAs [may not trigger `pull_request_target` at all](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#pull_request_target).
  While the check is merely advisory such a PR shows no guard run and no failing check, so
  it reads as clean. Marking the check **required** is what closes that: a check that never
  ran stays pending and blocks the merge.

# PR metadata gates enforce the merge bar

`pr-gates.yml` checks that a PR adding an API compatibility waiver has the `compatibility impact` label.
It runs the PR's copy of `pr_gates.py` with a read-only token. The separate `pr-review-gates.yml`
workflow runs default-branch `review_gates.py` with permission to publish commit statuses, so it
never executes contribution code with a write token. The workflows have distinct job names because
GitHub matches required checks by job name, even across workflows and event types.

The review status requires a clean douwebot approval for M/L/XL.
The status fails until exactly one size label matches the shared PR Bots calculation; changing a label cannot
lower the computed bar. PR Bots recalculates on open, push and PR edits, including base-branch changes.
Size S remains a manual policy judgment. Merge groups receive success without reevaluating queue
members because the review requirements are enforced before admission.

A clean douwebot approval is the latest non-dismissed completed verdict on the current head, with no inline or
review-body findings on that head. The one exception is an inline finding that an approval marks optional with the
`Optional:` prefix: it stops blocking once its review thread is resolved. Blocking findings, unresolved optional ones
and review-body findings still block, resolved or not. A clean retry may
supersede an empty incomplete review, but cannot erase earlier findings. PR Bots uses a read-only
reviewer and a trusted publisher that binds the result to its run and head. The gate verifies that
receipt, so `CI Review` cannot stand in for douwebot even though both can use the same bot account.
Maintainer-authorized protected-configuration reviews satisfy the same rule.
A clean OpenCode fallback approval satisfies the same gate as a Claude approval. The review body and receipt identify the engine that served the review: `engine=claude`, `engine=opencode` for OpenCode Go, or `engine=opencode-zai` for OpenCode on the GLM Coding Plan. Reviews from before the Coding Plan replaced DeepInfra carry `engine=opencode-deepinfra`, which the gate still reads.

`pr-review-gates-events.yml` observes PR opens, pushes, reopening, edits, label changes, reviews,
inline-comment changes and review-thread resolution with a read-only token, including on forks. Its completion wakes
`PR Review Gates`, which also runs on PR Bots completion. The latter covers writes by `GITHUB_TOKEN`,
which cannot trigger the observer. Do not add a `pull_request_target` trigger to `PR Review Gates`:
GitHub blocks that event by default in public repositories (#12961). The publisher binds each
wake-up to the unique open PR GitHub reports for its exact head SHA and repository. It ignores
titles and artifacts, and a stale or ambiguous event cannot write statuses to another head. Fork
observers may need contributor approval.

The `Gate: …` commit statuses carry the review verdict. The publisher job succeeds after publishing
those statuses, including a failing verdict for an unmet requirement. The job fails on publisher errors.
Publication is serialized per PR and marks the douwebot status pending before setup. API errors or
incomplete data fail closed. After the requirements are met, a new evaluation replaces the previous
failure through GitHub's latest-per-context commit statuses. Rerun an existing current-head,
default-branch `PR Review Gates` run to recover from an API failure or delayed observer; do not add
branch-selectable dispatch to this privileged workflow. Workflows and scripts remain owned in `.github/CODEOWNERS`.

# `PR Review Checks` reviews every pull request head

`pr-review-checks.yml` runs default-branch code on `pull_request_target` and publishes one
`PR Review Checks` commit status on the PR head. The status carries the verdict, not a job
conclusion: under `pull_request_target` the jobs' check runs attach to the default-branch commit.
Name no job `PR Review Checks`, because a check run with that name also matches the required context.
The status blocks merge only while the `main` ruleset lists `PR Review Checks` as a required check.

- `gate` reads the author's repository permission. An app, or an author without `write` or `admin`,
  gets a failing "Awaiting maintainer" status until a `User` with write access applies
  `pr-review-checks:run`. The label stays valid across pushes; the gate checks who applied it last.
- `detect` runs `pr_review_checks.py` on the head's `git diff`, read as data, and starts one review
  job per selected check. A check selects the files under its trigger paths, and can also select a
  file whose changed lines match its diff tokens. A check's joint paths select it without making it
  the file's owner. The `general` check reviews every changed source, workflow and configuration
  file that no selected check owns by path. The `tests-and-dependencies`
  check owns tests, cassettes, dependency manifests and lockfiles, and holds them to the
  repository's testing and dependency rules rather than the security rubric. The `general` check
  also reviews `THREAT_MODEL.md`, every `AGENTS.md` and `CLAUDE.md`, every file under an
  `agent_docs/`, `.agents/`, `.claude/` or `.opencode/` directory, and a file renamed from one, even
  when another check owns them. These are the instruction layer the review job strips from the
  head. A change only to the documentation `is_content` lists selects nothing and passes. Files
  under `.github/` always get a review, a generated `*.lock.yml` included: Actions runs the lock,
  and nothing checks that it matches its `.md` source.
- Each review job runs OpenCode with `opencode-go/deepseek-v4.1-flash`, its [Coding Plan fallback](#opencode-falls-back-from-opencode-go-to-the-glm-coding-plan), and only read, grep and glob,
  in the douwebot layout: the head is confined to `pr-head/` and its agent instruction files are
  deleted. The prompt lists and diffs only its check's files; the read tools reach both checkouts
  and the gathered PR context, so the model can trace callers. The job validates the model's answer
  against its schema. As in douwebot, the job unwraps a Markdown code fence that ends the answer, then
  validates the JSON object that ends the answer, so prose before the object passes; text after the
  object or the fence still fails. The check's `max_steps` and
  `max_minutes` end a review as `INCOMPLETE`; the defaults are permissive starting values, to be
  tuned down from recorded runs. Token totals are recorded and never capped. With `LOGFIRE_TOKEN`
  set, OpenCode exports its own spans and logs over OTLP, and the runner exports one span per review
  on every outcome, a failed, rejected or cancelled review included. The span names the provider and
  plan of the review's last OpenCode run, such as `the backup Coding Plan`, and how many runs the
  review took. When the review's run completes or reaches its time budget, the span adds its steps,
  tokens and duration. A validated answer adds the outcome and the number of findings. A review that
  raised has the outcome `FAILED`, the exception's class name and an error status. When the last
  run's process failed, the span adds the type, status and provider code of its stream's last
  `error` event, and whether the stream holds a `step_finish`. As each OpenCode run of the review
  ends, a run the chain moved past included, it exports a child
  [`OpenCode run` span](#opencode-falls-back-from-opencode-go-to-the-glm-coding-plan) with the same
  failure fields and its own steps and tokens. When Actions cancels a job, it sends SIGINT to the
  step's entry process and SIGTERM 7.5 s later, then kills the process tree 2.5 s after that; see
  [workflow cancellation](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-cancellation).
  The review step `exec`s the runner, so the runner is that process, and both signals raise
  `KeyboardInterrupt` in it, so the review span still exports.
  OpenCode's own spans carry no prompt or completion text
  and no token counts. A vulnerability that already exists on the default branch goes only to the
  private triage Slack webhook, escaped and quoted in a code block because the model wrote it. Result
  artifacts are public, so they carry no private report. A
  review job also leaves out a finding on a file outside its check or on a line the pull request did
  not add or remove, and its result keeps only how many it dropped of each severity.
- `publish` validates every finding against its check's files, and drops a finding on a line outside
  GitHub's patch for its file. The job log counts every dropped finding. A dropped `critical` or
  `high` finding marks its check `INCOMPLETE`, including one on a file whose patch GitHub omitted. A
  dropped `medium` finding leaves the verdict unchanged.
  When there are findings, it posts one `COMMENT` review with each check's usage and the receipt
  `<!-- pr-review-checks run=… head=… -->`, which douwebot's clean approval gate does not count as a
  douwebot finding. Each inline finding carries a hidden key built from its check, path, side,
  line text, severity and body. The review leaves out a finding whose key an earlier run already
  posted, and says how many it left out; the status still counts it. The job summary always carries
  the usage table. A `critical` or `high` finding,
  an `INCOMPLETE` check, or a check without a result fails the status. `medium` findings do not.
- A `User` with write access applies `pr-review-checks:accepted` to pass the current head despite its
  findings. Every event that reviews the head again, a push included, removes the label; apply it
  again after that review.
- Drafts are reviewed like any other pull request. A merge group gets success without a review.

To add a check, add a `SpecialistCheck` entry in `pr_review_checks.py` and its prompt
fragment in `.github/pr-review-checks/prompts/`. The workflow needs no change. Like `bots.yml`, a
PR cannot test its own change to this workflow or its scripts.

# `Category Classify` falls back to OpenCode on a Claude usage limit

`category-classify` in `bots.yml` asks Claude Code for the PR's category, and `category-apply` applies it.
When the Claude subscription rejects a five-hour or seven-day window, `category-classify-opencode` reruns the
classification with OpenCode and its [Coding Plan fallback](#opencode-falls-back-from-opencode-go-to-the-glm-coding-plan).
`category-classify` detects that rejection with douwebot's step, `Detect Claude subscription limit`, which runs
`.github/scripts/douwebot_usage_limit.py`. Every other Claude failure fails `category-classify`, a revoked or expired
credential included.

- `category-classify-opencode` runs on a fresh runner, so nothing the Claude run wrote reaches it.
- Its `Prefetch PR context` step saves the output of the `gh` commands the Claude prompt names: `gh pr view`,
  `gh pr diff --name-only`, and the full `gh pr diff` when a `pyproject.toml` changed.
- OpenCode runs in that context directory with read, grep and glob only, and without a GitHub token. It receives
  the Claude prompt, through the `category-prompt` YAML anchor, plus `TOOL_GUIDANCE` from
  `.github/scripts/category_classify_opencode.py`, which names each prefetched file. Edit the classification rules
  in the Claude step's prompt only.
- The answer must end with one JSON object whose only field is an allowed `category`. Prose before the object passes,
  and so does a Markdown code fence that wraps the object and ends the answer, as in `PR Review Checks`. Any other
  answer fails the job, and no label is applied. `category-apply` applies the category from whichever job set it.

# OpenCode falls back from OpenCode Go to the GLM Coding Plan

Every CI OpenCode run starts on `opencode-go/deepseek-v4.1-flash` with `OPENCODE_API_KEY`, a Console key for the Go
workspace: douwebot's OpenCode fallback, `Category Classify`'s OpenCode fallback, each `PR Review Checks` review and
OpenCode threat detection. `run_with_fallback` in `.github/scripts/douwebot_opencode_run.py` hands a refused run down
this chain:

1. `opencode-go/deepseek-v4.1-flash` with `OPENCODE_API_KEY`.
2. `zai-coding-plan/glm-5.3-flash` on the first Z.AI Coding Plan account: `ZAI_API_KEY`, the primary, or
   `ZAI_BACKUP_API_KEY`, the backup.
3. `zai-coding-plan/glm-5.3-flash` on the other account.

The workflow's run number plus the run's place in its job matrix picks the first account: the primary when the sum is
even, the backup when it is odd. Run numbers count up within a workflow, so consecutive runs alternate accounts. Each
`PR Review Checks` review passes its check's place in the `detect` matrix, so the checks of one run alternate too.
Every other caller is a single job and passes 0. Replayed on the 12 `PR Review Checks` runs that started together at
06:28Z on 2026-11-12, such as [run 43168376242](https://github.com/pydantic/pydantic-ai/actions/runs/43168376242),
this rule puts 16 of their 33 reviews on the primary first and 17 on the backup. Without `ZAI_BACKUP_API_KEY`, the
chain is Go, then the primary.

A run moves to the next provider when OpenCode exited non-zero, the next account's key is set, and one of these holds:

- The provider refused the run before its first step: the event stream has no `step_finish` event, and its last
  `error` event has a type the refusing provider hands over. That is `provider.go-limit` or `provider.quota` from Go,
  and `provider.quota` or `provider.rate-limit` from the first Coding Plan account. OpenCode 2.2.5 retries a rate
  limit within the run before it writes `provider.rate-limit`, and does not retry the other types; see `isRetryable`
  in [`provider-error.ts`](https://github.com/anomalyco/opencode/blob/v2.2.5/packages/ai/src/provider-error.ts) and
  `toSessionError` in
  [`to-session-error.ts`](https://github.com/anomalyco/opencode/blob/v2.2.5/packages/core/src/session/to-session-error.ts).
- The first Coding Plan account rate-limited the run after it took steps: the stream has a `step_finish` event, and
  its last `error` event is a `provider.rate-limit`. The job log says that account rate-limited the run after it took
  steps, and names the account that restarts it.

The next provider runs the whole run again from scratch, as its own `Attempt`. Each provider runs a run at most once,
so a run restarts at most once. Go never restarts a run. A `provider.quota` after a step fails the job. The chain
leaves out a Z.AI account without a key.

OpenCode 2.2.5 types any HTTP 402, and a Z.AI HTTP 429 whose message names a plan, quota or balance limit,
`provider.quota`. Those 429s include the 5-hour, weekly and monthly windows (`1308`, `1310`, `1316` to `1321`), an
expired plan (`1309`), a plan without the model (`1311`) and an empty balance (`1113`), so each hands the run to the
other account. OpenCode types every other 429 `provider.rate-limit` and retries it within the run, such as `1302`
request-rate limits and `1305` overload; see `classify` in
[`provider-error.ts`](https://github.com/anomalyco/opencode/blob/v2.2.5/packages/ai/src/provider-error.ts). A rate
limit that outlasts those retries hands the run to the other account too, before or after the run took steps.

Every other failure fails the job, including a rejected key, a provider outage, a time-budget stop, a timeout and an
invalid answer. A run the other account also refuses or rate-limits fails the job. A fallback run, a restart
included, gets only the rest of the time budget. OpenCode 2.2.5 reads the Coding Plan key from `ZHIPU_API_KEY` only,
so each Z.AI run sets that variable to its account's key. Each run's config allows only its provider and pins the `title`,
`compaction` and `summary` agents to its model. Its environment holds no provider key but that account's. The job log names the
model and the plan that served the run. When a run's process fails, the job log names that run's model, its plan, and the
`type` and `status` of its stream's last `error` event. It also names the provider's `error.code` from that event's
response body, such as Z.AI's `1302` request-rate limit or `1305` overload, both of which OpenCode types
`provider.rate-limit`. It prints a type only when the type is shaped like OpenCode's own, such as `provider.auth`. It
prints a code only when the code is one to six digits. It never prints the error message or the rest of the response body,
which can repeat the provider's response. When douwebot, `PR Review Checks` or threat detection rejects a run's
stream or answer, the job log names the reason: a fixed message from `final_text` in
`.github/scripts/douwebot_opencode_result.py`, the sorted validation error types, or the exception's class name. It
never prints model text, because the answer can hold a private report. A Z.AI run sends Z.AI the same content a Go
run sends Go, fork pull request content included. Start any new CI OpenCode run through `run_with_fallback`.

With `LOGFIRE_TOKEN` set, `run_with_fallback` makes each run export OpenCode's spans and logs to Logfire over OTLP.
douwebot's OpenCode fallback, `Category Classify`'s OpenCode fallback and `PR Review Checks` set the token. Each
run's resource attributes name the repository, `github.run_id`, `github.run_attempt` and `github.job`, then the
caller's attributes. They end with the run's plan as `opencode_fallback.plan`, and its place in the chain, counting
from 1, as `opencode_fallback.attempt`. OpenCode's own spans carry no model or plan attribute, and both Coding Plan
accounts call the same endpoint, so only `opencode_fallback.plan` tells their runs apart. A restart is its own
attempt. The `PR Review Checks` review span adds `pr_review_checks.restarted` when the review's chain restarted a run.

As each run ends, `run_with_fallback` also exports an `OpenCode run` span for it, under the service
`opencode-fallback`, so a later run's failure cannot drop it. Each span carries its run's resource attributes above and
`opencode_fallback.provider`. It also counts the `step_finish` events in the run's stream and the tokens they report,
every kind summed as in `pr_review_checks.tokens`, as `opencode_fallback.steps` and `opencode_fallback.tokens`, a failed
or handed-over run's included. A run that did not serve has an error status. When the run's process failed, the span
adds the error type, status and provider code the job log names, as `opencode_fallback.error_type`,
`opencode_fallback.error_status` and `opencode_fallback.error_code`, and whether the stream holds a `step_finish`, as
`opencode_fallback.step_finish`. Outside the job log, these spans are the only record of how a run the chain moved past
failed. A `PR Review Checks` review's run spans are children of its review span; every other chain's runs share a trace
of their own.

Threat detection does not set the token. Its `gh-aw.detector.result` span records each detection's usage and outcome,
and the provider and plan of its last run, which the detector step writes as its `provider` and `plan` outputs.

The Coding Plan runs draw on the same accounts as the gh-aw agent workflows, and the primary also serves Pydanty; see
[Shared Z.AI account limits](#shared-zai-account-limits). `PR Review Checks` starts several reviews per pull request
in parallel, so one Go limit can move all of them onto the Coding Plan at once. The alternating first account splits
them across both accounts.

# A copy of the repository picks the `CI` Linux runner

The `CI_LINUX_RUNNER` repository variable sets the runner label of `check` and of every Linux job
it depends on, including the jobs of `clai2-web.yml` and `latest-versions-canary.yml`. Those two
reusable workflows read it wherever they run, so it also moves the build that `clai2-bleeding.yml`
publishes and the scheduled canary of `latest-versions-canary-nightly.yml`. A fork or other copy
sets it when GitHub's standard runners are too small or unavailable for those jobs.
Leave it unset in pydantic/pydantic-ai: unset, each job runs on its fallback label,
`ubuntu-24.04` for `test-all-extras` and `ubuntu-latest` for the rest.

- Set it to one static label. Each job's `runs-on` reads the value as that job's only label.
- Choose a runner that matches GitHub's Ubuntu 24.04 image. `test-all-extras` starts the
  image's PostgreSQL 16, and several harness integration jobs start Docker containers.
- GitHub passes no configuration variables to a run for a pull request from a fork. Those runs
  use the fallback labels.
- The Linux leg of `test-clai2-clipboard` reads the variable. Its macOS and Windows legs stay on
  GitHub's runners.

The release and docs-deploy jobs run only in pydantic/pydantic-ai, so a tag pushed to a copy
starts none of them.

# Code owners review dependency, CI config and durable runtime changes

`.github/CODEOWNERS` owns three scopes:

- **Dependency and build config**, owned by `@DouweM`, `@adtyavrdhn`, `@mpfaffenberger` and
  `@dsfaccini`: every `pyproject.toml` and `uv.lock`, `.pre-commit-config.yaml` and `Makefile`.
- **Everything under `.github/`**, owned by `@dsfaccini` only: workflows, actions, scripts and the
  rest of the directory. `CODEOWNERS` itself is the exception: `@DouweM` and `@adtyavrdhn` own it
  too, so the durable runtime owners see every ownership change.
- **The durable runtime and its deep harness integrations**, owned by `@DouweM` and `@adtyavrdhn`
  only: the runtime under `pydantic_ai_slim/pydantic_ai/durable_exec/`, every harness durable
  engine (a `BaseDurabilityCapability` subclass), the harness's shared durable helper, and the
  `SubAgents` code that runs a delegated agent run inside a durable unit. Other capabilities that
  use `@durable_operation` or check for durable execution stay outside this scope. `CODEOWNERS`
  lists the exact paths.

GitHub requests a review from the owners of each path a PR changes. That request is not a merge
gate. The `main` ruleset has no `pull_request` rule, so it does not require code-owner review, and
the ruleset is meant to stay that way. Do not describe an owner's approval as required for a merge.

GitHub reads `CODEOWNERS` from the PR's base branch, so a PR that edits it requests review from the
current owners.

# Agentic workflows (`gh-aw`)

The `pydantic-ai-*` workflows in this directory are [agentic workflows](https://github.com/githubnext/gh-aw) authored as human-editable `<name>.md` sources (frontmatter + prompt) that **compile** to a generated `<name>.lock.yml`. GitHub Actions runs the `.lock.yml`, never the `.md`.

Compile with [gh-aw v1.2.1](https://github.com/github/gh-aw/releases/tag/v1.2.1), which ships AWF v0.29.7.
It is the first release that lets the review failure reporter read trusted pre-agent eligibility outputs
after detection errors ([gh-aw#74939](https://github.com/github/gh-aw/pull/74939)).
`shared/engine-zai.md` sets `max-ai-credits: -1`; AWF accepts that value only from v0.29.1.
`shared/engine-zai.md` imports the public Pydantic AI engine and explicitly pins runtime 0.64.0. Keep each workflow's
root engine block model-only; the shared import owns identity and credentials.
`stage-gh-aw-agent.sh` configures the existing request limits in an isolated CLAI
settings store. It also outputs the staged shim's source digest, which `shared/engine-zai.md`
passes to the engine as `LOGFIRE_SERVICE_VERSION`. Workflow `engine.args` is ignored by this compiler's public-engine path.
`shared/engine-zai.md` also sets the agent job to GitHub's 360-minute maximum and detection to 20 minutes;
gh-aw would otherwise cap them at 60 and 10.
The public engine ignores `tools:` allowlists, and gh-aw rejects `bash: []` or a command list for it.
`NO_SHELL_WORKFLOWS` in `pydantic_ai_gh_aw_shim/agent.py` withholds the shell by workflow name; mark each
such source with a comment naming that set. Every other workflow gets an unrestricted shell.

- **Never hand-edit a `*.lock.yml`.** It is generated — the header says so. Manual edits are silently overwritten on the next recompile, and until then the running workflow diverges from its source.
- **After editing a workflow `*.md` source, recompile and commit the regenerated `*.lock.yml` in the same change** — a `*.md` edit without its recompiled lock is incomplete, and source and lock drift apart:

  ```
  gh aw compile --action-mode action --action-tag v1.2.1
  ```

- **Recompilation is required for anything the lock bakes in:** a source's frontmatter (`on:` triggers, `permissions`, `tools`, `safe-outputs`, jobs, path/`detect` filters) and its `imports:` shared fragments (`shared/*.md`) are inlined into the lock at compile time.
- **Exception — runtime-resolved prompts need no recompile.** Agent prompts under `shared/prompts/` are fetched at run time (via the `fetch-dynamic-prompt` action / a Logfire-managed variable), not baked into the lock, so editing one takes effect on the next run without recompiling.

## OpenCode threat detection

gh-aw runs threat detection only on its Copilot, Claude and Codex engines.
`shared/engine-zai.md` sets `safe-outputs.threat-detection.engine: false` and runs our own detector as detection post-steps.
`.github/scripts/gh_aw_opencode_detect.py` runs OpenCode with `opencode-go/deepseek-v4.1-flash` and its [Coding Plan fallback](#opencode-falls-back-from-opencode-go-to-the-glm-coding-plan).
The post-steps check the script out at `github.workflow_sha` and install OpenCode with `setup-douwebot-tools`.

- Keep the detector step's id `detection_agentic_execution`. gh-aw's conclusion step reads that step's outcome.
- The detector sends the prompt gh-aw renders to `GH_AW_PROMPT`. It checks the agent's outputs, never the PR under review.
- OpenCode runs with douwebot's read-only lockdown in `/tmp/gh-aw/threat-detection`. It cannot read the repository workspace.
- A valid verdict writes `detection_result.json` without reasons and `detection_result_full.json` with them.
- An invalid verdict writes `THREAT_DETECTION_STATUS: reason=invalid_report_exhausted` to `detection.log`, which concludes as `parse_error`.
  An OpenCode failure is retried `GH_AW_HARNESS_MAX_RETRIES` times. A Coding Plan fallback runs inside its attempt and does not count as a retry.
  After the last retry fails, the detector writes `reason=engine_error`, which concludes as `agent_failure`.
  An invalid verdict is not retried.
  A run that reaches the script's deadline, two minutes before the step's 15, writes `reason=engine_timeout`, which concludes as `agent_failure`, and stops.
  Warn mode turns each of these three into a `warning` and a comment on the `[aw] Detection Runs` tracker.
- Keep gh-aw's `threat-detect` install: its conclusion step runs `threat-detect conclude`.
  gh-aw v1.2.1 also installs AWF, its container images and the Copilot CLI in the detection job. The detector uses none of them.

## Weekly agent instructions audit

`pydantic-ai-weekly-ce-audit.md` audits tracked context engineering (CE) Markdown weekly and on manual dispatch, proposing at most one focused trimming PR or a noop.
An existing workflow PR stops primary inference; threat detection remains enabled.

`.github/scripts/weekly_ce_audit.py` inventories committed objects, resolves tracked aliases, and rejects incomplete reported coverage or edits outside existing canonical instruction Markdown.
Workflow sources, imports, distributed package instructions and the packaged gh-aw definition remain report-only.
Keep candidate edits in a separate worktree and the reviewer checkout at the immutable policy base.

A fresh read-only Task must assess the actual diff, template exemption and scope of cited maintainer agreement.
Typos, broken links and small documentation improvements may omit an issue; nontrivial changes need an existing aligned issue assigned to the workflow actor, or noop.
For non-exempt proposals, host validation checks frozen issue identity, open state, assignment, cited maintainer authority and closing reference.
It also checks staged evidence and binds the review to the candidate, task and PR metadata; semantic classification belongs to the mandatory reviewer.
The agent job records the approved proposal in its `weekly-ce-audit` artifact.
The publication job publishes a pull request only when the ingested request equals that approved proposal.
Local Task evidence is an execution check, not attestation against a compromised coordinator.

The separate PR publication job uses the existing Docs app and retains protected-file review.
The validator's regressions live in `.github/scripts/test_weekly_ce_audit.py`; use the compilation rules above when changing the workflow source.

## Shared Z.AI account limits

Treat Pydanty, these workflows and the [OpenCode fallback](#opencode-falls-back-from-opencode-go-to-the-glm-coding-plan)
as consumers of the primary Z.AI account. The GitHub `ZAI_API_KEY` and Pydanty's production credential use that account.
`ZAI_BACKUP_API_KEY` holds a second Coding Plan account, the backup account. These workflows use it only when the
primary account cannot serve a run. The OpenCode fallback splits its runs across both accounts. Pydanty never uses it.
Correlate overlapping workflow and Pydanty requests when investigating rate-limit failures.
Include delegates and SDK retries in the request timeline.

Record the endpoint, model, HTTP status, provider `error.code`, and any `Retry-After` or quota reset time.
For abnormal streaming termination without an error code, record `finish_reason`.
Use [Z.AI's error codes](https://docs.z.ai/api-reference/api-code) to distinguish
`1302` request rate limits, `1305` temporary overload, and quota exhaustion such as `1308` or `1310`.
HTTP 429 alone does not identify the failure.

Check the [Coding Plan usage policy](https://docs.z.ai/devpack/usage-policy) for concurrency.
Check the [usage credit allowance](https://docs.z.ai/devpack/overview#usage-credit-allowance) for 5-hour and weekly quotas.
Concurrency changes with plan tier and resource availability.
Read account-specific values from the [rate-limit console](https://z.ai/manage-apikey/rate-limits).
Do not infer an account limit from Pydanty's observed concurrency or the provider's project-count recommendations.
A healthy usage-quota check does not establish spare request concurrency.

## Z.AI provider health

Every Z.AI-backed workflow must import `shared/provider-health.md` and include
`needs.provider_health.outputs.ready == 'true'` in its top-level gate. Keep its
existing eligibility and security conditions. Reference
`${{ needs.provider_health.outputs.ready }}` in the prompt body so gh-aw hoists
the custom job before activation; after compilation, verify `activation.needs`
includes `provider_health` and that `provider_health` does not depend on `activation`.
If a workflow has a local engine config, forward the same workflow, event, and
stable task-key values that `shared/engine-zai.md` forwards.

The provider-health job checks out the default branch and runs outside the agent
container. It writes a decision summary and the `provider-health` artifact for every
decision. A blocked gate must skip inference without emitting a passing review or other
fabricated agent result. The non-model provider-health monitor owns incident issue
creation and recovery. Do not enable gh-aw's generic failure-as-issue reporting or its
failed-job reporting (`report-failed-jobs`) for these workflows.

Every decision, the gate's and the agent-start recheck's, also exports one
`Z.AI provider health decision` span over OTLP/HTTP JSON:

- The controller reads the OTLP endpoint, `Authorization` header, service name and resource
  attributes that `shared/otel-logfire.md` compiles into each lock's workflow-level `env:`.
  Add no Logfire secret to these jobs or steps.
- Without the token, as on fork pull request events, it sends nothing. A failed export prints a
  warning and never changes the decision or the exit code.
- It sends with redirects disabled, as the Z.AI quota check does.
- `github.job` is `provider_health` for the gate and `agent` for the recheck. The recheck joins
  gh-aw's agent-job trace. The gate runs before gh-aw starts that trace, so its span starts its own;
  join the two by the `github.run_id` resource attribute.
- The span carries `provider_health.ready`, `.reason` and `.account`. Each checked account adds
  `provider_health.<account>.ready`, `.reason` and `.quota.*`, the artifact's quota fields. A gate
  that checked only the primary reports the decision as the primary's.
- `PR Review Checks` gives the controller to the `telemetry` check as a joint path, so the
  `general` check still reviews it.

The runner's `Workflow finished` log adds `failure_kind` and `http_status` from the failure the
record receives, and `zai_error_code` from a model HTTP error's body.

The gate checks the primary account first. When the primary's own provider incident or its quota
blocks it, and `ZAI_BACKUP_API_KEY` is set, the gate checks the backup account the same way. A
workflow or task incident blocks both accounts, so the gate does not check the backup for it. The run
blocks only when neither account can serve it. A ready decision outputs `account=primary` or
`account=backup`, and its artifact records each account's own decision when it checked both.
Without `ZAI_BACKUP_API_KEY`, the gate checks only the primary.

`shared/engine-zai.md` gives the engine `ZAI_BACKUP_API_KEY` when the agent-start recheck below
outputs `account=backup`, and `ZAI_API_KEY` otherwise. GitHub resolves that expression before
`awf` starts. The AWF api-proxy receives the selected key, and `awf --exclude-env` keeps it out of
the agent container. Keep the recheck's step id `agent_start_health`: the expression reads it.

The agent job waits for its engine concurrency slot after `provider_health` decides,
sometimes for hours. Incidents and quota exhaustion from that wait must still stop
inference. `shared/provider-health.md` therefore reruns the same check as the agent job's
first pre-agent step, before any workspace script runs:

- It fetches the controller from the default branch through the contents API. Do not run a
  workspace copy: the workspace can hold the PR head or its base branch.
- It forwards the same environment as the `provider_health` job.
- It replaces the `provider-health` artifact, so the monitor reconciles the decision the run
  acted on. When that upload fails, the record and the monitor read the gate's decision instead. If the
  recheck also chose another account than the gate, the agent job stops before inference, so neither
  attributes the run to the wrong account. The monitor never reads a failure of these steps as a setup failure. A recheck
  that errors opens no incident, like an error in the `provider_health` job.
- A blocked decision fails the agent job before the engine step. Do not write a `noop`
  instead: `detection` runs on any agent output and would still spend a detection pass.

A blocked recheck concludes the run as `failure`. For **CI Review**, `flag_failed_review`
then labels the PR `ci-review-failed`. Keep `ZAI_API_KEY` and `ZAI_BACKUP_API_KEY` in the
`provider_health` job, these host steps, the `record` step of the `provider_health_record` job
below, and the OpenCode threat detection step. gh-aw allows no secret in the agent job's post-steps. The agent container gets no Z.AI
credential beyond the engine's own.

The monitor reads a run's terminal result from a record a host job writes, not from the
agent's own output. After the agent job, including a failed one, `shared/provider-health.md`
runs the `provider_health_record` job. It checks out the default-branch controller, reads the
engine's terminal `session.result` from this run's `agent` artifact, and adds `run-result.json`
to the `provider-health` artifact. gh-aw's parser wrote that `agent-session.jsonl` on the agent
host from the engine's verified recording. The record keeps the failure kind the runner
reported. It marks a provider-scope kind confirmed only when the job's own Z.ai check observes
the same condition: the credential rejected with HTTP 401 or 403, or the plan's quota exhausted.
That check uses the key of the account the run used, which the record reads from the health decision.
gh-aw runs the job only after threat detection passes cleanly, and detection skips the model
when the agent left no output or patch. A failed record does not fail the run. Without a valid
record (the run predates the job, detection did not pass, or the record failed), the monitor
reads the `agent` artifact, which never confirms a provider failure.

The monitor serializes only incident work. Its `triage` job runs for each allowlisted
completion without a concurrency group and stops before it reads incidents. The `monitor`
job runs the completions `triage` hands over, plus scheduled and manual recovery, in one
job-level group with `queue: max`. Both jobs download the same artifacts with identical
steps; change them together. Keep the group on the job: a workflow-level group queues
every completion, and GitHub cancels new runs once 100 are pending, which drops
reconciliations and delays the hourly recovery. A job skipped by `if:` takes no queue
slot. GitHub does not document this. In CI Review's job-level `gh-aw-claude-CI Review`
group, the `agent` job of [run 39817346813](https://github.com/pydantic/pydantic-ai/actions/runs/39817346813),
skipped by its `if:`, resolved at once, while the queued `agent` job of
[run 39813059269](https://github.com/pydantic/pydantic-ai/actions/runs/39813059269) waited
71 minutes before it started. Scheduled recovery stays in the group because it can open
the same provider incident as a completion.

Older compiled workflows that pass `MINIMAX_API_KEY` without a `ZAI_API_KEY`
environment entry are intentionally blocked without an operational incident.
They fetch this controller from the default branch, so the skip applies even
before their branch receives the provider migration. Update the branch from
`main` to restore agent runs. An empty or missing Z.ai credential in current
configuration blocks the run; the provider-incident rules below say which runs open an incident for it.

The controller keeps one assigned operational incident for a matching failure
scope and marks it with both `agentic-workflows` and `pydanty:meta`. Leave both labels
in place: `pydanty:meta` keeps the automatic `@claude` issue/comment workflow from
starting on controller metadata. Do not create a second issue for the same open
provider, workflow, or task incident.

Each provider incident belongs to one account: its marker key is `zai` for the primary and
`zai:backup` for the backup. A provider incident blocks its account for every Z.AI workflow,
and runs fall back to the other account. Open one only on provider-wide evidence:
exhausted or unverifiable quota, rejected credentials, or a 429 whose Z.AI code is a
documented account or plan limit (runner kind `plan_limit`). A rejected credential or plan
limit from the agent run needs the host record's confirmation; unconfirmed, it opens a
workflow or task incident like other typed failures. Open a quota incident only when no account
can serve a run. Each account its quota blocks then gets its own incident. One exhausted
account alone opens no incident. Keep these outcomes local to their run:

- Any other 429 that outlasts the runner's retry window (runner kind `rate_limit`), such as
  `1302` request-rate limits and `1305` overload. A healthy quota check cannot show spare
  concurrency, so such an incident has no recovery signal. An exhausted plan still blocks
  the next run at its quota check.
- A gate on a `pull_request`, `pull_request_review` or `pull_request_review_comment` event
  that ran without `ZAI_API_KEY` (diagnostic `missing_api_key`). GitHub withholds secrets
  from these events on fork and Dependabot pull requests. Gates on other events, and the
  scheduled check, still open an incident for a missing key.

Before the gate passes or incident reuse creates an issue, re-read a list that returned no
readable incident. During a GitHub Issues degradation the labeled list returned HTTP 200 with
no readable incident of any age or scope. Each read writes its pages' HTTP status, issue and
unreadable counts, request IDs, and the incident numbers it saw to the job log and step summary.

The trusted health check calls `GET https://api.z.ai/api/monitor/usage/quota/limit`
with the raw key of the account it checks in the `Authorization` header. Recognize
`TOKENS_LIMIT` and `CREDIT_LIMIT` rows. Both quota windows must be valid:
unit `3` / number `5` is the 5-hour window, and unit `6` / number `1` is the weekly window.
`percentage` is the percentage used, so remaining quota is `100 - percentage`; `nextResetTime` is
epoch milliseconds. An omitted reset or explicit `null` is valid only when `percentage` is `0`;
a missing or null reset is invalid when quota has been consumed. A missing
or malformed required window makes health unknown and keeps agent workflows
gated. No quota-resource repository variable is needed.

Unknown quota results may include one of these fixed diagnostic categories:
`missing_api_key`, `invalid_request`, `http_error`, `timeout`, `tls_error`, `network_error`,
`invalid_encoding`, `invalid_json`, `invalid_response`, `invalid_response_code`,
`unsuccessful_response`, `invalid_limits`, `invalid_limit_row`,
`invalid_window_unit`, `invalid_window_number`, `duplicate_window`,
`invalid_percentage`, `invalid_reset_time`, or `missing_windows`. The artifact may
also include a numeric HTTP status. These values contain no provider response text,
request headers, URL, credential, or exception message. Version 1 artifacts without
diagnostic fields remain valid. Diagnostics explain the unknown result; they do not
change gate or recovery policy. Only `missing_api_key` changes incident creation, and only
on the pull request events listed above.

For manual provider recovery, run **Agent Provider Health** with `workflow_dispatch`
and enter the incident's number as `recover_issue`. The controller finds that one
open, marked incident whose provider key names a Z.ai account.
It never closes another incident in the same run. A fresh health check of the incident's own
account must pass before the controller closes the named incident. The hourly scheduled run closes Z.ai
plan-limit, quota-exhaustion, and unknown-quota incidents after a fresh healthy check of the
incident's own account, once the recorded reset time has passed or, when none was recorded, once the incident
is an hour old. Keep the hour: closes within minutes of opening were followed by a new
incident within minutes.
Credential and configuration incidents need operator attention rather than automatic recovery.

For a workflow-scoped incident, dispatch the affected workflow from the repository's
default branch and set `recovery_issue` to its incident number. This lets one fresh
manual run bypass that matching workflow blocker; it does not close the incident.

Task incidents block only the exact task key in their marker. Manual task keys include
the run ID, so a fresh dispatch has a new identity and can run with healthy quota
without `recovery_issue`. This input never closes an incident.

For **CI Review**, provide `aw_context` as JSON with `item_type` set to
`pull_request` and `item_number` set to the PR number. Verify the fresh run executed
the agent and model.

For workflow incidents, and task incidents the hourly run does not retire, inspect the
affected host stage and output, link that evidence to the incident, and close it manually.
A green run that skips the model or affected stage is not proof.

The hourly scheduled run retires a task incident that can no longer block a run.
That is the case when the pull request is merged or closed, when its head moved off the SHA in
the key, or when the manual dispatch run in the key has concluded. The run comments the state
it read once, then closes the incident. A retired incident records no recovery. Do not
describe a retired task as a same-target recovery. Close a task incident manually when its
key names neither a pull request head nor a manual dispatch run, such as a scheduled run's key.

The health controller never enables a disabled agent workflow. After a fresh healthy
provider-health check, an operator may deliberately enable a disabled workflow.

## Policy guard

`.github/scripts/agentic_workflow_guard.py` statically checks these workflows in CI. Every check encodes a defect that actually reached `main` and burned model budget before anyone noticed (see #6766) — a failure here is a real bug, not a style nit:

| Check | Rejects | Why it matters |
|---|---|---|
| `dangling-needs` | any `if:`, `outputs:`, `env:`, `with:` or `run:` referencing `needs.<job>` where `<job>` isn't a dependency of that job (outside `if:`, only inside `${{ }}` — elsewhere the text is literal) | The expression evaluates to empty rather than failing. In `if:` that skips the job or step — and **a job skipped by `if:` reports success**, so the required check stays green while the agent never runs. In `outputs:`/`env:`/`with:`/`run:` nothing skips at all: the step runs with an empty value, so a wrong action call or shell variable goes through looking healthy. This is the mechanical enforcement of ["A custom job named in `if:` must also appear in the prompt"](#a-custom-job-named-in-if-must-also-appear-in-the-prompt) — it reads the recompiled lock, so it catches the missing prompt reference whatever the cause. |
| `provider-health-*` | a Z.AI source missing the shared gate, generic failure-as-issue or failed-job reporting left enabled, a compiled activation graph that cannot see `provider_health`, an agent job that does not rerun the default-branch check with the gate's environment, both Z.AI keys and the step id `agent_start_health`, and stop before its engine step when blocked or when a failed replacement would hide the recheck's account, a workflow without the host job that records the agent job's terminal result, a monitor that accepts unlisted workflows or itself, or a monitor that queues completions `triage` did not hand over | A missing dependency can spend inference budget during a provider incident or make the recovery monitor loop on its own completions. A queue of every completion overflowed at 100 pending runs, dropped reconciliations, and delayed scheduled recovery ([run 39816847357](https://github.com/pydantic/pydantic-ai/actions/runs/39816847357)). |
| `provider-engine-config` | a shared threat-detection engine other than `false`, or a workflow-local engine override, whose endpoint or credential differs from the shared primary engine, or an empty shared primary endpoint or credential | Engine overrides drift when the provider endpoint or secret changes, which can send requests to the wrong provider or make them fail authentication. |
| `assigned-alert-metadata-gate` | the `@claude` issue and comment entry points lack an exclusion for `pydanty:meta` issues | Operational incidents may mention `@claude`; assigning or commenting on the alert must not start an agent on controller metadata. |
| `safe-output-job-max` | a `safe-outputs.jobs.*` entry with no `max:` | The default is 1; extra items land in an `errors` array nothing reads. Set it explicitly even when 1 is right. |
| `prompt-path-outside-workspace` | prompt text pointing at `/tmp/gh-aw/...` | Outside the agent's file-tool root — `Read` rejects it and the agent burns turns rediscovering the file. Stage context under `$GITHUB_WORKSPACE`. |
| `timeout-declared` | a source with no `timeout-minutes:` | An unbounded agent can spend a full run and be killed with nothing to show. |
| `job-timeout-env-missing` / `job-timeout-env-mismatch` | a source whose `env.PYDANTIC_AI_JOB_TIMEOUT_MINUTES` is absent, or set to something other than its `timeout-minutes` | The shim derives the agent's own wall-clock budget from that variable, because gh-aw's `GH_AW_TIMEOUT_MINUTES` is set only on the failure-handler step and never reaches the agent container. Drift means the agent either stops early and wastes the time it was granted, or overruns and is killed with nothing emitted. Every new workflow trips this until it declares the pair. |
| `job-timeout-too-short` | `timeout-minutes:` at or below the shim's teardown headroom (2) | The shim reserves that headroom for container teardown and artifact upload, so it falls back to 30 minutes rather than granting a non-positive budget — the agent then runs far past the point Actions kills the job. |
| `compiler-version-drift` | locks built by different gh-aw versions | Catches a partial `gh aw compile`. |
| `compiled-runner-contract` | missing public CLAI execution, runtime pin, workflow policy, Z.ai credential exclusion, or the engine key the agent-start recheck's `account` selects | The compiled job must install the pinned runtime and preserve the workflow policy and proxy boundary. |
| `awf-binary-version` | an installer pin differing from the bundled version or generated schema URL | The compiler installs AWF directly. Its binary must match the configuration schema. |
| `ai-credits-accounting` | an AWF provider overlay with unverified prices or per-million values in per-token fields | GLM accounting uses the verified Z.ai API-equivalent rates. Provider overlays use USD per token; default pricing uses USD per million. AWF retains its fixed 10,000-credit cap independently of configurable budgets. |
| `daily-aic-guardrail` | a lock that compiles gh-aw's daily AI-credits guardrail step, matched by the script it runs (`check_daily_aic_workflow_guardrail.cjs`) | On every activation the guardrail lists the artifacts of each run from the past 24h to total their AI credits: about 700 REST calls per CI Review activation in [run 39522281523](https://github.com/pydantic/pydantic-ai/actions/runs/39522281523). That spend exhausted the repo's `GITHUB_TOKEN` rate limit. AI credits do not measure the Z.AI Coding Plan quota; provider health gates that quota. Disable the guardrail with `max-daily-ai-credits: -1` in each source's own frontmatter. gh-aw removes the guardrail only for that value. A `-1` in an import such as `shared/engine-zai.md` only drops `GH_AW_MAX_DAILY_AI_CREDITS`: the scan step compiles but skips, and the conclusion job's `if: always()` cache restore, write, save and `aic-usage-cache` upload still run whenever the conclusion job runs. |
| `engine-sys-path-order` | a compiled engine wrapper (`AGENT_MODULE`) with a `sys.path.insert(0, ...)` statement, or a lock with no engine wrapper | The wrapper comes from the engine revision `shared/engine-zai.md` imports. #10347 imported one from before #12818, so every lock put the checkout ahead of installed packages on `sys.path` (#12815). |
| `logfire-import` / `logfire-compiled` / `logfire-network` | a Z.AI source missing `shared/otel-logfire.md`, a lock missing the compiled OTLP settings, or a compiled firewall missing any of the six Logfire ingest hosts | Three Z.AI workflows reached `main` without trace export (#19). The import and compiled settings enable export; the firewall must allow the ingest hosts or it silently blocks traces. |
| `lock-not-regenerated` | a changed `*.md` (or `shared/*.md` import) without its recompiled lock | Enforces the rule above. Checks that source and lock changed *together*, not that the lock was actually recompiled — `gh aw compile` is still on you. |

Run it locally before pushing:

```
uv run python .github/scripts/agentic_workflow_guard.py check --base-ref origin/main
```

Pass both the script project and changed file paths to Pyright, e.g. `uv run pyright -p .github/scripts .github/scripts/agent_provider_health.py`. The root Pyright project skips dot directories, even when a file is named explicitly.

When adding a check, pair it with a regression test in `test_agentic_workflow_guard.py` built from the configuration that actually broke — the existing cases are reconstructed from the parent commit of the PR that fixed each one.
