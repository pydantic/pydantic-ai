# Naming

Check names are what humans and agents refer to when they talk about CI, so they are
descriptive rather than generic. Two reviewers perform the same role — the
maintainer-voice standards review, driven by the repo's `AGENTS.md` and
`agent_docs/*.md` — on different engines and different cadences:

| Name | Where | Runs when |
|------|-------|-----------|
| `CI Review` | `pydantic-ai-pr-review.md` | when a user with write access or higher **approves** the PR's current head, on applying the **`final-review` label**, or on dispatch, once `CI` has passed on that head. Never in the merge queue. Z.AI Coding Plan engine, submits formal `REQUEST_CHANGES` for verified HIGH or CRITICAL findings, `COMMENT` for incomplete coverage without those findings, and `APPROVE` after complete coverage without them. Same-repo PRs only. |
| `douwebot` | `bots.yml` | only on applying the **`douwebot` label** — the fork-capable path (`pull_request_target`) and Claude Opus 5.5. Deletes the label when it finishes. Posts inline findings and a formal `APPROVE` or `REQUEST_CHANGES` review for the reviewed head. |
| `douwebot:review-agent-config` | `bots.yml` | one explicit review request for one PR head; the labeling sender must be a `User` with current `write` or `admin` permission. The label is not standing approval. |

`douwebot` treats every finding as blocking unless its comment starts with `Optional:`. Repository-rule violations in `AGENTS.md` or `agent_docs/*.md` are blocking and cannot be marked `Optional:`. It requests changes while any blocking finding remains outstanding, including unaddressed findings from earlier review rounds, without duplicating comments. It approves after a complete review with no outstanding blocking findings; optional findings may remain. It submits no verdict when the review is incomplete or the PR head changes before publication. The reviewer assesses choices against issue guidance and repository standards; a missing issue link or separate human sign-off alone does not block review.

The `douwebot:review-agent-config` label requests one review only from its first `labeled` event. The workflow checks the sender's current repository permission, the event's immutable head against the live PR head, and that the label is still present immediately before Claude starts. API errors, non-`User` senders, reruns, stale heads, and removed labels deny the request. An ordinary `douwebot` event never inherits approval from the label. To retry or approve a new head, a maintainer removes and reapplies the label; do not rerun the old event. The workflow never removes this label automatically because an old run could consume a newer request.

Removing the approval label cancels in-flight work on a best-effort basis; cancellation may not stop a reviewer that has already started. Both review labels share the per-PR concurrency group, so removing the approval label can also cancel an ordinary review, and reapplying it starts another workflow run that uses that shared slot.

Other label removals use unique concurrency groups so their no-op runs cannot replace queued review requests.

**They are independent.** Neither reads the other's state, and the label suppresses
nothing: `douwebot` is an on-demand deep pass on top of `CI Review`, requested when a
PR warrants a second opinion. Do not reintroduce a gate from one onto the other — the
previous label-as-switch design left a window in which a push got *neither* reviewer.

Not to be confused with `Pydantic AI UI Security Review`, a separate narrow reviewer
that only audits the UI-adapter trust boundary and never owns the merge-gate verdict.

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

The `CI Review` agent job queues in the engine concurrency group its own
`engine.concurrency` sets (`queue: max`), so one review runs at a time across all PRs.
`eligibility` can therefore decide hours before the model starts. The pre-agent step
`Skip a PR that closed or moved on while queued` writes a `noop` to safe outputs when the
PR has closed or moved to a new head. The engine's gh-aw harness must exit on a pre-written
`noop` before it starts the model. gh-aw's `claude_harness.cjs` does this
(`hasNoopInSafeOutputs`). Any other engine harness, including
`src/pydantic_ai_harness/gh-aw/pydantic.md`, must do the same, or every stale run
spends a full review. The `noop` does not skip the `detection` job, which still runs
one threat-detection pass outside the agent slot.

gh-aw drops its default agent group for a workflow with `pull_request*` triggers, and
allows one `engine` across a workflow and its imports. `CI Review` therefore inlines the
`shared/engine-zai.md` engine instead of importing it, to set `engine.concurrency`. Keep
the two in step.

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

That is not hypothetical: `pydantic-ai-ui-security-review` shipped this way and its agent
never fired once, reporting false-green on every PR (#6766). Both it and
`pydantic-ai-pr-review` now carry the prompt-body reference and a comment saying why.
After changing either, check the recompiled lock: `activation.needs` must list the job,
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
prompted the guard). **It only blocks merge once it is marked a required check in repo
settings**; until then it is advisory.

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

# Agentic workflows (`gh-aw`)

The `pydantic-ai-*` workflows in this directory are [agentic workflows](https://github.com/githubnext/gh-aw) authored as human-editable `<name>.md` sources (frontmatter + prompt) that **compile** to a generated `<name>.lock.yml`. GitHub Actions runs the `.lock.yml`, never the `.md`.

- **Never hand-edit a `*.lock.yml`.** It is generated — the header says so. Manual edits are silently overwritten on the next recompile, and until then the running workflow diverges from its source.
- **After editing a workflow `*.md` source, recompile and commit the regenerated `*.lock.yml` in the same change** — a `*.md` edit without its recompiled lock is incomplete, and source and lock drift apart:

  ```
  gh aw compile
  ```

- **Recompilation is required for anything the lock bakes in:** a source's frontmatter (`on:` triggers, `permissions`, `tools`, `safe-outputs`, jobs, path/`detect` filters) and its `imports:` shared fragments (`shared/*.md`) are inlined into the lock at compile time.
- **Exception — runtime-resolved prompts need no recompile.** Agent prompts under `shared/prompts/` are fetched at run time (via the `fetch-dynamic-prompt` action / a Logfire-managed variable), not baked into the lock, so editing one takes effect on the next run without recompiling.

## Shared Z.AI account limits

Treat Pydanty and these workflows as consumers of one Z.AI account.
The GitHub `ZAI_API_KEY` and Pydanty's production credential use that account.
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
container. Keep `ZAI_API_KEY` scoped to that job. It writes a decision summary
and the `provider-health` artifact on blocked decisions. A blocked gate must skip
inference without emitting a passing review or other fabricated agent result. The
non-model provider-health monitor owns incident issue creation and recovery; do not
enable gh-aw's generic failure-as-issue reporting for these workflows.

Older compiled workflows that pass `MINIMAX_API_KEY` without a `ZAI_API_KEY`
environment entry are intentionally blocked without an operational incident.
They fetch this controller from the default branch, so the skip applies even
before their branch receives the provider migration. Update the branch from
`main` to restore agent runs. An empty or missing Z.ai credential in current
configuration still follows the normal provider-health checks.

The controller keeps one assigned operational incident for a matching failure
scope and marks it with both `agentic-workflows` and `pydanty:meta`. Leave both labels
in place: `pydanty:meta` keeps the automatic `@claude` issue/comment workflow from
starting on controller metadata. Do not create a second issue for the same open
provider, workflow, or task incident.

The trusted health check calls `GET https://api.z.ai/api/monitor/usage/quota/limit`
with the raw `ZAI_API_KEY` value in the `Authorization` header. Recognize
`TOKENS_LIMIT` and `CREDIT_LIMIT` rows. Both quota windows must be valid:
unit `3` / number `5` is the 5-hour window, and unit `6` / number `1` is the weekly window.
`percentage` is the percentage used, so remaining quota is `100 - percentage`; `nextResetTime` is
epoch milliseconds. An explicit `null` is valid only when `percentage` is `0`;
a missing reset field or a null reset with consumed quota is invalid. A missing
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
change gate or recovery policy.

For manual provider recovery, run **Agent Provider Health** with `workflow_dispatch`
and enter the incident's number as `recover_issue`. The controller finds that one
open, marked incident whose provider key matches the currently configured provider.
It never closes another incident in the same run. A fresh health check must pass
before the controller closes the named incident. The hourly scheduled run closes Z.ai
rate-limit, quota-exhaustion, and unknown-quota incidents after a fresh healthy check,
once the recorded reset time has passed or, when none was recorded, once the incident
is an hour old. Keep the hour: a healthy quota check does not show spare concurrency,
and closes within minutes of opening were followed by a new incident within minutes.
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

For **Pydantic AI UI Security Review**, use a fresh eligible pull request event.
Manual dispatch does not provide `github.event.pull_request.number`, which this
workflow requires to run its agent and publish results. After the corrected workflow
is deployed, complete the existing UI Security review Environment approval for the
fresh PR run.

For workflow and task incidents, inspect the affected host stage and output, link that
evidence to the incident, and close it manually. A green run that skips the model or
affected stage is not proof.
Resolve closed-PR tasks as retired only after API verification and representative
workflow recovery. Record why each task is retired. Do not describe retired tasks as
same-target recoveries.

The health controller never enables a disabled agent workflow. After a fresh healthy
provider-health check, an operator may deliberately enable a disabled workflow.

## Policy guard

`.github/scripts/agentic_workflow_guard.py` statically checks these workflows in CI. Every check encodes a defect that actually reached `main` and burned model budget before anyone noticed (see #6766) — a failure here is a real bug, not a style nit:

| Check | Rejects | Why it matters |
|---|---|---|
| `dangling-needs` | any `if:`, `outputs:`, `env:`, `with:` or `run:` referencing `needs.<job>` where `<job>` isn't a dependency of that job (outside `if:`, only inside `${{ }}` — elsewhere the text is literal) | The expression evaluates to empty rather than failing. In `if:` that skips the job or step — and **a job skipped by `if:` reports success**, so the required check stays green while the agent never runs. In `outputs:`/`env:`/`with:`/`run:` nothing skips at all: the step runs with an empty value, so a wrong action call or shell variable goes through looking healthy. This is the mechanical enforcement of ["A custom job named in `if:` must also appear in the prompt"](#a-custom-job-named-in-if-must-also-appear-in-the-prompt) — it reads the recompiled lock, so it catches the missing prompt reference whatever the cause. |
| `provider-health-*` | a Z.AI source missing the shared gate, generic failure reporting left enabled, a compiled activation graph that cannot see `provider_health`, or a monitor that accepts unlisted workflows or itself | A missing dependency can spend inference budget during a provider incident or make the recovery monitor loop on its own completions. |
| `provider-engine-config` | a shared threat-detection engine or workflow-local engine override whose endpoint or credential differs from the shared primary engine, or an empty shared primary endpoint or credential | Engine overrides drift when the provider endpoint or secret changes, which can send requests to the wrong provider or make them fail authentication. |
| `assigned-alert-metadata-gate` | the `@claude` issue and comment entry points lack an exclusion for `pydanty:meta` issues | Operational incidents may mention `@claude`; assigning or commenting on the alert must not start an agent on controller metadata. |
| `safe-output-job-max` | a `safe-outputs.jobs.*` entry with no `max:` | The default is 1; extra items land in an `errors` array nothing reads. Set it explicitly even when 1 is right. |
| `prompt-path-outside-workspace` | prompt text pointing at `/tmp/gh-aw/...` | Outside the agent's file-tool root — `Read` rejects it and the agent burns turns rediscovering the file. Stage context under `$GITHUB_WORKSPACE`. |
| `timeout-declared` | a source with no `timeout-minutes:` | An unbounded agent can spend a full run and be killed with nothing to show. |
| `job-timeout-env-missing` / `job-timeout-env-mismatch` | a source whose `env.PYDANTIC_AI_JOB_TIMEOUT_MINUTES` is absent, or set to something other than its `timeout-minutes` | The shim derives the agent's own wall-clock budget from that variable, because gh-aw's `GH_AW_TIMEOUT_MINUTES` is set only on the failure-handler step and never reaches the agent container. Drift means the agent either stops early and wastes the time it was granted, or overruns and is killed with nothing emitted. Every new workflow trips this until it declares the pair. |
| `job-timeout-too-short` | `timeout-minutes:` at or below the shim's teardown headroom (2) | The shim reserves that headroom for container teardown and artifact upload, so it falls back to 30 minutes rather than granting a non-positive budget — the agent then runs far past the point Actions kills the job. |
| `compiler-version-drift` | locks built by different gh-aw versions | Catches a partial `gh aw compile`. |
| `awf-binary-version` | a lock whose `install_awf_binary.sh` pin differs from the `GH_AW_INFO_AWF_VERSION` gh-aw compiled it against, or that carries a pin the lock declares no such version for | `engine.command` makes gh-aw skip its own AWF install, so `shared/pre-steps.md` re-runs the installer with a version written by hand. A gh-aw upgrade moves the bundled version and leaves that pin behind, and the older binary then rejects the newer firewall config — the agent job dies before the model starts. Bump the pin in the same change as the compiler. The skip is tracked upstream at [github/gh-aw#58340](https://github.com/github/gh-aw/issues/58340) — retire this check and the hand-written pre-step once gh-aw installs the firewall binary independently of the engine. |
| `ai-credits-accounting` | an AWF config carrying `apiProxy.providers` | That key is the API proxy's AI-credits *pricing* table, not reporting metadata — a model priced there gets charged, and AWF caps a run at 10,000 credits that no frontmatter field lifts (`max-ai-credits` only drops gh-aw's own budget, and the schema types it `exclusiveMinimum: 0`). One ordinary request costs tens of thousands, so every agent job 403s on its first call with `ai_credits_limit_exceeded`. The weekly spend report reads token counts from each run's `agent_usage.json`, so nothing here needs the overlay. |
| `daily-aic-guardrail` | a lock that compiles gh-aw's daily AI-credits guardrail step, matched by the script it runs (`check_daily_aic_workflow_guardrail.cjs`) | On every activation the guardrail lists the artifacts of each run from the past 24h to total their AI credits: about 700 REST calls per CI Review activation in [run 39522281523](https://github.com/pydantic/pydantic-ai/actions/runs/39522281523). That spend exhausted the repo's `GITHUB_TOKEN` rate limit. AI credits do not measure the Z.AI Coding Plan quota; provider health gates that quota. Disable the guardrail with `max-daily-ai-credits: -1` in each source's own frontmatter. gh-aw removes the guardrail only for that value. A `-1` in an import such as `shared/engine-zai.md` only drops `GH_AW_MAX_DAILY_AI_CREDITS`: the scan step compiles but skips, and the conclusion job's `if: always()` cache restore, write, save and `aic-usage-cache` upload still run whenever the conclusion job runs. |
| `logfire-import` / `logfire-compiled` / `logfire-network` | a Z.AI source missing `shared/otel-logfire.md`, a lock missing the compiled OTLP settings, or a compiled firewall missing any of the six Logfire ingest hosts | Three Z.AI workflows reached `main` without trace export (#19). The import and compiled settings enable export; the firewall must allow the ingest hosts or it silently blocks traces. |
| `lock-not-regenerated` | a changed `*.md` (or `shared/*.md` import) without its recompiled lock | Enforces the rule above. Checks that source and lock changed *together*, not that the lock was actually recompiled — `gh aw compile` is still on you. |

Run it locally before pushing:

```
uv run python .github/scripts/agentic_workflow_guard.py check --base-ref origin/main
```

Pass both the script project and changed file paths to Pyright, e.g. `uv run pyright -p .github/scripts .github/scripts/agent_provider_health.py`. The root Pyright project skips dot directories, even when a file is named explicitly.

When adding a check, pair it with a regression test in `test_agentic_workflow_guard.py` built from the configuration that actually broke — the existing cases are reconstructed from the parent commit of the PR that fixed each one.
