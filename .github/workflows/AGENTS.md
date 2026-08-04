# Naming

Check names are what humans and agents refer to when they talk about CI, so they are
descriptive rather than generic. Two reviewers perform the same role — the
maintainer-voice standards review, driven by the repo's `AGENTS.md` and
`agent_docs/*.md` — on different engines and different cadences:

| Name | Where | Runs when |
|------|-------|-----------|
| `CI Review` | `pydantic-ai-pr-review.md` | automatically, once the `CI` workflow **succeeds** on the PR's current head. MiniMax engine, submits a formal `APPROVE`/`REQUEST_CHANGES` verdict. Same-repo PRs only. |
| `douwebot` | `bots.yml` | only on applying the **`douwebot` label** — the fork-capable path (`pull_request_target`) and the stronger model. Deletes the label when it finishes. Inline comments, no verdict. |

**They are independent.** Neither reads the other's state, and the label suppresses
nothing: `douwebot` is an on-demand deep pass on top of `CI Review`, requested when a
PR warrants a second opinion. Do not reintroduce a gate from one onto the other — the
previous label-as-switch design left a window in which a push got *neither* reviewer.

Not to be confused with `Pydantic AI UI Security Review`, a separate narrow reviewer
that only audits the UI-adapter trust boundary and never owns the merge-gate verdict.

`CI Review` runs on `workflow_run`, which carries no `github.event.pull_request`. Its
`eligibility` job resolves the PR, its current head and its refs, and every later step —
including the agent's safe outputs — consumes those as explicit values. Anything you add
that needs to know which PR this is must read them from `needs.eligibility.outputs`,
never from `github.event`. For the safe outputs specifically that means frontmatter
`target:` plus `safe-outputs.needs:` — an out-of-scope `needs.*` expression compiles
without error and evaluates to the empty string, silently discarding the review.

## What confines `CI Review` to same-repo PRs

Two different gates, guarding two different things. Neither is redundant, and the
distinction is the whole point: **`roles:` gates the actor, `eligibility` gates the
code.**

**`eligibility` is what rejects forks.** It compares
`workflow_run.head_repository.full_name` to `github.repository` on the event, and then
the resolved PR's `isCrossRepository` — and skips on either. That is the check to look
at when you want to know whether fork code can reach this workflow.

**`roles:` gates who may trigger it, and is not a fork filter.** It compiles to a
`check_membership` step validating `github.actor`, which under `workflow_run` is the
actor of the triggering CI run. It does exclude external contributors, who have `read`.
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
checkout succeeding, which never happens under this trigger.

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

## Policy guard

`.github/scripts/agentic_workflow_guard.py` statically checks these workflows in CI. Every check encodes a defect that actually reached `main` and burned model budget before anyone noticed (see #6766) — a failure here is a real bug, not a style nit:

| Check | Rejects | Why it matters |
|---|---|---|
| `dangling-needs` | any `if:`, `outputs:`, `env:`, `with:` or `run:` referencing `needs.<job>` where `<job>` isn't a dependency of that job (outside `if:`, only inside `${{ }}` — elsewhere the text is literal) | The expression evaluates to empty rather than failing. In `if:` that skips the job or step — and **a job skipped by `if:` reports success**, so the required check stays green while the agent never runs. In `outputs:`/`env:`/`with:`/`run:` nothing skips at all: the step runs with an empty value, so a wrong action call or shell variable goes through looking healthy. This is the mechanical enforcement of ["A custom job named in `if:` must also appear in the prompt"](#a-custom-job-named-in-if-must-also-appear-in-the-prompt) — it reads the recompiled lock, so it catches the missing prompt reference whatever the cause. |
| `safe-output-job-max` | a `safe-outputs.jobs.*` entry with no `max:` | The default is 1; extra items land in an `errors` array nothing reads. Set it explicitly even when 1 is right. |
| `prompt-path-outside-workspace` | prompt text pointing at `/tmp/gh-aw/...` | Outside the agent's file-tool root — `Read` rejects it and the agent burns turns rediscovering the file. Stage context under `$GITHUB_WORKSPACE`. |
| `timeout-declared` | a source with no `timeout-minutes:` | An unbounded agent can spend a full run and be killed with nothing to show. |
| `job-timeout-env-missing` / `job-timeout-env-mismatch` | a source whose `env.PYDANTIC_AI_JOB_TIMEOUT_MINUTES` is absent, or set to something other than its `timeout-minutes` | The shim derives the agent's own wall-clock budget from that variable, because gh-aw's `GH_AW_TIMEOUT_MINUTES` is set only on the failure-handler step and never reaches the agent container. Drift means the agent either stops early and wastes the time it was granted, or overruns and is killed with nothing emitted. Every new workflow trips this until it declares the pair. |
| `job-timeout-too-short` | `timeout-minutes:` at or below the shim's teardown headroom (2) | The shim reserves that headroom for container teardown and artifact upload, so it falls back to 30 minutes rather than granting a non-positive budget — the agent then runs far past the point Actions kills the job. |
| `compiler-version-drift` | locks built by different gh-aw versions | Catches a partial `gh aw compile`. |
| `lock-not-regenerated` | a changed `*.md` (or `shared/*.md` import) without its recompiled lock | Enforces the rule above. Checks that source and lock changed *together*, not that the lock was actually recompiled — `gh aw compile` is still on you. |

Run it locally before pushing:

```
uv run python .github/scripts/agentic_workflow_guard.py check --base-ref origin/main
```

When adding a check, pair it with a regression test in `test_agentic_workflow_guard.py` built from the configuration that actually broke — the existing cases are reconstructed from the parent commit of the PR that fixed each one.
