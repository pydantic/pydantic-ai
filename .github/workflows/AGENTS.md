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

# Agentic workflows (`gh-aw`)

The `pydantic-ai-*` workflows in this directory are [agentic workflows](https://github.com/githubnext/gh-aw) authored as human-editable `<name>.md` sources (frontmatter + prompt) that **compile** to a generated `<name>.lock.yml`. GitHub Actions runs the `.lock.yml`, never the `.md`.

- **Never hand-edit a `*.lock.yml`.** It is generated — the header says so. Manual edits are silently overwritten on the next recompile, and until then the running workflow diverges from its source.
- **After editing a workflow `*.md` source, recompile and commit the regenerated `*.lock.yml` in the same change** — a `*.md` edit without its recompiled lock is incomplete, and source and lock drift apart:

  ```
  gh aw compile
  ```

- **Recompilation is required for anything the lock bakes in:** a source's frontmatter (`on:` triggers, `permissions`, `tools`, `safe-outputs`, jobs, path/`detect` filters) and its `imports:` shared fragments (`shared/*.md`) are inlined into the lock at compile time.
- **Exception — runtime-resolved prompts need no recompile.** Agent prompts under `shared/prompts/` are fetched at run time (via the `fetch-dynamic-prompt` action / a Logfire-managed variable), not baked into the lock, so editing one takes effect on the next run without recompiling.

## Lock files are not reproducible right now

`gh aw compile` cannot currently regenerate the committed `*.lock.yml` files. Dependabot
(#6196) bumped the pinned action SHAs *inside* the generated locks — `actions/checkout`
v6.0.2 → v7.0.0, `actions/setup-python` v6.2.0 → v6.3.0, `astral-sh/setup-uv` v8.1.0 →
v8.2.0 — but the compiler pins those versions itself and overwrites `.github/aw/actions-lock.json`
rather than reading it. So a recompile silently reverts all three bumps across every lock.

Until that is resolved, **check `git diff` after any `gh aw compile`**: if the only changes
are action SHA downgrades, the recompile is reverting a security bump, not applying your
edit. Editing a `*.md` source is effectively blocked on this.
