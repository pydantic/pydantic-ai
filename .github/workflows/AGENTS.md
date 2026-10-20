# Which copy of a workflow runs for `v1`

GitHub picks the workflow file by trigger, and on this maintenance branch that decides whether
editing a file here changes anything at all.

- **`pull_request` and `push`** run the file at the PR's merge ref or the pushed commit, so `v1`'s
  own copy runs. That covers `ci.yml`, `pydantic-ai-pr-review.md`,
  `pydantic-ai-ui-security-review.md`, and the `pull_request_review` triggers of `at-claude.yml`.
  A `v1` PR that edits one of them tests its own change. A tag push runs the tagged commit's
  `ci.yml`, release jobs included.
- **`pull_request_target`, `workflow_run`, `schedule`, `issues` and `issue_comment`** run the
  [default branch's file](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#pull_request_target),
  `main`'s, whatever branch the PR targets. On a `v1` PR, `PR Bots` (`bots.yml`), `PR Guard`,
  `Package Label` and `Protect .github` are `main`'s workflows — the last two do not exist on
  `v1` at all. `v1`'s copies of `bots.yml`, `pr-guard.yml`, `after-ci.yml`, `stale.yml`,
  `gateway-model-health.yml`, `agentics-maintenance.yml` and the scheduled `pydantic-ai-*`
  workflows never run. Change those on `main`.
- A file that exists only on `v1` never runs under those triggers. `docs-preview.yml` is one:
  the `trigger:docs` label does nothing on a `v1` PR.

So a `v1` PR that edits one of the `main`-run workflows changes nothing that executes. A `main`
PR that edits one cannot test its own change either: the edit takes effect only after merge.
Reason about such a change by reading the diff.

# Naming

Check names are what humans and agents refer to when they talk about CI, so they are
descriptive rather than generic. These reviewers run on a `v1` PR:

| Name | Where | Runs when |
|------|-------|-----------|
| `Pydantic AI PR Review` | `pydantic-ai-pr-review.md` (`v1`'s copy) | automatically when a PR is opened, pushed to, or marked ready for review, unless it carries the `auto-review` label. Z.AI engine (`shared/engine-zai.md`). Submits `REQUEST_CHANGES` for any surviving HIGH or CRITICAL finding and `APPROVE` otherwise (`shared/prompts/pydantic-ai-pr-review.md`). |
| `douwebot` | `main`'s `bots.yml` | only on applying the **`douwebot` label**. It checks out the PR's base ref (`v1`) for its instructions, posts inline findings and a formal verdict, and deletes the label when it finishes. |

**Never apply `auto-review` to a `v1` PR.** It makes `Pydantic AI PR Review` skip, and the
`bots.yml` `review` job it used to route to never runs, because `main`'s `bots.yml` is the one
that executes. The PR is left with no automatic review.

`main`'s `CI Review` also runs on `workflow_run` after every `CI` run. Its `eligibility` job does
not filter by base branch.

Not to be confused with `Pydantic AI UI Security Review`, a separate narrow reviewer that only
audits the UI-adapter trust boundary. It always submits a `COMMENT` review and never owns the
merge-gate verdict (`shared/prompts/pydantic-ai-ui-security-review.md`).

## A custom job named in `if:` must also appear in the prompt

When a workflow's top-level `if:` references a custom job's output
(`needs.<job>.outputs.<x>`), **that job must also be referenced somewhere in the prompt
body**, even if only inside an HTML comment.

`gh aw compile` copies the top-level `if:` onto the generated `activation` job, but it
only adds jobs referenced by the *prompt* to `activation.needs`. A job named only in the
`if:` therefore resolves to the empty string inside `activation`, so `activation` skips —
and because the compiler makes custom jobs depend on `activation`, the named job and the
agent skip with it. A job skipped by `if:` reports as success, so the whole thing goes
green while never having run (#6766). After adding such an `if:`, check the recompiled lock:
`activation.needs` must list the job, and the job itself must **not** have `needs: activation`.

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

Treat Pydanty and the Z.AI-backed workflows as consumers of one Z.AI account.
On `v1` those are `pydantic-ai-pr-review.md` and `pydantic-ai-ui-security-review.md`, which import `shared/engine-zai.md`.
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
