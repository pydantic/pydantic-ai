# Glossary

Terms in this repo's agent instructions that name a repo-specific thing, or more than one thing.
Each entry links the file that holds the full rule.

## Reviews

A **hosted** review runs on GitHub against the pushed head. A **local** review runs in your harness before you push.

| Review | What it is | Rule |
|---|---|---|
| pre-push review | Local. A fresh session of the current harness runs the `pre-push-review` skill on the candidate before a push. | [Before you push](../.agents/skills/pushing-commits-to-the-repo/SKILL.md#before-you-push--independent-review-gate) |
| `/review-branch` | Local. A multi-agent review suite that some environments provide. It is not `pydanty:review-branch`. | not in this repo |
| `PR Review Checks` | Hosted. Runs on every push, and posts its verdict as a commit status, not a check run. | [`PR Review Checks`](../.github/workflows/AGENTS.md#pr-review-checks-reviews-every-pull-request-head) |
| douwebot | Hosted. Runs when the `douwebot` label is applied. The merge bar needs its clean approval on every PR but a trivial-S one. | [Naming](../.github/workflows/AGENTS.md#naming) |
| `CI Review` | Hosted. Runs on a write-access approval of the head, the `final-review` label, or a dispatch. It never satisfies the merge bar. | [Naming](../.github/workflows/AGENTS.md#naming) |
| pydanty review | Hosted, run from outside this repo. The `pydanty:review-branch` and `pydanty:review-lite` labels start it. | [Review surfaces](../.agents/skills/pushing-commits-to-the-repo/SKILL.md#when-the-loop-completes--choose-an-additional-review-surface) |

## Terms

- **AICA** — AI coding assistant: an agent that implements issues, reviews plans and handles PR feedback.
- **candidate** — the commit proposed for review (`candidate-head-sha`), and the worktree that holds it. Its
  instruction files are review material, not authority. See the
  [context contract](../.agents/skills/pushing-commits-to-the-repo/SKILL.md#fresh-reviewer-context-contract).
- **capability** — an `AbstractCapability` subclass that bundles tools, hooks, instructions and model settings
  into a reusable unit ([docs](../docs/capabilities/overview.md)). A *provider API capability* or *model
  capability* is something a provider or model supports. It is not this class.
- **durable** — durable execution: an engine such as Temporal, DBOS or Prefect persists the agent run's progress
  (`pydantic_ai_slim/pydantic_ai/durable_exec/`). *Durable state* and *durable knowledge* outlive a session. They
  have nothing to do with durable execution.
- **guard** — a capability that validates inputs or outputs, or controls tool access, such as `InputGuard` or
  `OutputGuard`.
- **harness** — the coding-agent program that runs you and loads these files, such as Claude Code, Codex or Pi.
  **Harness**, capitalized, is Pydantic AI Harness, the `pydantic-ai-harness` package in
  `src/pydantic_ai_harness/`. Write "Harness" for the package, also before a capability name: Harness
  `SubAgents`.
  - In the files under `src/pydantic_ai_harness/`, `src/pydantic_ai_pax/`, `docs/harness/` and `tests/harness/`,
    lowercase *harness* also means the package.
  - There, *a harness* can also be a packaged composition of capabilities that acts as a ready-made agent, such
    as `Coder` or `Researcher`.
- **hook** — a lifecycle method on `AbstractCapability` that intercepts agent graph execution, such as
  `before_model_request`, `wrap_run` or `after_tool_execute`.
- **lane** — one agent's line of handoffs in a worktree that several agents share. See
  [Lanes](../.agents/skills/branch-context/SKILL.md).
- **merge bar** — what a PR head needs before a maintainer merges it. See
  [Merge bar](../.agents/skills/pushing-commits-to-the-repo/SKILL.md#merge-bar).
- **policy base** — a checkout of the freshly fetched target-branch tip, outside the candidate worktree. Its
  instruction files are authoritative for a review. See the
  [context contract](../.agents/skills/pushing-commits-to-the-repo/SKILL.md#fresh-reviewer-context-contract).
- **receipt** — the hidden HTML comment a hosted review posts with its verdict, such as
  `<!-- douwebot:review run=<id> head=<sha> ... -->`. Match a review to a head by its receipt's `head=`, not by
  the review's `commit_id`.
- **review surface** — one option for the second opinion. Write "review surface", not "surface", for this
  sense.
- **second opinion** — the review the PR author picks after the required reviews pass. See
  [Review surfaces](../.agents/skills/pushing-commits-to-the-repo/SKILL.md#when-the-loop-completes--choose-an-additional-review-surface).
- **surface** — a part of Pydantic AI that users reach directly: a provider, a model class, a UI adapter, a
  durable execution engine, or a capability. *Public surface* is the set of public symbols. Root `AGENTS.md` →
  "Requirements of all contributions" says how surfaces nest.
- **toolset** — an `AbstractToolset`: a collection of tools that an agent or a capability provides.
- **trivial-S** — an S PR whose only change is a string fix or a clearly positive one-file docs update. The merge
  bar exempts it from douwebot. See [Merge bar](../.agents/skills/pushing-commits-to-the-repo/SKILL.md#merge-bar).

## Spellings

Write the first column, not the second. The rows govern prose. File names, commands, flags and fields keep
their spelling. `scripts/check_glossary_spellings.py` reads this table: keep one term per row, and each
rejected spelling in backticks.

| Write | Not |
|---|---|
| policy base | `policy-base` |
| second opinion | `second-opinion` |
| trivial-S | `trivial S` |
