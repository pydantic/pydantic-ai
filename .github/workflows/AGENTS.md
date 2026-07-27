# Naming

Check names are what humans and agents refer to when they talk about CI, so they are
descriptive rather than generic. Two reviewers perform the same role — the
maintainer-voice standards review, driven by the repo's `AGENTS.md` and
`agent_docs/*.md` — on different engines:

| Name | Where | Runs when |
|------|-------|-----------|
| `Pydantic AI PR Review` | `pydantic-ai-pr-review.md` | every PR from an `admin`/`maintainer`/`write` actor, **unless** the `auto-review` label is present. MiniMax engine, submits a formal `APPROVE`/`REQUEST_CHANGES` verdict. |
| `douwebot (label)` | `bots.yml` | only on applying the **`auto-review` label** — the fork-capable path (`pull_request_target`) and the stronger model. Deletes the label when it finishes, so the next push re-enables the gh-aw reviewer. |

Exactly one of the two runs per event; the label is the switch. Do not add a third
reviewer under a name that reads like either of these.

The gh-aw reviewer is still named for its file rather than its role, and the label is
still `auto-review`. Both are meant to become `douwebot (gh-aw)` / `douwebot`, so the
shared role is legible from the check name alone. That rename is blocked: the label
string is read in `pydantic-ai-pr-review.md`, renaming only one side of the switch makes
**both** reviewers run at once, and the gh-aw lock cannot currently be regenerated (see
"Lock files are not reproducible" below).

Not to be confused with `Pydantic AI UI Security Review`, a separate narrow reviewer
that only audits the UI-adapter trust boundary and never owns the merge-gate verdict.

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
