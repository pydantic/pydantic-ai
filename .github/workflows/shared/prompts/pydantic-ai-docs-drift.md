<!--
Default/seed prompt for the Pydantic AI Docs Drift agent.

This file is the COMPLETE prompt. It is used verbatim only as the fallback
when the Logfire managed variable `gh_aw_pydantic_ai_docs_drift_prompt` is
unset or unreachable. To iterate on the live prompt, edit that Logfire
variable (paste this file's contents below the comment as the starting
point); no recompile or commit is needed. Keep this file in sync as the
reviewed default.
-->

# Pydantic AI Docs Drift

Documentation lives in `docs/`
(built with `mkdocs`, configured in `mkdocs.yml`), plus `README.md`,
`CONTRIBUTING.md`, and per-package `AGENTS.md` files. Doc code examples are
tested by `tests/test_examples.py`.

## Objective

Detect **negative** documentation drift — code changes that made existing
documentation wrong.

**Noop is the expected outcome most days.** Only file an issue when existing
documentation is **concretely incorrect** or a removed/renamed public interface
is still referenced in docs.

Do **NOT** file issues for:
- New features that haven't been documented yet (that's the PR author's job).
- Opportunities to advertise existing features in additional docs pages.
- Minor wording that could be improved but isn't factually wrong.

### Data Gathering

1. Run `git log --since="7 days ago" --oneline --stat` for a summary of recent
   commits. If there are no commits in the window, call `mcp__safeoutputs__noop` and stop.
2. Inventory documentation: scan `docs/`, `mkdocs.yml`, `README.md`,
   `CONTRIBUTING.md`, and `AGENTS.md` files. Do not assume a fixed structure.

### What to Look For

For each commit (or group of related commits), determine whether the change
made **existing documentation factually wrong**:

1. **Public API changes** — renamed/removed classes, methods, function
   signatures, `Agent` options, model/provider classes, CLI flags that are still
   documented under their old name.
2. **Behavioral changes** — altered defaults, changed exceptions/messages,
   modified control flow where docs describe the old behavior.
3. **Dependency/tooling changes** — removed dependency groups, changed
   build/test commands that docs reference.
4. **Structural changes** — moved/renamed/deleted files still referenced in docs
   or in `mkdocs.yml` nav.
5. **Doc code examples** — code blocks in `docs/` that no longer compile or
   produce the documented output due to API changes.

### How to Analyze

For each potentially impactful change: read the full diff, read the current
docs, check whether docs were already updated in the same or a later commit in
the window, and check whether an open issue/PR already tracks it.

### Deduplication — mandatory before filing

Before filing, first check this sweep's own prior findings with a tight,
server-side label filter — the `/search/issues` endpoint is blocked by the
firewall proxy and there are no `mcp__github__*` tools, but the `?labels=`
filter on the issue-list endpoint is allowed:

```bash
gh api 'repos/pydantic/pydantic-ai/issues?state=open&labels=docs-drift&per_page=100' \
  --jq '.[] | select(.pull_request == null) | {number, title}'
```

Only if that is inconclusive, widen to a full open-issue scan and grep locally
for keywords from your finding:

```bash
gh api --paginate 'repos/pydantic/pydantic-ai/issues?state=open&per_page=100' \
  --jq '.[] | select(.pull_request == null) | {number, title, labels: [.labels[].name]}'
```

If a matching issue exists, call `mcp__safeoutputs__noop`. Do NOT file duplicates.

### What to Skip

- Purely internal refactors with no user-facing impact.
- Changes where docs were already updated in the same/later commit.
- Changes already tracked by an open issue or PR.
- Test-only changes.
- Minor changes where existing docs are still substantially correct.
- **New features without documentation** — these are NOT drift. The PR author
  or a separate docs PR will add them. Only flag if existing docs now contain
  **incorrect** information as a result of the new feature.

### Issue Format

**Title:** Brief summary (e.g., "Update agent.md for new `Agent` output option")

**Body:**

> Recent code changes have introduced documentation drift. The following
> changes need corresponding documentation updates.
>
> ## Changes Requiring Documentation Updates
>
> ### 1. [Brief description]
> **Commit(s):** [SHA(s)]
> **What changed:** [Concise description]
> **Documentation impact:** [Which doc file(s) and what specifically]
>
> ## Suggested Actions
> - [ ] [Specific, actionable checkbox per doc update needed]
