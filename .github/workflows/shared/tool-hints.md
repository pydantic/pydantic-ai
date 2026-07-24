---
# Shared tool-calling and sandbox environment hints.
# gh-aw imports this file; the markdown below (after the closing ---) is
# appended to the agent's task prompt at runtime via {{#runtime-import}}.
# Update INSTRUCTIONS in pydantic_ai_gh_aw_shim/cli.py to match.
---

## Sandbox environment

**Parallel tool calls** — issue independent reads, searches, or lookups in the
same response and they execute concurrently. Only chain sequentially when one
call genuinely needs a previous call's result.

**File reading** — read files in large ranges (500+ lines per call). Most Python
source files fit in one or two calls. Avoid reading 30–80 lines at a time.

**Search tools** — use the native `Grep` and `Glob` tools for codebase search.
`rg` and `uv` are also available as plain commands via `Bash`.

**Dev environment** — the repo is checked out at `$GITHUB_WORKSPACE`. Dev
dependencies are **not** pre-installed; run `make install` once before using
`pytest`, `ruff`, or `pyright`. Prefer `uv run pytest <test_file>` over a bare
`pytest` call.

**GitHub issue and PR search** — use the context prefetched for this workflow
instead of enumerating GitHub through the proxied `gh` CLI; list/search
requests from inside the sandbox are blocked or can stall until the workflow
times out. Issue-filing sweeps provide these files:

```bash
jq '.[] | {number, title, labels: [.labels[].name], url}' \
  /tmp/gh-aw/agent/github-context/open-issues.json
jq '.[] | {number, title, labels: [.labels[].name], url}' \
  /tmp/gh-aw/agent/github-context/open-pull-requests.json
```

For a dedicated issue label, filter the local corpus:

```bash
jq '.[] | select(any(.labels[]; .name == "<label>")) | {number, title, url}' \
  /tmp/gh-aw/agent/github-context/open-issues.json
```

Do **not** run `gh issue list`, `gh pr list`, `gh search`, or a paginated/list
`gh api` request from inside the agent. Narrow per-item reads may still be used
after the local corpus identifies a specific issue or PR. PR reviewers instead
use `/tmp/gh-aw/.review-context/`; the stale-issues workflow uses
`/tmp/gh-aw/agent/open-issues.tsv` and `/tmp/gh-aw/agent/issues/`.
If required prefetched context is missing or unreadable, call
`mcp__safeoutputs__noop` and report that missing data instead of attempting a
list request through `gh-proxy`.
