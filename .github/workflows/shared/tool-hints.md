---
# Shared tool-calling and sandbox environment hints.
# gh-aw imports this file; the markdown below (after the closing ---) is
# appended to the agent's task prompt at runtime via {{#runtime-import}}.
# Read-only Task delegates retain their own tool guidance.
---

## Sandbox environment

**CLAI 2 tools** -- use Coder's `read_file`, `grep`, `list_files`, and `shell`.
CLAI may expose these through `run_code`; follow that tool's instructions.
References to `Read`, `Grep`, `Glob`, `LS`, and `Bash` in task prompts refer to
those corresponding Coder tools. `Task` remains a read-only delegate tool.

**Safe outputs** -- invoke the `safeoutputs` executable through `shell`.
For a prompt referring to `mcp__safeoutputs__TOOL`, run `safeoutputs TOOL` with
one JSON object argument. For example, `safeoutputs noop '{"message":"No findings"}'`.
Use `safeoutputs --help` and the tool's help to inspect the allowed schema.
Keep the workflow's output limits and required review verdict unchanged.

**Parallel tool calls** — issue independent reads, searches, or lookups in the
same response and they execute concurrently. Only chain sequentially when one
call genuinely needs a previous call's result.

**File reading** — read files in large ranges (500+ lines per call). Most Python
source files fit in one or two calls. Avoid reading 30–80 lines at a time.

**Search tools** — use Coder's `grep` and `list_files` tools for codebase search.
`rg` and `uv` are also available as plain commands via `shell`.

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
use `$GITHUB_WORKSPACE/.review-context/`; the stale-issues workflow uses
`.github/.issue-context/open-issues.tsv` and `.github/.issue-context/issues/`.
If required prefetched context is missing or unreadable, call
`mcp__safeoutputs__noop` and report that missing data instead of attempting a
list request through `gh-proxy`.
