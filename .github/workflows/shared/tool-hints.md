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

**GitHub issue search** — `gh issue list --search` returns HTTP 403 via the AWF
firewall proxy. Use MCP tools instead:
`mcp__github__search_issues(query="repo:pydantic/pydantic-ai <keywords>")`
