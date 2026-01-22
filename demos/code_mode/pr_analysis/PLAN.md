# Minimal Code Mode Demo: PR Size vs Review Rounds

## Overview

Minimal demo using GitHub MCP directly. Goal: fit on one presentation slide.

## Task

Analyze correlation: **PR size vs review rounds** for pydantic/pydantic-ai

**Constraints:**
- Closed PRs only
- Last month
- >3 files changed
- Max 100 PRs
- Include PR duration (open → closed)

**Output:**
- Stats summary
- Top 10 PRs with most review rounds (what they were about, why iterations took long)

## Files

| File | Purpose |
|------|---------|
| `demo.py` | Shared code: prompt, constants, agent factories, datetime helpers |
| `web.py` | Web UI on ports 7934 (traditional) / 7935 (code mode) |
| `evals.py` | CLI evals: 3 models × 3 runs × 2 modes |

## How to Run

### Prerequisites

```bash
# Set GitHub token in .env
echo 'GITHUB_PERSONAL_ACCESS_TOKEN=ghp_xxx' >> .env
```

### Web UI Demo

```bash
source .env && uv run python -m demos.code_mode.pr_analysis.web
```

Opens two web UIs:
- http://localhost:7934 — Traditional mode (direct tool calls)
- http://localhost:7935 — Code mode (Python code generation)

**What to do:**
1. Open both URLs in separate tabs
2. Paste the prompt (shown in terminal) into both
3. Compare in Logfire: request count, token usage, tool call patterns
4. Code mode should show `run_code` → nested tool calls; traditional shows many individual calls

### CLI Evals

```bash
source .env && uv run python -m demos.code_mode.pr_analysis.evals
```

Runs 18 evaluations (3 models × 3 runs × 2 modes) and prints summary table.

### Prompt to Use

```
Analyze pydantic/pydantic-ai PRs:
- Closed PRs from last month, >3 files changed, max 100
- Find: PR size vs review rounds correlation
- Include: PR duration (days from open to close)
- Return: stats + top 10 PRs with most reviews (summarize what/why)
```

## Code Overview

```python
# demo.py - shared code
github = MCPServerStreamableHTTP(url='https://api.githubcopilot.com/mcp/', ...)
datetime_tools = FunctionToolset()  # datetime_now(), days_between()
combined = CombinedToolset([github, datetime_tools])

# Traditional: direct tool calls
traditional = Agent(model, toolsets=[github])

# Code mode: Python code generation wrapping tools
code_mode = Agent(model, toolsets=[CodeModeToolset(wrapped=combined, max_retries=5)])
```

## Metrics

| Metric | What to Compare |
|--------|-----------------|
| Request count | Code mode should use fewer LLM round-trips |
| Total tokens | Code mode may use more per-request but fewer overall |
| Retries | `ModelRetry` count from syntax/type errors in generated code |

## What Success Looks Like

**Traditional mode:** Many sequential tool calls, each requiring LLM round-trip
```
chat → search_issues → chat → get_pull_request → chat → get_pull_request → ...
```

**Code mode:** Single `run_code` with nested tool calls
```
chat → run_code (search_issues, get_pull_request ×N, days_between ×N) → chat
```

## Notes

- GitHub MCP remote server: `https://api.githubcopilot.com/mcp/` (no npx needed)
- Auth: Bearer token via `Authorization` header
- Env var: `GITHUB_PERSONAL_ACCESS_TOKEN`
- Optional headers: `X-MCP-Toolsets`, `X-MCP-Tools`, `X-MCP-Readonly`
- Code mode doesn't support `await` - all tool calls are sync in generated code
- CodeModeToolset wraps any Toolset via `wrapped=` parameter

Sources: [remote-server.md](https://github.com/github/github-mcp-server/blob/main/docs/remote-server.md)
