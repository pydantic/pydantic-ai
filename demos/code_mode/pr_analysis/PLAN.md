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
| `demos/code_mode/pr_analysis/demo.py` | Web UI demo (traditional vs code mode) |
| `demos/code_mode/pr_analysis/evals.py` | CLI evals with pydantic_evals |

## Demo Code (~50 lines)

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.toolsets.code_mode import CodeModeToolset
import os

# GitHub MCP remote server (no npx needed!)
github = MCPServerStreamableHTTP(
    url='https://api.githubcopilot.com/mcp/',
    headers={
        'Authorization': f'Bearer {os.environ["GITHUB_PERSONAL_ACCESS_TOKEN"]}',
        'X-MCP-Toolsets': 'repos,issues',  # optional: limit toolsets
        'X-MCP-Readonly': 'true',  # read-only mode
    }
)

PROMPT = '''Analyze pydantic/pydantic-ai PRs:
- Closed PRs from last month, >3 files changed, max 100
- Find: PR size vs review rounds correlation
- Include: PR duration (days from open to close)
- Return: stats + top 10 PRs with most reviews (summarize what/why)
'''

# Traditional agent
traditional = Agent('gateway/anthropic:claude-sonnet-4-5', toolsets=[github])

# Code mode agent
code_mode = Agent('gateway/anthropic:claude-sonnet-4-5', toolsets=[CodeModeToolset(wrapped=github, max_retries=5)])

# Web UI comparison
if __name__ == '__main__':
    import uvicorn, threading
    # ... run both on different ports
```

## Evals Structure

```python
# evals.py - IMPORTANT: Update eval name when prompt changes!
EVAL_NAME = 'pr_size_review_rounds_v1'  # <- bump version on prompt changes

MODELS = [
    'gateway/anthropic:claude-sonnet-4-5',
    'gateway/openai:gpt-5.2',
    'gateway/gemini:gemini-3-flash-preview',
]

# 3 models × 3 runs × 2 modes (with/without code mode) = 18 runs
```

**Metrics:**
- Request count (code mode should be lower)
- Total tokens
- Errors (ModelRetry count) - use max_retries=5

Note: Correctness is hard to eval - could compare code generated across runs but skipping for now.

## Implementation Steps

1. Add `max_retries` param to `CodeModeToolset` (currently hardcoded to 3 in code_mode.py:206)
2. Create `demos/code_mode/pr_analysis/` directory
3. Create `demo.py`:
   - MCPServerStreamableHTTP for GitHub (remote, no subprocess)
   - Two agents (traditional, code mode with max_retries=5)
   - Web UI on ports 7934/7935
4. Create `evals.py`:
   - Dataset with single prompt case
   - Run 3 models × 3 runs × 2 modes
   - Track: request count, tokens, error count (ModelRetry)
5. Test with `source .env && uv run python demos/code_mode/pr_analysis/demo.py`

## Verification

1. Run demo: `source .env && uv run python demos/code_mode/pr_analysis/demo.py`
2. Open both URLs, paste same prompt, compare Logfire traces
3. Run evals: `source .env && uv run python demos/code_mode/pr_analysis/evals.py`
4. Check request counts in output

## Notes

- GitHub MCP remote server: `https://api.githubcopilot.com/mcp/` (no npx needed)
- Auth: Bearer token via `Authorization` header
- Env var: `GITHUB_PERSONAL_ACCESS_TOKEN`
- Optional headers: `X-MCP-Toolsets`, `X-MCP-Tools`, `X-MCP-Readonly`
- Code mode doesn't support `await` - all tool calls are sync in generated code
- CodeModeToolset wraps any Toolset via `wrapped=` parameter

Sources: [remote-server.md](https://github.com/github/github-mcp-server/blob/main/docs/remote-server.md)
