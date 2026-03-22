"""Code Execution Example: Multi-Repo PR Deep Analysis via GitHub MCP.

Demonstrates code execution's advantage over traditional tool calling --
even when the model makes parallel tool calls.

The task requires a dependent fan-out:

    Level 1 (N=3):  List closed PRs for each repo         →  3 calls
    Level 2 (N*M*Z = 3*5*4 = 60):  Per PR, fetch files,   → 60 calls
                    reviews, review comments, issue comments
                                                           --------
                                                           63 total

Traditional parallel tool calling (best case):
    Roundtrip 1:  3 list-PRs calls fire in parallel → 3 JSON results enter context
    Roundtrip 2:  Model sees PR numbers, fires 60 detail calls in parallel
                  → 60 JSON results enter context (files, reviews, comments for
                    15 PRs -- easily 100k+ tokens of intermediate data)
    Roundtrip 3:  Model reads ALL 63 results and tries to aggregate mentally
    = 3 roundtrips, 63 tool results in context, ~100-200k tokens of raw JSON

Code execution:
    Roundtrip 1:  Model writes async code with nested asyncio.gather, all 63
                  API calls happen inside the sandbox, data is aggregated with
                  deterministic Python, only the final summary string is returned
    Roundtrip 2:  Model formats the summary as text
    = 2 roundtrips, ~1k tokens of intermediate data

The wins stack:
    - Context: 63 JSON payloads in context (traditional) vs ~1k token summary (code exec)
    - Cost: 100-200k tokens of I/O (traditional) vs ~5k tokens total (code exec)
    - Accuracy: Deterministic code aggregation vs model "mental math" over 60 JSON blobs
    - Latency: 3 serial roundtrips (traditional) vs 2 (code exec)

Requires:
    GITHUB_PERSONAL_ACCESS_TOKEN environment variable.

Run:
    uv run -m pydantic_ai_examples.code_execution.pr_comment_buckets
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import logfire

from pydantic_ai import Agent
from pydantic_ai.environments.monty import MontyEnvironment
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.messages import ModelResponse, RetryPromptPart
from pydantic_ai.run import AgentRunResult
from pydantic_ai.toolsets.code_execution import CodeExecutionToolset

# =============================================================================
# Configuration
# =============================================================================

REPOS = ['pydantic/pydantic', 'pydantic/pydantic-ai', 'pydantic/logfire']
PRS_PER_REPO = 5

# The prompt is designed to require cross-repo aggregation that's painful
# for a model to do "in its head" from 60+ JSON payloads, but trivial in code.
PROMPT = """\
Analyze the {prs_per_repo} most recent closed PRs in EACH of these repositories:
{repos}

For EACH PR across all repos, fetch ALL of the following:
1. Files changed (per_page=100, page=1)
2. Reviews (per_page=100, page=1)
3. Review comments -- the line-level code comments (per_page=100, page=1)
4. Issue comments -- the general discussion thread (per_page=100, page=1)

That's {total_detail_calls} detail API calls across {total_prs} PRs. Use asyncio.gather
aggressively -- fan out repo fetches, then fan out all detail fetches for all PRs at once.

From the collected data, compute and return ONLY these metrics (no raw data):

PER-REPO BREAKDOWN:
- Repo name
- Avg files changed per PR
- Avg reviews per PR
- Avg review comments (line-level) per PR
- Avg issue comments (discussion) per PR
- Total engagement score: sum of (reviews + review_comments + issue_comments) across all PRs
- File categories: tests (path contains "test") / docs (path contains "doc" or ends .md) / source (everything else)
- Review verdicts: approved / changes_requested / commented / dismissed

CROSS-REPO COMPARISON:
- Review engagement ratio per repo: total_engagement / total_files_changed
- Rank repos by engagement ratio (highest = most discussion per line of code changed)

HOT FILES (appear in 2+ PRs across ANY repo):
- List file paths that were modified in multiple PRs, with the count

HOTTEST PR:
- The single PR with the highest score = (reviews + review_comments + issue_comments + 1) * files_changed
- Include: repo, PR number, title, and the score breakdown

TOP 5 MOST-DISCUSSED PRs:
- Ranked by (review_comments + issue_comments), with repo, PR number, title, and counts

REVIEWER LEADERBOARD:
- Top 5 reviewers by total reviews given across all repos, with per-repo breakdown
"""

MODEL = 'gateway/anthropic:claude-sonnet-4-5'
MAX_RETRIES = 5

SYSTEM_PROMPT = (
    'You are a GitHub analyst. Use the available tools to fetch data and compute metrics. '
    'Do ALL data fetching and aggregation inside run_code -- return only the final summary as text in your response, '
    'not as code output. The point is to avoid polluting your context window with raw API data.'
)

# =============================================================================
# GitHub MCP
# =============================================================================


def create_github_mcp() -> MCPServerStreamableHTTP:
    """Create GitHub MCP server connection."""
    token = os.environ.get('GITHUB_PERSONAL_ACCESS_TOKEN')
    if not token:
        raise ValueError('GITHUB_PERSONAL_ACCESS_TOKEN environment variable is required')

    return MCPServerStreamableHTTP(
        url='https://api.githubcopilot.com/mcp/',
        timeout=30,
        headers={
            'Authorization': f'Bearer {token}',
            'X-MCP-Toolsets': 'issues,pull_requests',
            'X-MCP-Readonly': 'true',
        },
    )


# =============================================================================
# Agent Factories
# =============================================================================


def create_tool_calling_agent(github: MCPServerStreamableHTTP) -> Agent[None, str]:
    """Create agent with standard parallel tool calling.

    Even with parallel calls, the model needs:
      Roundtrip 1: list PRs per repo (3 calls)
      Roundtrip 2: fetch details per PR (60 calls) -- results ALL enter context
      Roundtrip 3: produce analysis from 63 JSON blobs in context
    """
    return Agent(MODEL, toolsets=[github], system_prompt=SYSTEM_PROMPT)


def create_code_execution_agent(github: MCPServerStreamableHTTP) -> Agent[None, str]:
    """Create agent with code execution.

    The model writes a single code block that:
      - Fans out all 63 API calls with asyncio.gather
      - Aggregates results in-memory with deterministic Python
      - Returns only the summary string
    Only ~1k tokens of tool I/O enter the context.
    """
    code_toolset: CodeExecutionToolset[None] = CodeExecutionToolset(
        MontyEnvironment(),
        toolset=github,
        max_retries=MAX_RETRIES,
    )
    return Agent(MODEL, toolsets=[code_toolset], system_prompt=SYSTEM_PROMPT)


# =============================================================================
# Metrics
# =============================================================================


@dataclass
class RunMetrics:
    mode: str
    request_count: int
    input_tokens: int
    output_tokens: int
    retry_count: int
    output: str

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def extract_metrics(result: AgentRunResult[str], mode: str) -> RunMetrics:
    request_count = 0
    input_tokens = 0
    output_tokens = 0
    retry_count = 0

    for msg in result.all_messages():
        if isinstance(msg, ModelResponse):
            request_count += 1
            if msg.usage:
                input_tokens += msg.usage.input_tokens or 0
                output_tokens += msg.usage.output_tokens or 0
        for part in getattr(msg, 'parts', []):
            if isinstance(part, RetryPromptPart):
                retry_count += 1

    return RunMetrics(
        mode=mode,
        request_count=request_count,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        retry_count=retry_count,
        output=result.output,
    )


def print_metrics(metrics: RunMetrics) -> None:
    print(f'\n{"=" * 70}')
    print(f'  {metrics.mode.upper()}')
    print(f'{"=" * 70}')
    print(f'  LLM roundtrips:  {metrics.request_count}')
    print(f'  Input tokens:    {metrics.input_tokens:,}')
    print(f'  Output tokens:   {metrics.output_tokens:,}')
    print(f'  Total tokens:    {metrics.total_tokens:,}')
    print(f'  Retries:         {metrics.retry_count}')
    print(f'{"=" * 70}')
    print(f'\n{metrics.output}\n')


# =============================================================================
# Run
# =============================================================================

TOTAL_PRS = len(REPOS) * PRS_PER_REPO
DETAIL_CALLS_PER_PR = 4  # files, reviews, review comments, issue comments
TOTAL_DETAIL_CALLS = TOTAL_PRS * DETAIL_CALLS_PER_PR
TOTAL_CALLS = len(REPOS) + TOTAL_DETAIL_CALLS

FORMATTED_PROMPT = PROMPT.format(
    prs_per_repo=PRS_PER_REPO,
    repos='\n'.join(f'- {r}' for r in REPOS),
    total_prs=TOTAL_PRS,
    total_detail_calls=TOTAL_DETAIL_CALLS,
)


async def run_tool_calling(github: MCPServerStreamableHTTP) -> RunMetrics:
    """Run with standard parallel tool calling."""
    with logfire.span('tool_calling'):
        agent = create_tool_calling_agent(github)
        result = await agent.run(FORMATTED_PROMPT)
    return extract_metrics(result, 'traditional parallel tool calling')


async def run_code_execution(github: MCPServerStreamableHTTP) -> RunMetrics:
    """Run with code execution."""
    with logfire.span('code_execution'):
        agent = create_code_execution_agent(github)
        code_toolset = agent.toolsets[0]
        async with code_toolset:
            result = await agent.run(FORMATTED_PROMPT)
    return extract_metrics(result, 'code execution')


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    logfire.configure(service_name='code-execution-pr-analysis')
    logfire.instrument_pydantic_ai()

    github = create_github_mcp()

    # Traditional parallel tool calling first
    async with github:
        trad = await run_tool_calling(github)
    print_metrics(trad)

    # Code execution (CodeExecutionToolset.__aenter__ enters the MCP server internally)
    code = await run_code_execution(github)
    print_metrics(code)


if __name__ == '__main__':
    asyncio.run(main())
