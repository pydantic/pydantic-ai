"""Code Execution Example: GitHub PR Velocity Analysis (Real API).

Analyzes the last 30 merged pull requests in a real GitHub repository using
the REST API — no mock data.

For each PR the agent fetches reviews and changed files, creating a dependent
fan-out of ~60 API calls that traditional tool calling must spread across
many LLM roundtrips while code execution handles in a single loop.

    Level 1:  List closed PRs, filter merged     →  1-2 API calls (paginated)
    Level 2:  Per PR, fetch reviews + files       → 60 API calls (30 × 2)
                                                    ≈ 62 total

Traditional parallel tool calling (best case):
    Roundtrip 1:  list PRs                        → 30 PR objects enter context
    Roundtrip 2:  30× get_pr_reviews in parallel  → 30 review payloads enter context
    Roundtrip 3:  30× get_pr_files in parallel    → 30 file-list payloads enter context
    Roundtrip 4:  model mentally aggregates ~60 JSON blobs into a report
    = 4 roundtrips, ~60 JSON payloads in context (~50-100k tokens of raw API data)

Code execution:
    Roundtrip 1:  model writes a loop that paginates, fans out, computes inline
    Roundtrip 2:  model formats the ~300 token summary
    = 2 roundtrips, only the final summary enters context

Requires:
    GITHUB_PERSONAL_ACCESS_TOKEN environment variable.

Run:
    uv run -m pydantic_ai_examples.code_execution.github_pr_analysis
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
import logfire

from pydantic_ai import Agent
from pydantic_ai.environments.monty import MontyEnvironment
from pydantic_ai.messages import ModelResponse, RetryPromptPart
from pydantic_ai.run import AgentRunResult
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.code_execution import CodeExecutionToolset

# =============================================================================
# Configuration
# =============================================================================

REPO = os.environ.get('GITHUB_REPO', 'pydantic/pydantic-ai')
TARGET_PRS = 30
MODEL = 'gateway/anthropic:claude-sonnet-4-5'
MAX_RETRIES = 5

PROMPT = f"""\
Analyze the last {TARGET_PRS} merged pull requests in "{REPO}".

Use list_pull_requests to fetch closed PRs (not all closed PRs are merged — \
filter for entries where merged_at is not null). Paginate if the first page \
doesn't yield {TARGET_PRS} merged PRs.

For each merged PR, fetch its reviews and changed files. Then produce a report:

1. Average and median hours_to_merge across all analyzed PRs
2. The 5 slowest PRs to merge — title, author, hours_to_merge
3. Top 5 reviewers by total number of reviews submitted
4. Top 10 most frequently modified file paths across all PRs
5. Average number of files changed per PR
6. PR size distribution: small (<5 files), medium (5-20), large (>20)
7. Review coverage: percentage of PRs that received at least one review\
"""

SYSTEM_PROMPT = (
    'You are a software engineering analyst. Use the available tools to query '
    'the GitHub API and produce data-driven reports. '
    'Each PR object includes an hours_to_merge field pre-computed from timestamps.'
)

# =============================================================================
# GitHub API Client
# =============================================================================

_client: httpx.Client | None = None


def _github_client() -> httpx.Client:
    global _client
    if _client is None:
        token = os.environ.get('GITHUB_PERSONAL_ACCESS_TOKEN')
        if not token:
            raise RuntimeError(
                'GITHUB_PERSONAL_ACCESS_TOKEN not set. '
                'Create a token at https://github.com/settings/tokens'
            )
        _client = httpx.Client(
            base_url='https://api.github.com',
            headers={
                'Authorization': f'token {token}',
                'Accept': 'application/vnd.github+json',
                'X-GitHub-Api-Version': '2022-11-28',
            },
            timeout=30.0,
        )
    return _client


def _parse_ts(ts: str) -> datetime:
    """Parse a GitHub API timestamp like '2025-01-15T14:22:00Z'."""
    return datetime.fromisoformat(ts.replace('Z', '+00:00'))


# =============================================================================
# Tool Call Tracking
# =============================================================================

_tool_calls: list[str] = []


# =============================================================================
# Tool Functions
# =============================================================================


def list_pull_requests(
    repo: str,
    state: str = 'closed',
    sort: str = 'updated',
    direction: str = 'desc',
    per_page: int = 30,
    page: int = 1,
) -> dict[str, Any]:
    """List pull requests for a GitHub repository.

    Args:
        repo: Repository in "owner/repo" format (e.g. "pydantic/pydantic-ai").
        state: Filter by state: "open", "closed", or "all".
        sort: Sort by: "created", "updated", or "popularity".
        direction: Sort direction: "asc" or "desc".
        per_page: Number of results per page (max 100).
        page: Page number (1-based).

    Returns:
        Pull request objects. Merged PRs include an hours_to_merge field.
    """
    _tool_calls.append('list_pull_requests')
    resp = _github_client().get(
        f'/repos/{repo}/pulls',
        params={
            'state': state,
            'sort': sort,
            'direction': direction,
            'per_page': per_page,
            'page': page,
        },
    )
    resp.raise_for_status()
    prs = resp.json()
    for pr in prs:
        if pr.get('merged_at') and pr.get('created_at'):
            delta = _parse_ts(pr['merged_at']) - _parse_ts(pr['created_at'])
            pr['hours_to_merge'] = round(delta.total_seconds() / 3600, 1)
    return {
        'page': page,
        'per_page': per_page,
        'count': len(prs),
        'pull_requests': prs,
    }


def get_pr_reviews(repo: str, pr_number: int) -> dict[str, Any]:
    """Get all reviews for a pull request.

    Args:
        repo: Repository in "owner/repo" format.
        pr_number: The pull request number.

    Returns:
        Review objects with reviewer login, state, and submitted_at timestamp.
    """
    _tool_calls.append('get_pr_reviews')
    resp = _github_client().get(f'/repos/{repo}/pulls/{pr_number}/reviews')
    resp.raise_for_status()
    return {'pr_number': pr_number, 'reviews': resp.json()}


def get_pr_files(repo: str, pr_number: int) -> dict[str, Any]:
    """Get files changed in a pull request.

    Args:
        repo: Repository in "owner/repo" format.
        pr_number: The pull request number.

    Returns:
        Changed file objects with filename, status, additions, and deletions.
    """
    _tool_calls.append('get_pr_files')
    resp = _github_client().get(
        f'/repos/{repo}/pulls/{pr_number}/files',
        params={'per_page': 100},
    )
    resp.raise_for_status()
    files = resp.json()
    # Drop patch diffs — they can be enormous and aren't needed for the analysis.
    for f in files:
        f.pop('patch', None)
    return {'pr_number': pr_number, 'files': files}


# =============================================================================
# Toolset & Agent Factories
# =============================================================================


def create_toolset() -> FunctionToolset[None]:
    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(list_pull_requests)
    toolset.add_function(get_pr_reviews)
    toolset.add_function(get_pr_files)
    return toolset


def create_tool_calling_agent(toolset: FunctionToolset[None]) -> Agent[None, str]:
    return Agent(MODEL, toolsets=[toolset], system_prompt=SYSTEM_PROMPT)


def create_code_execution_agent(toolset: FunctionToolset[None]) -> Agent[None, str]:
    code_toolset: CodeExecutionToolset[None] = CodeExecutionToolset(
        MontyEnvironment(),
        toolset=toolset,
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
    tool_calls: int
    output: str

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def extract_metrics(
    result: AgentRunResult[str], mode: str, tool_calls: int
) -> RunMetrics:
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
        tool_calls=tool_calls,
        output=result.output,
    )


# =============================================================================
# Run Functions
# =============================================================================


async def run_tool_calling(toolset: FunctionToolset[None]) -> RunMetrics | None:
    """Run with standard tool calling. Returns None if the context overflows."""
    _tool_calls.clear()
    try:
        with logfire.span('tool_calling'):
            agent = create_tool_calling_agent(toolset)
            result = await agent.run(PROMPT)
        return extract_metrics(result, 'tool_calling', len(_tool_calls))
    except Exception as e:
        error_str = str(e)
        if 'too long' in error_str or 'too many tokens' in error_str.lower():
            logfire.error(
                'tool_calling failed: context window overflow after {tool_calls} API calls',
                tool_calls=len(_tool_calls),
                error=error_str,
            )
            return None
        raise


async def run_code_execution(toolset: FunctionToolset[None]) -> RunMetrics:
    _tool_calls.clear()
    with logfire.span('code_execution'):
        agent = create_code_execution_agent(toolset)
        code_toolset = agent.toolsets[0]
        async with code_toolset:
            result = await agent.run(PROMPT)
    return extract_metrics(result, 'code_execution', len(_tool_calls))


# =============================================================================
# Output
# =============================================================================


def log_metrics(metrics: RunMetrics) -> None:
    logfire.info(
        '{mode}: {requests} requests, {tokens} tokens, {tool_calls} API calls',
        mode=metrics.mode,
        requests=metrics.request_count,
        tokens=metrics.total_tokens,
        input_tokens=metrics.input_tokens,
        output_tokens=metrics.output_tokens,
        tool_calls=metrics.tool_calls,
        retries=metrics.retry_count,
    )


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    logfire.configure(service_name='code-execution-github-pr-analysis')
    logfire.instrument_pydantic_ai()

    toolset = create_toolset()

    with logfire.span('demo_tool_calling'):
        trad = await run_tool_calling(toolset)
    if trad:
        log_metrics(trad)

    with logfire.span('demo_code_execution'):
        code = await run_code_execution(toolset)
    log_metrics(code)


if __name__ == '__main__':
    asyncio.run(main())
