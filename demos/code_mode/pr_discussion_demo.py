from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import logfire

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.messages import ModelResponse, RetryPromptPart
from pydantic_ai.run import AgentRunResult
from pydantic_ai.runtime.monty import MontyRuntime
from pydantic_ai.toolsets.code_mode import CodeModeToolset

# =============================================================================
# Configuration
# =============================================================================

PROMPT = """
Analyze the most recent 5 closed PRs in pydantic/pydantic.

Rules:
- Use list PRs with state=closed, sort=updated, direction=desc
- Use per_page=5 and page=1..1
- Only use GitHub MCP tools

For each PR:
- Fetch PR files (per_page=100, page=1..1)
- Fetch PR reviews (per_page=100, page=1..1)
- Fetch issue comments on the PR (per_page=100, page=1..1)

Compute a discussion intensity score:
score = (review_count + issue_comment_count + 1) * file_count

Return ONLY:
- Average file_count per PR
- Average review_count per PR
- Average issue_comment_count per PR
- The single PR with the highest score (number, title, file_count, review_count, issue_comment_count, score)
"""

MODEL = 'gateway/anthropic:claude-sonnet-4-5'
MAX_RETRIES = 5


# =============================================================================
# GitHub MCP Setup
# =============================================================================


def create_github_mcp() -> MCPServerStreamableHTTP:
    """Create GitHub MCP server connection."""
    token = os.environ.get('GITHUB_PERSONAL_ACCESS_TOKEN')
    if not token:
        raise ValueError('GITHUB_PERSONAL_ACCESS_TOKEN environment variable required')

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


def create_traditional_agent(github: MCPServerStreamableHTTP) -> Agent[None, str]:
    """Create agent with traditional tool calling."""
    return Agent(
        MODEL,
        toolsets=[github],
        system_prompt='You are a GitHub PR analyst. Use the available tools to analyze PRs.',
    )


def create_code_mode_agent(github: MCPServerStreamableHTTP) -> Agent[None, str]:
    """Create agent with CodeMode (tools as Python functions)."""
    runtime = MontyRuntime()
    code_toolset: CodeModeToolset[None] = CodeModeToolset(
        wrapped=github,
        max_retries=MAX_RETRIES,
        runtime=runtime,
    )
    return Agent(
        MODEL,
        toolsets=[code_toolset],
        system_prompt='You are a GitHub PR analyst. Use the available tools to analyze PRs.',
    )


# =============================================================================
# Metrics Collection
# =============================================================================


@dataclass
class RunMetrics:
    """Metrics collected from an agent run."""

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
    """Extract metrics from agent result."""
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


# =============================================================================
# Run Functions
# =============================================================================


async def run_traditional(github: MCPServerStreamableHTTP) -> RunMetrics:
    """Run with traditional tool calling."""
    with logfire.span('traditional_tool_calling'):
        agent = create_traditional_agent(github)
        result = await agent.run(PROMPT)
    return extract_metrics(result, 'traditional')


async def run_code_mode(github: MCPServerStreamableHTTP) -> RunMetrics:
    """Run with CodeMode tool calling."""
    with logfire.span('code_mode_tool_calling'):
        agent = create_code_mode_agent(github)
        code_toolset = agent.toolsets[0]
        async with code_toolset:
            result = await agent.run(PROMPT)
    return extract_metrics(result, 'code_mode')


# =============================================================================
# Main Demo
# =============================================================================


def print_metrics(metrics: RunMetrics) -> None:
    """Print metrics in formatted table."""
    print(f'  LLM Requests:  {metrics.request_count}')
    print(f'  Input Tokens:  {metrics.input_tokens:,}')
    print(f'  Output Tokens: {metrics.output_tokens:,}')
    print(f'  Total Tokens:  {metrics.total_tokens:,}')
    print(f'  Retries:       {metrics.retry_count}')
    print(f'\n  Output (truncated):\n  {metrics.output[:400]}...')


async def main() -> None:
    # Configure Logfire
    logfire.configure(service_name='code-mode-demo')
    logfire.instrument_pydantic_ai()

    print('=' * 70)
    print('CodeMode Demo: GitHub PR Discussion Intensity Analysis')
    print('=' * 70)
    print(f'\nModel: {MODEL}')
    print('Task: Analyze 5 closed PRs from pydantic/pydantic')

    github = create_github_mcp()

    async with github:
        # Run Traditional
        print('\n' + '-' * 70)
        print('Running TRADITIONAL tool calling...')
        print('(Expect 20+ LLM roundtrips - one per tool call)')
        print('-' * 70)

        with logfire.span('demo_traditional'):
            trad = await run_traditional(github)
        print_metrics(trad)

        # Run CodeMode
        print('\n' + '-' * 70)
        print('Running CODE MODE tool calling...')
        print('(Expect 2-3 LLM roundtrips - all tool calls in generated code)')
        print('-' * 70)

        with logfire.span('demo_code_mode'):
            code = await run_code_mode(github)
        print_metrics(code)

    # Comparison Summary
    print('\n' + '=' * 70)
    print('COMPARISON SUMMARY')
    print('=' * 70)

    request_reduction = trad.request_count - code.request_count
    token_diff = trad.total_tokens - code.total_tokens
    token_pct = (token_diff / trad.total_tokens * 100) if trad.total_tokens > 0 else 0

    print(
        f'\n  LLM Requests: {trad.request_count} → {code.request_count} '
        f'({request_reduction} fewer, {request_reduction / trad.request_count * 100:.0f}% reduction)'
    )
    print(
        f'  Total Tokens: {trad.total_tokens:,} → {code.total_tokens:,} '
        f'({token_pct:+.1f}% {"savings" if token_diff > 0 else "increase"})'
    )

    print('\n' + '=' * 70)
    print('View detailed traces: https://logfire.pydantic.dev')
    print('=' * 70)


if __name__ == '__main__':
    asyncio.run(main())
