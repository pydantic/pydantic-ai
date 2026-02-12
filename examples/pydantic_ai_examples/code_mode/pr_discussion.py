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


def create_tool_calling_agent(github: MCPServerStreamableHTTP) -> Agent[None, str]:
    """Create agent with standard tool calling."""
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


async def run_tool_calling(github: MCPServerStreamableHTTP) -> RunMetrics:
    """Run with standard tool calling."""
    with logfire.span('tool_calling'):
        agent = create_tool_calling_agent(github)
        result = await agent.run(PROMPT)
    return extract_metrics(result, 'tool_calling')


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


def log_metrics(metrics: RunMetrics) -> None:
    """Log metrics to logfire."""
    logfire.info(
        '{mode} completed: {requests} requests, {tokens} tokens',
        mode=metrics.mode,
        requests=metrics.request_count,
        tokens=metrics.total_tokens,
        input_tokens=metrics.input_tokens,
        output_tokens=metrics.output_tokens,
        retries=metrics.retry_count,
    )


async def main() -> None:
    logfire.configure(service_name='code-mode-demo')
    logfire.instrument_pydantic_ai()

    github = create_github_mcp()

    async with github:
        with logfire.span('demo_tool_calling'):
            trad = await run_tool_calling(github)
        log_metrics(trad)

        with logfire.span('demo_code_mode'):
            code = await run_code_mode(github)
        log_metrics(code)

    print('View traces: https://logfire.pydantic.dev')


if __name__ == '__main__':
    asyncio.run(main())
