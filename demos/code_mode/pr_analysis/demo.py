"""PR Analysis Demo - Shared code for web and evals.

Analyzes open PRs in pydantic/pydantic, calculating size scores based on
changed files (ignoring uv.lock and cassettes, weighting tests at 50%).

This demo shows how code mode reduces LLM round-trips:
- Traditional mode: 20+ API round-trips (list PRs, then get files one at a time)
- Code mode: 2-3 round-trips (generates loop in code, executes all API calls internally)
"""

from __future__ import annotations

import os

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.toolsets.code_mode import CodeModeToolset


PROMPT = """
Analyze all open PRs in pydantic/pydantic:
- For each PR, get the list of changed files
- Calculate size score: count files (ignore uv.lock and cassettes/, weight tests/ at 50%)
- Return: each PR's number, title, and size score
- Also return: total combined size across all open PRs
""".lstrip()

MODELS = [
    'gateway/anthropic:claude-sonnet-4-5',
    'gateway/openai:gpt-5.2',
    'gateway/gemini:gemini-3-flash-preview',
]

DEFAULT_MODEL = 'gateway/anthropic:claude-sonnet-4-5'
MAX_RETRIES = 5


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
            'X-MCP-Toolsets': 'repos,issues',
            'X-MCP-Readonly': 'true',
        },
    )


def create_traditional_agent(github: MCPServerStreamableHTTP, model: str = DEFAULT_MODEL) -> Agent[None, str]:
    """Create agent with traditional tool calling."""
    agent: Agent[None, str] = Agent(
        model,
        toolsets=[github],
        system_prompt='You are a GitHub PR analyst. Use the available tools to analyze PRs.',
    )
    return agent


def create_code_mode_agent(github: MCPServerStreamableHTTP, model: str = DEFAULT_MODEL) -> Agent[None, str]:
    """Create agent with code mode (tools as Python functions)."""
    code_toolset: CodeModeToolset[None] = CodeModeToolset(wrapped=github, max_retries=MAX_RETRIES)
    agent: Agent[None, str] = Agent(
        model,
        toolsets=[code_toolset],
        system_prompt='You are a GitHub PR analyst. Write Python code to analyze PRs efficiently.',
    )
    return agent
