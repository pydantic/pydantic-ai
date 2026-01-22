"""PR Analysis Demo - Shared code for web and evals.

Analyzes PR size vs review rounds correlation using GitHub MCP.
"""

from __future__ import annotations

import os
from datetime import datetime

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.toolsets.code_mode import CodeModeToolset
from pydantic_ai.toolsets.combined import CombinedToolset
from pydantic_ai.toolsets.function import FunctionToolset


# =============================================================================
# Datetime helpers (Monty sandbox doesn't support imports)
# =============================================================================


def datetime_now() -> str:
    """Get current datetime as ISO format string."""
    return datetime.now().isoformat()


def days_between(date1: str, date2: str) -> int:
    """Calculate days between two ISO format date strings.

    Args:
        date1: Start date in ISO format (e.g., '2025-01-10T00:00:00Z')
        date2: End date in ISO format (e.g., '2025-01-15T00:00:00Z')

    Returns:
        Number of days between the two dates (can be negative if date1 > date2).
    """
    dt1 = datetime.fromisoformat(date1.replace('Z', '+00:00'))
    dt2 = datetime.fromisoformat(date2.replace('Z', '+00:00'))
    return (dt2 - dt1).days


def create_datetime_toolset() -> FunctionToolset[None]:
    """Create toolset with datetime helper functions."""
    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(datetime_now, takes_ctx=False)
    toolset.add_function(days_between, takes_ctx=False)
    return toolset

PROMPT = """
Analyze pydantic/pydantic-ai PRs:
- Current date: January 20, 2026 (yes, 2026!)
- Closed PRs from last month, >3 files changed, max 100
- Find: PR size vs review rounds correlation
- Include: PR duration (days from open to close)
- Return: stats + top 10 PRs with most reviews (summarize what/why)
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
    # Combine GitHub MCP with datetime helpers
    datetime_tools = create_datetime_toolset()
    combined: CombinedToolset[None] = CombinedToolset([github, datetime_tools])
    code_toolset: CodeModeToolset[None] = CodeModeToolset(wrapped=combined, max_retries=MAX_RETRIES)
    agent: Agent[None, str] = Agent(
        model,
        toolsets=[code_toolset],
        system_prompt='You are a GitHub PR analyst. Write Python code to analyze PRs efficiently.',
    )
    return agent
