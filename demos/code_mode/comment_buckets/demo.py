"""Discussion Intensity Demo - Shared code for script.

Analyzes the most recent 5 closed PRs in pydantic/pydantic and scores
discussion intensity using GitHub MCP.
"""

from __future__ import annotations

import os

from pydantic_ai import Agent
from pydantic_ai.models import ModelSettings
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.toolsets.code_mode import CodeModeToolset

# Demo intent: nested PR -> files/reviews/comments fan-out; code mode does it in one run.
PROMPT = """
Analyze the most recent 5 closed PRs in pydantic/pydantic.

Rules:
- Use list PRs with state=closed, sort=updated, direction=desc
- Use per_page=5 and page=1..1 (no slicing, no cursor/after)
- Do not use datetime, slicing, break, or continue
- Do not use list comprehensions, map, or parallel tool calls
- Only use GitHub MCP tools
- All tool calls must be sequential

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
- Bucket totals: files (tests/docs/other), reviews (approved/changes requested/comment), comments (<=80, 81-200, 201+)
- The single PR with the highest score (number, title, file_count, review_count, issue_comment_count, score)
""".lstrip()

MODELS = [
    'gateway/openai:gpt-5.2',
    'gateway/anthropic:claude-sonnet-4-5',
    'gateway/gemini:gemini-3-flash-preview',
]

DEFAULT_MODEL = 'gateway/openai:gpt-5.2'
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
            'X-MCP-Toolsets': 'issues,pull_requests',
            'X-MCP-Readonly': 'true',
        },
    )


def create_traditional_agent(github: MCPServerStreamableHTTP, model: str = DEFAULT_MODEL) -> Agent[None, str]:
    """Create agent with traditional tool calling."""
    agent: Agent[None, str] = Agent(
        model,
        toolsets=[github],
        model_settings=ModelSettings(parallel_tool_calls=False),
        system_prompt=(
            'You are a GitHub PR analyst. Use the available tools to analyze PRs.'
        ),
    )
    return agent


def create_code_mode_agent(github: MCPServerStreamableHTTP, model: str = DEFAULT_MODEL) -> Agent[None, str]:
    """Create agent with code mode (tools as Python functions)."""
    code_toolset: CodeModeToolset[None] = CodeModeToolset(wrapped=github, max_retries=MAX_RETRIES)
    agent: Agent[None, str] = Agent(
        model,
        toolsets=[code_toolset],
        model_settings=ModelSettings(parallel_tool_calls=False),
        system_prompt=(
            'You are a GitHub PR analyst. Use the available tools to analyze PRs.'
        ),
    )
    return agent
