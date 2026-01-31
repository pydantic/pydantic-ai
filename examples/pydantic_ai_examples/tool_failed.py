"""Example demonstrating ToolFailed exception for graceful tool failure handling.

ToolFailed allows tools to fail and be marked as errors in telemetry, while allowing
the agent to continue execution. This is particularly useful for parallel tool execution
where partial failures are acceptable.

Key differences from ModelRetry:
- ToolFailed: Marks failure as an error in telemetry
- ModelRetry: Not an error, expected retry behavior

Run with:

    uv run -m pydantic_ai_examples.tool_failed
"""

from __future__ import annotations as _annotations

import asyncio
from dataclasses import dataclass

import logfire

from pydantic_ai import Agent, RunContext, ToolFailed

logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()


@dataclass
class DataSources:
    """Mock data sources with various failure states."""

    primary_db_available: bool = True
    cache_available: bool = True
    api_has_credentials: bool = False  # Simulates missing API credentials


agent = Agent(
    'openai:gpt-4o-mini',
    instructions=(
        'You are a data aggregation assistant. Query all available data sources. '
        'If some fail, continue with the successful ones and report what you found.'
    ),
    deps_type=DataSources,
    retries=1,
)


@agent.tool
async def query_primary_database(ctx: RunContext[DataSources]) -> str:
    """Query the primary database for user data."""
    if not ctx.deps.primary_db_available:
        # Temporary failure - database might recover
        # Tool stays available in case model wants to retry later
        raise ToolFailed('Primary database connection timeout', disable=False)

    return 'Primary DB: 1,234 active users, 5,678 total transactions'


@agent.tool
async def query_cache(ctx: RunContext[DataSources]) -> str:
    """Query the cache for recent statistics."""
    if not ctx.deps.cache_available:
        raise ToolFailed('Cache unavailable', disable=False)

    return 'Cache: Last hour - 42 new signups, 156 active sessions'


@agent.tool
async def query_external_api(ctx: RunContext[DataSources]) -> str:
    """Query external analytics API for detailed metrics."""
    if not ctx.deps.api_has_credentials:
        # Permanent failure - missing credentials won't fix themselves
        # Disable this tool so model knows not to try again this run
        raise ToolFailed(
            'External API authentication failed - missing credentials', disable=True
        )

    return 'External API: Conversion rate 3.2%, average session 4m 23s'


@agent.tool
async def query_backup_database(ctx: RunContext[DataSources]) -> str:
    """Query the backup database as fallback."""
    # This one always works
    return 'Backup DB: System healthy, last backup 2 hours ago'


async def main():
    """Demonstrate ToolFailed with parallel tool execution."""
    print('=== Scenario 1: All sources available ===')
    sources = DataSources(
        primary_db_available=True, cache_available=True, api_has_credentials=True
    )
    result = await agent.run(
        'Get me a summary of all available data from our systems', deps=sources
    )
    print(f'Result: {result.output}\n')

    print('=== Scenario 2: API has permanent failure (no credentials) ===')
    sources = DataSources(
        primary_db_available=True, cache_available=True, api_has_credentials=False
    )
    result = await agent.run(
        'Get me a summary of all available data from our systems', deps=sources
    )
    print(f'Result: {result.output}')
    print('(Note: External API tool was disabled after first failure)\n')

    print('=== Scenario 3: Multiple temporary failures ===')
    sources = DataSources(
        primary_db_available=False, cache_available=False, api_has_credentials=False
    )
    result = await agent.run(
        'Get me a summary of all available data from our systems', deps=sources
    )
    print(f'Result: {result.output}')
    print(
        '(Note: Only backup database succeeded, but failures are logged in telemetry)\n'
    )


if __name__ == '__main__':
    asyncio.run(main())
