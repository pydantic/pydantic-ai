"""CodeMode Demo: Batch Calendar Event Creation.

This demo shows how code mode reduces LLM roundtrips when creating multiple
calendar events. With traditional tool calling, each event requires a separate
roundtrip. With code mode, the LLM writes a loop that creates all events in one go.

Run:
    uv run python demos/code_mode/batch_operations_demo.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import logfire

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, RetryPromptPart
from pydantic_ai.run import AgentRunResult
from pydantic_ai.runtime.monty import MontyRuntime
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.code_mode import CodeModeToolset

# =============================================================================
# Configuration
# =============================================================================

PROMPT = """
Create calendar events for a daily standup meeting at 9:00 AM for every day
in the first week of January 2025 (January 1-7).

Return a summary with:
- Total events created
- List of all event dates
"""

MODEL = 'gateway/anthropic:claude-sonnet-4-5'
MAX_RETRIES = 3

# =============================================================================
# Mock Calendar Tools
# =============================================================================

# Simulated calendar storage
_calendar_events: list[dict[str, Any]] = []


def create_calendar_event(title: str, date: str, time: str) -> dict[str, Any]:
    """Create a calendar event.

    Args:
        title: The title of the event.
        date: The date of the event in YYYY-MM-DD format.
        time: The time of the event in HH:MM format.

    Returns:
        The created event with its ID.
    """
    event_id = len(_calendar_events) + 1
    event = {'id': event_id, 'title': title, 'date': date, 'time': time}
    _calendar_events.append(event)
    return {'success': True, 'event_id': event_id, 'event': event}


def list_calendar_events() -> list[dict[str, Any]]:
    """List all calendar events.

    Returns:
        List of all calendar events.
    """
    return _calendar_events.copy()


def create_toolset() -> FunctionToolset[None]:
    """Create the calendar toolset."""
    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(create_calendar_event)
    toolset.add_function(list_calendar_events)
    return toolset


# =============================================================================
# Agent Factories
# =============================================================================


def create_traditional_agent(toolset: FunctionToolset[None]) -> Agent[None, str]:
    """Create agent with traditional tool calling."""
    return Agent(
        MODEL,
        toolsets=[toolset],
        system_prompt='You are a calendar assistant. Use the available tools to manage calendar events.',
    )


def create_code_mode_agent(toolset: FunctionToolset[None]) -> Agent[None, str]:
    """Create agent with CodeMode (tools as Python functions)."""
    runtime = MontyRuntime()
    code_toolset: CodeModeToolset[None] = CodeModeToolset(
        wrapped=toolset,
        max_retries=MAX_RETRIES,
        runtime=runtime,
    )
    return Agent(
        MODEL,
        toolsets=[code_toolset],
        system_prompt='You are a calendar assistant. Use the available tools to manage calendar events.',
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


async def run_traditional(toolset: FunctionToolset[None]) -> RunMetrics:
    """Run with traditional tool calling."""
    global _calendar_events
    _calendar_events = []  # Reset calendar

    with logfire.span('traditional_tool_calling'):
        agent = create_traditional_agent(toolset)
        result = await agent.run(PROMPT)
    return extract_metrics(result, 'traditional')


async def run_code_mode(toolset: FunctionToolset[None]) -> RunMetrics:
    """Run with CodeMode tool calling."""
    global _calendar_events
    _calendar_events = []  # Reset calendar

    with logfire.span('code_mode_tool_calling'):
        agent = create_code_mode_agent(toolset)
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
    print(
        f'\n  Output:\n  {metrics.output[:500]}...' if len(metrics.output) > 500 else f'\n  Output:\n  {metrics.output}'
    )


async def main() -> None:
    # Configure Logfire
    logfire.configure(service_name='code-mode-batch-demo')
    logfire.instrument_pydantic_ai()

    print('=' * 70)
    print('CodeMode Demo: Batch Calendar Event Creation')
    print('=' * 70)
    print(f'\nModel: {MODEL}')
    print('Task: Create 7 calendar events (one per day, Jan 1-7)')

    toolset = create_toolset()

    # Run Traditional
    print('\n' + '-' * 70)
    print('Running TRADITIONAL tool calling...')
    print('(Expect 7+ LLM roundtrips - one per event creation)')
    print('-' * 70)

    with logfire.span('demo_traditional'):
        trad = await run_traditional(toolset)
    print_metrics(trad)

    # Run CodeMode
    print('\n' + '-' * 70)
    print('Running CODE MODE tool calling...')
    print('(Expect 1-2 LLM roundtrips - all events created in a loop)')
    print('-' * 70)

    with logfire.span('demo_code_mode'):
        code = await run_code_mode(toolset)
    print_metrics(code)

    # Comparison Summary
    print('\n' + '=' * 70)
    print('COMPARISON SUMMARY')
    print('=' * 70)

    request_reduction = trad.request_count - code.request_count
    if trad.request_count > 0:
        request_pct = request_reduction / trad.request_count * 100
    else:
        request_pct = 0

    token_diff = trad.total_tokens - code.total_tokens
    token_pct = (token_diff / trad.total_tokens * 100) if trad.total_tokens > 0 else 0

    print(
        f'\n  LLM Requests: {trad.request_count} → {code.request_count} ({request_reduction} fewer, {request_pct:.0f}% reduction)'
    )
    print(
        f'  Total Tokens: {trad.total_tokens:,} → {code.total_tokens:,} ({token_pct:+.1f}% {"savings" if token_diff > 0 else "increase"})'
    )

    print('\n' + '=' * 70)
    print('View detailed traces: https://logfire.pydantic.dev')
    print('=' * 70)


if __name__ == '__main__':
    asyncio.run(main())
