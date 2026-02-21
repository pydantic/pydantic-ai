"""Code Execution Example: Batch Calendar Event Creation.

This example shows how code execution reduces LLM roundtrips when creating multiple
calendar events. With traditional tool calling, each event requires a separate
roundtrip. With code execution, the LLM writes a loop that creates all events in one go.

Run:
    uv run -m pydantic_ai_examples.code_execution.batch_operations
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

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


def create_tool_calling_agent(toolset: FunctionToolset[None]) -> Agent[None, str]:
    """Create agent with standard tool calling."""
    return Agent(
        MODEL,
        toolsets=[toolset],
        system_prompt='You are a calendar assistant. Use the available tools to manage calendar events.',
    )


def create_code_execution_agent(toolset: FunctionToolset[None]) -> Agent[None, str]:
    """Create agent with code execution (tools as Python functions)."""
    environment = MontyEnvironment()
    code_toolset: CodeExecutionToolset[None] = CodeExecutionToolset(
        environment,
        toolset=toolset,
        max_retries=MAX_RETRIES,
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


async def run_tool_calling(toolset: FunctionToolset[None]) -> RunMetrics:
    """Run with standard tool calling."""
    global _calendar_events
    _calendar_events = []  # Reset calendar

    with logfire.span('tool_calling'):
        agent = create_tool_calling_agent(toolset)
        result = await agent.run(PROMPT)
    return extract_metrics(result, 'tool_calling')


async def run_code_execution(toolset: FunctionToolset[None]) -> RunMetrics:
    """Run with code execution tool calling."""
    global _calendar_events
    _calendar_events = []  # Reset calendar

    with logfire.span('code_execution_tool_calling'):
        agent = create_code_execution_agent(toolset)
        code_toolset = agent.toolsets[0]
        async with code_toolset:
            result = await agent.run(PROMPT)
    return extract_metrics(result, 'code_execution')


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
    logfire.configure(service_name='code-execution-batch-demo')
    logfire.instrument_pydantic_ai()

    toolset = create_toolset()

    with logfire.span('demo_tool_calling'):
        trad = await run_tool_calling(toolset)
    log_metrics(trad)

    with logfire.span('demo_code_execution'):
        code = await run_code_execution(toolset)
    log_metrics(code)

    print('View traces: https://logfire.pydantic.dev')


if __name__ == '__main__':
    asyncio.run(main())
