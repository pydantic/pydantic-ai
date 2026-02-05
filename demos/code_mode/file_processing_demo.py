"""CodeMode Demo: File Processing with Loops.

This demo shows how code mode handles unknown iteration counts. The LLM doesn't
know upfront how many files exist, but code mode can write a loop that processes
whatever it finds. Traditional tool calling requires multiple roundtrips.

Run:
    uv run python demos/code_mode/file_processing_demo.py
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
List all .txt files in the /data directory, read each one, and create a report.

For each file, include:
- Filename
- Character count
- Word count
- First 50 characters of content

Return a summary with:
- Total files found
- Total characters across all files
- List of file details
"""

MODEL = 'gateway/anthropic:claude-sonnet-4-5'
MAX_RETRIES = 3

# =============================================================================
# Mock File System
# =============================================================================

# Simulated file system
_files = {
    '/data/report.txt': 'This is the quarterly report for Q3 2024. Sales increased by 15% compared to last quarter, driven primarily by growth in the enterprise segment.',
    '/data/notes.txt': 'Meeting notes from October 15th. Action items: 1) Review budget proposal, 2) Schedule follow-up with marketing team, 3) Finalize product roadmap.',
    '/data/config.json': '{"setting": "value", "enabled": true}',  # Not a .txt file
    '/data/readme.txt': 'Welcome to the data directory. This folder contains various reports and documentation files for the project.',
    '/data/summary.txt': 'Executive Summary: The project is on track for delivery in November. Key milestones have been met and the team is performing well.',
    '/data/logs.txt': 'Application started at 10:00 AM. User login: admin. Data sync completed. Backup initiated. Cache cleared. Session ended at 5:00 PM.',
    '/data/archive.zip': 'binary data here',  # Not a .txt file
    '/data/draft.txt': 'DRAFT: This document is not yet finalized. Please do not distribute. Working title: "Improving Developer Productivity with AI Tools".',
}


# =============================================================================
# Mock File System Tools
# =============================================================================


def list_directory(path: str) -> dict[str, Any]:
    """List files in a directory.

    Args:
        path: The directory path to list.

    Returns:
        Dictionary with list of filenames.
    """
    if path != '/data':
        return {'error': f'Directory not found: {path}', 'files': []}

    files = [f.split('/')[-1] for f in _files.keys()]
    return {'path': path, 'files': files}


def read_file(path: str) -> dict[str, Any]:
    """Read the contents of a file.

    Args:
        path: The full path to the file.

    Returns:
        Dictionary with file content or error.
    """
    if path not in _files:
        return {'error': f'File not found: {path}'}

    content = _files[path]
    return {'path': path, 'content': content, 'size': len(content)}


def create_toolset() -> FunctionToolset[None]:
    """Create the file system toolset."""
    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(list_directory)
    toolset.add_function(read_file)
    return toolset


# =============================================================================
# Agent Factories
# =============================================================================


def create_tool_calling_agent(toolset: FunctionToolset[None]) -> Agent[None, str]:
    """Create agent with standard tool calling."""
    return Agent(
        MODEL,
        toolsets=[toolset],
        system_prompt='You are a file system assistant. Use the available tools to work with files.',
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
        system_prompt='You are a file system assistant. Use the available tools to work with files.',
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
    with logfire.span('tool_calling'):
        agent = create_tool_calling_agent(toolset)
        result = await agent.run(PROMPT)
    return extract_metrics(result, 'tool_calling')


async def run_code_mode(toolset: FunctionToolset[None]) -> RunMetrics:
    """Run with CodeMode tool calling."""
    with logfire.span('code_mode_tool_calling'):
        agent = create_code_mode_agent(toolset)
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
    logfire.configure(service_name='code-mode-files-demo')
    logfire.instrument_pydantic_ai()

    toolset = create_toolset()

    with logfire.span('demo_tool_calling'):
        trad = await run_tool_calling(toolset)
    log_metrics(trad)

    with logfire.span('demo_code_mode'):
        code = await run_code_mode(toolset)
    log_metrics(code)

    request_reduction = trad.request_count - code.request_count
    token_diff = trad.total_tokens - code.total_tokens

    print(f'Results: {trad.request_count} → {code.request_count} requests ({request_reduction} fewer)')
    print(f'Tokens: {trad.total_tokens:,} → {code.total_tokens:,} ({token_diff:+,} difference)')
    print('View traces: https://logfire.pydantic.dev')


if __name__ == '__main__':
    asyncio.run(main())
