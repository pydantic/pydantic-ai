"""RAG Pipeline Evals - Compare code mode vs traditional mode.

Runs 3 models x 3 runs x 2 modes = 18 runs to compare efficiency.

Usage (full mode):
    source .env && uv run python demos/code_mode/rag_pipeline/evals.py

Usage (zero-setup mode):
    uv run python demos/code_mode/rag_pipeline/evals.py --zero-setup

Metrics tracked:
- Request count (LLM round-trips)
- Total tokens (input + output)
- Errors (ModelRetry count)
"""

from __future__ import annotations

import argparse
import asyncio
import os
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel

from pydantic_ai import AgentRunResult
from pydantic_ai.messages import ModelResponse, RetryPromptPart
from pydantic_ai.mcp import MCPServerStdio, MCPServerStreamableHTTP

from .demo import (
    MODELS,
    PROMPT,
    PROMPT_ZERO_SETUP,
    create_code_mode_agent,
    create_code_mode_agent_zero_setup,
    create_pinecone_mcp,
    create_pinecone_mcp_zero_setup,
    create_tavily_mcp,
    create_traditional_agent,
    create_traditional_agent_zero_setup,
)

# IMPORTANT: Bump version when prompt changes!
EVAL_NAME = 'rag_pipeline_v1'

RUNS_PER_CONFIG = 3


class RunResult(BaseModel):
    """Result from a single eval run."""

    model: str
    mode: str  # 'traditional' or 'code_mode'
    run_number: int
    request_count: int
    total_input_tokens: int
    total_output_tokens: int
    retry_count: int
    success: bool
    error: str | None = None
    output_preview: str | None = None


@dataclass
class EvalConfig:
    """Configuration for an eval run."""

    model: str
    mode: str
    run_number: int


async def run_eval_full(
    config: EvalConfig,
    pinecone: MCPServerStdio,
    tavily: MCPServerStreamableHTTP,
) -> RunResult:
    """Run a single eval with full mode (Pinecone + Tavily)."""
    try:
        if config.mode == 'code_mode':
            agent = create_code_mode_agent(pinecone, tavily, model=config.model)
            code_toolset = agent.toolsets[0]
            async with code_toolset:
                result = await agent.run(PROMPT)
        else:
            agent = create_traditional_agent(pinecone, tavily, model=config.model)
            result = await agent.run(PROMPT)

        return extract_metrics(config, result)

    except Exception as e:
        return error_result(config, str(e))


async def run_eval_zero_setup(
    config: EvalConfig,
    pinecone: MCPServerStdio,
) -> RunResult:
    """Run a single eval with zero-setup mode (search_docs only)."""
    try:
        if config.mode == 'code_mode':
            agent = create_code_mode_agent_zero_setup(pinecone, model=config.model)
            code_toolset = agent.toolsets[0]
            async with code_toolset:
                result = await agent.run(PROMPT_ZERO_SETUP)
        else:
            agent = create_traditional_agent_zero_setup(pinecone, model=config.model)
            result = await agent.run(PROMPT_ZERO_SETUP)

        return extract_metrics(config, result)

    except Exception as e:
        return error_result(config, str(e))


def extract_metrics(config: EvalConfig, result: AgentRunResult[str]) -> RunResult:
    """Extract metrics from agent result."""
    request_count = 0
    total_input = 0
    total_output = 0
    retry_count = 0

    for msg in result.all_messages():
        if isinstance(msg, ModelResponse):
            request_count += 1
            total_input += msg.usage.input_tokens
            total_output += msg.usage.output_tokens
        for part in getattr(msg, 'parts', []):
            if isinstance(part, RetryPromptPart):
                retry_count += 1

    return RunResult(
        model=config.model,
        mode=config.mode,
        run_number=config.run_number,
        request_count=request_count,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        retry_count=retry_count,
        success=True,
        output_preview=result.output[:200] if result.output else None,
    )


def error_result(config: EvalConfig, error: str) -> RunResult:
    """Create error result."""
    return RunResult(
        model=config.model,
        mode=config.mode,
        run_number=config.run_number,
        request_count=0,
        total_input_tokens=0,
        total_output_tokens=0,
        retry_count=0,
        success=False,
        error=error,
    )


async def main():
    """Run all evaluations and print results."""
    parser = argparse.ArgumentParser(description='RAG Pipeline Evals')
    parser.add_argument(
        '--zero-setup',
        action='store_true',
        help='Run in zero-setup mode (no API keys needed)',
    )
    args = parser.parse_args()

    print('=' * 70)
    print(f'RAG Pipeline Eval: {EVAL_NAME}')
    print(f'Mode: {"Zero-Setup" if args.zero_setup else "Full"}')
    print(f'Started: {datetime.now().isoformat()}')
    print('=' * 70)
    print()
    print(f'Models: {len(MODELS)}')
    print(f'Runs per config: {RUNS_PER_CONFIG}')
    print(f'Modes: 2 (traditional, code_mode)')
    print(f'Total runs: {len(MODELS) * 2 * RUNS_PER_CONFIG}')
    print()

    # Build all configs
    configs: list[EvalConfig] = []
    for model in MODELS:
        for mode in ['traditional', 'code_mode']:
            for run_num in range(1, RUNS_PER_CONFIG + 1):
                configs.append(EvalConfig(model=model, mode=mode, run_number=run_num))

    results: list[RunResult] = []

    if args.zero_setup:
        pinecone = create_pinecone_mcp_zero_setup()
        async with pinecone:
            for i, config in enumerate(configs, 1):
                print(f'[{i}/{len(configs)}] {config.model} | {config.mode} | run {config.run_number}')
                result = await run_eval_zero_setup(config, pinecone)
                results.append(result)
                print_result(result)
    else:
        if not os.environ.get('PINECONE_API_KEY') or not os.environ.get('TAVILY_API_KEY'):
            print('ERROR: PINECONE_API_KEY and TAVILY_API_KEY required. Use --zero-setup for no-key mode.')
            return

        pinecone = create_pinecone_mcp()
        tavily = create_tavily_mcp()

        async with pinecone, tavily:
            for i, config in enumerate(configs, 1):
                print(f'[{i}/{len(configs)}] {config.model} | {config.mode} | run {config.run_number}')
                result = await run_eval_full(config, pinecone, tavily)
                results.append(result)
                print_result(result)

    print_summary(results)


def print_result(result: RunResult):
    """Print single result."""
    if result.success:
        total_tokens = result.total_input_tokens + result.total_output_tokens
        print(f'  -> requests={result.request_count}, tokens={total_tokens}, retries={result.retry_count}')
    else:
        print(f'  -> FAILED: {result.error}')
    print()


def print_summary(results: list[RunResult]):
    """Print aggregated summary."""
    print('=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print()
    print(f'{"Model":<40} {"Mode":<12} {"Requests":>10} {"Tokens":>12} {"Retries":>10}')
    print('-' * 90)

    for model in MODELS:
        for mode in ['traditional', 'code_mode']:
            mode_results = [r for r in results if r.model == model and r.mode == mode and r.success]
            if mode_results:
                avg_requests = sum(r.request_count for r in mode_results) / len(mode_results)
                avg_tokens = sum(r.total_input_tokens + r.total_output_tokens for r in mode_results) / len(mode_results)
                avg_retries = sum(r.retry_count for r in mode_results) / len(mode_results)
                print(f'{model:<40} {mode:<12} {avg_requests:>10.1f} {avg_tokens:>12.0f} {avg_retries:>10.1f}')
            else:
                print(f'{model:<40} {mode:<12} {"FAILED":>10} {"":>12} {"":>10}')

    print()
    print('=' * 70)
    print(f'Completed: {datetime.now().isoformat()}')
    print('=' * 70)


if __name__ == '__main__':
    asyncio.run(main())
