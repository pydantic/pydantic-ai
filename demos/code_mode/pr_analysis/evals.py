"""PR Analysis Evals - Compare code mode vs traditional mode.

Runs 3 models × 2 runs × 2 modes = 12 runs to compare efficiency.

Usage:
    source .env && uv run python demos/code_mode/pr_analysis/evals.py

Metrics tracked:
- Request count (LLM round-trips)
- Total tokens (input + output)
- Errors (ModelRetry count)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime

import logfire
from pydantic import BaseModel

from pydantic_ai.messages import ModelResponse, RetryPromptPart
from pydantic_ai.mcp import MCPServerStreamableHTTP

from .demo import MODELS, PROMPT, create_code_mode_agent, create_github_mcp, create_traditional_agent

# IMPORTANT: Bump version when prompt changes!
EVAL_NAME = 'pr_size_review_rounds_v1'

RUNS_PER_CONFIG = 1


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


async def run_eval(config: EvalConfig, github: MCPServerStreamableHTTP) -> RunResult:
    """Run a single eval with the given configuration."""
    with logfire.span('eval_run', model=config.model, mode=config.mode, run_number=config.run_number):
        try:
            if config.mode == 'code_mode':
                agent = create_code_mode_agent(github, model=config.model)
                # CodeModeToolset needs context manager for sandbox lifecycle
                code_toolset = agent.toolsets[0]
                async with code_toolset:
                    result = await agent.run(PROMPT)
            else:
                agent = create_traditional_agent(github, model=config.model)
                result = await agent.run(PROMPT)

            # Count metrics from messages
            request_count = 0
            total_input = 0
            total_output = 0
            retry_count = 0

            for msg in result.all_messages():
                if isinstance(msg, ModelResponse):
                    request_count += 1
                    total_input += msg.usage.input_tokens
                    total_output += msg.usage.output_tokens
                # Count retries
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

        except Exception as e:
            return RunResult(
                model=config.model,
                mode=config.mode,
                run_number=config.run_number,
                request_count=0,
                total_input_tokens=0,
                total_output_tokens=0,
                retry_count=0,
                success=False,
                error=str(e),
            )


async def main():
    """Run all evaluations and print results."""
    print('=' * 70)
    print(f'PR Analysis Eval: {EVAL_NAME}')
    print(f'Started: {datetime.now().isoformat()}')
    print('=' * 70)
    print()
    print(f'Models: {len(MODELS)}')
    print(f'Runs per config: {RUNS_PER_CONFIG}')
    print(f'Modes: 2 (traditional, code_mode)')
    print(f'Total runs: {len(MODELS) * 2 * RUNS_PER_CONFIG}')
    print()

    github = create_github_mcp()

    # Build all configs
    configs: list[EvalConfig] = []
    for model in MODELS:
        for mode in ['traditional', 'code_mode']:
            for run_num in range(1, RUNS_PER_CONFIG + 1):
                configs.append(EvalConfig(model=model, mode=mode, run_number=run_num))

    results: list[RunResult] = []

    async with github:
        for i, config in enumerate(configs, 1):
            print(f'[{i}/{len(configs)}] {config.model} | {config.mode} | run {config.run_number}')
            result = await run_eval(config, github)
            results.append(result)

            if result.success:
                total_tokens = result.total_input_tokens + result.total_output_tokens
                print(f'  -> requests={result.request_count}, tokens={total_tokens}, retries={result.retry_count}')
            else:
                print(f'  -> FAILED: {result.error}')
            print()

    # Aggregate results
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
    print('View traces at https://logfire.pydantic.dev')
    print('=' * 70)


if __name__ == '__main__':
    logfire.configure(service_name='pr-analysis-eval')
    logfire.instrument_pydantic_ai()
    asyncio.run(main())
