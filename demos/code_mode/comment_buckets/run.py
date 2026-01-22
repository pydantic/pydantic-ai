"""Discussion Intensity Demo - Script runner for code mode vs traditional mode.

Usage:
    source .env && uv run python demos/code_mode/comment_buckets/run.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import logfire

from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.messages import ModelResponse, RetryPromptPart

from .demo import MODELS, PROMPT, create_code_mode_agent, create_github_mcp, create_traditional_agent


@dataclass
class RunResult:
    model: str
    mode: str
    request_count: int
    total_input_tokens: int
    total_output_tokens: int
    retry_count: int
    output: str | None


async def run_mode(mode: str, github: MCPServerStreamableHTTP, model: str) -> RunResult:
    with logfire.span(f'demo_run.{mode}', mode=mode, model=model):
        logfire.info('start_run', mode=mode, model=model)
        if mode == 'code_mode':
            agent = create_code_mode_agent(github, model=model)
            code_toolset = agent.toolsets[0]
            async with code_toolset:
                result = await agent.run(PROMPT)
        else:
            agent = create_traditional_agent(github, model=model)
            result = await agent.run(PROMPT)

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
        model=model,
        mode=mode,
        request_count=request_count,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        retry_count=retry_count,
        output=result.output,
    )


def print_summary(result: RunResult) -> None:
    total_tokens = result.total_input_tokens + result.total_output_tokens
    print(
        f'{result.mode:<12} requests={result.request_count} tokens={total_tokens} retries={result.retry_count}'
    )


def print_model_header(model: str) -> None:
    print('=' * 70)
    print(f'MODEL: {model}')
    print('=' * 70)


async def main() -> None:
    with logfire.span('discussion_intensity_demo'):
        print('=' * 70)
        print('Discussion Intensity Demo: Code Mode vs Traditional Mode')
        print('=' * 70)
        print()
        print(PROMPT)
        print()

        github: MCPServerStreamableHTTP = create_github_mcp()

        async with github:
            for model in MODELS:
                print_model_header(model)
                with logfire.span('model_run', model=model):
                    traditional_result = await run_mode('traditional', github, model)
                    code_mode_result = await run_mode('code_mode', github, model)

                print('=' * 70)
                print('RESULTS')
                print('=' * 70)
                print_summary(traditional_result)
                print_summary(code_mode_result)
                print()
                print('=' * 70)
                print('TRADITIONAL OUTPUT')
                print('=' * 70)
                print(traditional_result.output or 'No output')
                print()
                print('=' * 70)
                print('CODE MODE OUTPUT')
                print('=' * 70)
                print(code_mode_result.output or 'No output')
                print()

    print('View traces at https://logfire.pydantic.dev')


if __name__ == '__main__':
    logfire.configure(service_name='discussion-intensity-demo')
    logfire.instrument_pydantic_ai()
    asyncio.run(main())
