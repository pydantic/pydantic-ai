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
from pydantic_ai.runtime import CodeRuntime
from pydantic_ai.runtime.monty import MontyRuntime

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


async def run_mode(
    mode: str, github: MCPServerStreamableHTTP, model: str, runtime: CodeRuntime | None = None
) -> RunResult:
    with logfire.span('demo_run.{mode}', mode=mode, model=model):
        if mode == 'code_mode':
            agent = create_code_mode_agent(github, model=model, runtime=runtime)
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


async def main() -> None:
    github: MCPServerStreamableHTTP = create_github_mcp()
    runtime = MontyRuntime()

    async with github:
        for model in MODELS:
            with logfire.span('model_run {model}', model=model):
                tool_calling_result = await run_mode('tool_calling', github, model)
                code_mode_result = await run_mode('code_mode', github, model, runtime=runtime)

            logfire.info(
                'tool_calling_result: {request_count=} {token_count=} {retry_count=}',
                request_count=tool_calling_result.request_count,
                token_count=tool_calling_result.total_input_tokens + tool_calling_result.total_output_tokens,
                retry_count=tool_calling_result.retry_count,
                output=tool_calling_result.output,
            )
            logfire.info(
                'code_mode_result: {request_count=} {token_count=} {token_count=}',
                request_count=code_mode_result.request_count,
                token_count=code_mode_result.total_input_tokens + code_mode_result.total_output_tokens,
                retry_count=code_mode_result.retry_count,
                output=code_mode_result.output,
            )


if __name__ == '__main__':
    logfire.configure(service_name='discussion-intensity-demo')
    logfire.instrument_pydantic_ai()
    asyncio.run(main())
