"""Compare HTTP and WebSocket latency for the same sequential tool workload.

Run from the repository root with an OpenAI API key in the environment:
    uv run scripts/benchmark_openai_responses_websocket.py --model gpt-6-astra --service-tier ultrafast

Both transports reuse their connection and use stored-response chaining, identical
settings, and the same tools. One warmup run per transport precedes alternating
trials. Connection setup is excluded. Calls use the selected tier and incur API charges.

TTFT is measured per model request, from opening its stream to the first generated
text, reasoning, or tool-argument content. Total time includes tools and all requests.
The JSON report includes actual service tiers and token usage so tier and output-length differences remain visible.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from statistics import median
from time import perf_counter
from typing import Literal, get_args

import httpx2
import logfire
from openai import AsyncOpenAI
from opentelemetry import trace
from pydantic import TypeAdapter

from pydantic_ai import (
    Agent,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    RunContext,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

Transport = Literal['http', 'websocket']
ServiceTier = Literal['auto', 'default', 'flex', 'priority', 'ultrafast']


@dataclass
class Workload:
    """Per-run state for a fixed number of sequential tool calls."""

    rounds: int
    completed: int = 0


@dataclass
class Trial:
    """Latency and usage measurements for one complete agent run."""

    transport: Transport
    trial: int
    total_ms: float
    ttft_ms: list[float]
    requests: int
    tool_calls: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    served_tiers: list[str | None]


async def measure(
    agent: Agent[Workload, str], model: OpenAIResponsesModel, transport: Transport, trial: int, rounds: int
) -> Trial:
    """Measure first-token latency per request and total time for one tool loop."""
    workload = Workload(rounds)
    times: list[float] = []
    started = perf_counter()
    async with agent.iter('Complete the steps.', model=model, deps=workload) as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                request_started = perf_counter()
                first_token: float | None = None
                async with node.stream(run.ctx) as stream:
                    async for event in stream:
                        has_content = False
                        if isinstance(event, PartStartEvent):
                            if isinstance(event.part, (TextPart, ThinkingPart)):
                                has_content = bool(event.part.content)
                            elif isinstance(event.part, ToolCallPart):
                                has_content = bool(event.part.args)
                        elif isinstance(event, PartDeltaEvent):
                            if isinstance(event.delta, (TextPartDelta, ThinkingPartDelta)):
                                has_content = bool(event.delta.content_delta)
                            elif isinstance(event.delta, ToolCallPartDelta):
                                has_content = bool(event.delta.args_delta)
                        if first_token is None and has_content:
                            first_token = perf_counter() - request_started
                if first_token is None:
                    raise RuntimeError('A response contained no generated tokens; the trial is not comparable.')
                times.append(first_token * 1000)
    elapsed = perf_counter() - started
    assert run.result is not None
    if workload.completed != rounds or run.result.output.strip().lower() != 'done':
        raise RuntimeError(f'The model did not complete exactly {rounds} tool calls and answer done.')
    usage = run.result.usage
    served_tiers: list[str | None] = []
    for message in run.result.all_messages():
        if isinstance(message, ModelResponse):
            served_tier: object = (message.provider_details or {}).get('service_tier')
            served_tiers.append(served_tier if isinstance(served_tier, str) else None)
    return Trial(
        transport=transport,
        trial=trial,
        total_ms=elapsed * 1000,
        ttft_ms=times,
        requests=usage.requests,
        tool_calls=workload.completed,
        input_tokens=usage.input_tokens,
        cached_input_tokens=usage.cache_read_tokens,
        output_tokens=usage.output_tokens,
        served_tiers=served_tiers,
    )


async def main() -> None:
    """Warm both transports, alternate trials, and print a comparable JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--service-tier', choices=get_args(ServiceTier), required=True)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--tool-rounds', type=int, default=3)
    args = parser.parse_args()
    if args.trials < 1 or args.tool_rounds < 1:
        parser.error('--trials and --tool-rounds must be positive')

    model_name: str = args.model
    tier = TypeAdapter[ServiceTier](ServiceTier).validate_python(args.service_tier)
    settings: OpenAIResponsesModelSettings = {
        'openai_responses_service_tier': tier,
        'openai_previous_response_id': 'auto',
        'openai_store': True,
        'openai_reasoning_effort': 'low',
        'parallel_tool_calls': False,
    }
    logfire.configure(send_to_logfire='if-token-present', console=False)
    logfire.instrument_pydantic_ai()
    trials: list[Trial] = []
    async with httpx2.AsyncClient(timeout=60, limits=httpx2.Limits(keepalive_expiry=None)) as http_client:
        logfire.instrument_httpx(http_client, capture_all=True)
        async with AsyncOpenAI(http_client=http_client, max_retries=0, timeout=60) as client:
            source = OpenAIResponsesModel(model_name, provider=OpenAIProvider(openai_client=client), settings=settings)
            agent = Agent(
                source,
                deps_type=Workload,
                name='responses_transport_benchmark',
                instructions='Call next_step until it returns done=true. Call it once per response. Then answer exactly done.',
            )

            @agent.tool
            async def next_step(ctx: RunContext[Workload]) -> dict[str, int | bool]:
                """Perform the next sequential step and report whether the workload is complete."""
                ctx.deps.completed += 1
                return {'step': ctx.deps.completed, 'done': ctx.deps.completed >= ctx.deps.rounds}

            with logfire.span('Responses transport benchmark', model=model_name, service_tier=tier):
                trace_id = format(trace.get_current_span().get_span_context().trace_id, '032x')
                async with source, source.connect() as connected:
                    models: dict[Transport, OpenAIResponsesModel] = {'http': source, 'websocket': connected}
                    for transport, model in models.items():
                        await measure(agent, model, transport, 0, args.tool_rounds)
                    for trial in range(1, args.trials + 1):
                        order: tuple[Transport, Transport] = (
                            ('http', 'websocket') if trial % 2 else ('websocket', 'http')
                        )
                        for transport in order:
                            trials.append(await measure(agent, models[transport], transport, trial, args.tool_rounds))

    summary = {
        transport: {
            'median_total_ms': median(row.total_ms for row in trials if row.transport == transport),
            'median_ttft_ms': median(value for row in trials if row.transport == transport for value in row.ttft_ms),
        }
        for transport in ('http', 'websocket')
    }
    print(
        json.dumps(
            {
                'model': model_name,
                'settings': settings,
                'trace_id': trace_id,
                'trials': [asdict(row) for row in trials],
                'summary': summary,
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    asyncio.run(main())
