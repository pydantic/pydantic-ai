"""Agents with a structured output type whose model streams text before a tool call, for each durability engine.

A durable run replays the unit's stream against the parameters the unit's model prepared, in which `'auto'` has
resolved to tool output, so the text isn't the final result, as in a plain run. Behind a `FallbackModel`, `'auto'` is
resolved to the mode the candidate that answered resolved it to.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Callable
from dataclasses import dataclass

from pydantic_ai import (
    Agent,
    AgentStreamEvent,
    FinalResultEvent,
    ModelMessage,
    ModelResponse,
    ModelResponsePart,
    RunContext,
)
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel


async def preamble_text_before_function_tool(
    messages: list[ModelMessage], _info: AgentInfo
) -> AsyncIterator[str | DeltaToolCalls]:
    if len(messages) == 1:
        yield 'Let me look that up.'
        yield {0: DeltaToolCall(name='look_up_city', json_args='{}', tool_call_id='call_1')}
    else:
        yield {0: DeltaToolCall(name='final_result', json_args='{"response":"Paris"}', tool_call_id='call_2')}


async def preamble_text_before_output_tool(
    _messages: list[ModelMessage], _info: AgentInfo
) -> AsyncIterator[str | DeltaToolCalls]:
    yield 'Preamble text.'
    yield {0: DeltaToolCall(name='final_result', json_args='{"response":"Paris"}', tool_call_id='call_2')}


def look_up_city() -> str:
    return 'Paris'


@dataclass
class StructuredOutput:
    response: str


def preamble_text_agents(
    durability: Callable[[], AbstractCapability[object]] | None = None,
    *,
    fallback: bool = False,
) -> list[Agent[object, StructuredOutput]]:
    """One agent per stream shape: text before a function tool, and text before the output tool.

    Without `durability`, the agents run as a plain `Agent` does, which a durable run must match. With `fallback`, the
    model is a `FallbackModel` around it, whose `prepare_request` leaves the output mode to the candidate that answers.

    The tests built on these agents aren't VCR tests. What they check is how the workflow side classifies the
    `ModelResponse` the durable unit returned. The unit has already parsed the provider's answer by then, so the wire
    format a recording would pin never reaches that code. A scripted model streams the exact shape on every engine,
    where a real model writes text before a tool call only when it chooses to.
    """
    return [
        Agent(
            FallbackModel(FunctionModel(stream_function=stream_function))
            if fallback
            else FunctionModel(stream_function=stream_function),
            name=f'durability_{"fallback_" if fallback else ""}{stream_function.__name__}',
            output_type=StructuredOutput,
            tools=[look_up_city],
            capabilities=[durability()] if durability is not None else [],
        )
        for stream_function in (preamble_text_before_function_tool, preamble_text_before_output_tool)
    ]


@dataclass
class RunAndStreamResult:
    """What `run()` and `run_stream()` return, and the tool name of each `FinalResultEvent` the `run()` handler gets."""

    run_output: str
    stream_output: str
    final_result_tool_names: list[str | None]
    run_responses: list[list[ModelResponsePart]]
    """The parts of each response `run()` replayed."""
    stream_responses: list[list[ModelResponsePart]]
    """The parts of each response `run_stream()` replayed."""


async def run_and_stream(agent: Agent[object, StructuredOutput]) -> RunAndStreamResult:
    final_result_tool_names: list[str | None] = []

    async def handler(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            if isinstance(event, FinalResultEvent):
                final_result_tool_names.append(event.tool_name)

    run_result = await agent.run('hello', event_stream_handler=handler)
    async with agent.run_stream('hello') as stream_result:
        stream_output = await stream_result.get_output()
        stream_messages = stream_result.new_messages()
    return RunAndStreamResult(
        run_output=run_result.output.response,
        stream_output=stream_output.response,
        final_result_tool_names=final_result_tool_names,
        run_responses=[
            list(message.parts) for message in run_result.new_messages() if isinstance(message, ModelResponse)
        ],
        stream_responses=[list(message.parts) for message in stream_messages if isinstance(message, ModelResponse)],
    )


async def plain_run_and_stream() -> list[RunAndStreamResult]:
    """What `run_and_stream` returns for each agent run as a plain `Agent`, without durability."""
    return [await run_and_stream(agent) for agent in preamble_text_agents()]
