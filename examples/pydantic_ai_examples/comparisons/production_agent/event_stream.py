"""The run is an event stream you can observe or transform.

Part deltas, tool calls, results, and the final result — all typed, all
streamed. No framework opinion on what you do with them.
"""
import asyncio
from pydantic_ai import Agent, AgentStreamEvent
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai.messages import FinalResultEvent, FunctionToolCallEvent


async def stream(messages, info):
    if len(messages) == 1:
        yield {0: DeltaToolCall(name='twice', json_args='{"n": 21}', tool_call_id='c1')}
    else:
        yield '42'


agent = Agent(FunctionModel(stream_function=stream))


@agent.tool
def twice(ctx, n: int) -> int:
    return n * 2


async def main():
    kinds = []
    async with agent.run_stream_events('what is 21*2?') as run:
        async for event in run:
            kinds.append(type(event).__name__)
        final = run.result.output
    assert 'FunctionToolCallEvent' in kinds and 'FinalResultEvent' in kinds
    print(f'events observed: {kinds}')
    print(f'final output: {final!r} (streamed while it happened)')


if __name__ == '__main__':
    asyncio.run(main())
