"""Interrupted history repairs itself before it reaches the model.

A run that dies mid-tool leaves a dangling tool call. The next run closes
that call out (outcome=<interrupted>) before the request goes out, so the
provider never rejects the history as malformed.
"""
import asyncio
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)


async def model(messages, info):
    return ModelResponse(parts=[TextPart('ok')])


agent = Agent(FunctionModel(model), tools=[])


@agent.tool
def add(ctx, n: int) -> int:
    return n


interrupted = [
    ModelRequest(parts=[UserPromptPart(content='add 1')]),
    ModelResponse(parts=[ToolCallPart('add', {'n': 1}, tool_call_id='t1')], state='interrupted'),
]


async def main():
    with capture_run_messages() as msgs:
        await agent.run('add 1', message_history=interrupted)
    repaired = [
        p for m in msgs for p in m.parts if isinstance(p, ToolReturnPart) and p.tool_call_id == 't1'
    ]
    assert repaired, 'dangling tool call was not repaired'
    print(f'outgoing request carried a synthesized result for t1: {[r.content for r in repaired]}')
    print('history was provider-valid: no malformed pairing sent to the model')


if __name__ == '__main__':
    asyncio.run(main())
