"""Deferred capability: the model cannot touch what it hasn't loaded.

Request 1: the deferred tool is absent from the request payload.
Request 2: the model asks to load the capability; request 3 sees the tool.
"""
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

CAP_INSTRUCTION = 'Always confirm the order ID before issuing a refund.'

refunds = Capability(
    id='refunds', description='Use for refunds.', instructions=CAP_INSTRUCTION, defer_loading=True
)


@refunds.tool_plain
def refund_status(order_id: str) -> str:
    """Look up refund status."""
    return f'Order {order_id}: refunded.'


seen = []  # (tool names, cap_instruction_in_messages)


async def model(messages, info):
    tools = sorted(t.name for t in info.function_tools)
    seen.append((tools, CAP_INSTRUCTION in str(messages)))
    n = len(seen)
    if n == 1:
        return ModelResponse(parts=[ToolCallPart('refund_status', {'order_id': 'X'})])  # blocked: not loaded
    if n == 2:
        return ModelResponse(parts=[ToolCallPart('load_capability', {'id': 'refunds'})])
    if n == 3:
        return ModelResponse(parts=[ToolCallPart('refund_status', {'order_id': 'Y'})])
    return ModelResponse(parts=[TextPart('done')])


agent = Agent(FunctionModel(model), capabilities=[refunds])


def main() -> None:
    res = agent.run_sync('go')
    print('requests:', len(seen))
    for i, (tools, instr) in enumerate(seen, 1):
        print(f'  req{i}: tools={tools} cap_instruction_in_messages={instr}')
    late = [t for t in seen[0][0] if 'refund' in t]
    assert not late, 'deferred tool was offered before loading'
    print('deferred tool visible before load_capability:', bool(late))


if __name__ == '__main__':
    main()
