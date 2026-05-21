"""v1: FunctionToolCallEvent.call_id."""
from pydantic_ai.messages import FunctionToolCallEvent, ToolCallPart


def trigger():
    ev = FunctionToolCallEvent(part=ToolCallPart(tool_name='x', args={}, tool_call_id='t1'))
    # DEPRECATION: F3_call_id
    _ = ev.call_id
    return ev


EXPECT = '`call_id` is deprecated, use `tool_call_id` instead'

if __name__ == '__main__':
    trigger()
