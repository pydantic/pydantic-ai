"""v2 form: .tool_call_id."""
from pydantic_ai.messages import FunctionToolCallEvent, ToolCallPart


def trigger():
    ev = FunctionToolCallEvent(part=ToolCallPart(tool_name='x', args={}, tool_call_id='t1'))
    _ = ev.tool_call_id
    return ev


if __name__ == '__main__':
    trigger()
