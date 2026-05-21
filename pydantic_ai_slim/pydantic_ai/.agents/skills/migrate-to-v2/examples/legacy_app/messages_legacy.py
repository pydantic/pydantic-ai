"""v1: vendor_*, call_id, Usage."""
from pydantic_ai.messages import ModelResponse, TextPart, FunctionToolCallEvent, ToolCallPart
from pydantic_ai.usage import Usage


def touch_legacy_fields() -> tuple:
    r = ModelResponse(parts=[TextPart(content='x')])
    # DEPRECATION: F2_vendor_details
    _ = r.vendor_details
    _ = r.vendor_id
    ev = FunctionToolCallEvent(part=ToolCallPart(tool_name='t', args={}, tool_call_id='t1'))
    # DEPRECATION: F3_call_id
    _ = ev.call_id
    # DEPRECATION: F1_usage_class
    u = Usage()
    return r, ev, u
