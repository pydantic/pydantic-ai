"""v2 form: provider_*, tool_call_id, RunUsage."""
from pydantic_ai.messages import ModelResponse, TextPart, FunctionToolCallEvent, ToolCallPart
from pydantic_ai.usage import RunUsage


def touch_legacy_fields() -> tuple:
    r = ModelResponse(parts=[TextPart(content='x')])
    _ = r.provider_details
    _ = r.provider_response_id
    ev = FunctionToolCallEvent(part=ToolCallPart(tool_name='t', args={}, tool_call_id='t1'))
    _ = ev.tool_call_id
    u = RunUsage()
    return r, ev, u
