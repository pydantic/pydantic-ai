"""Helpers for a capability, toolset or workspace to add spans to an agent run's trace.

Each helper takes the run's `ctx` and uses the run's tracer and its `include_content` setting, so a
span appears only when the run is instrumented, and a value that can quote the run's data (passed as
`content`) is recorded only when the run captures content:

- [`open_span`][pydantic_ai.telemetry.open_span]: a span around work that takes time and can fail.
- [`record_decision`][pydantic_ai.telemetry.record_decision]: a decision that changed the run's course,
  such as a guard blocking a value.
- [`current_span`][pydantic_ai.telemetry.current_span]: the span already open where the caller runs,
  such as a tool hook's `execute_tool` span, to add attributes to.
- [`set_content_attribute`][pydantic_ai.telemetry.set_content_attribute]: an attribute that can quote
  content, set only when the run captures content.
- [`records_content`][pydantic_ai.telemetry.records_content]: whether content is recorded, to skip
  building a costly content value.

Code that runs during a run but has no `RunContext`, such as a workspace that several runs share,
passes [`current_run_telemetry()`][pydantic_ai.telemetry.current_run_telemetry] instead of `ctx`.
"""

from ._telemetry import (
    RunTelemetry,
    current_run_telemetry,
    current_span,
    open_span,
    record_decision,
    records_content,
    set_content_attribute,
)

__all__ = (
    'RunTelemetry',
    'current_run_telemetry',
    'current_span',
    'open_span',
    'record_decision',
    'records_content',
    'set_content_attribute',
)
