"""Events that show tools called from inside `run_code` the same way as direct tool calls.

Harness `CodeMode` runs sandbox calls through capability hooks but emits no call or result events
for them, so the renderer would print only the `run_code` header. `ShowSandboxCalls` in
`speculative_mode` reports them with these events, and `SandboxCallOrder` turns each report back
into the core event the renderer already draws. This module stays free of harness imports because
the renderer loads it at startup.
"""

from dataclasses import dataclass, field

from pydantic_ai.messages import (
    AgentStreamEvent,
    CapabilityEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
)


@dataclass(kw_only=True)
class SandboxCallStartedEvent(CapabilityEvent, namespace='pydantic_clai2'):
    """A tool called from inside `run_code` started, or a speculative launch was claimed."""

    call: ToolCallPart


@dataclass(kw_only=True)
class SandboxCallFinishedEvent(CapabilityEvent, namespace='pydantic_clai2'):
    """A tool called from inside `run_code` returned or failed."""

    result: ToolReturnPart | RetryPromptPart


@dataclass
class SandboxCallOrder:
    """Put each `run_code` header above the calls it made.

    Eager execution runs sandbox calls while `run_code` is still streaming, but core emits the
    `run_code` call event only once streaming ends. Whichever comes first announces the header:
    the first sandbox call under a `run_code`, or core's event when the snippet ran after the
    stream. The other is not repeated. Announced ids are kept for the renderer's lifetime because
    calls whose statements close at the end of the stream still arrive after the late event.
    """

    _announced: set[str] = field(default_factory=set[str], init=False, repr=False)

    def tool_events(self, event: AgentStreamEvent) -> list[FunctionToolCallEvent | FunctionToolResultEvent] | None:
        """Core tool events to render in place of `event`, or `None` to render `event` as usual."""
        if isinstance(event, SandboxCallStartedEvent):
            parent = event.call.tool_call_id.rpartition('__')[0]
            header: list[FunctionToolCallEvent | FunctionToolResultEvent] = []
            if parent not in self._announced:
                self._announced.add(parent)
                header.append(FunctionToolCallEvent(ToolCallPart(tool_name='run_code', tool_call_id=parent)))
            return [*header, FunctionToolCallEvent(event.call)]
        if isinstance(event, SandboxCallFinishedEvent):
            return [FunctionToolResultEvent(event.result)]
        if isinstance(event, FunctionToolCallEvent) and event.part.tool_name == 'run_code':
            if event.part.tool_call_id in self._announced:
                return []
            self._announced.add(event.part.tool_call_id)
        return None
