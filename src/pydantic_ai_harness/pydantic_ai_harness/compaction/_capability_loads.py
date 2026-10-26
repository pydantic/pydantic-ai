"""Keep deferred capabilities loaded across compaction, or tell the model which ones it unloaded.

Core counts a deferred capability as loaded while its `load_capability` exchange is in the
[`post_compaction_window`][pydantic_ai.messages.post_compaction_window], the same rule as tool search's
discoveries. A strategy that summarizes or drops the exchange therefore unloads the capability: its tools
leave the request and its instructions stop being delivered, while a summary of the same turns may still
say it is loaded. So every shipped strategy that removes history records each such load again after what
replaced it, and core then delivers the capability's current instructions again. With
`preserve_loaded_capabilities=False`, it leaves them unloaded and says so instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256

from typing_extensions import TypeIs

from pydantic_ai._deferred_capabilities import LoadExchanges
from pydantic_ai._utils import is_str_dict
from pydantic_ai.messages import (
    CompactionPart,
    LoadCapabilityCallPart,
    LoadCapabilityReturnPart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    TextContent,
    ToolAvailabilityDeltaPart,
    UserPromptPart,
    post_compaction_window,
)
from pydantic_ai.tools import AgentDepsT, RunContext

RECORDED_LOAD_METADATA_KEY = 'pydantic-ai-harness.compaction.recorded-load.v1'
"""Marks a `load_capability` exchange a compaction recorded, rather than one the model made.

Set on the exchange's `ModelResponse.metadata` and on its `LoadCapabilityReturnPart.metadata`, neither of
which reaches the model. The return's marker is the one that identifies the exchange: its call is the one
with the same `tool_call_id`.
"""

UNLOADED_CAPABILITIES_ORIGIN = 'pydantic-ai-harness.compaction.unloaded-capabilities.v1'
"""The `origin` of the note `preserve_loaded_capabilities=False` leaves, naming the capabilities a compaction unloaded."""


@dataclass(frozen=True)
class _Load:
    capability_id: str
    tool_call_id: str
    returned: LoadCapabilityReturnPart
    tools: list[str] | None


def _loads(messages: Sequence[ModelMessage]) -> dict[str, _Load]:
    """The capabilities loaded in the post-compaction window of *messages*, in load order, with what recorded each load."""
    exchanges = LoadExchanges()
    loads: dict[str, _Load] = {}
    tools: dict[str, list[str]] = {}
    for message in post_compaction_window(messages):
        for part in message.parts:
            if isinstance(part, ToolAvailabilityDeltaPart) and part.tool_call_id is not None:
                tools[part.tool_call_id] = part.tools_added
            elif (capability_id := exchanges.observe(part)) is not None:
                assert isinstance(part, LoadCapabilityReturnPart)
                loads.pop(capability_id, None)
                loads[capability_id] = _Load(capability_id, part.tool_call_id, part, None)
    return {
        capability_id: _Load(capability_id, load.tool_call_id, load.returned, tools.get(load.tool_call_id))
        for capability_id, load in loads.items()
    }


def _still_deferred(capability_id: str, ctx: RunContext[AgentDepsT]) -> bool:
    """Whether a load of *capability_id* still means something to the run.

    Outside a run, as with `compact_now`, there is no registry to ask, and core ignores a load of a
    capability the next run doesn't register.
    """
    if not ctx.capabilities:
        return True
    capability = ctx.capabilities.get(capability_id)
    return capability is not None and capability.defer_loading is True


def _insert_index(compacted: Sequence[ModelMessage], protected_tail: int) -> int:
    """Where records go so they are inside the post-compaction window, without splitting a tool call from its result.

    That's right after the latest compaction part, which a summarizing strategy has just written into a
    request of its own. A provider's compaction part is in a response, which may also hold tool calls, so
    there they go after the requests that follow it. Without one the whole history is the window, and they
    go after the leading requests, ahead of the first response. Either way the newest message stays last.
    """
    end = len(compacted) - max(protected_tail, 1)
    for index in range(len(compacted) - 1, -1, -1):
        if any(isinstance(part, CompactionPart) for part in compacted[index].parts):
            after = index + 1
            if isinstance(compacted[index], ModelResponse):
                while after < end and isinstance(compacted[after], ModelRequest):
                    after += 1
            return after
    leading = next((i for i, message in enumerate(compacted) if isinstance(message, ModelResponse)), len(compacted))
    return min(leading, end)


def _recorded_tool_call_id(load: _Load) -> str:
    """A tool call id for the recorded load, derived from the one it records so a durable replay records the same one."""
    digest = sha256(f'{load.tool_call_id}:{load.capability_id}'.encode()).hexdigest()
    return f'pyd_ai_{digest[:32]}'


def _recorded_exchange(loads: Sequence[_Load]) -> list[ModelMessage]:
    """The `load_capability` exchange recording *loads* again, marked as a compaction's own."""
    calls: list[ModelResponsePart] = []
    results: list[ModelRequestPart] = []
    for load in loads:
        tool_call_id = _recorded_tool_call_id(load)
        calls.append(LoadCapabilityCallPart(args={'id': load.capability_id}, tool_call_id=tool_call_id))
        results.append(
            LoadCapabilityReturnPart(
                content=load.returned.content.copy(),
                tool_call_id=tool_call_id,
                metadata={RECORDED_LOAD_METADATA_KEY: True},
            )
        )
        if load.tools:
            results.append(ToolAvailabilityDeltaPart(tools_added=list(load.tools), tool_call_id=tool_call_id))
    return [
        ModelResponse(parts=calls, metadata={RECORDED_LOAD_METADATA_KEY: True}),
        ModelRequest(parts=results),
    ]


def _unloaded_note(capability_ids: Sequence[str]) -> ModelRequest:
    """The note naming the capabilities a compaction unloaded, as harness data on the user channel."""
    names = ', '.join(repr(capability_id) for capability_id in capability_ids)
    text = (
        f'Compacting the conversation unloaded these capabilities: {names}. Their instructions and tools are '
        'no longer available. Call `load_capability` for any you still need before using its tools.'
    )
    metadata = {'origin': UNLOADED_CAPABILITIES_ORIGIN, 'capabilities': list(capability_ids)}
    return ModelRequest(parts=[UserPromptPart(content=[TextContent(text, metadata=metadata)])])


def keep_capability_loads(
    original: Sequence[ModelMessage],
    compacted: list[ModelMessage],
    ctx: RunContext[AgentDepsT],
    *,
    preserve: bool = True,
    protected_tail: int = 0,
) -> list[ModelMessage]:
    """Record again each deferred capability load *compacted* lost from *original*, or note that it was lost.

    A load is lost when its exchange is in *original*'s post-compaction window but not in *compacted*'s.
    With *preserve*, each is recorded again with the content of its original return, after the compaction
    that replaced it, so core keeps the capability loaded and delivers its current instructions again.
    Without, a note names the capabilities that are no longer loaded. A no-op when nothing was lost.
    `protected_tail` bounds where records go when there is no compaction part, as for `reinject_pinned`.
    """
    after = _loads(compacted)
    lost = [
        load
        for capability_id, load in _loads(original).items()
        if capability_id not in after and _still_deferred(capability_id, ctx)
    ]
    if not lost:
        return compacted
    records = _recorded_exchange(lost) if preserve else [_unloaded_note([load.capability_id for load in lost])]
    index = _insert_index(compacted, protected_tail)
    return [*compacted[:index], *records, *compacted[index:]]


def recorded_load_ids(messages: Sequence[ModelMessage]) -> set[str]:
    """The tool call ids of the `load_capability` exchanges compactions recorded in *messages*."""
    return {
        part.tool_call_id
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if is_recorded_load_return(part)
    }


def is_recorded_load_return(part: ModelRequestPart) -> TypeIs[LoadCapabilityReturnPart]:
    """Whether *part* is the return of a `load_capability` exchange a compaction recorded."""
    return (
        isinstance(part, LoadCapabilityReturnPart)
        and is_str_dict(part.metadata)
        and part.metadata.get(RECORDED_LOAD_METADATA_KEY) is True
    )


def is_recorded_load_part(part: ModelRequestPart | ModelResponsePart, recorded: set[str]) -> bool:
    """Whether *part* belongs to a recorded `load_capability` exchange, given the exchanges' tool call ids."""
    if isinstance(part, LoadCapabilityCallPart | LoadCapabilityReturnPart | ToolAvailabilityDeltaPart):
        return part.tool_call_id in recorded
    return False
