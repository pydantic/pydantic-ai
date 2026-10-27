"""Keep deferred capabilities loaded across compaction, or tell the model which ones it unloaded.

Core counts a deferred capability as loaded while its `load_capability` exchange is in the
[`post_compaction_window`][pydantic_ai.messages.post_compaction_window], the same rule as tool search's
discoveries. A strategy that summarizes or drops the exchange therefore unloads the capability: its tools
leave the request and its instructions stop being delivered, while a summary of the same turns may still
say it is loaded. So every shipped strategy that removes history records each such load again after what
replaced it, and core then delivers the capability's current instructions again. With
`preserve_loaded_capabilities=False`, it leaves them unloaded and says so instead.

This module imports only Pydantic AI, and `keep_capability_loads` is a plain function of messages and
capability ids, so it runs workflow-side in a durable run and can move into core's compaction step
(#11454), which will call it once per attempt after whichever compaction ran, a provider's included.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from hashlib import sha256

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
    returned: LoadCapabilityReturnPart
    tools: list[str] | None


def _loads(messages: Sequence[ModelMessage]) -> dict[str, _Load]:
    """The capabilities loaded in the post-compaction window of *messages*, in load order, with what recorded each load."""
    exchanges = LoadExchanges()
    returns: dict[str, LoadCapabilityReturnPart] = {}
    tools: dict[str, list[str]] = {}
    for message in post_compaction_window(messages):
        for part in message.parts:
            if isinstance(part, ToolAvailabilityDeltaPart) and part.tool_call_id is not None:
                tools[part.tool_call_id] = part.tools_added
            elif (capability_id := exchanges.observe(part)) is not None:
                assert isinstance(part, LoadCapabilityReturnPart)
                # Moved to the end, so a capability loaded again keeps its latest place in the load order.
                returns.pop(capability_id, None)
                returns[capability_id] = part
    return {
        capability_id: _Load(capability_id, returned, tools.get(returned.tool_call_id))
        for capability_id, returned in returns.items()
    }


def deferred_capability_ids(ctx: RunContext[AgentDepsT]) -> set[str] | None:
    """The ids of the run's deferred capabilities, whose loads mean something to it, or `None` outside a run.

    Outside a run, as with `compact_now`, there is no registry to ask; core ignores a load of a capability
    the next run doesn't register. A stand-in context without a registry at all counts as outside a run.
    """
    capabilities = getattr(ctx, 'capabilities', None)
    if not capabilities:
        return None
    return {capability_id for capability_id, capability in capabilities.items() if capability.defer_loading}


def _insert_index(compacted: Sequence[ModelMessage], kept_from: int) -> int:
    """Where records go so they are inside the post-compaction window, without splitting a tool call from its result.

    That's right after the latest compaction part, which a summarizing strategy writes into a request of its
    own. A provider's compaction part is in a response, which may also hold tool calls (or, like Anthropic's
    signed block, must stay first), so there they go after it and the requests that follow it. Without one the
    whole history is the window, and they go after the leading requests and ahead of the kept messages, which
    start at *kept_from*. Either way the newest message stays last.

    TODO(#7449): a sliding-window truncation marker will be a `CompactionPart` too, and then counts as a
    boundary here like a summary does.
    """
    last = len(compacted) - 1
    for index in range(last, -1, -1):
        if any(isinstance(part, CompactionPart) for part in compacted[index].parts):
            after = index + 1
            if isinstance(compacted[index], ModelResponse):
                while after < last and isinstance(compacted[after], ModelRequest):
                    after += 1
            return after
    leading = next((i for i, message in enumerate(compacted) if isinstance(message, ModelResponse)), len(compacted))
    return min(leading, kept_from, last)


def _recorded_tool_call_id(load: _Load) -> str:
    """A tool call id for the recorded load, derived from the one it records so a durable replay records the same one."""
    digest = sha256(f'{load.returned.tool_call_id}:{load.capability_id}'.encode()).hexdigest()
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


def _noted_unloaded(messages: Sequence[ModelMessage]) -> set[str]:
    """The capabilities an unloaded-capabilities note in the post-compaction window of *messages* names."""
    noted: set[str] = set()
    for message in post_compaction_window(messages):
        for part in message.parts:
            if (
                isinstance(part, UserPromptPart)
                and not isinstance(part.content, str)
                and part.content
                and isinstance(note := part.content[0], TextContent)
                and is_str_dict(note.metadata)
                and note.metadata.get('origin') == UNLOADED_CAPABILITIES_ORIGIN
            ):
                noted.update(_unloaded_note_ids(note.metadata.get('capabilities')))
    return noted


def _unloaded_note_ids(value: object) -> list[str]:
    """The capability ids a note's `capabilities` metadata names."""
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []  # pyright: ignore[reportUnknownVariableType]


def keep_capability_loads(
    original: Sequence[ModelMessage],
    compacted: list[ModelMessage],
    *,
    kept_from: int,
    deferred_ids: Collection[str] | None,
    preserve: bool = True,
) -> list[ModelMessage]:
    """Record again each deferred capability load *compacted* lost from *original*, or note that it was lost.

    A load is lost when its exchange is in *original*'s post-compaction window but not in *compacted*'s, and
    its capability is in *deferred_ids*, the capabilities the run still registers as deferred (`None` when
    that's unknown, which keeps every load). With *preserve*, each is recorded again with the content of its
    original return, after the compaction that replaced it, so core keeps the capability loaded and delivers
    its current instructions again. Without, a note names the capabilities that are no longer loaded.
    *kept_from* is the index in *compacted* where the kept messages start, which records go ahead of when
    there is no compaction part.

    It depends only on the two histories, not on how one became the other: the latest compaction part may
    be a summary a strategy wrote into a request or a provider's in a response. A no-op when nothing was
    lost, including on its own result, so calling it again on what it returned changes nothing.

    TODO: tool search discoveries follow the same window rule and are lost the same way (#11264). A local
    `search_tools` exchange could be recorded again like a load; a provider's native search can't be, so
    that's left for the core compaction step (#11454) to settle.
    """
    after = _loads(compacted)
    lost = [
        load
        for capability_id, load in _loads(original).items()
        if capability_id not in after and (deferred_ids is None or capability_id in deferred_ids)
    ]
    if not preserve:
        noted = _noted_unloaded(compacted)
        lost = [load for load in lost if load.capability_id not in noted]
    if not lost:
        return compacted
    records = _recorded_exchange(lost) if preserve else [_unloaded_note([load.capability_id for load in lost])]
    index = _insert_index(compacted, kept_from)
    return [*compacted[:index], *records, *compacted[index:]]


def is_load_part(part: ModelRequestPart | ModelResponsePart) -> bool:
    """Whether *part* records which capabilities are loaded: a `load_capability` call or return, or a tool reveal.

    The summarizer leaves these out: load state is carried by the exchanges themselves, recorded again after
    each compaction, or by the note saying they were unloaded, never by the summary.
    """
    return isinstance(part, LoadCapabilityCallPart | LoadCapabilityReturnPart | ToolAvailabilityDeltaPart)
