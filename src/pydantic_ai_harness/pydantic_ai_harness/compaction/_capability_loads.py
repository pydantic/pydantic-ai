"""Keep deferred capabilities loaded and discovered tools discovered across compaction.

Core counts a deferred capability as loaded while the record of its load is in the
[`post_compaction_window`][pydantic_ai.messages.post_compaction_window], the same rule as tool search's
discoveries: the model's `load_capability` exchange, or else a `CapabilityLoadDeltaPart`, which a load from
code leaves. A strategy that summarizes or drops the record therefore unloads the capability: its tools
leave the request and its instructions stop being delivered, while a summary of the same turns may still
say it is loaded. So every shipped strategy that removes history records each such load again after what
replaced it, as a `CapabilityLoadDeltaPart` rather than an exchange the model never made, and core then
delivers the capability's current instructions again. With `preserve_loaded_capabilities=False`, it leaves
them unloaded and says so instead.

This module imports only Pydantic AI, and `keep_loaded_state` is a plain function of messages and
capability ids, so it runs workflow-side in a durable run and can move into core's compaction step
(#11454), which will call it once per attempt after whichever compaction ran, a provider's included.
"""

from __future__ import annotations

from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass

from pydantic_ai._deferred_capabilities import LoadExchanges
from pydantic_ai._utils import is_str_dict
from pydantic_ai.messages import (
    CapabilityLoadDeltaPart,
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
from pydantic_ai.toolsets._tool_search import discovered_tool_names_in_order, parse_discovered_tools

UNLOADED_CAPABILITIES_ORIGIN = 'pydantic-ai-harness.compaction.unloaded-capabilities.v1'
"""The `origin` of the note `preserve_loaded_capabilities=False` leaves, naming the capabilities a compaction unloaded."""


@dataclass(frozen=True)
class _Load:
    capability_id: str
    tools: list[str] | None
    """The tools the load revealed, if any."""


def _loads(messages: Sequence[ModelMessage]) -> dict[str, _Load]:
    """The capabilities loaded in the post-compaction window of *messages*, in load order, with the tools each revealed.

    A load is the model's `load_capability` exchange or a `CapabilityLoadDeltaPart`. A model's load reveals its
    tools in a `ToolAvailabilityDeltaPart` with its call's `tool_call_id`, and a load from code in the one right
    after its delta with the same `tool_call_id`: that of the tool call it was made in, or `None`. A load
    without tools leaves no reveal, so adjacency alone would credit it with someone else's.
    """
    exchanges = LoadExchanges()
    loads: dict[str, _Load] = {}
    calls: dict[str, str] = {}
    """The capability each completed `load_capability` call loaded, by its `tool_call_id`."""
    for part, previous in _parts_with_previous(post_compaction_window(messages)):
        if isinstance(part, ToolAvailabilityDeltaPart) and (owner := _reveal_owner(part, previous, calls)):
            loads[owner] = _Load(owner, part.tools_added)
        for capability_id in exchanges.loads(part):
            # Moved to the end, so a capability loaded again keeps its latest place in the load order.
            loads.pop(capability_id, None)
            loads[capability_id] = _Load(capability_id, None)
            if isinstance(part, LoadCapabilityReturnPart):
                calls[part.tool_call_id] = capability_id
    return loads


def _parts_with_previous(
    messages: Sequence[ModelMessage],
) -> Iterator[tuple[ModelRequestPart | ModelResponsePart, ModelRequestPart | ModelResponsePart | None]]:
    """Each part of *messages* with the one before it in the same message, if any."""
    for message in messages:
        previous: ModelRequestPart | ModelResponsePart | None = None
        for part in message.parts:
            yield part, previous
            previous = part


def _reveal_owner(
    reveal: ToolAvailabilityDeltaPart, previous: ModelRequestPart | ModelResponsePart | None, calls: dict[str, str]
) -> str | None:
    """The capability whose load *reveal* belongs to, by `tool_call_id`: a load from code in *previous*, or a model's load."""
    if (
        isinstance(previous, CapabilityLoadDeltaPart)
        and previous.loaded
        and reveal.tool_call_id == previous.tool_call_id
    ):
        return previous.loaded[-1]
    return calls.get(reveal.tool_call_id) if reveal.tool_call_id is not None else None


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


def _recorded_loads(loads: Sequence[_Load]) -> ModelRequest:
    """The deltas recording *loads* again: per load, a `CapabilityLoadDeltaPart` and the reveal of its tools, if any.

    The records a load from code leaves, without a tool call: the model made no exchange after the compaction.
    """
    parts: list[ModelRequestPart] = []
    for load in loads:
        parts.append(CapabilityLoadDeltaPart(loaded=[load.capability_id]))
        if load.tools:
            parts.append(ToolAvailabilityDeltaPart(tools_added=list(load.tools)))
    return ModelRequest(parts=parts)


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

    A load is lost when its record, the model's `load_capability` exchange or a `CapabilityLoadDeltaPart`, is
    in *original*'s post-compaction window but not in *compacted*'s, and its capability is in *deferred_ids*,
    the capabilities the run still registers as deferred (`None` when that's unknown, which keeps every load).
    With *preserve*, each is recorded again after the compaction that replaced it, in load order, as a
    `CapabilityLoadDeltaPart` followed by the `ToolAvailabilityDeltaPart` revealing its tools, if the load
    revealed any, so core keeps the capability loaded and delivers its current instructions again. Without, a
    note names the capabilities that are no longer loaded.
    *kept_from* is the index in *compacted* where the kept messages start, which records go ahead of when
    there is no compaction part.

    It depends only on the two histories, not on how one became the other: the latest compaction part may
    be a summary a strategy wrote into a request or a provider's in a response. A no-op when nothing was
    lost, including on its own result, so calling it again on what it returned changes nothing.
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
    record = _recorded_loads(lost) if preserve else _unloaded_note([load.capability_id for load in lost])
    index = _insert_index(compacted, kept_from)
    return [*compacted[:index], record, *compacted[index:]]


def _load_reveals(messages: Sequence[ModelMessage]) -> set[str]:
    """The tools a `load_capability` exchange in the post-compaction window of *messages* revealed."""
    window = post_compaction_window(messages)
    loads = {
        part.tool_call_id for message in window for part in message.parts if isinstance(part, LoadCapabilityReturnPart)
    }
    return {
        name
        for message in window
        for part in message.parts
        if isinstance(part, ToolAvailabilityDeltaPart) and part.tool_call_id in loads
        for name in part.tools_added
    }


def keep_tool_discoveries(
    original: Sequence[ModelMessage], compacted: list[ModelMessage], *, kept_from: int
) -> list[ModelMessage]:
    """Record again each tool discovery *compacted* lost from *original*, as a `ToolAvailabilityDeltaPart`.

    A discovery is lost when *original*'s post-compaction window records it and *compacted*'s doesn't, whether
    it was a local `search_tools` exchange, a provider's native search result or a tool's `ToolReturn(tools=...)`.
    The reveal names the lost tools in the order they were discovered and goes where `keep_capability_loads`
    puts its records. A tool a capability load revealed is left to that load: it is recorded with it, or
    unloaded with it. A no-op when nothing was lost, including on its own result.
    """
    # Every reveal counts, a plain tool's cache-safe addition included: each records what the model was shown.
    # Tools the run no longer registers aren't filtered out, as adapters drop names a request doesn't declare.
    after = parse_discovered_tools(compacted)
    by_loads = _load_reveals(original)
    lost = [name for name in discovered_tool_names_in_order(original) if name not in after and name not in by_loads]
    if not lost:
        return compacted
    index = _insert_index(compacted, kept_from)
    return [*compacted[:index], ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=lost)]), *compacted[index:]]


def keep_loaded_state(
    original: Sequence[ModelMessage],
    compacted: list[ModelMessage],
    *,
    kept_from: int,
    deferred_ids: Collection[str] | None,
    preserve: bool = True,
) -> list[ModelMessage]:
    """Keep the capability loads and tool discoveries *compacted* lost from *original*.

    See `keep_capability_loads` and `keep_tool_discoveries`; *kept_from* is the index in *compacted* where the
    kept messages start.
    """
    result = keep_capability_loads(
        original, compacted, kept_from=kept_from, deferred_ids=deferred_ids, preserve=preserve
    )
    return keep_tool_discoveries(original, result, kept_from=kept_from + len(result) - len(compacted))


def is_load_part(part: ModelRequestPart | ModelResponsePart) -> bool:
    """Whether *part* records which capabilities are loaded: a `load_capability` call or return, a load from code, or a tool reveal.

    The summarizer leaves these out: load state is carried by the exchanges themselves, recorded again after
    each compaction, or by the note saying they were unloaded, never by the summary.
    """
    return isinstance(
        part,
        LoadCapabilityCallPart | LoadCapabilityReturnPart | CapabilityLoadDeltaPart | ToolAvailabilityDeltaPart,
    )
