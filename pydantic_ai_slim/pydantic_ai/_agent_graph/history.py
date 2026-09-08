from __future__ import annotations as _annotations

import dataclasses
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import TypeGuard

from .. import messages as _messages

__all__ = (
    'SYNTHESIZED_TOOL_RETURN_METADATA_KEY',
    '_clean_message_history',
    '_dangling_tool_calls_by_response',
    '_first_new_message_index',
    '_first_run_id_index',
    '_repair_dangling_tool_calls',
    'capture_run_messages',
    'get_captured_run_messages',
)


@dataclasses.dataclass
class _RunMessages:
    messages: list[_messages.ModelMessage]
    used: bool = False


_messages_ctx_var: ContextVar[_RunMessages] = ContextVar('var')


@contextmanager
def capture_run_messages() -> Generator[list[_messages.ModelMessage]]:
    """Context manager to access the messages used in a [`run`][pydantic_ai.agent.AbstractAgent.run], [`run_sync`][pydantic_ai.agent.AbstractAgent.run_sync], or [`run_stream`][pydantic_ai.agent.AbstractAgent.run_stream] call.

    Useful when a run may raise an exception, see [model errors](../agent.md#model-errors) for more information.

    Examples:
    ```python
    from pydantic_ai import Agent, capture_run_messages

    agent = Agent('test')

    with capture_run_messages() as messages:
        try:
            result = agent.run_sync('foobar')
        except Exception:
            print(messages)
            raise
    ```

    !!! note
        If you call `run`, `run_sync`, or `run_stream` more than once within a single `capture_run_messages` context,
        `messages` will represent the messages exchanged during the first call only.

        Contexts can be nested: each `capture_run_messages` context captures the runs for which it is the
        innermost active context. A run started inside a nested context is captured by that nested context,
        not by any enclosing one, so wrapping a nested agent run (e.g. inside a tool) in its own
        `capture_run_messages` lets you inspect that inner run's messages independently.

    If a run is interrupted by an exception or cancellation while streaming a response or executing
    tool calls, the partial [`ModelResponse`][pydantic_ai.messages.ModelResponse] or
    [`ModelRequest`][pydantic_ai.messages.ModelRequest] is still captured here with
    `state='interrupted'`, so consumers can detect and inspect partial state.
    """
    messages: list[_messages.ModelMessage] = []
    # Always push a fresh context so nested `capture_run_messages` contexts each capture their own runs,
    # rather than sharing (and overwriting) the enclosing context's messages.
    token = _messages_ctx_var.set(_RunMessages(messages))
    try:
        yield messages
    finally:
        _messages_ctx_var.reset(token)


def get_captured_run_messages() -> _RunMessages:
    return _messages_ctx_var.get()


SYNTHESIZED_TOOL_RETURN_METADATA_KEY = 'pydantic_ai_synthesized_tool_return'
"""Metadata key set to `True` on `ToolReturnPart`s synthesized for tool calls that never received a result."""


def _first_run_id_index(messages: Sequence[_messages.ModelMessage], run_id: str) -> int:
    """Return the index of the first message for the current run, or len(messages) if none are found."""
    for index, message in enumerate(messages):
        if message.run_id == run_id:
            return index
    return len(messages)


def _first_new_message_index(
    messages: list[_messages.ModelMessage],
    run_id: str,
    *,
    resumed_request: _messages.ModelRequest | None,
    resumed_request_index: int | None,
) -> int:
    """Return the first index that should be included in `new_messages()`.

    When resuming from `message_history` without a new user prompt, the trailing
    `ModelRequest` is prior context even though the framework stamps it with the current
    `run_id` for adapter bookkeeping, so it must be excluded. A capability or history processor
    can mutate the message list before this runs, so the resumed request is located by trying
    progressively looser fallbacks, each robust to a different kind of mutation:

    1. Object identity (`is`) — survives reordering, insertion, and removal of *other* messages.
    2. Value match (`_is_same_request`) — survives loss of identity (e.g. a deep-copying
       processor) as long as the request's fields are unchanged.
    3. Position (`resumed_request_index`, pinned while preparing the request) — survives an
       in-place rebuild that changes the request's fields (e.g. system-prompt reinjection),
       which defeats both matches above.

    Falling back to the first message carrying the current `run_id` is the last resort. Note the
    layers cover different *single* mutations: a rebuild that also shifts the request's position
    by adding/removing messages after it on the same step defeats all three, and detection falls
    back to `run_id` (which includes the resumed request); this is rarer than any layer's own
    blind spot and no built-in capability triggers it.
    """
    if resumed_request is not None:
        for index, message in enumerate(messages):
            if message is resumed_request:
                return index + 1

        for index in range(len(messages) - 1, -1, -1):
            if _is_same_request(messages[index], resumed_request):
                return index + 1

    if resumed_request_index is not None and 0 <= resumed_request_index < len(messages):
        return resumed_request_index + 1

    return _first_run_id_index(messages, run_id)


def _is_same_request(message: _messages.ModelMessage, request: _messages.ModelRequest) -> bool:
    if not isinstance(message, _messages.ModelRequest):
        return False
    if message is request:  # pragma: no cover
        return True
    # Intentionally excludes `run_id`: the resumed request may not have `run_id` set yet when
    # this comparison is performed.
    return (
        message.parts == request.parts
        and message.timestamp == request.timestamp
        and message.instructions == request.instructions
        and message.metadata == request.metadata
    )


def _dangling_tool_calls_by_response(messages: list[_messages.ModelMessage]) -> dict[int, list[_messages.ToolCallPart]]:
    """Find tool calls that will never receive a result, keyed by the index of their response.

    Matching is an ordered walk: a tool result (`_is_tool_result_part` — a `ToolReturnPart` or
    *tool-bound* `RetryPromptPart`; plain validation feedback doesn't answer a call even if its
    `tool_call_id` collides) only answers a call that is open (produced by an earlier response and
    not already answered) at that point. An out-of-place result — one preceding its call, a
    duplicate, or one reusing the ID of an already-answered call — doesn't mask a genuinely
    dangling call.
    """
    open_calls: dict[str, tuple[int, _messages.ToolCallPart]] = {}
    dangling_by_response: dict[int, list[_messages.ToolCallPart]] = {}
    for index, message in enumerate(messages):
        if isinstance(message, _messages.ModelResponse):
            for part in message.parts:
                if isinstance(part, _messages.ToolCallPart):
                    if shadowed := open_calls.get(part.tool_call_id):
                        # A new call reusing the ID of an open call means the open call can no
                        # longer be answered: any later result answers the new call instead.
                        dangling_by_response.setdefault(shadowed[0], []).append(shadowed[1])
                    open_calls[part.tool_call_id] = (index, part)
        elif isinstance(message, _messages.ModelRequest):  # pragma: no branch
            for part in message.parts:
                if _is_tool_result_part(part):
                    open_calls.pop(part.tool_call_id, None)
    for response_index, call in open_calls.values():
        dangling_by_response.setdefault(response_index, []).append(call)
    return dangling_by_response


def _insert_synthesized_returns(
    request: _messages.ModelRequest, synthesized: list[_messages.ToolReturnPart]
) -> _messages.ModelRequest:
    """Insert synthesized returns after the request's existing tool results (if any).

    They go ahead of user-facing parts — including a plain (non-tool-bound) `RetryPromptPart`,
    which renders as user text — matching where providers expect tool results.
    """
    insert_at = next(
        (
            part_index + 1
            for part_index in range(len(request.parts) - 1, -1, -1)
            if _is_tool_result_part(request.parts[part_index])
        ),
        0,
    )
    return replace(request, parts=[*request.parts[:insert_at], *synthesized, *request.parts[insert_at:]])


def _is_tool_result_part(
    part: _messages.ModelRequestPart | _messages.ModelResponsePart,
) -> TypeGuard[_messages.ToolReturnPart | _messages.RetryPromptPart]:
    """Whether a part is a regular (locally-executed) tool result answering a `ToolCallPart`.

    A `RetryPromptPart` with no `tool_name` is validation feedback rendered as a plain user message,
    not a tool result, so it doesn't need (or have) a matching tool call. `NativeToolReturnPart` (a
    sibling `BaseToolReturnPart` subclass) is intentionally excluded: native/builtin results are
    co-located with their call in one `ModelResponse` and shaped by each model's own serializer, so
    the pipeline leaves them alone.
    """
    return isinstance(part, _messages.ToolReturnPart) or (
        isinstance(part, _messages.RetryPromptPart) and part.tool_name is not None
    )


def _drop_orphaned_tool_results(messages: list[_messages.ModelMessage]) -> list[_messages.ModelMessage]:
    """REMOVE regular tool results whose call is missing.

    A `ToolReturnPart` or tool-bound `RetryPromptPart` (in a `ModelRequest`) whose `tool_call_id`
    never appeared as a regular `ToolCallPart` in any preceding `ModelResponse` is "orphaned".
    Providers reject a tool result without a matching tool call (Anthropic rejects a `tool_result`
    with no `tool_use`; OpenAI Responses raises `No tool call found for function call output`).
    Orphans arise from context eviction dropping the response that made the call, from a result
    placed before its call in a hand-built history, or from adapter round-trips.

    This operates only on regular, locally-executed tool call/result pairing across message
    boundaries. Native/builtin parts (`NativeToolCallPart`/`NativeToolReturnPart`) are left untouched:
    they're produced and resulted by the provider inline and shaped by each model's own serializer,
    and a native result can even arrive in a *later* response (e.g. Anthropic tool search), so the
    core pipeline must not treat them as droppable.

    Removal, not reordering: a result that precedes its call could in principle be moved after it,
    but that crosses message boundaries and reorders content, so the fundamentally-invalid result is
    dropped instead (its now-unanswered call is later synthesized a result by
    `_repair_dangling_tool_calls`). If dropping empties an interior `ModelRequest` the request is
    dropped; if it empties the last message an empty `ModelRequest` is kept so the history still ends
    on a request. Returns the input unchanged when there are no orphans.
    """
    seen_call_ids: set[str] = set()
    repaired: list[_messages.ModelMessage] = []
    changed = False
    for index, message in enumerate(messages):
        for part in message.parts:
            if isinstance(part, _messages.ToolCallPart):
                seen_call_ids.add(part.tool_call_id)
        kept_parts = [
            part
            for part in message.parts
            if not (_is_tool_result_part(part) and part.tool_call_id not in seen_call_ids)
        ]
        if len(kept_parts) == len(message.parts):
            repaired.append(message)
            continue
        changed = True
        if kept_parts or (isinstance(message, _messages.ModelRequest) and index == len(messages) - 1):
            repaired.append(replace(message, parts=kept_parts))
        # else: interior emptied `ModelRequest` — drop the message
    return repaired if changed else messages


def _repair_dangling_tool_calls(
    messages: list[_messages.ModelMessage], *, repair_last_response: bool = False
) -> list[_messages.ModelMessage]:
    """Repair tool calls that are missing a matching result ("dangling" tool calls).

    A run that was cancelled or crashed mid-tool-execution — or a hand-built history — can contain
    `ToolCallPart`s with no matching `ToolReturnPart`/`RetryPromptPart` in a later `ModelRequest`.
    Providers reject histories with dangling tool calls, so before a request is sent, every dangling
    tool call gets a synthesized `ToolReturnPart` — marked with `SYNTHESIZED_TOOL_RETURN_METADATA_KEY`
    in its `metadata` — inserted after the existing tool returns of the immediately following
    `ModelRequest`, or as a new `ModelRequest` when no request follows.

    This includes a call whose args string was cut off mid-stream (unparsable JSON): the call is
    kept verbatim and closed out like any other dangling call, never removed. Malformed args are
    already sendable — serializers degrade them gracefully (see `ToolCallPart.args_as_dict` and
    `ToolCallPart.args_as_json_str`), as the tool-call retry flow relies on — and removing the call
    would disturb the response's shape, e.g. leaving a thinking-only response whose signature was
    computed over a turn that included the call.

    The last `ModelResponse` is only repaired when `repair_last_response` is set: its tool calls
    are the live frontier that run resumption and `deferred_tool_results` may still answer, and a
    trailing unparsable-args call is left for local args validation to turn into a retry prompt.

    Matching is an ordered walk: a result only answers a call that is open (produced by an earlier
    response and not already answered) at that point. An out-of-place result — one preceding its
    call, a duplicate, or one reusing the ID of an already-answered call — doesn't mask a genuinely
    dangling call; such orphaned results themselves are not repaired.

    The repair is deterministic and idempotent: synthesized parts derive their timestamp from the
    response they repair and contain no wall-clock or random data, so repairing the same history
    twice (or on every run) yields the same output and never churns provider prompt-cache prefixes.
    If there is nothing to repair, the input list is returned unchanged. Repair is silent — like
    the other pipeline passes — with the `SYNTHESIZED_TOOL_RETURN_METADATA_KEY` marker as the
    mechanism for inspecting what was synthesized.
    """
    dangling_by_response = _dangling_tool_calls_by_response(messages)
    if not repair_last_response:
        last_response_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if isinstance(messages[index], _messages.ModelResponse)
            ),
            None,
        )
        if last_response_index is not None:
            dangling_by_response.pop(last_response_index, None)
    if not dangling_by_response:
        return messages

    repaired: list[_messages.ModelMessage] = []
    synthesized: list[_messages.ToolReturnPart] = []
    for index, message in enumerate(messages):
        if isinstance(message, _messages.ModelResponse):
            if synthesized:
                # The dangling calls of the previous response are followed by another response,
                # so the synthesized returns need a new request in between.
                repaired.append(_messages.ModelRequest(parts=synthesized))
                synthesized = []

            if dangling := dangling_by_response.get(index):
                for call in dangling:
                    synthesized.append(
                        _messages.ToolReturnPart(
                            tool_name=call.tool_name,
                            content=_messages.INTERRUPTED_TOOL_RETURN_CONTENT,
                            tool_call_id=call.tool_call_id,
                            metadata={SYNTHESIZED_TOOL_RETURN_METADATA_KEY: True},
                            timestamp=message.timestamp,
                            outcome='interrupted',
                        )
                    )
            repaired.append(message)
        elif isinstance(message, _messages.ModelRequest):  # pragma: no branch
            if synthesized:
                message = _insert_synthesized_returns(message, synthesized)
                synthesized = []
            repaired.append(message)

    if synthesized:
        repaired.append(_messages.ModelRequest(parts=synthesized))

    return repaired


def _merge_consecutive_messages(messages: list[_messages.ModelMessage]) -> list[_messages.ModelMessage]:
    """Normalize the history's shape by merging consecutive same-role messages into one.

    Neither adds nor removes content — it only combines adjacent `ModelRequest`s (or adjacent
    synthetic `ModelResponse`s) that providers expect as a single turn, and within a merged request
    hoists tool results ahead of user-facing parts (where providers require them). Runs last, after
    the repair passes have settled call/result pairing, so it operates on a valid history and never
    separates a result from the call it answers.
    """
    clean_messages: list[_messages.ModelMessage] = []
    for message in messages:
        last_message = clean_messages[-1] if len(clean_messages) > 0 else None

        if isinstance(message, _messages.ModelRequest):
            if (
                last_message
                and isinstance(last_message, _messages.ModelRequest)
                # Requests can only be merged if they have the same instructions
                and (
                    not last_message.instructions
                    or not message.instructions
                    or last_message.instructions == message.instructions
                )
                # We intentionally don't block merging when `conversation_id` or `metadata` differ,
                # nor try to preserve them across the merge. These fields are only bookkeeping for
                # callers; they're never part of what gets sent to the model. Refusing to merge on a
                # mismatch would leave two consecutive requests where the model expects one, breaking
                # providers (and provider-side conversation state) that require a single request per
                # turn -- a real regression -- just to preserve fields the model request node never reads.
            ):
                parts = [*last_message.parts, *message.parts]
                parts.sort(key=_messages._tool_results_first_sort_key)  # pyright: ignore[reportPrivateUsage]
                merged_message = _messages.ModelRequest(
                    parts=parts,
                    instructions=last_message.instructions or message.instructions,
                    timestamp=message.timestamp or last_message.timestamp,
                )
                clean_messages[-1] = merged_message
            else:
                clean_messages.append(message)
        elif isinstance(message, _messages.ModelResponse):  # pragma: no branch
            if (
                last_message
                and isinstance(last_message, _messages.ModelResponse)
                # Responses can only be merged if they didn't really come from an API
                and last_message.provider_response_id is None
                and last_message.provider_name is None
                and last_message.model_name is None
                and message.provider_response_id is None
                and message.provider_name is None
                and message.model_name is None
            ):
                merged_message = replace(last_message, parts=[*last_message.parts, *message.parts])
                clean_messages[-1] = merged_message
            else:
                clean_messages.append(message)
    return clean_messages


def _clean_message_history(
    messages: list[_messages.ModelMessage], *, repair_last_response: bool = False
) -> list[_messages.ModelMessage]:
    """Make the message history provider-valid and normalize its shape, out of the box.

    An ordered pipeline of pure `list[ModelMessage] -> list[ModelMessage]` passes over regular,
    locally-executed tool call/result pairing across message boundaries. Following the principle
    "massage the history however we can to make the model API accept it, and drop only what's
    fundamentally unsendable", each pass ADDs (synthesizes) or REMOVEs content, never silently
    dropping anything a provider could accept. Native/builtin parts are left entirely untouched:
    they're produced and resulted by the provider inline and shaped by each model's own serializer
    (which handles their own dangling/empty-id cases), and a native result can even arrive in a
    later response, so the core pipeline must not touch them. Ordering matters:

    1. `_drop_orphaned_tool_results` (REMOVE) — first, so an orphaned result can't survive into the
       merge (which would hoist it to the front of a request) and so dropping it can expose a call
       that then needs a synthesized result in pass 2.
    2. `_repair_dangling_tool_calls` (ADD synthesized results) — the matching-graph repair; runs
       before the merge changes message boundaries. Frontier-gated by `repair_last_response` so
       the last response's still-answerable calls are left alone.
    3. `_merge_consecutive_messages` (normalize) — last, once call/result pairing is valid, so it
       never separates a result from its call.
    """
    messages = _drop_orphaned_tool_results(messages)
    messages = _repair_dangling_tool_calls(messages, repair_last_response=repair_last_response)
    messages = _merge_consecutive_messages(messages)
    return messages
