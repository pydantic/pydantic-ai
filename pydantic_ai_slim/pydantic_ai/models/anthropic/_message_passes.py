"""Passes over the whole mapped Anthropic message list, run once every message is mapped."""

from __future__ import annotations as _annotations

from typing import Any, cast

from anthropic.types.beta import BetaContentBlockParam, BetaMessageParam, BetaTextBlockParam

from ..._utils import is_str_dict
from ._cache_control import (
    ANTHROPIC_CACHEABLE_PARAM_TYPES,
    add_cache_control_param,
    is_effort_entry,
    last_message_content,
)
from ._containers import is_tool_result_only

_USER_ANCHOR_TEXT = '.'
"""The minimal user turn a `system` entry is given when nothing else precedes it.

The API takes a `system` entry only after a user turn (or an assistant turn ending in a server tool
result), so an instruction enqueued after the model's last response has nothing legal to sit behind.
A single period is the cheapest thing that satisfies the rule while asserting nothing on the user's
behalf; the alternative is degrading the instruction to `<system>`-tagged text, which the recordings
show the model reads as a preference it may overrule.
"""


def drop_unpaired_native_tool_calls(anthropic_messages: list[BetaMessageParam]) -> None:
    """Drop native tool calls that Anthropic will reject without a rendered result.

    A result can arrive in a later response, or exist as a `NativeToolReturnPart` but fail to render.
    Anthropic accepts an unpaired native call only while every later message is a user turn containing
    only concurrent client-tool results. That shape is `is_tool_result_only`, shared with the
    `container_upload` injection so the two cannot drift.
    """
    returned_native_tool_call_ids: set[str] = set()
    for message in anthropic_messages:
        content = message['content']
        assert isinstance(content, list)
        for block in content:
            if is_str_dict(block) and (tool_use_id := block.get('tool_use_id')) is not None:
                returned_native_tool_call_ids.add(tool_use_id)

    suffix_is_tool_result_only = True
    for index in range(len(anthropic_messages) - 1, -1, -1):
        message = anthropic_messages[index]
        content = message['content']
        assert isinstance(content, list)
        kept: list[BetaContentBlockParam] = []
        for block in content:
            # Preserve the union type before `is_str_dict` narrows `block` to `dict[str, Any]`.
            block_param = block
            # Every block `_map_message` emits is a request param TypedDict.
            assert is_str_dict(block)

            unpaired = (
                block['type'] in ('server_tool_use', 'mcp_tool_use')
                and block['id'] not in returned_native_tool_call_ids
            )
            if unpaired and not suffix_is_tool_result_only:
                # Keep a cache boundary on the nearest preceding cacheable block. Moving it
                # forward would cache content the user placed outside the boundary. If no such
                # block exists, the boundary disappears with the block that carried it.
                if (cache_control := block.get('cache_control')) is not None:
                    carriers: list[BetaContentBlockParam] = []
                    for preceding_content in [
                        preceding_message['content'] for preceding_message in anthropic_messages[:index]
                    ] + [kept]:
                        assert isinstance(preceding_content, list)
                        for param in preceding_content:
                            # Preserve the union type before `is_str_dict` narrows `param`.
                            param_value = param
                            if is_str_dict(param) and param['type'] in ANTHROPIC_CACHEABLE_PARAM_TYPES:
                                carriers.append(param_value)
                    if carriers:
                        add_cache_control_param(carriers, cache_control)
                continue
            kept.append(block_param)

        if len(kept) != len(content):
            if not kept:
                # Anthropic rejects empty messages; an assistant turn containing only the
                # dropped call has nothing left to send.
                del anthropic_messages[index]
                continue
            message['content'] = kept

        suffix_is_tool_result_only = (
            suffix_is_tool_result_only and message['role'] == 'user' and is_tool_result_only(kept)
        )


def anchor_system_messages(anthropic_messages: list[BetaMessageParam]) -> None:
    """Give each `system` section a user turn to follow, if it doesn't already have one.

    This runs *after* `place_system_messages_before_generation`, on final positions, because an
    entry's predecessor isn't known until the slide has finished with it. Anchoring while mapping
    instead put the anchor between an assistant `tool_use` and the `tool_result` that answers it,
    whenever the two were separated by a system-only request: the slide then moved the entry past the
    result and left the anchor stranded, and the API rejects that with `tool_use ids were found
    without tool_result blocks immediately after`. Deciding here, the same history needs no anchor at
    all — the entry lands behind the tool result, which is a user turn.

    Walking in reverse keeps the untouched indexes valid. A `system` entry behind another only has to
    satisfy the rule as a section, so it defers to the entry ahead of it. An assistant turn ending in
    a server tool result is a legal predecessor too, but it gets an anchor anyway rather than a live
    verification we can't cheaply record: the cost is one wasted turn in a shape that needs a response
    truncated right after a server tool call, and the alternative is an unproven exception.
    """
    for index in range(len(anthropic_messages) - 1, -1, -1):
        # An effort-only entry can sit anywhere, so it neither moves nor needs an anchor.
        if anthropic_messages[index]['role'] != 'system' or is_effort_entry(anthropic_messages[index]):
            continue
        if index > 0 and anthropic_messages[index - 1]['role'] in ('user', 'system'):
            continue
        anthropic_messages.insert(
            index, {'role': 'user', 'content': [BetaTextBlockParam(text=_USER_ANCHOR_TEXT, type='text')]}
        )


def place_system_messages_before_generation(anthropic_messages: list[BetaMessageParam]) -> None:
    """Move each `system` entry to just before the assistant turn it governs.

    A `{'role': 'system'}` entry is accepted only immediately before an assistant turn or at the end
    of the array. An entry that a *user* turn ended up behind is therefore illegal where it stands —
    which can't be decided while mapping, because a `ModelResponse` whose parts all drop out (an
    empty `TextPart`, an orphan tool-search call) renders to nothing, so reading ahead in the message
    list calls placements legal that the wire rejects.

    Sliding it forward past those user turns rather than degrading it keeps it an operator
    instruction. It costs nothing semantically: an instruction only ever governs the generation that
    follows it, and that's the same assistant turn either way — the entry just stops matching the
    position it was authored at. This is the same trade the mapping loop makes inside a single
    request, where `[user, system, user]` parts render as `user, user, system`.

    Only *user* turns are hopped over, so an earlier instruction stops behind a later one instead of
    overtaking it and inverting the order they were given in. A `system` entry directly after another
    is a placement the API accepts — the group as a whole still precedes the generation.

    Walking in reverse keeps the indexes still to be visited valid, since moving an entry later never
    disturbs anything before it. The mapping loop is the only thing that emits the `system` role, so
    scanning for it finds exactly the entries it wrote, with no bookkeeping to keep in step.
    """
    for index in range(len(anthropic_messages) - 1, -1, -1):
        # An effort-only entry can sit anywhere, so it neither moves nor needs an anchor.
        if anthropic_messages[index]['role'] != 'system' or is_effort_entry(anthropic_messages[index]):
            continue
        target = index + 1
        while target < len(anthropic_messages) and anthropic_messages[target]['role'] == 'user':
            target += 1
        if target == index + 1:
            continue
        leave_cache_boundary_behind(anthropic_messages, index)
        anthropic_messages.insert(target - 1, anthropic_messages.pop(index))


def leave_cache_boundary_behind(anthropic_messages: list[BetaMessageParam], index: int) -> None:
    """Keep a sliding `system` entry's cache boundary at the position it was authored at.

    A `CachePoint` that ends a request lands on the entry's final block, because the entry renders
    after the request's user blocks even though the instruction was authored before them. Carrying it
    along on the slide would drag the boundary over the turns the entry hops and cache content nobody
    marked, so the boundary stays and the entry travels without it. What that costs is the instruction
    itself, which is the one thing that can't be both inside the boundary and after the turns it now
    follows; the cached prefix stays a prefix of what was authored before the marker.

    The block it moves to is the end of the message the entry currently sits behind — which the entry
    is about to stop sitting behind — so an unmoved entry never reaches here and never loses its
    boundary. `add_cache_control_param` raises if that block can't carry one, or if there's nothing
    behind the entry at all, which is the same `CachePoint` with nothing to attach to that the mapping
    loop raises on.
    """
    content = anthropic_messages[index]['content']
    if not isinstance(content, list):  # pragma: no cover
        return
    last_block = cast(dict[str, Any], content[-1])
    if (cache_control := last_block.pop('cache_control', None)) is None:
        return
    add_cache_control_param(last_message_content(anthropic_messages[:index]), cache_control)
