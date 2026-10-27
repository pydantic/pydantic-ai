"""Placing and checking `cache_control` breakpoints on mapped Anthropic content blocks."""

from __future__ import annotations as _annotations

from typing import Any, Literal, cast

from anthropic.types.beta import (
    BetaCacheControlEphemeralParam,
    BetaContentBlockParam,
    BetaMessageParam,
    BetaTextBlockParam,
    BetaToolUnionParam,
)

from ..._utils import is_str_dict
from ...exceptions import UserError
from .. import (
    _request_scoped_suffix_start,  # pyright: ignore[reportPrivateUsage]
)
from .._prompt_cache import raise_earlier_cache_ttls

# See https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#what-can-be-cached
# `tool_addition` accepted and honored verified live on `claude-opus-4-8`: a boundary on the block
# writes and reads back the full prefix. It can end a mid-conversation `system` entry, where a
# terminal `CachePoint`'s boundary lands on the entry's final block. `mcp_tool_result` verified the
# same way on `claude-opus-4-8` and `claude-sonnet-4-5`: a replayed MCP tool result can end the
# cached prefix, which is where a `CachePoint` opening the next user message puts its boundary.
ANTHROPIC_CACHEABLE_PARAM_TYPES = frozenset(
    {'text', 'tool_use', 'server_tool_use', 'image', 'tool_result', 'document', 'tool_addition', 'mcp_tool_result'}
)


def is_effort_entry(message: BetaMessageParam) -> bool:
    """Whether `message` is an effort-only `system` entry, which has no content block to carry `cache_control`."""
    return 'output_config' in message


def add_cache_control_param(params: list[BetaContentBlockParam], cache_control: BetaCacheControlEphemeralParam) -> None:
    """Attach an already-built `cache_control` to the last content block param.

    See https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching for more information.
    """
    if not params:
        raise UserError(
            'CachePoint cannot be the first content in a user message - there must be previous content to attach the CachePoint to. '
            'To cache system instructions or tool definitions, use the `anthropic_cache_instructions` or `anthropic_cache_tool_definitions` settings instead.'
        )

    # Cast needed because BetaContentBlockParam is a union including response Block types (Pydantic models)
    # that don't support dict operations, even though at runtime we only have request Param types (TypedDicts).
    last_param = cast(dict[str, Any], params[-1])
    if last_param['type'] not in ANTHROPIC_CACHEABLE_PARAM_TYPES:
        raise UserError(f'Cache control not supported for param type: {last_param["type"]}')

    last_param['cache_control'] = cache_control


def last_message_content(anthropic_messages: list[BetaMessageParam]) -> list[BetaContentBlockParam]:
    """The content blocks of the last rendered message that isn't request-scoped, or an empty list if there's nothing to attach to.

    Only used to give a leading `CachePoint` somewhere to land. A `str` content body can't carry
    `cache_control`, and neither can a conversation that hasn't rendered a message yet, so both return
    empty and let the caller raise the error that explains the situation.
    """
    # A request-scoped entry can't carry `cache_control`, and is never part of a cached prefix anyway.
    message = next(
        (
            m
            for m in reversed(anthropic_messages)
            if m.get('clear_at') != 'next_user_message' and not is_effort_entry(m)
        ),
        None,
    )
    if message is None:
        return []
    content = message['content']
    # Returned as-is, not copied: the caller attaches `cache_control` by mutating the block in place, so
    # it has to be the list the message actually holds.
    return content if isinstance(content, list) else []


def blocks_before_request_scoped_text(
    message: BetaMessageParam, texts: tuple[str, ...]
) -> tuple[list[BetaContentBlockParam], tuple[str, ...]]:
    """`message`'s content blocks before the trailing text blocks rendering the last of `texts`, and the texts left over.

    The blocks are the message's own, so `cache_control` added to one lands on the wire. Every message
    `_map_message` builds has a list body.
    """
    blocks = cast(list[BetaContentBlockParam], message['content'])
    end, remaining = _request_scoped_suffix_start(blocks, texts, text_block_text)
    return blocks[:end], remaining


def text_block_text(block: BetaContentBlockParam) -> str | None:
    return block.get('text') if is_str_dict(block) and block.get('type') == 'text' else None


def raise_earlier_breakpoint_ttls(
    system_prompt: str | list[BetaTextBlockParam],
    anthropic_messages: list[BetaMessageParam],
    tools: list[BetaToolUnionParam],
    automatic_cache_control: BetaCacheControlEphemeralParam | None,
) -> None:
    """Raise each breakpoint's TTL to the longest TTL of a breakpoint after it, as Anthropic requires.

    Anthropic processes `tools`, then `system`, then `messages`, and the top-level automatic breakpoint lands on the
    last cacheable block, after all of them.
    """
    blocks: list[object] = [*tools, *(system_prompt if isinstance(system_prompt, list) else [])]
    for message in anthropic_messages:
        content = message['content']
        blocks.extend([] if isinstance(content, str) else content)
    carriers = [block for block in blocks if is_str_dict(block) and block.get('cache_control')]
    ttls: list[Literal['5m', '1h']] = [carrier['cache_control'].get('ttl', '5m') for carrier in carriers]
    automatic_ttls: list[Literal['5m', '1h']] = (
        [automatic_cache_control.get('ttl', '5m')] if automatic_cache_control is not None else []
    )
    raised = raise_earlier_cache_ttls([*ttls, *automatic_ttls])
    for carrier, ttl, raised_ttl in zip(carriers, ttls, raised[: len(carriers)], strict=True):
        if raised_ttl != ttl:
            carrier['cache_control'] = BetaCacheControlEphemeralParam(type='ephemeral', ttl=raised_ttl)


def last_cacheable_block_has_cache_control(anthropic_messages: list[BetaMessageParam]) -> bool:
    """Whether the block automatic caching would put its breakpoint on already carries an explicit `cache_control`.

    That's the last cacheable block of the last message that isn't a request-scoped entry.
    https://platform.claude.com/docs/en/build-with-claude/prompt-caching
    """
    # Automatic caching skips a request-scoped entry, which can't carry a breakpoint.
    message = next(
        (
            m
            for m in reversed(anthropic_messages)
            if m.get('clear_at') != 'next_user_message' and not is_effort_entry(m)
        ),
        None,
    )
    content = message['content'] if message is not None else ''
    blocks = [] if isinstance(content, str) else cast(list[dict[str, Any]], content)
    last = next(
        (
            block
            for block in reversed(blocks)
            if 'cache_control' in block or block['type'] in ANTHROPIC_CACHEABLE_PARAM_TYPES
        ),
        None,
    )
    return last is not None and 'cache_control' in last
