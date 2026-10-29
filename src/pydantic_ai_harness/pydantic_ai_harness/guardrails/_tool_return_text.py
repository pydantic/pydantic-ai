"""Text detection over a `ToolReturn`'s model-directed `content`, for `for_tool_result_text`.

The model reads adjacent text parts of `ToolReturn.content` as one span, so a detector runs
over each part and over each span joined. This module holds that span logic, kept apart from
the public detectors that use it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeGuard, cast

from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import CachePoint, TextContent, ToolReturn, UserContent, is_multi_modal_content
from pydantic_ai_harness._message_text import with_text
from pydantic_ai_harness.guardrails._shared import GuardrailResult, TextDetector


def _is_content_list(value: object) -> TypeGuard[list[UserContent]]:
    """Whether `value` is a list of content parts, such as text beside a screenshot, as a tool can return."""
    if not isinstance(value, list):
        return False
    return all(
        isinstance(part, str | TextContent | CachePoint) or is_multi_modal_content(part)
        for part in cast(list[Any], value)
    )


def _detect_content_text(detector: TextDetector, text: str) -> tuple[GuardrailResult | None, str | None]:
    """Run `detector` over one text part of `ToolReturn.content`.

    Returns a terminal verdict, or the replacement text -- `None` when the part is left alone.
    """
    verdict = detector(text)
    if verdict.action not in ('allow', 'replace'):
        return verdict, None
    if verdict.action == 'allow':
        return None, None
    replacement = verdict.replacement
    if not isinstance(replacement, str):
        raise UserError('A text detector used with for_tool_result_text() must replace tool content with text.')
    return None, replacement


def _text_of(part: str | TextContent) -> str:
    """The model-visible text of one `ToolReturn.content` part."""
    return part if isinstance(part, str) else part.content


def _carries_no_model_content(part: UserContent) -> bool:
    """Whether `part` puts nothing in front of the model, so the text either side of it is contiguous.

    `CachePoint` is a prompt-caching marker that providers without caching drop, so it separates
    two text parts on the page but not in what the model reads. It is the only member of
    `UserContent` like that: `str` and `TextContent` carry text, and `ImageUrl`, `AudioUrl`,
    `DocumentUrl`, `VideoUrl`, `BinaryContent` and `UploadedFile` each put a payload in the
    prompt, so those do separate the text around them.
    """
    return isinstance(part, CachePoint)


def _merge_span(
    run: Sequence[UserContent], text_indexes: Sequence[int], first_text: str | TextContent, text: str
) -> list[UserContent]:
    """Collapse a span's text into one part, keeping every marker on a side it still has.

    A marker before the first text part or after the last one keeps its side. The text it
    bounded is still on that side, so moving it would change what the caller asked to be
    cached for nothing. Only a marker between two merged texts has lost the split it marked,
    and that one goes ahead of the merged text, which narrows the cached prefix rather than
    widening it over text the caller kept outside.
    """
    first, last = text_indexes[0], text_indexes[-1]
    interior = [part for part in run[first:last] if not isinstance(part, str | TextContent)]
    return [*run[:first], *interior, with_text(first_text, text), *run[last + 1 :]]


def _detect_text_run(
    detector: TextDetector, run: Sequence[UserContent]
) -> tuple[GuardrailResult | None, Sequence[UserContent], bool]:
    """Run `detector` over one stretch of text the model reads as one span, per part and then joined.

    The model reads that span as one string, so a secret split across two parts of it
    matches nothing in either part on its own. The joined pass catches that, and it scans
    the span's *original* text, not the per-part rewrites. Redacting a fragment can strip
    the very characters a pattern anchors on -- `sk-` at the front of a key -- which would
    leave the rest of the value exposed while the joined pass reported the span clean. It
    can hide a terminal verdict the same way.

    The span keeps its parts when sanitizing them one by one already produced the text that
    sanitizing the whole span produces. Then the boundaries carry nothing, and each part
    keeps its own metadata. When the two differ, something is only reachable across a
    boundary, and the span collapses to the whole-span sanitization -- the text the same
    content would have got as a single string.

    A span holding one text part is scanned once. Its whole-span sanitization is by
    definition its per-part one, so the joined pass could only agree with itself. That
    matters beyond the wasted call: `TextDetector` is a public extension point, so a
    detector that is stateful or bills per call would otherwise see `content='x'` and
    `content=['x']` differently.

    A span can hold parts that carry no model content, which is why the joined pass keys on
    the number of text parts rather than the length of `run`.

    Returns a terminal verdict, the parts to keep, and whether any text changed.
    """
    text_parts: list[str | TextContent] = []
    text_indexes: list[int] = []
    originals: list[str] = []
    texts: list[str] = []
    replaced = False
    for index, part in enumerate(run):
        if not isinstance(part, str | TextContent):
            continue
        original = _text_of(part)
        verdict, replacement = _detect_content_text(detector, original)
        if verdict is not None:
            return verdict, run, False
        replaced |= replacement is not None
        text_parts.append(part)
        text_indexes.append(index)
        originals.append(original)
        texts.append(original if replacement is None else replacement)
    if len(texts) > 1:
        verdict, joined = _detect_content_text(detector, ''.join(originals))
        if verdict is not None:
            return verdict, run, False
        if joined is not None and joined != ''.join(texts):
            return None, _merge_span(run, text_indexes, text_parts[0], joined), True
    rewritten: list[UserContent] = []
    replacements = iter(texts)
    for part in run:
        if isinstance(part, str | TextContent):
            rewritten.append(with_text(part, next(replacements)))
        else:
            rewritten.append(part)
    return None, rewritten, replaced


def detect_tool_return_content(
    detector: TextDetector, content: str | Sequence[UserContent] | None
) -> tuple[GuardrailResult | None, str | Sequence[UserContent] | None]:
    """Run `detector` over every text-bearing part of `ToolReturn.content`.

    Returns the first terminal verdict a part produced, or the rewritten content --
    `None` when no part was redacted, so the caller keeps the original shape.
    """
    if isinstance(content, str):
        return _detect_content_text(detector, content)
    if content is None:
        return None, None
    rewritten: list[UserContent] = []
    run: list[UserContent] = []
    replaced = False
    # The trailing `None` flushes the last run without repeating the flush after the loop.
    for part in [*content, None]:
        if part is not None and (isinstance(part, str | TextContent) or _carries_no_model_content(part)):
            run.append(part)
            continue
        verdict, parts, run_replaced = _detect_text_run(detector, run)
        if verdict is not None:
            return verdict, None
        rewritten.extend(parts)
        replaced |= run_replaced
        run = []
        if part is not None:
            rewritten.append(part)
    return None, rewritten if replaced else None


def is_checked_tool_return(value: object) -> TypeGuard[ToolReturn[Any]]:
    """Whether `value` is a `ToolReturn` whose `return_value` is a string or a list of content parts."""
    return isinstance(value, ToolReturn) and (
        isinstance(value.return_value, str) or _is_content_list(cast(ToolReturn[Any], value).return_value)
    )


def detect_return_value(
    detector: TextDetector, value: str | list[UserContent]
) -> tuple[GuardrailResult | None, str | list[UserContent] | None]:
    """Run `detector` over a tool result's string, or the text of its content parts, keeping images.

    Returns a terminal verdict, or the replacement -- `None` when nothing was redacted.
    """
    if isinstance(value, str):
        verdict = detector(value)
        if verdict.action not in ('allow', 'replace'):
            return verdict, None
        if verdict.action == 'allow':
            return None, None
        if not isinstance(verdict.replacement, str):
            raise UserError('A text detector used with for_tool_result_text() must replace a tool result with text.')
        return None, verdict.replacement
    verdict, rewritten = detect_tool_return_content(detector, value)
    return verdict, None if rewritten is None else list(rewritten)


def detect_content_list_result(detector: TextDetector, result: object) -> GuardrailResult | None:
    """Run `detector` over a result made of content parts, such as text beside a screenshot, keeping the images.

    Returns `None` for any other result.
    """
    if not _is_content_list(result):
        return None
    verdict, rewritten = detect_return_value(detector, result)
    if verdict is not None:
        return verdict
    return GuardrailResult.allow() if rewritten is None else GuardrailResult.replace(rewritten)
