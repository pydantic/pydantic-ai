"""Text extraction from message parts, shared by capabilities that read history as text.

Compaction (token estimates, summarizer prompts) and conversation search (the BM25 index)
both need a part's text without its media: a file's `repr` carries its bytes at ~3
characters per byte, so folding it in would bloat estimates, prompts, and indexes alike.
"""

from __future__ import annotations

from dataclasses import replace

from pydantic_ai.messages import (
    StructuredContent,
    TextContent,
    ToolReturnPart,
    UserPromptPart,
    is_multi_modal_content,
)


def with_text(item: str | TextContent, text: str) -> str | TextContent:
    """`item` with its text replaced by `text`, keeping its shape and any `TextContent.metadata`.

    For every capability that rewrites prompt text, such as a redaction or a truncation. An item whose
    text is unchanged is returned as it is. A `StructuredContent` whose text changes becomes a plain
    `TextContent`: a model that takes structured input natively is sent its `value`, which keeping the
    item would let through unredacted or untruncated.
    """
    if isinstance(item, str):
        return text
    if text == item.content:
        return item
    if isinstance(item, StructuredContent):
        return TextContent(text, metadata=item.metadata)
    return replace(item, content=text)


def user_prompt_texts(part: UserPromptPart) -> list[str]:
    """The textual items of a user prompt, in order, skipping binary and URL content.

    Only `str` and `TextContent` carry text. Callers choose how to join the items.
    """
    if isinstance(part.content, str):
        return [part.content]
    texts: list[str] = []
    for item in part.content:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, TextContent):
            texts.append(item.content)
    return texts


def tool_return_text(part: ToolReturnPart) -> str:
    """The tool return's content as text, with each file named by identifier rather than spelled out.

    The reference is the one a text-only tool result gives the model in its place
    (`ToolReturnPart.model_response_str_and_user_content`).
    """
    if not part.files:
        return str(part.content)
    return ' '.join(
        f'See file {item.identifier}.' if is_multi_modal_content(item) else str(item) for item in part.content_items()
    )
