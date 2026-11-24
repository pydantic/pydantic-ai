"""Message-text helpers shared by capabilities that read or rewrite history as text.

These return a part's text without its media: a file's `repr` carries its bytes at ~3 characters
per byte, so folding it in would bloat token estimates, prompts, and search indexes alike.
"""

from __future__ import annotations

from dataclasses import replace

from pydantic_ai.messages import (
    CapabilityPart,
    StructuredContent,
    TextContent,
    ToolReturnPart,
    UserPromptPart,
    is_multi_modal_content,
)

CHARS_PER_TOKEN = 4
"""Rough approximation: ~4 characters per token on average."""


def with_text(item: str | TextContent, text: str) -> str | TextContent:
    """`item` with its text replaced by `text`, keeping its shape and any `TextContent.metadata`.

    An item whose text is unchanged is returned as it is. A `StructuredContent` whose text changes
    becomes a plain `TextContent`: a model that takes structured input natively is sent its `value`,
    which keeping the item would let through unredacted or untruncated.
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


def user_prompt_text(part: UserPromptPart) -> str:
    """The textual items of a user prompt joined with spaces, skipping binary and URL content."""
    return ' '.join(user_prompt_texts(part))


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


def capability_part_text(part: CapabilityPart) -> str:
    """The text of a capability part's content, as the model sees it, without a data kind's frame or note."""
    content = part.model_content()
    return ''.join(user_prompt_texts(UserPromptPart([content] if isinstance(content, StructuredContent) else content)))
