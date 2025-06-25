from __future__ import annotations as _annotations

from pydantic_ai.messages import TextPart, ThinkingPart

START_THINK_TAG = '<think>'
END_THINK_TAG = '</think>'


def split_content_into_text_and_thinking(content: str) -> list[ThinkingPart | TextPart]:
    """Split a string into text and thinking parts.

    Some models don't return the thinking part as a separate part, but rather as a tag in the content.
    This function splits the content into text and thinking parts.

    We use the `<think>` tag because that's how Groq uses it in the `raw` format, so instead of using `<Thinking>` or
    something else, we just match the tag to make it easier for other models that don't support the `ThinkingPart`.
    """
    parts: list[ThinkingPart | TextPart] = []

    start_index = content.find(START_THINK_TAG)
    while start_index >= 0:
        before_think, content = content[:start_index], content[start_index + len(START_THINK_TAG) :]
        if before_think:
            parts.append(TextPart(content=before_think))
        end_index = content.find(END_THINK_TAG)
        if end_index >= 0:
            think_content, content = content[:end_index], content[end_index + len(END_THINK_TAG) :]
            parts.append(ThinkingPart(content=think_content))
        else:
            # We lose the `<think>` tag, but it shouldn't matter.
            parts.append(TextPart(content=content))
            content = ''
        start_index = content.find(START_THINK_TAG)
    if content:
        parts.append(TextPart(content=content))
    return parts
