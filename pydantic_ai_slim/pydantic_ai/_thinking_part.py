from __future__ import annotations as _annotations

from pydantic_ai import TextPart, ThinkingPart

FOREIGN_THINKING_INTRO = 'Earlier reasoning, possibly from another model, provided for context only:'


def render_foreign_thinking(content: str) -> str:
    """Render a `ThinkingPart` that can't be sent back through the model's own native reasoning channel.

    Such a part reaches this fallback when it has no signature (a model's own reasoning round-tripped
    through storage, or a provider like xAI that returns reasoning unsigned by default) or was produced by
    a different provider (e.g. another model in a `FallbackModel` chain). It is prefixed with an explicit
    marker rather than the profile's `<thinking>` tags: providers like Anthropic document that `<thinking>`
    tags in the prompt get generalized into the model's own output, so re-rendering the reasoning as tags
    teaches the model to leak that format into user-visible answers. The wording stays accurate whether the
    reasoning is the model's own (unsigned) or genuinely foreign, since the two are indistinguishable here.
    """
    return f'{FOREIGN_THINKING_INTRO}\n\n{content}'


def split_content_into_text_and_thinking(content: str, thinking_tags: tuple[str, str]) -> list[ThinkingPart | TextPart]:
    """Split a string into text and thinking parts.

    Some models don't return the thinking part as a separate part, but rather as a tag in the content.
    This function splits the content into text and thinking parts.
    """
    start_tag, end_tag = thinking_tags
    parts: list[ThinkingPart | TextPart] = []

    start_index = content.find(start_tag)
    while start_index >= 0:
        before_think, content = content[:start_index], content[start_index + len(start_tag) :]
        if before_think:
            parts.append(TextPart(content=before_think))
        end_index = content.find(end_tag)
        if end_index >= 0:
            think_content, content = content[:end_index], content[end_index + len(end_tag) :]
            parts.append(ThinkingPart(content=think_content))
        else:
            # We lose the `<think>` tag, but it shouldn't matter.
            parts.append(TextPart(content=content))
            content = ''
        start_index = content.find(start_tag)
    if content:
        parts.append(TextPart(content=content))
    return parts
