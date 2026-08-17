from __future__ import annotations as _annotations

from pydantic_ai import TextPart, ThinkingPart

from .profiles import DEFAULT_THINKING_TAGS, ModelProfile


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


def render_replayed_thinking(content: str, profile: ModelProfile) -> tuple[str, bool]:
    """Render reasoning that can't be replayed through a model's native channel, as tagged text.

    The inverse of [`split_content_into_text_and_thinking`][pydantic_ai._thinking_part.split_content_into_text_and_thinking]:
    a `ThinkingPart` that arrived unsigned or from another provider has no native block to ride, so the
    only way to keep it in the history at all is to write it out in the model's own thinking tags.

    Returns the text and whether it has to go in a *user* message rather than the assistant turn it was
    produced in, which is what
    [`mimics_assistant_message_formatting`][pydantic_ai.profiles.ModelProfile.mimics_assistant_message_formatting]
    decides. Both answers come from the profile, so the tag names and the meaning of that flag live in
    one place across every adapter that replays reasoning as text; where the carried text ends up is the
    caller's, because only the adapter knows what its wire accepts next to a tool result or a mid-conversation
    instruction.
    """
    start_tag, end_tag = profile.get('thinking_tags', DEFAULT_THINKING_TAGS)
    return (
        '\n'.join([start_tag, content, end_tag]),
        profile.get('mimics_assistant_message_formatting', False),
    )
