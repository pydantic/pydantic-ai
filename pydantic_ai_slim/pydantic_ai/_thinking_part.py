from __future__ import annotations as _annotations

from pydantic_ai import TextPart, ThinkingPart

from ._framing import ASSISTANT_THINKING_TAG, wrap
from .profiles import DEFAULT_THINKING_TAGS, ModelProfile

REDACTED_THINKING_PART_IDS = frozenset({'redacted_thinking', 'redacted_content'})
"""The `ThinkingPart.id`s of Claude's redacted thinking: `AnthropicModel` records it as `redacted_thinking` and
`BedrockConverseModel` as `redacted_content`, both with the same encrypted data as the `signature`."""


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
    only way to keep it in the history at all is to write it out as text.

    Returns the text and whether it has to go in a *user* message rather than the assistant turn it was
    produced in, which is what
    [`mimics_assistant_message_formatting`][pydantic_ai.profiles.ModelProfile.mimics_assistant_message_formatting]
    decides. Reasoning that stays in its own turn keeps the model's own thinking tags, which is how it
    would have written the reasoning itself. Reasoning carried into a user message can't: attributed to
    the user, it reads as something the user thought, so it gets an `assistant_thinking` wrapper instead.

    The wrapper carries no attributes. Naming the producing model would need `ModelResponse.model_name`,
    which UI adapters deliberately don't send to the client (it can hold an account id or org name, see
    https://github.com/pydantic/pydantic-ai/issues/5174), so a history that went through a UI would render
    differently from the server's copy and miss the prompt cache. And a `provider` attribute on its own makes
    `claude-opus-4-5` copy the wrapper into its answers (8/8 live, against 0/8 bare), which is the leak
    this placement exists to stop. The tag is rendered here, for the model the request goes to, and never
    stored.

    Both answers come from the profile, so the tag names and the meaning of that flag live in one place
    across every adapter that replays reasoning as text; where the carried text ends up is the caller's,
    because only the adapter knows what its wire accepts next to a tool result or a mid-conversation
    instruction.
    """
    carry_in_user_turn = profile.get('mimics_assistant_message_formatting', False)
    if carry_in_user_turn:
        # Reasoning that talks about the wrapper would otherwise close it early and spill the rest into
        # the message as if the user had written it, or open a `<system>` block of its own. `wrap` escapes
        # only the `<` of each opening or closing framing tag, so the rest goes in exactly as written.
        text = wrap(ASSISTANT_THINKING_TAG, f'\n{content}\n')
    else:
        start_tag, end_tag = profile.get('thinking_tags', DEFAULT_THINKING_TAGS)
        text = '\n'.join([start_tag, content, end_tag])
    return text, carry_in_user_turn
