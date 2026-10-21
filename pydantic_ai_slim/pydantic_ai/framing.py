"""Frame data in the tags Pydantic AI writes into prompt text, so that content can't break out of them.

Where a provider has no channel for something, Pydantic AI says it in prompt text and frames it in a
tag: a mid-conversation instruction as `<system>`, a tool's file on the user channel as `<tool_result>`.
A capability that puts its own data into prompt text, such as a memory or a reviewer's feedback, frames
it the same way, with a tag it registers with [`register_tag`][pydantic_ai.framing.register_tag] and
renders with [`wrap`][pydantic_ai.framing.wrap].

`<system>` is the one voice tag: it means the application is speaking. Every other tag, built-in or
registered, is a data tag, whose body can't open or close any framing tag, so a tool result can't
close your frame and your frame's content can't forge a `<system>` block.

See [Framing data in prompt text](../capabilities/custom.md#framing-data-in-prompt-text).
"""

from ._framing import FramingTag, RegisteredTag, neutralize_tags, register_tag, wrap

__all__ = ('FramingTag', 'RegisteredTag', 'neutralize_tags', 'register_tag', 'wrap')
