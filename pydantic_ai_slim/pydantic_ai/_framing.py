"""The tags the framework writes into prompt text, and the one way to write them.

When a provider has no channel for something, the framework says it in text instead and frames it
in a tag: a mid-conversation instruction as `<system>`, a tool's file on the user channel as
`<tool_result>`, another model's reasoning carried into a user turn as `<assistant_thinking>`.

There are two kinds. `<system>` is the one voice tag: it means the application is speaking, not the
person the model is talking to. The rest are data tags: they say where content came from, so the
model can treat it as data rather than as instructions.

Either way, text inside a tag must not be able to end it, open another, or speak in the framework's
voice: on a model with no mid-conversation system role, the framework's own `<system>` block is user
text, so a `<system>` written into a tool result would read exactly like it.
[`wrap`][pydantic_ai._framing.wrap] neutralizes every opening and closing spelling of every framing
tag in a data tag's body, and of `<system>` in the voice tag's own, escapes attribute values, and
changes nothing else, so content that mentions none of them goes out exactly as written.
"""

from __future__ import annotations as _annotations

import html
import re
from typing import Final, Literal, get_args

FramingTag = Literal[
    'system', 'tool_result', 'assistant_thinking', 'context', 'context_diff', 'validation_errors', 'retry_feedback'
]
"""Every tag the framework writes into prompt text."""

SYSTEM_TAG: Final = 'system'
"""The voice tag: the application speaking, on a channel with no system role of its own."""

TOOL_RESULT_TAG: Final = 'tool_result'
"""Frames a tool's output that has to travel on the user channel, naming the call it came from."""

ASSISTANT_THINKING_TAG: Final = 'assistant_thinking'
"""Frames another model's reasoning carried into a user turn, so it doesn't read as the user's."""

CONTEXT_TAG: Final = 'context'
"""Frames an instruction block's text with the id a later `<context>` element replaces it by."""

CONTEXT_DIFF_TAG: Final = 'context_diff'
"""Frames a unified diff from an instruction block's previous text to its new text."""


def _tag_openers(*tags: str) -> re.Pattern[str]:
    """A pattern matching only the `<` of an opening or closing tag among `tags`, in every spelling a model might reach for.

    That covers any case, whitespace around the `/`, a tag that carries attributes or ends in `/>`, and
    one cut off at the end of the text, which the wrapper's own closing tag or newline would otherwise
    complete, but not a longer tag name that merely starts with one of `tags`. The whitespace after the
    `/` is only tried when there is one, so a `<` followed by a long run of whitespace costs linear time.
    """
    names = '|'.join(re.escape(tag) for tag in tags)
    return re.compile(rf'<(?=\s*(?:/\s*)?(?:{names})(?=[\s/>]|$))', re.IGNORECASE)


VALIDATION_ERRORS_TAG: Final = 'validation_errors'
"""Frames validation errors fed back to the model in the user voice, behind a line saying the application is speaking."""

RETRY_FEEDBACK_TAG: Final = 'retry_feedback'
"""Frames a `ModelRetry` message fed back to the model in the user voice, behind the same line."""

_FRAMING_TAG_OPENERS: Final = _tag_openers(*get_args(FramingTag))
_VOICE_TAG_OPENERS: Final = _tag_openers(SYSTEM_TAG)


def neutralize_tags(text: str) -> str:
    """Escape the `<` of every opening or closing framing tag in `text` as `&lt;`, and leave everything else as written.

    For untrusted text that will sit inside a framing tag without being wrapped here, such as a value
    interpolated into a statement that a later step wraps.
    """
    return _FRAMING_TAG_OPENERS.sub('&lt;', text)


def open_tag(tag: FramingTag, /, **attrs: str | None) -> str:
    """Render the opening `tag`, with each attribute's value escaped and those that are `None` left off.

    Attributes keep the order they're given in, so the same call always renders the same bytes.
    """
    rendered = ''.join(f' {name}="{html.escape(value)}"' for name, value in attrs.items() if value is not None)
    return f'<{tag}{rendered}>'


def close_tag(tag: FramingTag) -> str:
    """Render the closing `tag`."""
    return f'</{tag}>'


def wrap(tag: FramingTag, body: str, /, **attrs: str | None) -> str:
    """Wrap `body` in `tag` so that nothing in it can end the tag early, open another, or speak in the framework's voice.

    In a data tag's body, the `<` of every opening or closing framing tag is escaped, as by
    [`neutralize_tags`][pydantic_ai._framing.neutralize_tags]. The `<system>` voice tag's body is
    framework or operator text, which can carry data tags the framework built itself, such as a
    `<context>` element on a model without mid-conversation system messages, so only `<system>` tags
    are escaped there. Either way the rest of the body goes in exactly as written. Attribute values are
    escaped as by [`open_tag`][pydantic_ai._framing.open_tag].
    """
    openers = _VOICE_TAG_OPENERS if tag == SYSTEM_TAG else _FRAMING_TAG_OPENERS
    return f'{open_tag(tag, **attrs)}{openers.sub("&lt;", body)}{close_tag(tag)}'
