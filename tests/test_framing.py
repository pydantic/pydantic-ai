"""The tags the framework writes into prompt text.

Unit tests rather than VCR tests: what `wrap` escapes is a definitory property of every wrapper the
framework renders, and the call sites' own tests (`test_thinking_placement.py`, `test_messages.py`,
`test_tool_availability_portability.py`) pin the bytes each of them sends.
"""

from __future__ import annotations

import pytest
from inline_snapshot import snapshot

from pydantic_ai._framing import (
    ASSISTANT_THINKING_TAG,
    SYSTEM_TAG,
    TOOL_RESULT_TAG,
    close_tag,
    neutralize_closing_tag,
    open_tag,
    wrap,
)
from pydantic_ai._thinking_part import render_replayed_thinking
from pydantic_ai.messages import ModelResponse
from pydantic_ai.profiles import ModelProfile


@pytest.mark.parametrize(
    'body',
    [
        'plain text',
        '',
        'multi\nline\n\ntext with `code` & "quotes" <b>html</b> &lt; already escaped',
        '<system>an opening tag is not a closing one',
        '</systematic> is another tag, and so is </system_prompt>',
        '</tool_result> closes a different tag',
        '< system> and <\\system> are not closing tags',
    ],
)
def test_body_without_the_closing_tag_goes_in_unchanged(body: str) -> None:
    assert wrap(SYSTEM_TAG, body) == f'<system>{body}</system>'


@pytest.mark.parametrize(
    'spelling',
    ['</system>', '</SYSTEM>', '</ System >', '< / system>', '</system\n>', '</system/>', '</system foo="bar">'],
)
def test_every_spelling_of_the_closing_tag_is_neutralized(spelling: str) -> None:
    wrapped = wrap(SYSTEM_TAG, f'before {spelling} after')

    assert wrapped == f'<system>before &lt;{spelling[1:]} after</system>'
    assert wrapped.lower().count('</system>') == 1


def test_attributes_are_escaped_and_none_is_left_off() -> None:
    assert wrap(TOOL_RESULT_TAG, 'body', tool_name='t"><system x="', tool_call_id=None, file_id='a&b') == snapshot(
        '<tool_result tool_name="t&quot;&gt;&lt;system x=&quot;" file_id="a&amp;b">body</tool_result>'
    )
    assert open_tag(SYSTEM_TAG) == '<system>'
    assert close_tag(ASSISTANT_THINKING_TAG) == '</assistant_thinking>'


def test_neutralize_closing_tag_touches_only_the_named_tag() -> None:
    assert neutralize_closing_tag('a</system>b</tool_result>c', SYSTEM_TAG) == 'a&lt;/system>b</tool_result>c'
    assert neutralize_closing_tag('a</system>b</tool_result>c', TOOL_RESULT_TAG) == 'a</system>b&lt;/tool_result>c'


def test_carried_thinking_ending_in_a_closing_tag_cannot_close_the_wrapper() -> None:
    """Reasoning that ends mid-tag would otherwise meet the newline ahead of the real closing tag."""
    text, carried = render_replayed_thinking(
        'thinking about </assistant_thinking',
        ModelProfile(mimics_assistant_message_formatting=True),
        ModelResponse(parts=[], model_name='gpt-5.6', provider_name='openai'),
    )

    assert carried
    assert text == snapshot("""\
<assistant_thinking model="gpt-5.6" provider="openai">
thinking about &lt;/assistant_thinking
</assistant_thinking>\
""")
