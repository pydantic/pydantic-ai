from __future__ import annotations as _annotations

import pytest

from pydantic_ai._thinking_part import split_content_into_text_and_thinking
from pydantic_ai.messages import ModelResponsePart, TextPart, ThinkingPart


@pytest.mark.parametrize(
    ('content', 'parts'),
    [
        ('foo bar', [TextPart(content='foo bar')]),
        (
            'foo bar<think>thinking</think>',
            [TextPart(content='foo bar'), ThinkingPart(content='thinking')],
        ),
        (
            'foo bar<think>thinking</think>baz',
            [TextPart(content='foo bar'), ThinkingPart(content='thinking'), TextPart(content='baz')],
        ),
        (
            'foo bar<think>thinking',
            [TextPart(content='foo bar'), TextPart(content='thinking')],
        ),
    ],
)
def test_split_content_into_text_and_thinking(content: str, parts: list[ModelResponsePart]):
    assert split_content_into_text_and_thinking(content) == parts
