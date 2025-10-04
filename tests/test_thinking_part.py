from __future__ import annotations as _annotations

import pytest
from inline_snapshot import snapshot

from pydantic_ai import ModelResponsePart, TextPart, ThinkingPart, ThinkingPartDelta
from pydantic_ai._thinking_part import split_content_into_text_and_thinking


@pytest.mark.parametrize(
    'thinking_tags, content, parts',
    [
        # default <think>…</think> cases
        (
            ('<think>', '</think>'),
            'foo bar',
            [TextPart(content='foo bar')],
        ),
        (
            ('<think>', '</think>'),
            'foo bar<think>thinking</think>',
            [TextPart(content='foo bar'), ThinkingPart(content='thinking')],
        ),
        (
            ('<think>', '</think>'),
            'foo bar<think>thinking</think>baz',
            [
                TextPart(content='foo bar'),
                ThinkingPart(content='thinking'),
                TextPart(content='baz'),
            ],
        ),
        (
            ('<think>', '</think>'),
            'foo bar<think>thinking',
            [TextPart(content='foo bar'), TextPart(content='thinking')],
        ),
        (
            ('<think>', '</think>'),
            'foo bar<custom>thinking</custom>baz',
            [TextPart(content='foo bar<custom>thinking</custom>baz')],
        ),
        # custom <custom>…</custom> cases
        (
            ('<custom>', '</custom>'),
            'foo bar',
            [TextPart(content='foo bar')],
        ),
        (
            ('<custom>', '</custom>'),
            'foo bar<custom>thinking</custom>',
            [TextPart(content='foo bar'), ThinkingPart(content='thinking')],
        ),
        (
            ('<custom>', '</custom>'),
            'foo bar<custom>thinking</custom>baz',
            [
                TextPart(content='foo bar'),
                ThinkingPart(content='thinking'),
                TextPart(content='baz'),
            ],
        ),
        (
            ('<custom>', '</custom>'),
            'foo bar<custom>thinking',
            [TextPart(content='foo bar'), TextPart(content='thinking')],
        ),
        (
            ('<custom>', '</custom>'),
            'foo bar<think>thinking</think>baz',
            [TextPart(content='foo bar<think>thinking</think>baz')],
        ),
    ],
)
def test_split_content(thinking_tags: tuple[str, str], content: str, parts: list[ModelResponsePart]):
    assert split_content_into_text_and_thinking(content, thinking_tags) == parts


def test_thinking_part_delta_applies_both_content_and_signature():
    thinking_part = ThinkingPart(content='Initial content', signature='initial_sig')
    delta = ThinkingPartDelta(content_delta=' added', signature_delta='new_sig')

    result = delta.apply(thinking_part)

    # The content is appended, and the signature is updated.
    assert result == snapshot(ThinkingPart(content='Initial content added', signature='new_sig'))


def test_thinking_part_delta_applies_signature_only():
    thinking_part = ThinkingPart(content='Initial content', signature='initial_sig')
    delta_sig_only = ThinkingPartDelta(content_delta=None, signature_delta='sig_only')

    result_sig_only = delta_sig_only.apply(thinking_part)

    # The content is unchanged, and the signature is updated.
    assert result_sig_only == snapshot(ThinkingPart(content='Initial content', signature='sig_only'))


def test_thinking_part_delta_applies_content_only_preserves_signature():
    thinking_part = ThinkingPart(content='Initial content', signature='initial_sig')
    delta_content_only = ThinkingPartDelta(content_delta=' more', signature_delta=None)

    result_content_only = delta_content_only.apply(thinking_part)

    # The content is appended, and the signature is preserved.
    assert result_content_only == snapshot(ThinkingPart(content='Initial content more', signature='initial_sig'))


def test_thinking_part_delta_applies_to_part_with_none_signature():
    thinking_part_no_sig = ThinkingPart(content='No sig content', signature=None)
    delta_to_none_sig = ThinkingPartDelta(content_delta=' extra', signature_delta='added_sig')

    result_none_sig = delta_to_none_sig.apply(thinking_part_no_sig)

    # The content is appended, and the signature is updated.
    assert result_none_sig == snapshot(ThinkingPart(content='No sig content extra', signature='added_sig'))
