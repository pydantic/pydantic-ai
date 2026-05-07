"""Unit tests for the shared MCP utility helpers."""

from __future__ import annotations

import pytest

from .conftest import try_import

with try_import() as imports_successful:
    from mcp.types import Annotations, ImageContent, TextContent

    from pydantic_ai._mcp_util import is_for_assistant, merge_meta, partition_content


pytestmark = pytest.mark.skipif(not imports_successful(), reason='mcp not installed')


def _text(text: str, *, audience: list[str] | None = None) -> TextContent:
    return TextContent(
        type='text',
        text=text,
        annotations=Annotations(audience=audience) if audience is not None else None,  # type: ignore[arg-type]
    )


def test_is_for_assistant_no_annotations():
    assert is_for_assistant(_text('a')) is True


def test_is_for_assistant_audience_unset():
    assert is_for_assistant(TextContent(type='text', text='a', annotations=Annotations(priority=0.5))) is True


def test_is_for_assistant_with_assistant_in_audience():
    assert is_for_assistant(_text('a', audience=['assistant'])) is True
    assert is_for_assistant(_text('a', audience=['user', 'assistant'])) is True


def test_is_for_assistant_excludes_user_only():
    assert is_for_assistant(_text('a', audience=['user'])) is False


def test_partition_content_splits_by_audience():
    assistant_part = _text('for-assistant', audience=['assistant'])
    user_part = _text('for-user', audience=['user'])
    unannotated_image = ImageContent(type='image', data='', mimeType='image/png')
    assistant_visible, user_only = partition_content([assistant_part, user_part, unannotated_image])
    assert assistant_visible == [assistant_part, unannotated_image]
    assert user_only == [user_part]


def test_partition_content_empty():
    assert partition_content([]) == ([], [])


def test_merge_meta_returns_none_when_all_empty():
    assert merge_meta(None, None) is None
    assert merge_meta(None, {}) is None


def test_merge_meta_combines_dicts_with_later_overriding():
    assert merge_meta({'a': 1, 'b': 2}, {'b': 3, 'c': 4}) == {'a': 1, 'b': 3, 'c': 4}


def test_merge_meta_skips_none_arguments():
    assert merge_meta(None, {'a': 1}, None, {'b': 2}) == {'a': 1, 'b': 2}
