"""Unit tests for the shared MCP utility helpers."""

from __future__ import annotations

import base64

import pytest
from pydantic import AnyUrl

from .conftest import try_import

with try_import() as imports_successful:
    from mcp.types import (
        Annotations,
        AudioContent,
        BlobResourceContents,
        EmbeddedResource,
        ImageContent,
        ResourceLink,
        TextContent,
        TextResourceContents,
    )

    from pydantic_ai import messages
    from pydantic_ai._mcp_util import (
        clean_meta,
        is_for_assistant,
        map_tool_result_part,
        map_tool_result_user_content,
        merge_meta,
        partition_content,
    )


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


def test_clean_meta_strips_fastmcp_marker():
    assert clean_meta({'fastmcp': {'wrap_result': True}, 'k': 'v'}) == {'k': 'v'}


def test_clean_meta_returns_none_when_only_fastmcp_marker():
    assert clean_meta({'fastmcp': {'wrap_result': True}}) is None


def test_clean_meta_returns_none_for_empty_or_none():
    assert clean_meta(None) is None
    assert clean_meta({}) is None


def test_map_tool_result_part_text_without_meta_is_str():
    assert map_tool_result_part(TextContent(type='text', text='hello')) == 'hello'


def test_map_tool_result_part_text_with_meta_wraps_in_text_content():
    out = map_tool_result_part(TextContent(type='text', text='hello', _meta={'k': 'v'}))
    assert out == messages.TextContent(content='hello', metadata={'k': 'v'})


def test_map_tool_result_part_text_with_meta_skips_json_parse():
    """`_meta` preservation wins over the JSON-parse convenience — the wrapper survives."""
    out = map_tool_result_part(TextContent(type='text', text='[1, 2, 3]', _meta={'k': 'v'}))
    assert out == messages.TextContent(content='[1, 2, 3]', metadata={'k': 'v'})


def test_map_tool_result_part_image_with_meta():
    encoded = base64.b64encode(b'bytes').decode()
    out = map_tool_result_part(ImageContent(type='image', data=encoded, mimeType='image/png', _meta={'k': 'v'}))
    assert isinstance(out, messages.BinaryImage)
    assert out.metadata == {'k': 'v'}


def test_map_tool_result_part_audio_with_meta():
    encoded = base64.b64encode(b'bytes').decode()
    out = map_tool_result_part(AudioContent(type='audio', data=encoded, mimeType='audio/mpeg', _meta={'k': 'v'}))
    assert isinstance(out, messages.BinaryContent)
    assert out.metadata == {'k': 'v'}


def test_map_tool_result_part_embedded_text_merges_outer_and_inner_meta():
    out = map_tool_result_part(
        EmbeddedResource(
            type='resource',
            resource=TextResourceContents(
                uri=AnyUrl('resource://doc'), text='body', _meta={'inner': True, 'shared': 'inner'}
            ),
            _meta={'outer': True, 'shared': 'outer'},
        )
    )
    # Inner (resource-level) meta wins on shared keys.
    assert out == messages.TextContent(content='body', metadata={'outer': True, 'inner': True, 'shared': 'inner'})


def test_map_tool_result_part_embedded_blob_propagates_meta():
    encoded = base64.b64encode(b'data').decode()
    out = map_tool_result_part(
        EmbeddedResource(
            type='resource',
            resource=BlobResourceContents(
                uri=AnyUrl('resource://blob'), blob=encoded, mimeType='application/octet-stream'
            ),
            _meta={'k': 'v'},
        )
    )
    assert isinstance(out, messages.BinaryContent)
    assert out.metadata == {'k': 'v'}


def test_map_tool_result_part_resource_link_with_meta_wraps_uri():
    out = map_tool_result_part(
        ResourceLink(type='resource_link', uri=AnyUrl('resource://x'), name='x', _meta={'k': 'v'})
    )
    assert out == messages.TextContent(content='resource://x', metadata={'k': 'v'})


def test_map_tool_result_part_resource_link_without_meta_returns_str():
    out = map_tool_result_part(ResourceLink(type='resource_link', uri=AnyUrl('resource://x'), name='x'))
    assert out == 'resource://x'


def test_map_tool_result_user_content_text_without_meta_is_str():
    assert map_tool_result_user_content(TextContent(type='text', text='hi')) == 'hi'


def test_map_tool_result_user_content_text_with_meta_preserves_wrapper():
    out = map_tool_result_user_content(TextContent(type='text', text='hi', _meta={'k': 'v'}))
    assert out == messages.TextContent(content='hi', metadata={'k': 'v'})


def test_map_tool_result_user_content_skips_json_parse():
    """Unlike `map_tool_result_part`, `UserContent` doesn't admit dict/list — JSON-encoded
    text stays as a string."""
    out = map_tool_result_user_content(TextContent(type='text', text='[1, 2, 3]'))
    assert out == '[1, 2, 3]'


def test_map_tool_result_user_content_image():
    encoded = base64.b64encode(b'bytes').decode()
    out = map_tool_result_user_content(ImageContent(type='image', data=encoded, mimeType='image/png'))
    assert isinstance(out, messages.BinaryImage)


def test_map_tool_result_user_content_audio():
    encoded = base64.b64encode(b'bytes').decode()
    out = map_tool_result_user_content(AudioContent(type='audio', data=encoded, mimeType='audio/mpeg'))
    assert isinstance(out, messages.BinaryContent)


def test_map_tool_result_user_content_resource_link_with_meta():
    out = map_tool_result_user_content(
        ResourceLink(type='resource_link', uri=AnyUrl('resource://x'), name='x', _meta={'k': 'v'})
    )
    assert out == messages.TextContent(content='resource://x', metadata={'k': 'v'})


def test_map_tool_result_user_content_resource_link_without_meta_is_str():
    out = map_tool_result_user_content(ResourceLink(type='resource_link', uri=AnyUrl('resource://x'), name='x'))
    assert out == 'resource://x'


def test_map_tool_result_user_content_embedded_resource_round_trips():
    out = map_tool_result_user_content(
        EmbeddedResource(
            type='resource',
            resource=TextResourceContents(uri=AnyUrl('resource://x'), text='body'),
        )
    )
    assert out == 'body'
