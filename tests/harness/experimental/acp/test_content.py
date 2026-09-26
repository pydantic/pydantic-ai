"""Tests for ACP content-block to Pydantic AI user-content translation."""

from __future__ import annotations

import base64

import acp
from pydantic_ai.messages import BinaryContent, ImageUrl

from pydantic_ai_harness.experimental.acp._content import prompt_blocks_to_user_content


def test_text_block_becomes_string() -> None:
    content = prompt_blocks_to_user_content([acp.text_block('hello')])
    assert content == ['hello']


def test_image_with_data_becomes_binary_content() -> None:
    raw = b'\x89PNG\r\n'
    block = acp.image_block(base64.b64encode(raw).decode(), 'image/png')
    [item] = prompt_blocks_to_user_content([block])
    assert isinstance(item, BinaryContent)
    assert item.data == raw
    assert item.media_type == 'image/png'


def test_image_with_uri_becomes_image_url() -> None:
    block = acp.image_block('', 'image/png', uri='https://example.com/a.png')
    [item] = prompt_blocks_to_user_content([block])
    assert isinstance(item, ImageUrl)
    assert item.url == 'https://example.com/a.png'


def test_image_with_both_data_and_uri_prefers_inline_data() -> None:
    # `data` is required and authoritative; `uri` is only a source reference. A block carrying both
    # must keep the inline bytes rather than be replaced by a link the model may be unable to fetch.
    raw = b'\x89PNG\r\n'
    block = acp.image_block(base64.b64encode(raw).decode(), 'image/png', uri='https://example.com/a.png')
    [item] = prompt_blocks_to_user_content([block])
    assert isinstance(item, BinaryContent)
    assert item.data == raw


def test_audio_block_becomes_binary_content() -> None:
    block = acp.audio_block(base64.b64encode(b'snd').decode(), 'audio/wav')
    [item] = prompt_blocks_to_user_content([block])
    assert isinstance(item, BinaryContent)
    assert item.media_type == 'audio/wav'


def test_embedded_text_resource_becomes_string() -> None:
    block = acp.resource_block(acp.embedded_text_resource('file:///a.txt', 'from file'))
    assert prompt_blocks_to_user_content([block]) == ['from file']


def test_embedded_blob_resource_becomes_binary_content() -> None:
    block = acp.resource_block(
        acp.embedded_blob_resource('file:///a.bin', base64.b64encode(b'xx').decode(), mime_type='application/x-thing')
    )
    [item] = prompt_blocks_to_user_content([block])
    assert isinstance(item, BinaryContent)
    assert item.data == b'xx'
    assert item.media_type == 'application/x-thing'


def test_embedded_blob_resource_without_mime_type_uses_octet_stream() -> None:
    block = acp.resource_block(acp.embedded_blob_resource('file:///a.bin', base64.b64encode(b'xx').decode()))
    [item] = prompt_blocks_to_user_content([block])
    assert isinstance(item, BinaryContent)
    # Falls back to the standard "unknown binary" type rather than an invalid empty string.
    assert item.media_type == 'application/octet-stream'


def test_resource_link_labels_uri_with_name() -> None:
    block = acp.resource_link_block('a', 'file:///a.txt')
    assert prompt_blocks_to_user_content([block]) == ['a (file:///a.txt)']


def test_resource_link_prefers_title_over_name() -> None:
    block = acp.resource_link_block('a.txt', 'file:///a.txt', title='My File')
    assert prompt_blocks_to_user_content([block]) == ['My File (file:///a.txt)']


def test_resource_link_without_label_falls_back_to_uri() -> None:
    block = acp.resource_link_block('', 'file:///a.txt')
    assert prompt_blocks_to_user_content([block]) == ['file:///a.txt']


def test_mixed_blocks_preserve_order() -> None:
    blocks = [acp.text_block('look at'), acp.image_block('', 'image/png', uri='https://example.com/a.png')]
    content = prompt_blocks_to_user_content(blocks)
    assert content[0] == 'look at'
    assert isinstance(content[1], ImageUrl)
