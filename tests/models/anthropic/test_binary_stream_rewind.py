"""Retries must re-send real image bytes, not exhausted streams (#9999)."""

from __future__ import annotations

import io

import pytest

from pydantic_ai.models.anthropic import AnthropicModel

IMAGE_BYTES = b'\x89PNG\r\n\x1a\nfake-image'


def _messages_with_image() -> list[dict]:
    stream = io.BytesIO(IMAGE_BYTES)
    return [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'image',
                    'source': {'type': 'base64', 'media_type': 'image/png', 'data': stream},
                },
            ],
        }
    ]


class TestRewindBinaryStreams:
    def test_rewinds_exhausted_image_stream(self):
        messages = _messages_with_image()
        stream = messages[0]['content'][0]['source']['data']
        # Simulate the SDK serializing the first request: read the stream to its end.
        assert stream.read() == IMAGE_BYTES

        AnthropicModel._rewind_binary_streams(messages)

        assert stream.tell() == 0
        assert stream.read() == IMAGE_BYTES

    def test_rewind_is_noop_for_url_sources_and_non_binary_blocks(self):
        messages = [
            {'role': 'user', 'content': [{'type': 'text', 'text': 'hi'}]},
            {'role': 'user', 'content': [{'type': 'image', 'source': {'type': 'url', 'url': 'https://x/y.png'}}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': 'ok'}]},
        ]
        AnthropicModel._rewind_binary_streams(messages)  # must not raise

    def test_rewinds_pdf_document_stream(self):
        stream = io.BytesIO(b'%PDF-1.4 fake')
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'document', 'source': {'type': 'base64', 'media_type': 'application/pdf', 'data': stream}}
                ],
            }
        ]
        assert stream.read() == b'%PDF-1.4 fake'
        AnthropicModel._rewind_binary_streams(messages)
        assert stream.tell() == 0
