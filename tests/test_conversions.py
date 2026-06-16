"""Tests for _conversions.py — round-tripping OTEL messages and OpenAI format conversion."""

from __future__ import annotations

import base64
import json

from inline_snapshot import snapshot

from pydantic_ai import model_messages_to_openai_format, otel_messages_to_model_messages
from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    CachePoint,
    DocumentUrl,
    FilePart,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    RetryPromptPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UploadedFile,
    UserPromptPart,
    VideoUrl,
)

from .conftest import IsDatetime

# ── otel_messages_to_model_messages: ChatMessage format ────────────────


class TestChatMessagesToModelMessages:
    def test_empty(self):
        assert otel_messages_to_model_messages([]) == []

    def test_simple_text_conversation(self):
        otel = [
            {'role': 'user', 'parts': [{'type': 'text', 'content': 'Hello'}]},
            {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'Hi there!'}]},
        ]
        assert otel_messages_to_model_messages(otel) == snapshot(
            [
                ModelRequest(parts=[UserPromptPart(content='Hello', timestamp=IsDatetime())]),
                ModelResponse(parts=[TextPart(content='Hi there!')], timestamp=IsDatetime()),
            ]
        )

    def test_json_string_input(self):
        otel_json = json.dumps(
            [
                {'role': 'user', 'parts': [{'type': 'text', 'content': 'Hello'}]},
                {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'Hi'}]},
            ]
        )
        assert otel_messages_to_model_messages(otel_json) == snapshot(
            [
                ModelRequest(parts=[UserPromptPart(content='Hello', timestamp=IsDatetime())]),
                ModelResponse(parts=[TextPart(content='Hi')], timestamp=IsDatetime()),
            ]
        )

    def test_system_and_user_merged_into_request(self):
        otel = [
            {'role': 'system', 'parts': [{'type': 'text', 'content': 'Be helpful.'}]},
            {'role': 'user', 'parts': [{'type': 'text', 'content': 'Hello'}]},
            {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'Hi'}]},
        ]
        assert otel_messages_to_model_messages(otel) == snapshot(
            [
                ModelRequest(
                    parts=[
                        SystemPromptPart(content='Be helpful.', timestamp=IsDatetime()),
                        UserPromptPart(content='Hello', timestamp=IsDatetime()),
                    ]
                ),
                ModelResponse(parts=[TextPart(content='Hi')], timestamp=IsDatetime()),
            ]
        )

    def test_tool_call_and_return(self):
        otel = [
            {'role': 'user', 'parts': [{'type': 'text', 'content': 'What is the weather?'}]},
            {
                'role': 'assistant',
                'parts': [
                    {'type': 'tool_call', 'id': 'call_1', 'name': 'get_weather', 'arguments': {'city': 'London'}},
                ],
            },
            {
                'role': 'user',
                'parts': [
                    {'type': 'tool_call_response', 'id': 'call_1', 'name': 'get_weather', 'result': 'Sunny, 21°C'},
                ],
            },
            {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'It is sunny and 21°C in London.'}]},
        ]
        assert otel_messages_to_model_messages(otel) == snapshot(
            [
                ModelRequest(parts=[UserPromptPart(content='What is the weather?', timestamp=IsDatetime())]),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='get_weather', args={'city': 'London'}, tool_call_id='call_1')],
                    timestamp=IsDatetime(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 21°C',
                            tool_call_id='call_1',
                            timestamp=IsDatetime(),
                        )
                    ]
                ),
                ModelResponse(parts=[TextPart(content='It is sunny and 21°C in London.')], timestamp=IsDatetime()),
            ]
        )

    def test_thinking_part(self):
        otel = [
            {'role': 'user', 'parts': [{'type': 'text', 'content': 'Think about this'}]},
            {
                'role': 'assistant',
                'parts': [
                    {'type': 'thinking', 'content': 'Let me think...'},
                    {'type': 'text', 'content': 'Here is my answer.'},
                ],
            },
        ]
        assert otel_messages_to_model_messages(otel) == snapshot(
            [
                ModelRequest(parts=[UserPromptPart(content='Think about this', timestamp=IsDatetime())]),
                ModelResponse(
                    parts=[ThinkingPart(content='Let me think...'), TextPart(content='Here is my answer.')],
                    timestamp=IsDatetime(),
                ),
            ]
        )

    def test_builtin_tool_call(self):
        otel = [
            {'role': 'user', 'parts': [{'type': 'text', 'content': 'Search for cats'}]},
            {
                'role': 'assistant',
                'parts': [
                    {
                        'type': 'tool_call',
                        'id': 'call_1',
                        'name': 'web_search',
                        'arguments': {'query': 'cats'},
                        'builtin': True,
                    },
                    {
                        'type': 'tool_call_response',
                        'id': 'call_1',
                        'name': 'web_search',
                        'result': 'Cats are great',
                        'builtin': True,
                    },
                    {'type': 'text', 'content': 'Cats are great.'},
                ],
            },
        ]
        assert otel_messages_to_model_messages(otel) == snapshot(
            [
                ModelRequest(parts=[UserPromptPart(content='Search for cats', timestamp=IsDatetime())]),
                ModelResponse(
                    parts=[
                        NativeToolCallPart(tool_name='web_search', args={'query': 'cats'}, tool_call_id='call_1'),
                        NativeToolReturnPart(
                            tool_name='web_search',
                            content='Cats are great',
                            tool_call_id='call_1',
                            timestamp=IsDatetime(),
                        ),
                        TextPart(content='Cats are great.'),
                    ],
                    timestamp=IsDatetime(),
                ),
            ]
        )

    def test_image_url_in_user_message(self):
        otel = [
            {
                'role': 'user',
                'parts': [
                    {'type': 'text', 'content': 'What is in this image?'},
                    {'type': 'image-url', 'url': 'https://example.com/cat.png'},
                ],
            },
            {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'A cat.'}]},
        ]
        assert otel_messages_to_model_messages(otel) == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content=[
                                'What is in this image?',
                                ImageUrl(
                                    url='https://example.com/cat.png', media_type='image/png', identifier='750ff4'
                                ),
                            ],
                            timestamp=IsDatetime(),
                        )
                    ]
                ),
                ModelResponse(parts=[TextPart(content='A cat.')], timestamp=IsDatetime()),
            ]
        )

    def test_media_urls_in_user_message(self):
        """Audio, video, and document URLs are all converted to their respective types."""
        otel = [
            {
                'role': 'user',
                'parts': [
                    {'type': 'audio-url', 'url': 'https://example.com/a.mp3'},
                    {'type': 'video-url', 'url': 'https://example.com/v.mp4'},
                    {'type': 'document-url', 'url': 'https://example.com/d.pdf'},
                ],
            },
        ]
        assert otel_messages_to_model_messages(otel) == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content=[
                                AudioUrl(url='https://example.com/a.mp3'),
                                VideoUrl(url='https://example.com/v.mp4', media_type='video/mp4', identifier='2ed291'),
                                DocumentUrl(
                                    url='https://example.com/d.pdf', media_type='application/pdf', identifier='96c6c6'
                                ),
                            ],
                            timestamp=IsDatetime(),
                        )
                    ]
                )
            ]
        )

    def test_media_urls_with_empty_url_are_skipped(self):
        """Media parts with a missing/empty URL are dropped, leaving only the text."""
        otel = [
            {
                'role': 'user',
                'parts': [
                    {'type': 'text', 'content': 'Just text'},
                    {'type': 'image-url', 'url': ''},
                    {'type': 'audio-url'},
                    {'type': 'video-url'},
                    {'type': 'document-url'},
                ],
            },
        ]
        assert otel_messages_to_model_messages(otel) == snapshot(
            [ModelRequest(parts=[UserPromptPart(content='Just text', timestamp=IsDatetime())])]
        )

    def test_binary_content_in_user_message(self):
        data = b'fake image data'
        b64 = base64.b64encode(data).decode()
        otel = [
            {
                'role': 'user',
                'parts': [
                    {'type': 'text', 'content': 'Describe this'},
                    {'type': 'binary', 'media_type': 'image/png', 'content': b64},
                ],
            },
            {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'An image.'}]},
        ]
        assert otel_messages_to_model_messages(otel) == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content=[
                                'Describe this',
                                BinaryContent(data=b'fake image data', media_type='image/png', identifier='d7c7d6'),
                            ],
                            timestamp=IsDatetime(),
                        )
                    ]
                ),
                ModelResponse(parts=[TextPart(content='An image.')], timestamp=IsDatetime()),
            ]
        )

    def test_binary_content_in_assistant_message(self):
        data = b'fake image data'
        b64 = base64.b64encode(data).decode()
        otel = [
            {'role': 'user', 'parts': [{'type': 'text', 'content': 'Generate an image'}]},
            {
                'role': 'assistant',
                'parts': [
                    {'type': 'binary', 'media_type': 'image/png', 'content': b64},
                ],
            },
        ]
        assert otel_messages_to_model_messages(otel) == snapshot(
            [
                ModelRequest(parts=[UserPromptPart(content='Generate an image', timestamp=IsDatetime())]),
                ModelResponse(
                    parts=[
                        FilePart(
                            content=BinaryContent(data=b'fake image data', media_type='image/png', identifier='d7c7d6')
                        )
                    ],
                    timestamp=IsDatetime(),
                ),
            ]
        )

    def test_v4_uri_parts_by_modality(self):
        """v4+ OTEL `uri` parts map to the URL type implied by their modality."""
        otel = [
            {
                'role': 'user',
                'parts': [
                    {'type': 'uri', 'modality': 'image', 'uri': 'https://example.com/a.png', 'mime_type': 'image/png'},
                    {'type': 'uri', 'modality': 'audio', 'uri': 'https://example.com/a.mp3'},
                    {'type': 'uri', 'modality': 'video', 'uri': 'https://example.com/a.mp4'},
                    {'type': 'uri', 'uri': 'https://example.com/a.pdf'},
                ],
            },
        ]
        assert otel_messages_to_model_messages(otel) == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content=[
                                ImageUrl(url='https://example.com/a.png', media_type='image/png', identifier='b86daf'),
                                AudioUrl(url='https://example.com/a.mp3'),
                                VideoUrl(url='https://example.com/a.mp4', media_type='video/mp4', identifier='1228be'),
                                DocumentUrl(
                                    url='https://example.com/a.pdf', media_type='application/pdf', identifier='390e3c'
                                ),
                            ],
                            timestamp=IsDatetime(),
                        )
                    ]
                )
            ]
        )

    def test_v4_uri_part_without_url_is_skipped(self):
        """A `uri` part recorded with `include_content=False` (no `uri`) is dropped."""
        otel = [
            {
                'role': 'user',
                'parts': [
                    {'type': 'text', 'content': 'See attached'},
                    {'type': 'uri', 'modality': 'image', 'mime_type': 'image/png'},
                ],
            },
        ]
        assert otel_messages_to_model_messages(otel) == snapshot(
            [ModelRequest(parts=[UserPromptPart(content='See attached', timestamp=IsDatetime())])]
        )

    def test_v4_blob_parts(self):
        """v4+ OTEL `blob` parts map to `BinaryContent` in both user and assistant messages."""
        b64 = base64.b64encode(b'blob bytes').decode()
        otel = [
            {
                'role': 'user',
                'parts': [{'type': 'blob', 'modality': 'image', 'mime_type': 'image/png', 'content': b64}],
            },
            {'role': 'assistant', 'parts': [{'type': 'blob', 'mime_type': 'image/png', 'content': b64}]},
        ]
        assert otel_messages_to_model_messages(otel) == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content=[BinaryContent(data=b'blob bytes', media_type='image/png', identifier='2261db')],
                            timestamp=IsDatetime(),
                        )
                    ]
                ),
                ModelResponse(
                    parts=[
                        FilePart(content=BinaryContent(data=b'blob bytes', media_type='image/png', identifier='2261db'))
                    ],
                    timestamp=IsDatetime(),
                ),
            ]
        )

    def test_v4_file_part_reconstructs_uploaded_file(self):
        """A `file` part with `file_id` and `provider_name` round-trips to an `UploadedFile`."""
        otel = [
            {
                'role': 'user',
                'parts': [
                    {
                        'type': 'file',
                        'modality': 'document',
                        'file_id': 'file-abc',
                        'mime_type': 'application/pdf',
                        'provider_name': 'anthropic',
                    }
                ],
            },
        ]
        assert otel_messages_to_model_messages(otel) == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content=[
                                UploadedFile(
                                    file_id='file-abc', provider_name='anthropic', _media_type='application/pdf'
                                )
                            ],
                            timestamp=IsDatetime(),
                        )
                    ]
                )
            ]
        )

    def test_v4_file_part_without_provider_falls_back_to_marker(self):
        """Without `provider_name` (older traces / `include_content=False`) the file can't be rebuilt."""
        otel = [
            {
                'role': 'user',
                'parts': [
                    {'type': 'file', 'modality': 'document', 'file_id': 'file-abc', 'mime_type': 'application/pdf'}
                ],
            },
        ]
        assert otel_messages_to_model_messages(otel) == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='[unavailable file (application/pdf): provider-hosted reference not captured in OTEL]',
                            timestamp=IsDatetime(),
                        )
                    ]
                )
            ]
        )

    def test_v4_file_part_with_unknown_provider_falls_back_to_marker(self):
        """An unrecognized `provider_name` can't construct an `UploadedFile`, so a marker is used."""
        otel = [
            {
                'role': 'user',
                'parts': [
                    {
                        'type': 'file',
                        'modality': 'document',
                        'file_id': 'file-abc',
                        'mime_type': 'application/pdf',
                        'provider_name': 'not-a-real-provider',
                    }
                ],
            },
        ]
        assert otel_messages_to_model_messages(otel) == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='[unavailable file (application/pdf): provider-hosted reference not captured in OTEL]',
                            timestamp=IsDatetime(),
                        )
                    ]
                )
            ]
        )

    def test_finish_reason_preserved(self):
        otel = [
            {'role': 'user', 'parts': [{'type': 'text', 'content': 'Hello'}]},
            {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'Hi'}], 'finish_reason': 'stop'},
        ]
        assert otel_messages_to_model_messages(otel) == snapshot(
            [
                ModelRequest(parts=[UserPromptPart(content='Hello', timestamp=IsDatetime())]),
                ModelResponse(parts=[TextPart(content='Hi')], timestamp=IsDatetime(), finish_reason='stop'),
            ]
        )

    def test_missing_content_uses_empty_string(self):
        """When include_content=False was used, parts lack content fields."""
        otel = [
            {'role': 'user', 'parts': [{'type': 'text'}]},
            {'role': 'assistant', 'parts': [{'type': 'text'}]},
        ]
        assert otel_messages_to_model_messages(otel) == snapshot(
            [
                ModelRequest(parts=[UserPromptPart(content='', timestamp=IsDatetime())]),
                ModelResponse(parts=[TextPart(content='')], timestamp=IsDatetime()),
            ]
        )

    def test_logfire_semconv_response_field(self):
        """Logfire semconv uses 'response' instead of 'result' for tool call responses."""
        otel = [
            {
                'role': 'user',
                'parts': [
                    {'type': 'tool_call_response', 'id': 'call_1', 'name': 'my_tool', 'response': 'tool result'},
                ],
            },
        ]
        assert otel_messages_to_model_messages(otel) == snapshot(
            [
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='my_tool', content='tool result', tool_call_id='call_1', timestamp=IsDatetime()
                        )
                    ]
                )
            ]
        )

    def test_mixed_user_content_and_tool_returns(self):
        """User messages can contain both text parts and tool_call_response parts."""
        otel = [
            {
                'role': 'user',
                'parts': [
                    {'type': 'text', 'content': 'Here is context'},
                    {'type': 'tool_call_response', 'id': 'call_1', 'name': 'my_tool', 'result': 'result'},
                ],
            },
        ]
        assert otel_messages_to_model_messages(otel) == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(content='Here is context', timestamp=IsDatetime()),
                        ToolReturnPart(
                            tool_name='my_tool', content='result', tool_call_id='call_1', timestamp=IsDatetime()
                        ),
                    ]
                )
            ]
        )

    def test_leading_assistant_without_pending_request(self):
        """An assistant message with no preceding user/system message starts the result."""
        otel = [
            {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'Hello!'}]},
        ]
        assert otel_messages_to_model_messages(otel) == snapshot(
            [ModelResponse(parts=[TextPart(content='Hello!')], timestamp=IsDatetime())]
        )

    def test_unknown_part_types_are_skipped(self):
        """Unrecognized part types in system/user/assistant messages are dropped."""
        otel = [
            {
                'role': 'system',
                'parts': [{'type': 'mystery'}, {'type': 'text', 'content': 'Be helpful.'}],
            },
            {
                'role': 'user',
                'parts': [{'type': 'text', 'content': 'Hi'}, {'type': 'mystery'}],
            },
            {
                'role': 'assistant',
                'parts': [
                    {'type': 'tool_call_response', 'id': 'call_1', 'name': 't', 'result': 'r'},
                    {'type': 'mystery'},
                    {'type': 'text', 'content': 'Done.'},
                ],
            },
        ]
        assert otel_messages_to_model_messages(otel) == snapshot(
            [
                ModelRequest(
                    parts=[
                        SystemPromptPart(content='Be helpful.', timestamp=IsDatetime()),
                        UserPromptPart(content='Hi', timestamp=IsDatetime()),
                    ]
                ),
                ModelResponse(parts=[TextPart(content='Done.')], timestamp=IsDatetime()),
            ]
        )


# ── otel_messages_to_model_messages: Legacy events format ─────────────


class TestLegacyEventsToModelMessages:
    def test_simple_conversation(self):
        events = [
            {
                'event.name': 'gen_ai.system.message',
                'role': 'system',
                'content': 'Be concise.',
                'gen_ai.message.index': 0,
            },
            {'event.name': 'gen_ai.user.message', 'role': 'user', 'content': 'Hello', 'gen_ai.message.index': 1},
            {
                'event.name': 'gen_ai.choice',
                'gen_ai.message.index': 2,
                'message': {'role': 'assistant', 'content': 'Hi!'},
            },
        ]
        assert otel_messages_to_model_messages(events) == snapshot(
            [
                ModelRequest(
                    parts=[
                        SystemPromptPart(content='Be concise.', timestamp=IsDatetime()),
                        UserPromptPart(content='Hello', timestamp=IsDatetime()),
                    ]
                ),
                ModelResponse(parts=[TextPart(content='Hi!')], timestamp=IsDatetime()),
            ]
        )

    def test_empty_system_and_user_content_skipped(self):
        """Empty/non-str system content and empty user content produce no parts."""
        events = [
            {'event.name': 'gen_ai.system.message', 'role': 'system', 'content': '', 'gen_ai.message.index': 0},
            {'event.name': 'gen_ai.user.message', 'role': 'user', 'content': '', 'gen_ai.message.index': 1},
            {
                'event.name': 'gen_ai.choice',
                'gen_ai.message.index': 2,
                'message': {'role': 'assistant', 'content': 'Hi!'},
            },
        ]
        assert otel_messages_to_model_messages(events) == snapshot(
            [ModelResponse(parts=[TextPart(content='Hi!')], timestamp=IsDatetime())]
        )

    def test_non_str_user_content_is_stringified(self):
        events = [
            {'event.name': 'gen_ai.user.message', 'role': 'user', 'content': [1, 2], 'gen_ai.message.index': 0},
            {
                'event.name': 'gen_ai.choice',
                'gen_ai.message.index': 1,
                'message': {'role': 'assistant', 'content': 'ok'},
            },
        ]
        assert otel_messages_to_model_messages(events) == snapshot(
            [
                ModelRequest(parts=[UserPromptPart(content='[1, 2]', timestamp=IsDatetime())]),
                ModelResponse(parts=[TextPart(content='ok')], timestamp=IsDatetime()),
            ]
        )

    def test_tool_calls_and_returns(self):
        events = [
            {'event.name': 'gen_ai.user.message', 'role': 'user', 'content': 'Weather?', 'gen_ai.message.index': 0},
            {
                'event.name': 'gen_ai.assistant.message',
                'role': 'assistant',
                'gen_ai.message.index': 1,
                'tool_calls': [
                    {
                        'id': 'call_1',
                        'type': 'function',
                        'function': {'name': 'get_weather', 'arguments': '{"city":"London"}'},
                    },
                ],
            },
            {
                'event.name': 'gen_ai.tool.message',
                'role': 'tool',
                'id': 'call_1',
                'name': 'get_weather',
                'content': 'Sunny',
                'gen_ai.message.index': 2,
            },
            {
                'event.name': 'gen_ai.choice',
                'gen_ai.message.index': 3,
                'message': {'role': 'assistant', 'content': 'It is sunny in London.'},
            },
        ]
        assert otel_messages_to_model_messages(events) == snapshot(
            [
                ModelRequest(parts=[UserPromptPart(content='Weather?', timestamp=IsDatetime())]),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='get_weather', args='{"city":"London"}', tool_call_id='call_1')],
                    timestamp=IsDatetime(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather', content='Sunny', tool_call_id='call_1', timestamp=IsDatetime()
                        )
                    ]
                ),
                ModelResponse(parts=[TextPart(content='It is sunny in London.')], timestamp=IsDatetime()),
            ]
        )

    def test_tool_calls_in_choice(self):
        events = [
            {'event.name': 'gen_ai.user.message', 'role': 'user', 'content': 'Weather?', 'gen_ai.message.index': 0},
            {
                'event.name': 'gen_ai.choice',
                'gen_ai.message.index': 1,
                'message': {
                    'role': 'assistant',
                    'tool_calls': [
                        {
                            'id': 'call_1',
                            'type': 'function',
                            'function': {'name': 'get_weather', 'arguments': '{"city":"London"}'},
                        },
                    ],
                },
            },
            {
                'event.name': 'gen_ai.tool.message',
                'role': 'tool',
                'id': 'call_1',
                'name': 'get_weather',
                'content': 'Sunny',
                'gen_ai.message.index': 2,
            },
            {
                'event.name': 'gen_ai.choice',
                'gen_ai.message.index': 3,
                'message': {
                    'role': 'assistant',
                    'content': 'It is sunny.',
                    'tool_calls': [
                        {
                            'id': 'call_2',
                            'type': 'function',
                            'function': {'name': 'get_temp', 'arguments': '{"city":"London"}'},
                        },
                    ],
                },
            },
        ]
        assert otel_messages_to_model_messages(events) == snapshot(
            [
                ModelRequest(parts=[UserPromptPart(content='Weather?', timestamp=IsDatetime())]),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='get_weather', args='{"city":"London"}', tool_call_id='call_1')],
                    timestamp=IsDatetime(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather', content='Sunny', tool_call_id='call_1', timestamp=IsDatetime()
                        )
                    ]
                ),
                ModelResponse(
                    parts=[
                        TextPart(content='It is sunny.'),
                        ToolCallPart(tool_name='get_temp', args='{"city":"London"}', tool_call_id='call_2'),
                    ],
                    timestamp=IsDatetime(),
                ),
            ]
        )

    def test_thinking_in_choice(self):
        events = [
            {'event.name': 'gen_ai.user.message', 'role': 'user', 'content': 'Think', 'gen_ai.message.index': 0},
            {
                'event.name': 'gen_ai.choice',
                'gen_ai.message.index': 1,
                'message': {
                    'role': 'assistant',
                    'content': [
                        {'kind': 'thinking', 'text': 'Hmm...'},
                        {'kind': 'text', 'text': 'Answer'},
                    ],
                },
            },
        ]
        assert otel_messages_to_model_messages(events) == snapshot(
            [
                ModelRequest(parts=[UserPromptPart(content='Think', timestamp=IsDatetime())]),
                ModelResponse(
                    parts=[ThinkingPart(content='Hmm...'), TextPart(content='Answer')], timestamp=IsDatetime()
                ),
            ]
        )

    def test_choice_without_message_key(self):
        """A `gen_ai.choice` event may carry its body inline rather than under `message`."""
        events = [
            {
                'event.name': 'gen_ai.choice',
                'gen_ai.message.index': 0,
                'content': 'Inline content',
            },
        ]
        assert otel_messages_to_model_messages(events) == snapshot(
            [ModelResponse(parts=[TextPart(content='Inline content')], timestamp=IsDatetime())]
        )

    def test_empty_choice_produces_no_response(self):
        """A choice with no content or tool calls is dropped entirely."""
        events = [
            {'event.name': 'gen_ai.user.message', 'role': 'user', 'content': 'Hi', 'gen_ai.message.index': 0},
            {
                'event.name': 'gen_ai.choice',
                'gen_ai.message.index': 1,
                'message': {'role': 'assistant'},
            },
        ]
        assert otel_messages_to_model_messages(events) == snapshot(
            [ModelRequest(parts=[UserPromptPart(content='Hi', timestamp=IsDatetime())])]
        )

    def test_trailing_request_is_flushed(self):
        """A conversation ending with a request event flushes the pending parts."""
        events = [
            {'event.name': 'gen_ai.user.message', 'role': 'user', 'content': 'Hello?', 'gen_ai.message.index': 0},
        ]
        assert otel_messages_to_model_messages(events) == snapshot(
            [ModelRequest(parts=[UserPromptPart(content='Hello?', timestamp=IsDatetime())])]
        )

    def test_unknown_event_name_is_skipped(self):
        """An unrecognized request-side event name produces no parts."""
        events = [
            {'event.name': 'gen_ai.mystery.message', 'content': 'ignored', 'gen_ai.message.index': 0},
            {
                'event.name': 'gen_ai.choice',
                'gen_ai.message.index': 1,
                'message': {'role': 'assistant', 'content': 'Hi'},
            },
        ]
        assert otel_messages_to_model_messages(events) == snapshot(
            [ModelResponse(parts=[TextPart(content='Hi')], timestamp=IsDatetime())]
        )

    def test_choice_with_non_text_content(self):
        """Choice content that is neither a string nor a list is ignored, tool calls still parsed."""
        events = [
            {
                'event.name': 'gen_ai.choice',
                'gen_ai.message.index': 0,
                'message': {
                    'role': 'assistant',
                    'content': None,
                    'tool_calls': [
                        {'id': 'call_1', 'type': 'function', 'function': {'name': 'f', 'arguments': '{}'}},
                    ],
                },
            },
        ]
        assert otel_messages_to_model_messages(events) == snapshot(
            [
                ModelResponse(
                    parts=[ToolCallPart(tool_name='f', args='{}', tool_call_id='call_1')], timestamp=IsDatetime()
                )
            ]
        )

    def test_unknown_content_kind_is_skipped(self):
        """Items in a legacy content list with an unrecognized kind are dropped."""
        events = [
            {
                'event.name': 'gen_ai.choice',
                'gen_ai.message.index': 0,
                'message': {
                    'role': 'assistant',
                    'content': [
                        {'kind': 'mystery', 'text': 'dropped'},
                        {'kind': 'text', 'text': 'kept'},
                    ],
                },
            },
        ]
        assert otel_messages_to_model_messages(events) == snapshot(
            [ModelResponse(parts=[TextPart(content='kept')], timestamp=IsDatetime())]
        )


# ── model_messages_to_openai_format ──────────────────────────────────


class TestModelMessagesToOpenaiFormat:
    def test_empty(self):
        assert model_messages_to_openai_format([]) == []

    def test_simple_conversation(self):
        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    SystemPromptPart('Be helpful.'),
                    UserPromptPart('Hello'),
                ]
            ),
            ModelResponse(parts=[TextPart('Hi there!')]),
        ]
        result = model_messages_to_openai_format(messages)
        assert result == snapshot(
            [
                {'role': 'system', 'content': 'Be helpful.'},
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': 'Hi there!'},
            ]
        )

    def test_tool_call_and_return(self):
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart('Weather?')]),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='get_weather', args='{"city":"London"}', tool_call_id='call_1'),
                ]
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name='get_weather', content='Sunny', tool_call_id='call_1'),
                ]
            ),
            ModelResponse(parts=[TextPart('It is sunny.')]),
        ]
        result = model_messages_to_openai_format(messages)
        assert result == snapshot(
            [
                {'role': 'user', 'content': 'Weather?'},
                {
                    'role': 'assistant',
                    'content': None,
                    'tool_calls': [
                        {
                            'id': 'call_1',
                            'type': 'function',
                            'function': {'name': 'get_weather', 'arguments': '{"city":"London"}'},
                        },
                    ],
                },
                {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'Sunny'},
                {'role': 'assistant', 'content': 'It is sunny.'},
            ]
        )

    def test_text_with_tool_calls(self):
        """Assistant message with both text and tool calls."""
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart('Hello')]),
            ModelResponse(
                parts=[
                    TextPart('Let me check.'),
                    ToolCallPart(tool_name='search', args='{"q":"test"}', tool_call_id='call_1'),
                ]
            ),
        ]
        result = model_messages_to_openai_format(messages)
        assert result == snapshot(
            [
                {'role': 'user', 'content': 'Hello'},
                {
                    'role': 'assistant',
                    'content': 'Let me check.',
                    'tool_calls': [
                        {
                            'id': 'call_1',
                            'type': 'function',
                            'function': {'name': 'search', 'arguments': '{"q":"test"}'},
                        },
                    ],
                },
            ]
        )

    def test_thinking_parts_excluded(self):
        """ThinkingParts should not appear in OpenAI format."""
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart('Think')]),
            ModelResponse(
                parts=[
                    ThinkingPart('Let me think...'),
                    TextPart('Here is my answer.'),
                ]
            ),
        ]
        result = model_messages_to_openai_format(messages)
        assert result == snapshot(
            [
                {'role': 'user', 'content': 'Think'},
                {'role': 'assistant', 'content': 'Here is my answer.'},
            ]
        )

    def test_response_with_only_thinking_is_skipped(self):
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart('Think')]),
            ModelResponse(
                parts=[
                    ThinkingPart('Let me think...'),
                ]
            ),
        ]
        result = model_messages_to_openai_format(messages)
        assert result == snapshot(
            [
                {'role': 'user', 'content': 'Think'},
            ]
        )

    def test_image_url_in_user_content(self):
        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        [
                            'What is in this image?',
                            ImageUrl('https://example.com/cat.png'),
                        ]
                    ),
                ]
            ),
        ]
        result = model_messages_to_openai_format(messages)
        assert result == snapshot(
            [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': 'What is in this image?'},
                        {'type': 'image_url', 'image_url': {'url': 'https://example.com/cat.png'}},
                    ],
                },
            ]
        )

    def test_image_url_detail_metadata_preserved(self):
        """`vendor_metadata['detail']` is carried over to the OpenAI `image_url` fidelity hint."""
        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    UserPromptPart([ImageUrl('https://example.com/cat.png', vendor_metadata={'detail': 'high'})]),
                ]
            ),
        ]
        result = model_messages_to_openai_format(messages)
        assert result == snapshot(
            [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image_url', 'image_url': {'url': 'https://example.com/cat.png', 'detail': 'high'}}
                    ],
                }
            ]
        )

    def test_binary_image_and_audio_in_user_content(self):
        """Image binary becomes a data URI; audio binary becomes input_audio."""
        png = base64.b64encode(b'png-bytes').decode()
        wav = base64.b64encode(b'wav-bytes').decode()
        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        [
                            BinaryContent(data=base64.b64decode(png), media_type='image/png'),
                            BinaryContent(data=base64.b64decode(wav), media_type='audio/wav'),
                        ]
                    ),
                ]
            ),
        ]
        result = model_messages_to_openai_format(messages)
        assert result == snapshot(
            [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,cG5nLWJ5dGVz'}},
                        {'type': 'input_audio', 'input_audio': {'data': 'd2F2LWJ5dGVz', 'format': 'wav'}},
                    ],
                }
            ]
        )

    def test_non_media_binary_falls_back_to_text(self):
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart([BinaryContent(data=b'%PDF', media_type='application/pdf')])]),
        ]
        result = model_messages_to_openai_format(messages)
        assert result == snapshot([{'role': 'user', 'content': '[Binary: application/pdf]'}])

    def test_audio_url_and_other_urls_fall_back_to_text(self):
        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        [
                            AudioUrl('https://example.com/a.mp3'),
                            DocumentUrl('https://example.com/d.pdf'),
                            VideoUrl('https://example.com/v.mp4'),
                        ]
                    ),
                ]
            ),
        ]
        result = model_messages_to_openai_format(messages)
        assert result == snapshot(
            [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': '[Audio: https://example.com/a.mp3]'},
                        {'type': 'text', 'text': '[DocumentUrl: https://example.com/d.pdf]'},
                        {'type': 'text', 'text': '[VideoUrl: https://example.com/v.mp4]'},
                    ],
                }
            ]
        )

    def test_cache_point_is_dropped(self):
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(['Hello', CachePoint()])]),
        ]
        result = model_messages_to_openai_format(messages)
        assert result == snapshot([{'role': 'user', 'content': 'Hello'}])

    def test_text_content_in_user_content(self):
        """`TextContent` contributes its `.content` as a text part, like a plain `str`."""
        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    UserPromptPart(['Plain text', TextContent(content='Tagged text', metadata={'tag': 'important'})]),
                ]
            ),
        ]
        result = model_messages_to_openai_format(messages)
        assert result == snapshot(
            [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': 'Plain text'},
                        {'type': 'text', 'text': 'Tagged text'},
                    ],
                }
            ]
        )

    def test_uploaded_file_falls_back_to_text(self):
        """`UploadedFile` is a provider-hosted reference with no Chat Completions equivalent, so it becomes a marker."""
        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    UserPromptPart(['What is in this file?', UploadedFile(file_id='file-abc', provider_name='openai')]),
                ]
            ),
        ]
        result = model_messages_to_openai_format(messages)
        assert result == snapshot(
            [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': 'What is in this file?'},
                        {'type': 'text', 'text': '[UploadedFile: file-abc (openai)]'},
                    ],
                }
            ]
        )

    def test_retry_prompt_with_tool(self):
        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    RetryPromptPart(content='Try again', tool_name='my_tool', tool_call_id='call_1'),
                ]
            ),
        ]
        result = model_messages_to_openai_format(messages)
        assert result == snapshot(
            [
                {
                    'role': 'tool',
                    'tool_call_id': 'call_1',
                    'content': """\
Try again

Fix the errors and try again.\
""",
                }
            ]
        )

    def test_retry_prompt_without_tool(self):
        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    RetryPromptPart(content='Invalid output'),
                ]
            ),
        ]
        result = model_messages_to_openai_format(messages)
        assert result == snapshot(
            [
                {
                    'role': 'user',
                    'content': """\
Validation feedback:
Invalid output

Fix the errors and try again.\
""",
                }
            ]
        )

    def test_unknown_audio_format_falls_back_to_text(self):
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart([BinaryContent(data=b'abc', media_type='audio/webm')])]),
        ]
        result = model_messages_to_openai_format(messages)
        assert result == snapshot(
            [
                {'role': 'user', 'content': '[Audio: audio/webm]'},
            ]
        )

    def test_non_openai_audio_format_falls_back_to_text(self):
        """Audio formats OpenAI doesn't accept (e.g. flac) fall back to a text marker."""
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart([BinaryContent(data=b'abc', media_type='audio/flac')])]),
        ]
        result = model_messages_to_openai_format(messages)
        assert result == snapshot([{'role': 'user', 'content': '[Audio: audio/flac]'}])


# ── Round-trip tests ─────────────────────────────────────────────────


class TestRoundTrip:
    """Test that converting ModelMessages → OTEL → ModelMessages preserves semantics."""

    def test_round_trip_simple(self):
        """Simple text conversation round-trips through OTEL format."""
        otel = [
            {'role': 'user', 'parts': [{'type': 'text', 'content': 'Hello'}]},
            {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'Hi'}]},
        ]
        messages = otel_messages_to_model_messages(otel)
        openai_fmt = model_messages_to_openai_format(messages)
        assert openai_fmt == snapshot(
            [
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': 'Hi'},
            ]
        )

    def test_round_trip_with_tools(self):
        otel = [
            {'role': 'user', 'parts': [{'type': 'text', 'content': 'Weather?'}]},
            {
                'role': 'assistant',
                'parts': [
                    {'type': 'tool_call', 'id': 'call_1', 'name': 'weather', 'arguments': {'city': 'NYC'}},
                ],
            },
            {
                'role': 'user',
                'parts': [
                    {'type': 'tool_call_response', 'id': 'call_1', 'name': 'weather', 'result': 'Rainy'},
                ],
            },
            {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'It is rainy.'}]},
        ]
        messages = otel_messages_to_model_messages(otel)
        openai_fmt = model_messages_to_openai_format(messages)
        assert openai_fmt == snapshot(
            [
                {'role': 'user', 'content': 'Weather?'},
                {
                    'role': 'assistant',
                    'content': None,
                    'tool_calls': [
                        {
                            'id': 'call_1',
                            'type': 'function',
                            'function': {'name': 'weather', 'arguments': '{"city":"NYC"}'},
                        },
                    ],
                },
                {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'Rainy'},
                {'role': 'assistant', 'content': 'It is rainy.'},
            ]
        )
