"""Tests for _conversions.py — round-tripping OTEL messages and OpenAI format conversion."""

from __future__ import annotations

import json

from inline_snapshot import snapshot

from pydantic_ai._conversions import model_messages_to_openai_format, otel_messages_to_model_messages
from pydantic_ai.messages import (
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    FilePart,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

# ── otel_messages_to_model_messages: ChatMessage format ────────────────


class TestChatMessagesToModelMessages:
    def test_empty(self):
        assert otel_messages_to_model_messages([]) == []

    def test_simple_text_conversation(self):
        otel = [
            {'role': 'user', 'parts': [{'type': 'text', 'content': 'Hello'}]},
            {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'Hi there!'}]},
        ]
        result = otel_messages_to_model_messages(otel)
        assert len(result) == 2
        assert isinstance(result[0], ModelRequest)
        assert len(result[0].parts) == 1
        assert isinstance(result[0].parts[0], UserPromptPart)
        assert result[0].parts[0].content == 'Hello'
        assert isinstance(result[1], ModelResponse)
        assert result[1].text == 'Hi there!'

    def test_json_string_input(self):
        otel_json = json.dumps(
            [
                {'role': 'user', 'parts': [{'type': 'text', 'content': 'Hello'}]},
                {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'Hi'}]},
            ]
        )
        result = otel_messages_to_model_messages(otel_json)
        assert len(result) == 2
        assert isinstance(result[0], ModelRequest)
        assert isinstance(result[1], ModelResponse)

    def test_system_and_user_merged_into_request(self):
        otel = [
            {'role': 'system', 'parts': [{'type': 'text', 'content': 'Be helpful.'}]},
            {'role': 'user', 'parts': [{'type': 'text', 'content': 'Hello'}]},
            {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'Hi'}]},
        ]
        result = otel_messages_to_model_messages(otel)
        assert len(result) == 2
        # System and user should be merged into one ModelRequest
        assert isinstance(result[0], ModelRequest)
        assert len(result[0].parts) == 2
        assert isinstance(result[0].parts[0], SystemPromptPart)
        assert result[0].parts[0].content == 'Be helpful.'
        assert isinstance(result[0].parts[1], UserPromptPart)
        assert result[0].parts[1].content == 'Hello'

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
        result = otel_messages_to_model_messages(otel)
        assert len(result) == 4

        # User prompt
        assert isinstance(result[0], ModelRequest)
        assert isinstance(result[0].parts[0], UserPromptPart)

        # Assistant tool call
        assert isinstance(result[1], ModelResponse)
        assert len(result[1].parts) == 1
        tc = result[1].parts[0]
        assert isinstance(tc, ToolCallPart)
        assert tc.tool_name == 'get_weather'
        assert tc.tool_call_id == 'call_1'
        assert tc.args == {'city': 'London'}

        # Tool return
        assert isinstance(result[2], ModelRequest)
        assert len(result[2].parts) == 1
        tr = result[2].parts[0]
        assert isinstance(tr, ToolReturnPart)
        assert tr.tool_name == 'get_weather'
        assert tr.tool_call_id == 'call_1'
        assert tr.content == 'Sunny, 21°C'

        # Final response
        assert isinstance(result[3], ModelResponse)
        assert result[3].text == 'It is sunny and 21°C in London.'

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
        result = otel_messages_to_model_messages(otel)
        assert isinstance(result[1], ModelResponse)
        assert len(result[1].parts) == 2
        assert isinstance(result[1].parts[0], ThinkingPart)
        assert result[1].parts[0].content == 'Let me think...'
        assert isinstance(result[1].parts[1], TextPart)

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
        result = otel_messages_to_model_messages(otel)
        assert isinstance(result[1], ModelResponse)
        parts = result[1].parts
        assert isinstance(parts[0], BuiltinToolCallPart)
        assert parts[0].tool_name == 'web_search'
        assert isinstance(parts[1], BuiltinToolReturnPart)
        assert parts[1].content == 'Cats are great'
        assert isinstance(parts[2], TextPart)

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
        result = otel_messages_to_model_messages(otel)
        assert isinstance(result[0], ModelRequest)
        part = result[0].parts[0]
        assert isinstance(part, UserPromptPart)
        assert isinstance(part.content, list)
        assert len(part.content) == 2
        assert part.content[0] == 'What is in this image?'
        assert isinstance(part.content[1], ImageUrl)
        assert part.content[1].url == 'https://example.com/cat.png'

    def test_binary_content_in_user_message(self):
        import base64

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
        result = otel_messages_to_model_messages(otel)
        part = result[0].parts[0]
        assert isinstance(part, UserPromptPart)
        assert isinstance(part.content, list)
        assert isinstance(part.content[1], BinaryContent)
        assert part.content[1].data == data
        assert part.content[1].media_type == 'image/png'

    def test_binary_content_in_assistant_message(self):
        import base64

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
        result = otel_messages_to_model_messages(otel)
        assert isinstance(result[1], ModelResponse)
        part = result[1].parts[0]
        assert isinstance(part, FilePart)
        assert part.content.data == data

    def test_finish_reason_preserved(self):
        otel = [
            {'role': 'user', 'parts': [{'type': 'text', 'content': 'Hello'}]},
            {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'Hi'}], 'finish_reason': 'stop'},
        ]
        result = otel_messages_to_model_messages(otel)
        assert isinstance(result[1], ModelResponse)
        assert result[1].finish_reason == 'stop'

    def test_missing_content_uses_empty_string(self):
        """When include_content=False was used, parts lack content fields."""
        otel = [
            {'role': 'user', 'parts': [{'type': 'text'}]},
            {'role': 'assistant', 'parts': [{'type': 'text'}]},
        ]
        result = otel_messages_to_model_messages(otel)
        assert isinstance(result[0], ModelRequest)
        assert isinstance(result[0].parts[0], UserPromptPart)
        assert result[0].parts[0].content == ''
        assert isinstance(result[1], ModelResponse)
        assert result[1].text == ''

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
        result = otel_messages_to_model_messages(otel)
        assert isinstance(result[0], ModelRequest)
        tr = result[0].parts[0]
        assert isinstance(tr, ToolReturnPart)
        assert tr.content == 'tool result'

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
        result = otel_messages_to_model_messages(otel)
        assert isinstance(result[0], ModelRequest)
        assert len(result[0].parts) == 2
        assert isinstance(result[0].parts[0], UserPromptPart)
        assert result[0].parts[0].content == 'Here is context'
        assert isinstance(result[0].parts[1], ToolReturnPart)


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
        result = otel_messages_to_model_messages(events)
        assert len(result) == 2
        assert isinstance(result[0], ModelRequest)
        assert len(result[0].parts) == 2
        assert isinstance(result[0].parts[0], SystemPromptPart)
        assert isinstance(result[0].parts[1], UserPromptPart)
        assert isinstance(result[1], ModelResponse)
        assert result[1].text == 'Hi!'

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
        result = otel_messages_to_model_messages(events)
        assert len(result) == 4

        # User
        assert isinstance(result[0], ModelRequest)
        assert isinstance(result[0].parts[0], UserPromptPart)

        # Assistant with tool call
        assert isinstance(result[1], ModelResponse)
        tc = result[1].parts[0]
        assert isinstance(tc, ToolCallPart)
        assert tc.tool_name == 'get_weather'
        assert tc.args == '{"city":"London"}'

        # Tool return
        assert isinstance(result[2], ModelRequest)
        tr = result[2].parts[0]
        assert isinstance(tr, ToolReturnPart)
        assert tr.tool_name == 'get_weather'
        assert tr.content == 'Sunny'

        # Final choice
        assert isinstance(result[3], ModelResponse)
        assert result[3].text == 'It is sunny in London.'

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
        result = otel_messages_to_model_messages(events)
        assert len(result) == 4

        # User
        assert isinstance(result[0], ModelRequest)

        # Choice with only tool calls
        assert isinstance(result[1], ModelResponse)
        assert len(result[1].parts) == 1
        tc = result[1].parts[0]
        assert isinstance(tc, ToolCallPart)
        assert tc.tool_name == 'get_weather'
        assert tc.args == '{"city":"London"}'
        assert tc.tool_call_id == 'call_1'

        # Tool return
        assert isinstance(result[2], ModelRequest)

        # Choice with both content and tool calls
        assert isinstance(result[3], ModelResponse)
        assert len(result[3].parts) == 2
        assert isinstance(result[3].parts[0], TextPart)
        assert result[3].parts[0].content == 'It is sunny.'
        tc2 = result[3].parts[1]
        assert isinstance(tc2, ToolCallPart)
        assert tc2.tool_name == 'get_temp'
        assert tc2.tool_call_id == 'call_2'

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
        result = otel_messages_to_model_messages(events)
        assert isinstance(result[1], ModelResponse)
        assert len(result[1].parts) == 2
        assert isinstance(result[1].parts[0], ThinkingPart)
        assert result[1].parts[0].content == 'Hmm...'
        assert isinstance(result[1].parts[1], TextPart)
        assert result[1].parts[1].content == 'Answer'


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

    def test_retry_prompt_with_tool(self):
        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    RetryPromptPart(content='Try again', tool_name='my_tool', tool_call_id='call_1'),
                ]
            ),
        ]
        result = model_messages_to_openai_format(messages)
        assert len(result) == 1
        assert result[0]['role'] == 'tool'
        assert result[0]['tool_call_id'] == 'call_1'
        assert 'Try again' in result[0]['content']

    def test_retry_prompt_without_tool(self):
        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    RetryPromptPart(content='Invalid output'),
                ]
            ),
        ]
        result = model_messages_to_openai_format(messages)
        assert len(result) == 1
        assert result[0]['role'] == 'user'
        assert 'Invalid output' in result[0]['content']

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
