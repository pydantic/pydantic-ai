"""Tests for unified typed schemas for builtin tool return content.

This tests the feature from GitHub issue #3561 - adding TypedDict schemas for
builtin tool return content with unified field definitions across providers.
"""

from __future__ import annotations

import pydantic

from pydantic_ai.messages import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    CodeExecutionCallPart,
    CodeExecutionReturnContent,
    CodeExecutionReturnPart,
    FileSearchCallPart,
    FileSearchResult,
    FileSearchReturnContent,
    FileSearchReturnPart,
    ImageGenerationCallPart,
    ImageGenerationReturnContent,
    ImageGenerationReturnPart,
    ModelResponse,
    WebFetchCallPart,
    WebFetchPage,
    WebFetchReturnContent,
    WebFetchReturnPart,
    WebSearchCallPart,
    WebSearchReturnContent,
    WebSearchReturnPart,
    WebSearchSource,
)


class TestTypedReturnParts:
    """Test the specialized return part subclasses with typed content."""

    def test_code_execution_return_part_anthropic_format(self):
        """Test CodeExecutionReturnPart with Anthropic-style content."""
        content: CodeExecutionReturnContent = {
            'stdout': 'Hello, World!',
            'stderr': '',
            'return_code': 0,
            'type': 'code_execution_result',
        }
        part = CodeExecutionReturnPart(
            tool_name='code_execution',
            tool_call_id='test-id',
            content=content,
        )
        assert part.content['stdout'] == 'Hello, World!'
        assert part.content.get('stderr') == ''
        assert part.content['return_code'] == 0
        assert part.part_kind == 'code-execution-return'

    def test_code_execution_return_part_openai_format(self):
        """Test CodeExecutionReturnPart with OpenAI-style content."""
        content: CodeExecutionReturnContent = {
            'status': 'completed',
            'logs': ['Line 1', 'Line 2'],
        }
        part = CodeExecutionReturnPart(
            tool_name='code_execution',
            tool_call_id='test-id',
            content=content,
        )
        assert part.content['status'] == 'completed'
        assert part.content['logs'] == ['Line 1', 'Line 2']
        assert part.part_kind == 'code-execution-return'

    def test_code_execution_return_part_google_format(self):
        """Test CodeExecutionReturnPart with Google-style content."""
        content: CodeExecutionReturnContent = {
            'outcome': 'OUTCOME_OK',
            'output': 'Result: 42',
        }
        part = CodeExecutionReturnPart(
            tool_name='code_execution',
            tool_call_id='test-id',
            content=content,
        )
        assert part.content['output'] == 'Result: 42'
        assert part.content['outcome'] == 'OUTCOME_OK'
        assert part.part_kind == 'code-execution-return'

    def test_web_search_return_part_openai_format(self):
        """Test WebSearchReturnPart with OpenAI-style dict content."""
        content: WebSearchReturnContent = {
            'status': 'completed',
            'sources': [
                {'title': 'Example', 'url': 'https://example.com'},
                {'title': 'Test', 'url': 'https://test.com'},
            ],
        }
        part = WebSearchReturnPart(
            tool_name='web_search',
            tool_call_id='test-id',
            content=content,
        )
        assert part.content['status'] == 'completed'
        assert len(part.content['sources']) == 2
        assert part.content['sources'][0]['title'] == 'Example'
        assert part.part_kind == 'web-search-return'

    def test_web_search_return_part_list_format(self):
        """Test WebSearchReturnPart with list content (Google/Anthropic style)."""
        content: list[WebSearchSource] = [
            {'title': 'Result 1', 'uri': 'https://example1.com'},
            {'title': 'Result 2', 'uri': 'https://example2.com'},
        ]
        part = WebSearchReturnPart(
            tool_name='web_search',
            tool_call_id='test-id',
            content=content,
        )
        assert len(part.content) == 2
        assert part.content[0]['uri'] == 'https://example1.com'
        assert part.part_kind == 'web-search-return'

    def test_web_fetch_return_part_anthropic_format(self):
        """Test WebFetchReturnPart with Anthropic-style single dict content."""
        content: WebFetchReturnContent = {
            'content': 'Page content here...',
            'type': 'text/html',
            'url': 'https://example.com',
            'retrieved_at': '2024-01-01T00:00:00Z',
        }
        part = WebFetchReturnPart(
            tool_name='web_fetch',
            tool_call_id='test-id',
            content=content,
        )
        assert part.content['content'] == 'Page content here...'
        assert part.content['url'] == 'https://example.com'
        assert part.part_kind == 'web-fetch-return'

    def test_web_fetch_return_part_list_format(self):
        """Test WebFetchReturnPart with list content (Google style)."""
        content: list[WebFetchPage] = [
            {'retrieved_url': 'https://example1.com', 'content': 'Content 1'},
            {'retrieved_url': 'https://example2.com', 'content': 'Content 2'},
        ]
        part = WebFetchReturnPart(
            tool_name='web_fetch',
            tool_call_id='test-id',
            content=content,
        )
        assert len(part.content) == 2
        assert part.content[0]['retrieved_url'] == 'https://example1.com'
        assert part.part_kind == 'web-fetch-return'

    def test_file_search_return_part_openai_format(self):
        """Test FileSearchReturnPart with OpenAI-style dict content."""
        content: FileSearchReturnContent = {
            'status': 'completed',
            'results': [
                {'id': 'file-1', 'filename': 'doc.pdf', 'score': 0.95, 'content': 'Relevant content'},
            ],
        }
        part = FileSearchReturnPart(
            tool_name='file_search',
            tool_call_id='test-id',
            content=content,
        )
        assert part.content['status'] == 'completed'
        assert len(part.content['results']) == 1
        assert part.content['results'][0]['filename'] == 'doc.pdf'
        assert part.part_kind == 'file-search-return'

    def test_file_search_return_part_list_format(self):
        """Test FileSearchReturnPart with list content (Google style)."""
        content: list[FileSearchResult] = [
            {'file_search_store': 'store-1', 'content': 'Retrieved context 1'},
            {'file_search_store': 'store-2', 'content': 'Retrieved context 2'},
        ]
        part = FileSearchReturnPart(
            tool_name='file_search',
            tool_call_id='test-id',
            content=content,
        )
        assert len(part.content) == 2
        assert part.content[0]['file_search_store'] == 'store-1'
        assert part.part_kind == 'file-search-return'

    def test_image_generation_return_part(self):
        """Test ImageGenerationReturnPart with OpenAI-style content."""
        content: ImageGenerationReturnContent = {
            'status': 'completed',
            'revised_prompt': 'A beautiful sunset over mountains',
            'size': '1024x1024',
            'quality': 'high',
            'background': 'opaque',
        }
        part = ImageGenerationReturnPart(
            tool_name='image_generation',
            tool_call_id='test-id',
            content=content,
        )
        assert part.content['status'] == 'completed'
        assert part.content['revised_prompt'] == 'A beautiful sunset over mountains'
        assert part.content['size'] == '1024x1024'
        assert part.content['quality'] == 'high'
        assert part.content['background'] == 'opaque'
        assert part.part_kind == 'image-generation-return'


class TestTypedCallParts:
    """Test the specialized call part subclasses."""

    def test_code_execution_call_part(self):
        """Test CodeExecutionCallPart with typed args."""
        part = CodeExecutionCallPart(
            tool_name='code_execution',
            tool_call_id='test-id',
            args={'code': 'print("Hello")', 'container_id': 'container-123'},
        )
        assert part.code == 'print("Hello")'
        assert part.part_kind == 'code-execution-call'

    def test_web_search_call_part(self):
        """Test WebSearchCallPart with typed args."""
        part = WebSearchCallPart(
            tool_name='web_search',
            tool_call_id='test-id',
            args={'query': 'Python programming'},
        )
        assert part.query == 'Python programming'
        assert part.part_kind == 'web-search-call'

    def test_web_fetch_call_part_single_url(self):
        """Test WebFetchCallPart with single URL."""
        part = WebFetchCallPart(
            tool_name='web_fetch',
            tool_call_id='test-id',
            args={'url': 'https://example.com'},
        )
        assert part.urls == ['https://example.com']
        assert part.part_kind == 'web-fetch-call'

    def test_web_fetch_call_part_multiple_urls(self):
        """Test WebFetchCallPart with multiple URLs."""
        part = WebFetchCallPart(
            tool_name='web_fetch',
            tool_call_id='test-id',
            args={'urls': ['https://example1.com', 'https://example2.com']},
        )
        assert len(part.urls) == 2
        assert part.part_kind == 'web-fetch-call'

    def test_file_search_call_part_single_query(self):
        """Test FileSearchCallPart with single query."""
        part = FileSearchCallPart(
            tool_name='file_search',
            tool_call_id='test-id',
            args={'query': 'search term'},
        )
        assert part.queries == ['search term']
        assert part.part_kind == 'file-search-call'

    def test_file_search_call_part_multiple_queries(self):
        """Test FileSearchCallPart with multiple queries."""
        part = FileSearchCallPart(
            tool_name='file_search',
            tool_call_id='test-id',
            args={'queries': ['term1', 'term2']},
        )
        assert len(part.queries) == 2
        assert part.part_kind == 'file-search-call'

    def test_image_generation_call_part(self):
        """Test ImageGenerationCallPart."""
        part = ImageGenerationCallPart(
            tool_name='image_generation',
            tool_call_id='test-id',
        )
        assert part.part_kind == 'image-generation-call'


class TestBackwardsCompatibility:
    """Test backwards compatibility with old serialized data."""

    def test_old_builtin_tool_return_part_migration_code_execution(self):
        """Test that old part_kind='builtin-tool-return' migrates to specific subclass."""
        old_data = {
            'parts': [
                {
                    'part_kind': 'builtin-tool-return',
                    'tool_name': 'code_execution',
                    'tool_call_id': 'test-id',
                    'content': {'stdout': 'Hello', 'return_code': 0},
                }
            ],
            'kind': 'response',
        }
        response = pydantic.TypeAdapter(ModelResponse).validate_python(old_data)
        assert len(response.parts) == 1
        part = response.parts[0]
        assert isinstance(part, CodeExecutionReturnPart)
        assert part.content['stdout'] == 'Hello'
        assert part.content['return_code'] == 0

    def test_old_builtin_tool_return_part_migration_web_search(self):
        """Test that old web search returns migrate correctly."""
        old_data = {
            'parts': [
                {
                    'part_kind': 'builtin-tool-return',
                    'tool_name': 'web_search',
                    'tool_call_id': 'test-id',
                    'content': {'status': 'completed', 'sources': [{'url': 'https://example.com'}]},
                }
            ],
            'kind': 'response',
        }
        response = pydantic.TypeAdapter(ModelResponse).validate_python(old_data)
        part = response.parts[0]
        assert isinstance(part, WebSearchReturnPart)
        assert part.content['status'] == 'completed'

    def test_old_builtin_tool_call_part_migration(self):
        """Test that old part_kind='builtin-tool-call' migrates to specific subclass."""
        old_data = {
            'parts': [
                {
                    'part_kind': 'builtin-tool-call',
                    'tool_name': 'code_execution',
                    'tool_call_id': 'test-id',
                    'args': {'code': 'print(1)'},
                }
            ],
            'kind': 'response',
        }
        response = pydantic.TypeAdapter(ModelResponse).validate_python(old_data)
        part = response.parts[0]
        assert isinstance(part, CodeExecutionCallPart)
        assert part.code == 'print(1)'

    def test_mcp_server_stays_as_base_class(self):
        """Test that MCP server parts stay as base BuiltinToolReturnPart."""
        old_data = {
            'parts': [
                {
                    'part_kind': 'builtin-tool-return',
                    'tool_name': 'mcp_server:my-server',
                    'tool_call_id': 'test-id',
                    'content': {'output': 'result'},
                }
            ],
            'kind': 'response',
        }
        response = pydantic.TypeAdapter(ModelResponse).validate_python(old_data)
        part = response.parts[0]
        # Should stay as base class since MCP schema is not stable
        assert type(part) is BuiltinToolReturnPart

    def test_memory_stays_as_base_class(self):
        """Test that memory tool parts stay as base BuiltinToolReturnPart."""
        old_data: dict[str, object] = {
            'parts': [
                {
                    'part_kind': 'builtin-tool-return',
                    'tool_name': 'memory',
                    'tool_call_id': 'test-id',
                    'content': {'memories': []},
                }
            ],
            'kind': 'response',
        }
        response = pydantic.TypeAdapter(ModelResponse).validate_python(old_data)
        part = response.parts[0]
        assert type(part) is BuiltinToolReturnPart

    def test_url_context_migrates_to_web_fetch(self):
        """Test that deprecated url_context tool migrates to WebFetchReturnPart."""
        old_data = {
            'parts': [
                {
                    'part_kind': 'builtin-tool-return',
                    'tool_name': 'url_context',
                    'tool_call_id': 'test-id',
                    'content': {'url': 'https://example.com', 'content': 'page content'},
                }
            ],
            'kind': 'response',
        }
        response = pydantic.TypeAdapter(ModelResponse).validate_python(old_data)
        part = response.parts[0]
        assert isinstance(part, WebFetchReturnPart)
        assert part.content['url'] == 'https://example.com'

    def test_unknown_tool_stays_as_base_class(self):
        """Test that unknown tool names stay as base BuiltinToolReturnPart."""
        old_data = {
            'parts': [
                {
                    'part_kind': 'builtin-tool-return',
                    'tool_name': 'some_future_tool',
                    'tool_call_id': 'test-id',
                    'content': {'data': 'value'},
                }
            ],
            'kind': 'response',
        }
        response = pydantic.TypeAdapter(ModelResponse).validate_python(old_data)
        part = response.parts[0]
        assert type(part) is BuiltinToolReturnPart

    def test_new_part_kind_serializes_correctly(self):
        """Test that new specialized parts serialize with their specific part_kind."""
        part = CodeExecutionReturnPart(
            tool_name='code_execution',
            tool_call_id='test-id',
            content={'stdout': 'Hello'},
        )
        response = ModelResponse(parts=[part])
        serialized = pydantic.TypeAdapter(ModelResponse).dump_python(response, mode='json')
        assert serialized['parts'][0]['part_kind'] == 'code-execution-return'

    def test_roundtrip_serialization(self):
        """Test that specialized parts survive JSON roundtrip."""
        original = CodeExecutionReturnPart(
            tool_name='code_execution',
            tool_call_id='test-id',
            content={'stdout': 'Hello', 'return_code': 0},
        )
        response = ModelResponse(parts=[original])

        # Serialize to JSON
        serialized = pydantic.TypeAdapter(ModelResponse).dump_json(response)

        # Deserialize back
        deserialized = pydantic.TypeAdapter(ModelResponse).validate_json(serialized)

        part = deserialized.parts[0]
        assert isinstance(part, CodeExecutionReturnPart)
        assert part.content['stdout'] == 'Hello'
        assert part.content['return_code'] == 0


class TestIsinstanceChecks:
    """Test that isinstance checks work correctly with the new types."""

    def test_isinstance_code_execution_return_part(self):
        """Test isinstance check for CodeExecutionReturnPart."""
        part = CodeExecutionReturnPart(
            tool_name='code_execution',
            tool_call_id='test-id',
            content={'stdout': 'Hello'},
        )
        assert isinstance(part, CodeExecutionReturnPart)
        assert isinstance(part, BuiltinToolReturnPart)

    def test_isinstance_web_search_return_part(self):
        """Test isinstance check for WebSearchReturnPart."""
        part = WebSearchReturnPart(
            tool_name='web_search',
            tool_call_id='test-id',
            content=[],
        )
        assert isinstance(part, WebSearchReturnPart)
        assert isinstance(part, BuiltinToolReturnPart)

    def test_isinstance_code_execution_call_part(self):
        """Test isinstance check for CodeExecutionCallPart."""
        part = CodeExecutionCallPart(
            tool_name='code_execution',
            tool_call_id='test-id',
            args={'code': 'print(1)'},
        )
        assert isinstance(part, CodeExecutionCallPart)
        assert isinstance(part, BuiltinToolCallPart)

    def test_filter_by_type_in_response(self):
        """Test filtering response parts by specific type."""
        response = ModelResponse(
            parts=[
                CodeExecutionCallPart(
                    tool_name='code_execution',
                    tool_call_id='call-1',
                    args={'code': 'print(1)'},
                ),
                CodeExecutionReturnPart(
                    tool_name='code_execution',
                    tool_call_id='call-1',
                    content={'stdout': '1'},
                ),
                WebSearchCallPart(
                    tool_name='web_search',
                    tool_call_id='call-2',
                    args={'query': 'test'},
                ),
            ]
        )

        code_exec_returns = [p for p in response.parts if isinstance(p, CodeExecutionReturnPart)]
        assert len(code_exec_returns) == 1
        assert code_exec_returns[0].content['stdout'] == '1'

        web_search_calls = [p for p in response.parts if isinstance(p, WebSearchCallPart)]
        assert len(web_search_calls) == 1
        assert web_search_calls[0].query == 'test'


class TestTypedDictSchemas:
    """Test that TypedDict schemas provide correct type information."""

    def test_code_execution_content_type_hints(self):
        """Test that CodeExecutionReturnContent TypedDict has expected keys."""
        # This test verifies the TypedDict can be used for type annotation
        content: CodeExecutionReturnContent = {
            'stdout': 'test',
            'status': 'completed',
        }
        part = CodeExecutionReturnPart(
            tool_name='code_execution',
            tool_call_id='test-id',
            content=content,
        )
        # Accessing typed content
        assert part.content.get('stdout') == 'test'
        assert part.content.get('status') == 'completed'

    def test_web_search_source_type_hints(self):
        """Test that WebSearchSource TypedDict has expected keys."""
        source: WebSearchSource = {
            'title': 'Test Result',
            'url': 'https://example.com',
            'snippet': 'A snippet of text',
        }
        assert source['title'] == 'Test Result'
        assert source.get('relevance_score') is None  # Optional field

    def test_file_search_result_type_hints(self):
        """Test that FileSearchResult TypedDict has expected keys."""
        result: FileSearchResult = {
            'id': 'file-123',
            'filename': 'test.pdf',
            'score': 0.95,
            'content': 'Matched content',
        }
        assert result['filename'] == 'test.pdf'
        assert result['score'] == 0.95

    def test_image_generation_content_type_hints(self):
        """Test that ImageGenerationReturnContent TypedDict has expected keys."""
        content: ImageGenerationReturnContent = {
            'status': 'completed',
            'size': '1024x1024',
        }
        part = ImageGenerationReturnPart(
            tool_name='image_generation',
            tool_call_id='test-id',
            content=content,
        )
        assert part.content.get('status') == 'completed'
        assert part.content.get('revised_prompt') is None  # Optional field
