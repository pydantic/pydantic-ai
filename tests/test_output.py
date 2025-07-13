"""Tests for the TraceContext class and related output processing functionality.

This test module provides comprehensive coverage of the TraceContext class which is responsible
for managing OpenTelemetry tracing and span creation during output processing in pydantic-ai.

The tests cover:
- Basic initialization and property access
- Tool call context management (with_call context manager)
- Span creation for both tool calls and direct function calls
- Response serialization for various data types (Pydantic models, primitives, etc.)
- JSON schema generation for span attributes
- Error handling and edge cases
- Integration with the build_trace_context function
"""

import json
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from pydantic_ai._output import TraceContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ToolCallPart


class WeatherInfo(BaseModel):
    temperature: float
    description: str


# pyright: reportPrivateUsage=false
class TestTraceContext:
    """Test cases for the TraceContext class."""

    def test_init_basic(self):
        """Test basic initialization of TraceContext."""
        mock_tracer = Mock()
        trace_context = TraceContext(tracer=mock_tracer, include_content=True)

        assert trace_context.tracer is mock_tracer
        assert trace_context.include_content is True
        assert not trace_context.has_call()

    def test_init_default_include_content(self):
        """Test TraceContext initialization with default include_content."""
        mock_tracer = Mock()
        trace_context = TraceContext(tracer=mock_tracer)

        assert trace_context.tracer is mock_tracer
        assert trace_context.include_content is False
        assert not trace_context.has_call()

    def test_properties(self):
        """Test TraceContext property access."""
        mock_tracer = Mock()
        trace_context = TraceContext(tracer=mock_tracer, include_content=True)

        assert trace_context.tracer is mock_tracer
        assert trace_context.include_content is True

    def test_call_property_no_call_set(self):
        """Test accessing call property when no call is set raises UserError."""
        mock_tracer = Mock()
        trace_context = TraceContext(tracer=mock_tracer)

        with pytest.raises(UserError, match='No tool call is set in the trace context.'):
            _ = trace_context.call

    def test_has_call_false_when_no_call(self):
        """Test has_call returns False when no call is set."""
        mock_tracer = Mock()
        trace_context = TraceContext(tracer=mock_tracer)

        assert not trace_context.has_call()

    def test_has_call_true_when_call_set(self):
        """Test has_call returns True when call is set via context manager."""
        mock_tracer = Mock()
        trace_context = TraceContext(tracer=mock_tracer)
        tool_call = ToolCallPart(tool_name='test_tool', args={'arg1': 'value1'})

        with trace_context.with_call(tool_call):
            assert trace_context.has_call()
            assert trace_context.call is tool_call

    def test_with_call_context_manager(self):
        """Test with_call context manager properly sets and clears call."""
        mock_tracer = Mock()
        trace_context = TraceContext(tracer=mock_tracer)
        tool_call = ToolCallPart(tool_name='test_tool', args={'arg1': 'value1'})

        # Before context manager
        assert not trace_context.has_call()

        # Inside context manager
        with trace_context.with_call(tool_call):
            assert trace_context.has_call()
            assert trace_context.call is tool_call

        # After context manager
        assert not trace_context.has_call()

    def test_with_call_nested_error(self):
        """Test that nesting with_call raises UserError."""
        mock_tracer = Mock()
        trace_context = TraceContext(tracer=mock_tracer)
        tool_call1 = ToolCallPart(tool_name='tool1', args={'arg1': 'value1'})
        tool_call2 = ToolCallPart(tool_name='tool2', args={'arg2': 'value2'})

        with trace_context.with_call(tool_call1):
            with pytest.raises(UserError, match='Cannot set a tool call while another one is already set.'):
                with trace_context.with_call(tool_call2):
                    pass

    def test_span_for_tool_call_function_without_call(self):
        """Test span_for_tool_call_function raises error when no call is set."""
        mock_tracer = Mock()
        trace_context = TraceContext(tracer=mock_tracer)

        with pytest.raises(UserError, match='Cannot create tool call span without a tool call context.'):
            with trace_context.span_for_tool_call_function('test_function', '{"arg": "value"}'):
                pass

    def test_span_for_tool_call_function_with_call(self):
        """Test span_for_tool_call_function creates span with proper attributes."""
        mock_tracer = Mock()
        mock_span = Mock()
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_span)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_tracer.start_as_current_span.return_value = mock_context_manager

        trace_context = TraceContext(tracer=mock_tracer, include_content=True)
        tool_call = ToolCallPart(tool_name='weather_tool', args={'location': 'London'}, tool_call_id='call_123')

        with trace_context.with_call(tool_call):
            with trace_context.span_for_tool_call_function('get_weather', '{"location": "London"}') as span:
                assert span is mock_span

        # Verify span was created with correct attributes
        mock_tracer.start_as_current_span.assert_called_once()
        call_args = mock_tracer.start_as_current_span.call_args
        assert call_args[0][0] == 'running output function'

        expected_attributes = call_args[1]['attributes']
        assert expected_attributes['gen_ai.tool.name'] == 'get_weather'
        assert expected_attributes['gen_ai.tool.call.id'] == 'call_123'
        assert expected_attributes['tool_arguments'] == '{"location": "London"}'
        assert expected_attributes['logfire.msg'] == 'running output function: weather_tool'

    def test_span_for_tool_call_function_without_content(self):
        """Test span_for_tool_call_function excludes tool_arguments when include_content=False."""
        mock_tracer = Mock()
        mock_span = Mock()
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_span)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_tracer.start_as_current_span.return_value = mock_context_manager

        trace_context = TraceContext(tracer=mock_tracer, include_content=False)
        tool_call = ToolCallPart(tool_name='weather_tool', args={'location': 'London'}, tool_call_id='call_123')

        with trace_context.with_call(tool_call):
            with trace_context.span_for_tool_call_function('get_weather', '{"location": "London"}'):
                pass

        call_args = mock_tracer.start_as_current_span.call_args
        expected_attributes = call_args[1]['attributes']
        assert 'tool_arguments' not in expected_attributes

    def test_span_for_direct_function_call(self):
        """Test span_for_direct_function_call creates span with proper attributes."""
        mock_tracer = Mock()
        mock_span = Mock()
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_span)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_tracer.start_as_current_span.return_value = mock_context_manager

        trace_context = TraceContext(tracer=mock_tracer, include_content=True)

        with trace_context.span_for_direct_function_call('process_text', '{"text": "hello"}') as span:
            assert span is mock_span

        call_args = mock_tracer.start_as_current_span.call_args
        expected_attributes = call_args[1]['attributes']
        assert expected_attributes['tool_arguments'] == '{"text": "hello"}'
        assert expected_attributes['logfire.msg'] == 'running output function: process_text'
        # Should not include gen_ai.tool.name or gen_ai.tool.call.id for direct calls
        assert 'gen_ai.tool.name' not in expected_attributes
        assert 'gen_ai.tool.call.id' not in expected_attributes

    def test_record_response_pydantic_model(self):
        """Test record_response serializes Pydantic models correctly."""
        mock_tracer = Mock()
        mock_span = Mock()
        mock_span.is_recording.return_value = True

        trace_context = TraceContext(tracer=mock_tracer, include_content=True)
        weather = WeatherInfo(temperature=25.5, description='sunny')

        trace_context.record_response(mock_span, weather)

        mock_span.set_attribute.assert_called_once_with(
            'tool_response', '{"temperature": 25.5, "description": "sunny"}'
        )

    def test_record_response_dict_like_object(self):
        """Test record_response handles objects with dict() method."""
        mock_tracer = Mock()
        mock_span = Mock()
        mock_span.is_recording.return_value = True

        trace_context = TraceContext(tracer=mock_tracer, include_content=True)

        # Mock object with dict() method that doesn't have model_dump
        class MockDictObject:
            def dict(self):
                return {'key': 'value'}

        mock_obj = MockDictObject()
        trace_context.record_response(mock_span, mock_obj)

        mock_span.set_attribute.assert_called_once_with('tool_response', '{"key": "value"}')

    def test_record_response_primitive_types(self):
        """Test record_response handles primitive types correctly."""
        mock_tracer = Mock()
        mock_span = Mock()
        mock_span.is_recording.return_value = True

        trace_context = TraceContext(tracer=mock_tracer, include_content=True)

        # Test string
        trace_context.record_response(mock_span, 'hello')
        mock_span.set_attribute.assert_called_with('tool_response', 'hello')

        # Test integer
        mock_span.reset_mock()
        trace_context.record_response(mock_span, 42)
        mock_span.set_attribute.assert_called_with('tool_response', '42')

        # Test float
        mock_span.reset_mock()
        trace_context.record_response(mock_span, 3.14)
        mock_span.set_attribute.assert_called_with('tool_response', '3.14')

        # Test boolean
        mock_span.reset_mock()
        trace_context.record_response(mock_span, True)
        mock_span.set_attribute.assert_called_with('tool_response', 'True')

    def test_record_response_json_serializable_object(self):
        """Test record_response handles JSON-serializable objects."""
        mock_tracer = Mock()
        mock_span = Mock()
        mock_span.is_recording.return_value = True

        trace_context = TraceContext(tracer=mock_tracer, include_content=True)
        data = {'key': 'value', 'number': 123}

        trace_context.record_response(mock_span, data)

        mock_span.set_attribute.assert_called_once_with('tool_response', '{"key": "value", "number": 123}')

    def test_record_response_non_serializable_object(self):
        """Test record_response handles non-JSON-serializable objects with fallback."""
        mock_tracer = Mock()
        mock_span = Mock()
        mock_span.is_recording.return_value = True

        trace_context = TraceContext(tracer=mock_tracer, include_content=True)

        # Object that can't be serialized to JSON
        class NonSerializable:
            def __str__(self):
                return 'NonSerializable instance'

        obj = NonSerializable()
        trace_context.record_response(mock_span, obj)

        mock_span.set_attribute.assert_called_once_with('tool_response', 'NonSerializable instance')

    def test_record_response_exception_fallback(self):
        """Test record_response handles exceptions with fallback."""
        mock_tracer = Mock()
        mock_span = Mock()
        mock_span.is_recording.return_value = True

        trace_context = TraceContext(tracer=mock_tracer, include_content=True)

        # Object that raises exception during serialization
        class ProblematicObject:
            def model_dump(self):
                raise ValueError("Can't serialize")

        obj = ProblematicObject()
        trace_context.record_response(mock_span, obj)

        mock_span.set_attribute.assert_called_once_with('tool_response', '<ProblematicObject object>')

    def test_record_response_include_content_false(self):
        """Test record_response does nothing when include_content=False."""
        mock_tracer = Mock()
        mock_span = Mock()
        mock_span.is_recording.return_value = True

        trace_context = TraceContext(tracer=mock_tracer, include_content=False)
        weather = WeatherInfo(temperature=25.5, description='sunny')

        trace_context.record_response(mock_span, weather)

        mock_span.set_attribute.assert_not_called()

    def test_record_response_span_not_recording(self):
        """Test record_response does nothing when span is not recording."""
        mock_tracer = Mock()
        mock_span = Mock()
        mock_span.is_recording.return_value = False

        trace_context = TraceContext(tracer=mock_tracer, include_content=True)
        weather = WeatherInfo(temperature=25.5, description='sunny')

        trace_context.record_response(mock_span, weather)

        mock_span.set_attribute.assert_not_called()

    def test_build_json_schema_with_content_and_tool_attrs(self):
        """Test _build_json_schema includes all properties when include_content=True and include_tool_attrs=True."""
        mock_tracer = Mock()
        trace_context = TraceContext(tracer=mock_tracer, include_content=True)

        schema_str = trace_context._build_json_schema(include_tool_attrs=True)
        schema = json.loads(schema_str)

        expected_properties = {
            'tool_arguments': {'type': 'object'},
            'tool_response': {'type': 'object'},
            'gen_ai.tool.name': {},
            'gen_ai.tool.call.id': {},
        }

        assert schema['type'] == 'object'
        assert schema['properties'] == expected_properties

    def test_build_json_schema_with_content_no_tool_attrs(self):
        """Test _build_json_schema excludes tool attributes when include_tool_attrs=False."""
        mock_tracer = Mock()
        trace_context = TraceContext(tracer=mock_tracer, include_content=True)

        schema_str = trace_context._build_json_schema(include_tool_attrs=False)
        schema = json.loads(schema_str)

        expected_properties = {'tool_arguments': {'type': 'object'}, 'tool_response': {'type': 'object'}}

        assert schema['type'] == 'object'
        assert schema['properties'] == expected_properties

    def test_build_json_schema_no_content_with_tool_attrs(self):
        """Test _build_json_schema includes only tool attributes when include_content=False."""
        mock_tracer = Mock()
        trace_context = TraceContext(tracer=mock_tracer, include_content=False)

        schema_str = trace_context._build_json_schema(include_tool_attrs=True)
        schema = json.loads(schema_str)

        expected_properties = {'gen_ai.tool.name': {}, 'gen_ai.tool.call.id': {}}  # type: ignore

        assert schema['type'] == 'object'
        assert schema['properties'] == expected_properties

    def test_build_json_schema_no_content_no_tool_attrs(self):
        """Test _build_json_schema returns minimal schema when both flags are False."""
        mock_tracer = Mock()
        trace_context = TraceContext(tracer=mock_tracer, include_content=False)

        schema_str = trace_context._build_json_schema(include_tool_attrs=False)
        schema = json.loads(schema_str)

        assert schema['type'] == 'object'
        assert schema['properties'] == {}

    def test_record_response_with_json_import_error(self):
        """Test record_response handles json import issues gracefully."""
        mock_tracer = Mock()
        mock_span = Mock()
        mock_span.is_recording.return_value = True

        trace_context = TraceContext(tracer=mock_tracer, include_content=True)

        # Mock an object that will cause issues during JSON serialization
        class ProblematicJsonObject:
            def model_dump(self):
                # This should simulate a situation where json.dumps fails
                return {'circular': self}  # circular reference

        obj = ProblematicJsonObject()
        trace_context.record_response(mock_span, obj)

        # Should fall back to the exception handling
        mock_span.set_attribute.assert_called_once_with('tool_response', '<ProblematicJsonObject object>')

    def test_with_call_exception_cleanup(self):
        """Test that with_call properly cleans up even if an exception occurs."""
        mock_tracer = Mock()
        trace_context = TraceContext(tracer=mock_tracer)
        tool_call = ToolCallPart(tool_name='test_tool', args={'arg1': 'value1'})

        try:
            with trace_context.with_call(tool_call):
                assert trace_context.has_call()
                raise ValueError('Test exception')
        except ValueError:
            pass

        # Should be cleaned up even after exception
        assert not trace_context.has_call()

    def test_record_response_none_value(self):
        """Test record_response handles None values correctly."""
        mock_tracer = Mock()
        mock_span = Mock()
        mock_span.is_recording.return_value = True

        trace_context = TraceContext(tracer=mock_tracer, include_content=True)

        trace_context.record_response(mock_span, None)

        mock_span.set_attribute.assert_called_once_with('tool_response', 'null')

    def test_record_response_list_value(self):
        """Test record_response handles list values correctly."""
        mock_tracer = Mock()
        mock_span = Mock()
        mock_span.is_recording.return_value = True

        trace_context = TraceContext(tracer=mock_tracer, include_content=True)

        trace_context.record_response(mock_span, [1, 2, 3])

        mock_span.set_attribute.assert_called_once_with('tool_response', '[1, 2, 3]')

    def test_span_attributes_json_schema_structure(self):
        """Test that JSON schema is properly structured in span attributes."""
        mock_tracer = Mock()
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=Mock())
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_tracer.start_as_current_span.return_value = mock_context_manager

        trace_context = TraceContext(tracer=mock_tracer, include_content=True)
        tool_call = ToolCallPart(tool_name='test_tool', tool_call_id='call_123')

        with trace_context.with_call(tool_call):
            with trace_context.span_for_tool_call_function('test_func', '{}'):
                pass

        call_args = mock_tracer.start_as_current_span.call_args
        json_schema_str = call_args[1]['attributes']['logfire.json_schema']

        # Verify it's valid JSON
        schema = json.loads(json_schema_str)
        assert schema['type'] == 'object'
        assert 'properties' in schema

    def test_build_trace_context_function(self):
        """Test the build_trace_context function."""
        from unittest.mock import Mock

        from pydantic_ai._output import build_trace_context

        # Mock the GraphRunContext and its dependencies
        mock_ctx = Mock()
        mock_deps = Mock()
        mock_tracer = Mock()
        mock_instrumentation_settings = Mock()
        mock_instrumentation_settings.include_content = True

        mock_deps.tracer = mock_tracer
        mock_deps.instrumentation_settings = mock_instrumentation_settings
        mock_ctx.deps = mock_deps

        result = build_trace_context(mock_ctx)

        assert isinstance(result, TraceContext)
        assert result.tracer is mock_tracer
        assert result.include_content is True

    def test_build_trace_context_no_instrumentation_settings(self):
        """Test build_trace_context when instrumentation_settings is None."""
        from unittest.mock import Mock

        from pydantic_ai._output import build_trace_context

        mock_ctx = Mock()
        mock_deps = Mock()
        mock_tracer = Mock()

        mock_deps.tracer = mock_tracer
        mock_deps.instrumentation_settings = None
        mock_ctx.deps = mock_deps

        result = build_trace_context(mock_ctx)

        assert isinstance(result, TraceContext)
        assert result.tracer is mock_tracer
        assert result.include_content is False
