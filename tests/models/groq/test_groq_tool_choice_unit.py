"""Groq tool_choice translation unit tests.

These tests verify the translation from PydanticAI's internal tool_choice format
to the Groq API format. They complement the VCR-based integration tests by
explicitly asserting on the exact API values sent.

Key Groq API format details:
- 'auto', 'none', 'required' are passed as strings
- Single tool: {type: 'function', function: {name: X}}
- Multiple tools: 'required' + filtered tool list
"""

from __future__ import annotations

import pytest
from inline_snapshot import snapshot

from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ToolsPlusOutput
from pydantic_ai.tools import ToolDefinition, ToolKind

from ...conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.groq import GroqModel, GroqModelSettings
    from pydantic_ai.providers.groq import GroqProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='groq not installed')


def make_tool(name: str, *, kind: ToolKind = 'function') -> ToolDefinition:
    """Create a minimal ToolDefinition for testing."""
    return ToolDefinition(
        name=name,
        description=f'Test tool {name}',
        parameters_json_schema={'type': 'object', 'properties': {'x': {'type': 'string'}}},
        kind=kind,
    )


@pytest.fixture
def groq_model() -> GroqModel:
    """Create a Groq model for testing (no API calls made)."""
    return GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(api_key='test-key'))


class TestToolChoiceTranslation:
    """Unit tests for Groq tool_choice API format translation."""

    def test_auto_format(self, groq_model: GroqModel):
        """tool_choice='auto' passes 'auto' string to API."""
        settings: GroqModelSettings = {'tool_choice': 'auto'}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather')])

        tools, tool_choice = groq_model._get_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        assert tool_choice == 'auto'
        assert len(tools) == 1
        assert tools[0]['function']['name'] == 'get_weather'

    def test_none_format(self, groq_model: GroqModel):
        """tool_choice='none' passes 'none' string to API."""
        settings: GroqModelSettings = {'tool_choice': 'none'}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather')])

        tools, tool_choice = groq_model._get_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        assert tool_choice == 'none'
        assert len(tools) == 1

    def test_required_format(self, groq_model: GroqModel):
        """tool_choice='required' passes 'required' string to API."""
        settings: GroqModelSettings = {'tool_choice': 'required'}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather')])

        tools, tool_choice = groq_model._get_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        assert tool_choice == 'required'
        assert len(tools) == 1

    def test_single_tool_in_list_format(self, groq_model: GroqModel):
        """Single tool in list uses named tool choice format."""
        settings: GroqModelSettings = {'tool_choice': ['get_weather']}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather'), make_tool('get_time')])

        tools, tool_choice = groq_model._get_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        # Explicit API format: forces the specific tool
        assert tool_choice == snapshot({'type': 'function', 'function': {'name': 'get_weather'}})
        # Tools are filtered to only include the specified tool
        assert [t['function']['name'] for t in tools] == ['get_weather']

    def test_multiple_tools_in_list_format(self, groq_model: GroqModel):
        """Multiple tools in list fall back to 'required' with filtered tools."""
        settings: GroqModelSettings = {'tool_choice': ['get_weather', 'get_time']}
        params = ModelRequestParameters(
            function_tools=[make_tool('get_weather'), make_tool('get_time'), make_tool('get_population')]
        )

        tools, tool_choice = groq_model._get_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        # Falls back to 'required' since Groq only supports forcing a single tool
        assert tool_choice == 'required'
        # Tools are filtered to only the specified ones
        tool_names = {t['function']['name'] for t in tools}
        assert tool_names == {'get_weather', 'get_time'}

    def test_tools_plus_output_with_single_function_tool(self, groq_model: GroqModel):
        """ToolsPlusOutput with single function tool uses named format."""
        settings: GroqModelSettings = {'tool_choice': ToolsPlusOutput(function_tools=['get_weather'])}
        params = ModelRequestParameters(
            function_tools=[make_tool('get_weather'), make_tool('get_time')],
            output_tools=[make_tool('final_result', kind='output')],
        )

        tools, tool_choice = groq_model._get_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        # With 2 tools (get_weather + final_result), falls back to 'required'
        assert tool_choice == 'required'
        tool_names = {t['function']['name'] for t in tools}
        assert tool_names == {'get_weather', 'final_result'}

    def test_tools_plus_output_with_multiple_function_tools(self, groq_model: GroqModel):
        """ToolsPlusOutput with multiple function tools uses 'required' with filtered tools."""
        settings: GroqModelSettings = {'tool_choice': ToolsPlusOutput(function_tools=['get_weather', 'get_time'])}
        params = ModelRequestParameters(
            function_tools=[make_tool('get_weather'), make_tool('get_time'), make_tool('get_population')],
            output_tools=[make_tool('final_result', kind='output')],
        )

        tools, tool_choice = groq_model._get_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        assert tool_choice == 'required'
        tool_names = {t['function']['name'] for t in tools}
        assert tool_names == {'get_weather', 'get_time', 'final_result'}

    def test_none_with_output_tools_uses_auto_mode(self, groq_model: GroqModel):
        """tool_choice='none' with output tools returns filtered tools with 'auto' mode."""
        settings: GroqModelSettings = {'tool_choice': 'none'}
        params = ModelRequestParameters(
            function_tools=[make_tool('get_weather')],
            output_tools=[make_tool('final_result', kind='output')],
            allow_text_output=True,
        )

        with pytest.warns(UserWarning, match="tool_choice='none' but output tools"):
            tools, tool_choice = groq_model._get_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        # Returns 'auto' to allow output tools, with only output tools in the list
        assert tool_choice == 'auto'
        assert [t['function']['name'] for t in tools] == ['final_result']

    def test_no_tools_returns_none(self, groq_model: GroqModel):
        """When no tools are available, returns empty list and None."""
        settings: GroqModelSettings = {'tool_choice': 'auto'}
        params = ModelRequestParameters(function_tools=[])

        tools, tool_choice = groq_model._get_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        assert tools == []
        assert tool_choice is None

    def test_tool_definition_format(self, groq_model: GroqModel):
        """Verify the complete tool definition format sent to API."""
        settings: GroqModelSettings = {'tool_choice': 'auto'}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather')])

        tools, _tool_choice = groq_model._get_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        assert tools == snapshot(
            [
                {
                    'type': 'function',
                    'function': {
                        'name': 'get_weather',
                        'description': 'Test tool get_weather',
                        'parameters': {'type': 'object', 'properties': {'x': {'type': 'string'}}},
                    },
                }
            ]
        )
