"""Anthropic API tool_choice translation unit tests.

These tests verify the translation from PydanticAI's internal tool_choice format
to the Anthropic API format. They complement the VCR-based integration tests by
explicitly asserting on the exact API values sent.

Key Anthropic API format details:
- All modes are dicts: {type: 'auto'}, {type: 'none'}, {type: 'any'}, {type: 'tool', name: X}
- 'required' maps to 'any' (Anthropic terminology)
- Single specific tool: {type: 'tool', name: X}
- Multiple specific tools: {type: 'any'} + filtered tool list
- Thinking mode has restrictions: cannot use 'any' or force specific tools
"""

from __future__ import annotations

import pytest
from inline_snapshot import snapshot

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ToolsPlusOutput
from pydantic_ai.tools import ToolDefinition, ToolKind

from ...conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='anthropic not installed')


def make_tool(name: str, *, kind: ToolKind = 'function') -> ToolDefinition:
    """Create a minimal ToolDefinition for testing."""
    return ToolDefinition(
        name=name,
        description=f'Test tool {name}',
        parameters_json_schema={'type': 'object', 'properties': {'x': {'type': 'string'}}},
        kind=kind,
    )


@pytest.fixture
def anthropic_model() -> AnthropicModel:
    """Create an Anthropic model for testing (no API calls made)."""
    return AnthropicModel('claude-sonnet-4-20250514', provider=AnthropicProvider(api_key='test-key'))


class TestToolChoiceTranslation:
    """Unit tests for Anthropic tool_choice API format translation."""

    def test_auto_format(self, anthropic_model: AnthropicModel):
        """tool_choice='auto' passes {type: 'auto'} to API."""
        settings: AnthropicModelSettings = {'tool_choice': 'auto'}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather')])

        tools, tool_choice = anthropic_model._prepare_tools_and_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]  # pyright: ignore[reportPrivateUsage]

        assert tool_choice == snapshot({'type': 'auto'})
        assert len(tools) == 1
        assert tools[0].get('name') == 'get_weather'

    def test_none_format(self, anthropic_model: AnthropicModel):
        """tool_choice='none' passes {type: 'none'} to API."""
        settings: AnthropicModelSettings = {'tool_choice': 'none'}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather')])

        tools, tool_choice = anthropic_model._prepare_tools_and_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        assert tool_choice == snapshot({'type': 'none'})
        assert len(tools) == 1

    def test_required_format(self, anthropic_model: AnthropicModel):
        """tool_choice='required' passes {type: 'any'} to API (Anthropic terminology)."""
        settings: AnthropicModelSettings = {'tool_choice': 'required'}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather')])

        tools, tool_choice = anthropic_model._prepare_tools_and_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        # 'required' maps to 'any' in Anthropic terminology
        assert tool_choice == snapshot({'type': 'any'})
        assert len(tools) == 1

    def test_single_tool_in_list_format(self, anthropic_model: AnthropicModel):
        """Single tool in list uses {type: 'tool', name: X} format."""
        settings: AnthropicModelSettings = {'tool_choice': ['get_weather']}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather'), make_tool('get_time')])

        tools, tool_choice = anthropic_model._prepare_tools_and_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        # Single tool uses named format which forces that specific tool
        assert tool_choice == snapshot({'type': 'tool', 'name': 'get_weather'})
        # All tools are sent - the {type: 'tool', name: X} enforces the restriction
        assert {t['name'] for t in tools if 'name' in t} == {'get_weather', 'get_time'}

    def test_multiple_tools_in_list_format(self, anthropic_model: AnthropicModel):
        """Multiple tools in list uses {type: 'any'} with filtered tools."""
        settings: AnthropicModelSettings = {'tool_choice': ['get_weather', 'get_time']}
        params = ModelRequestParameters(
            function_tools=[make_tool('get_weather'), make_tool('get_time'), make_tool('get_population')]
        )

        tools, tool_choice = anthropic_model._prepare_tools_and_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        # Multiple tools fall back to 'any' with filtered list
        assert tool_choice == snapshot({'type': 'any'})
        tool_names: set[str] = {t['name'] for t in tools if 'name' in t}
        assert tool_names == {'get_weather', 'get_time'}

    def test_tools_plus_output_with_single_function_tool(self, anthropic_model: AnthropicModel):
        """ToolsPlusOutput with single function tool becomes {type: 'any'} (multiple tools)."""
        settings: AnthropicModelSettings = {'tool_choice': ToolsPlusOutput(function_tools=['get_weather'])}
        params = ModelRequestParameters(
            function_tools=[make_tool('get_weather'), make_tool('get_time')],
            output_tools=[make_tool('final_result', kind='output')],
        )

        tools, tool_choice = anthropic_model._prepare_tools_and_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        # 2 tools total (get_weather + final_result) so uses 'any'
        assert tool_choice == snapshot({'type': 'any'})
        tool_names: set[str] = {t['name'] for t in tools if 'name' in t}
        assert tool_names == {'get_weather', 'final_result'}

    def test_tools_plus_output_with_multiple_function_tools(self, anthropic_model: AnthropicModel):
        """ToolsPlusOutput with multiple function tools uses {type: 'any'}."""
        settings: AnthropicModelSettings = {'tool_choice': ToolsPlusOutput(function_tools=['get_weather', 'get_time'])}
        params = ModelRequestParameters(
            function_tools=[make_tool('get_weather'), make_tool('get_time'), make_tool('get_population')],
            output_tools=[make_tool('final_result', kind='output')],
        )

        tools, tool_choice = anthropic_model._prepare_tools_and_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        assert tool_choice == snapshot({'type': 'any'})
        tool_names: set[str] = {t['name'] for t in tools if 'name' in t}
        assert tool_names == {'get_weather', 'get_time', 'final_result'}

    def test_none_with_output_tools_uses_auto_mode(self, anthropic_model: AnthropicModel):
        """tool_choice='none' with output tools returns {type: 'auto'} with filtered tools."""
        settings: AnthropicModelSettings = {'tool_choice': 'none'}
        params = ModelRequestParameters(
            function_tools=[make_tool('get_weather')],
            output_tools=[make_tool('final_result', kind='output')],
            allow_text_output=True,
        )

        with pytest.warns(UserWarning, match="tool_choice='none' but output tools"):
            tools, tool_choice = anthropic_model._prepare_tools_and_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        # Returns 'auto' to allow output tools while text is allowed
        assert tool_choice == snapshot({'type': 'auto'})
        assert [t['name'] for t in tools if 'name' in t] == ['final_result']

    def test_no_tools_returns_none(self, anthropic_model: AnthropicModel):
        """When no tools are available, returns empty list and None."""
        settings: AnthropicModelSettings = {'tool_choice': 'auto'}
        params = ModelRequestParameters(function_tools=[])

        tools, tool_choice = anthropic_model._prepare_tools_and_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        assert tools == []
        assert tool_choice is None

    def test_tool_definition_format(self, anthropic_model: AnthropicModel):
        """Verify the complete tool definition format sent to API."""
        settings: AnthropicModelSettings = {'tool_choice': 'auto'}
        params = ModelRequestParameters(function_tools=[make_tool('get_weather')])

        tools, _tool_choice = anthropic_model._prepare_tools_and_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        assert tools == snapshot(
            [
                {
                    'name': 'get_weather',
                    'description': 'Test tool get_weather',
                    'input_schema': {'type': 'object', 'properties': {'x': {'type': 'string'}}},
                }
            ]
        )


class TestThinkingModeRestrictions:
    """Tests for Anthropic thinking mode tool_choice restrictions."""

    def test_required_with_thinking_raises_error(self, anthropic_model: AnthropicModel):
        """tool_choice='required' with thinking mode raises UserError."""
        settings: AnthropicModelSettings = {
            'tool_choice': 'required',
            'anthropic_thinking': {'type': 'enabled', 'budget_tokens': 1000},
        }
        params = ModelRequestParameters(function_tools=[make_tool('get_weather')])

        with pytest.raises(UserError, match="tool_choice='required'.*thinking mode"):
            anthropic_model._prepare_tools_and_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

    def test_specific_tool_with_thinking_raises_error(self, anthropic_model: AnthropicModel):
        """Forcing specific tools with thinking mode raises UserError."""
        settings: AnthropicModelSettings = {
            'tool_choice': ['get_weather'],
            'anthropic_thinking': {'type': 'enabled', 'budget_tokens': 1000},
        }
        params = ModelRequestParameters(function_tools=[make_tool('get_weather'), make_tool('get_time')])

        with pytest.raises(UserError, match='forcing specific tools.*thinking mode'):
            anthropic_model._prepare_tools_and_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

    def test_auto_with_thinking_works(self, anthropic_model: AnthropicModel):
        """tool_choice='auto' works with thinking mode."""
        settings: AnthropicModelSettings = {
            'tool_choice': 'auto',
            'anthropic_thinking': {'type': 'enabled', 'budget_tokens': 1000},
        }
        params = ModelRequestParameters(function_tools=[make_tool('get_weather')])

        _tools, tool_choice = anthropic_model._prepare_tools_and_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        assert tool_choice == snapshot({'type': 'auto'})

    def test_none_with_thinking_works(self, anthropic_model: AnthropicModel):
        """tool_choice='none' works with thinking mode."""
        settings: AnthropicModelSettings = {
            'tool_choice': 'none',
            'anthropic_thinking': {'type': 'enabled', 'budget_tokens': 1000},
        }
        params = ModelRequestParameters(function_tools=[make_tool('get_weather')])

        _tools, tool_choice = anthropic_model._prepare_tools_and_tool_choice(settings, params)  # pyright: ignore[reportPrivateUsage]

        assert tool_choice == snapshot({'type': 'none'})
